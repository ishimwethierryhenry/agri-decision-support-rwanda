"""Third round of tuning (training protocol 1.4, section 13): a steadier comparison of the settings, crop weights at
decision time, and a blend whose members are chosen inside the training part.

Version 1.3 (tune_v2.py) showed that a wider search alone can choose worse settings, because the comparison inside
the training part is one noisy split. So this round keeps the candidates of version 1.3 and changes what is done with
them, always inside the training part of each outer fold:
  1. Steadier comparison (new-fields test): the 8 candidates with the best inner score of version 1.3 (seed 0, from
     tuning_v2/inner_scores.csv) are scored again on two more splits into 3 block folds (seeds 1 and 2), and the best
     mean over the three splits wins (a tie goes to the candidate drawn first). The main test keeps the version 1.3
     choice: its inner comparison leaves each training district out and has no random split.
  2. Crop weights: the chosen model's inner out-of-fold probabilities are multiplied by one weight per crop (2/3, 1 or
     3/2, 81 combinations) and the best inner main score wins (a tie goes to no change).
  3. Blend: among the 7 groups of the three tuned models, the group whose averaged inner probabilities score best wins
     (a tie goes to the smaller group, then Random Forest, Extra Trees, XGBoost); then its crop weights, as in step 2.

Methods Random Forest, Extra Trees and XGBoost on the views field_sat and all_sat, both nested tests. Every (view,
test, outer fold) is one task, saved in <out>/_work, so a stopped run starts again where it stopped. The built-in
check fits the version 1.3 choice of every fold again and must give exactly the tuning_v2 scores.

Run from the repo root (about an hour on two cores):
    python 01-crop-recommendation/src/models/tune_v3.py
Options: --workers N (default 2), --quick (a tiny run into a temporary folder, to check the code; its numbers are
never reported), --assemble-only (rebuild the report from the saved tasks), --final (after the run: the settings and
crop weights of the model the map serves, chosen on all fields), --table, --out
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import platform
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import tune as r1  # noqa: E402
import tune_v2 as t2  # noqa: E402

PROTOCOL_VERSION = "1.4"
TABLE = r1.TABLE
DEFAULT_OUT = ev.REPORTS / "tuning_v3"
ROUND1_DIR, ROUND2_DIR = ev.REPORTS / "tuning_v1", ev.REPORTS / "tuning_v2"
SEED = ev.SEED
CROPS, LABELS, TUNED = r1.CROPS, r1.LABELS, r1.TUNED
METHODS = ("RF", "ET", "XGB")
VIEWS = ("field_sat", "all_sat")
TOP_K = 8                                   # candidates compared again on the extra splits
EXTRA_SEEDS = (1, 2)                        # the two extra splits into 3 block folds
WEIGHT_STEPS = (1.0, 2 / 3, 1.5)            # 1.0 first, so the all-ones combination comes first and wins a tie
GROUPS = [g for n in (1, 2, 3) for g in itertools.combinations(METHODS, n)]   # smaller groups first
MAP_MODEL = "RF[all_sat]" + TUNED           # the model on the map since 23 Nov (tuning_v1)
MAP_MARGIN, TOP1_MARGIN = 0.01, 0.01
VARIANTS = ("v1.3", "steady", "steady+weights")
BLEND = "Blend"
# The baselines do not depend on this round: they are copied from tuning_v2 (same rows, same folds), so that this folder
# holds everything the dashboard map reads about the model on the map and the rules.
REFERENCES = ("B0_majority", "B0d_district_majority", ev.BASELINE)
FINAL_FILE = "final_model.json"


# ------------------------------------------------------------------ small steps
def weight_grid() -> list[np.ndarray]:
    """The 81 combinations of one weight per crop. The first one is all ones (no change)."""
    return [np.array(w) for w in itertools.product(WEIGHT_STEPS, repeat=len(CROPS))]


def best_weights(df: pd.DataFrame, proba: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The crop weights with the best inner main score on these out-of-fold probabilities (columns follow CROPS).
    A tie goes to the combination listed first, so no change wins every tie. Returns weights, score, score with no
    change."""
    scores = [r1.block_score(df, proba * w) for w in weight_grid()]
    i = r1.pick(scores)
    return weight_grid()[i], float(scores[i]), float(scores[0])


def best_group(df: pd.DataFrame, inner: dict[str, np.ndarray]) -> tuple[tuple, list[float]]:
    """The group of tuned models whose averaged inner probabilities score best. A tie goes to the group listed first:
    the smaller group, then Random Forest, Extra Trees, XGBoost."""
    scores = [r1.block_score(df, np.mean([inner[m] for m in g], axis=0)) for g in GROUPS]
    return GROUPS[r1.pick(scores)], [float(s) for s in scores]


def apply_weights(proba: np.ndarray, weights) -> np.ndarray:
    """Probabilities times one weight per crop, normalised again so each row sums to 1. The first crop is the same
    as without normalising; the map also needs the gap between the first two crops, so it uses this form."""
    p = np.asarray(proba) * np.asarray(weights, dtype=float)
    total = p.sum(axis=1, keepdims=True)
    return np.divide(p, total, out=np.zeros_like(p), where=total > 0)


def _outer_split(df: pd.DataFrame, test: str, outer) -> tuple[np.ndarray, np.ndarray, str]:
    if test == "blockcv":
        tr, te = ev.block_folds(df)[outer]
        return tr, te, f"fold {outer + 1}"
    dist = df[ev.GROUP_COL].to_numpy()
    return np.flatnonzero(dist != outer), np.flatnonzero(dist == outer), outer


def _fit_proba(method: str, params: dict, view: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    m = t2.Tunable(method, params, view).fit(train, train["label"])
    out = np.zeros((len(test), len(CROPS)))
    out[:, [CROPS.index(c) for c in m.classes_]] = m.predict_proba(test)
    return out


# ------------------------------------------------------------------ one task: a view, a test, an outer fold
def run_task(view: str, test: str, outer, df: pd.DataFrame, cands: dict, inner13: pd.DataFrame,
             chosen13: pd.DataFrame, top_k: int = TOP_K, log: Callable = print) -> dict:
    t0 = time.time()
    tr, te, outer_name = _outer_split(df, test, outer)
    train, testdf = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    members, inner = {}, {}
    for method in METHODS:
        name = f"{method}[{view}]"
        c13 = int(chosen13[(chosen13["test"] == test) & (chosen13["model"] == name)
                           & (chosen13["outer"] == outer_name)]["candidate"].iloc[0])

        def make(c, method=method):
            return lambda: t2.Tunable(method, cands[method][c], view)

        if test == "blockcv":
            s0 = inner13[(inner13["test"] == test) & (inner13["model"] == name) & (inner13["outer"] == outer_name)]
            s0 = s0.set_index("candidate")["inner_score"].sort_index()
            assert r1.pick(s0.tolist()) == c13, (name, outer_name, "version 1.3 choice not reproduced")
            top = [int(c) for c in sorted(s0.index, key=lambda c: (-s0[c], c))[:top_k]]
            cache, means = {}, {}
            for c in top:
                vals = [float(s0[c])]
                for seed in EXTRA_SEEDS:
                    p = r1.oof_proba(train, ev.block_folds(train, n_splits=r1.INNER_BLOCK_FOLDS, seed=seed), make(c))
                    cache[(c, seed)] = p
                    vals.append(r1.block_score(train, p))
                means[c] = float(np.mean(vals))
            chosen = min((c for c in top if means[c] == max(means.values())))
            p0 = r1.oof_proba(train, ev.block_folds(train, n_splits=r1.INNER_BLOCK_FOLDS, seed=SEED), make(chosen))
            inner_p = np.mean([p0] + [cache[(chosen, s)] for s in EXTRA_SEEDS], axis=0)
        else:                                   # the main test keeps the version 1.3 choice
            chosen, means, top = c13, {}, [c13]
            inner_p = r1.oof_proba(train, r1.lodo_folds(train), make(c13))
        w, w_score, w_none = best_weights(train, inner_p)
        p_steady = _fit_proba(method, cands[method][chosen], view, train, testdf)
        p13 = p_steady if chosen == c13 else _fit_proba(method, cands[method][c13], view, train, testdf)
        members[method] = {"candidate_v13": c13, "candidate": chosen, "top": top, "means": means,
                           "weights": w.tolist(), "inner_score": w_none, "inner_score_weighted": w_score,
                           "proba_v13": p13, "proba": p_steady}
        inner[method] = inner_p
        log(f"{name} {test} {outer_name}: candidate {chosen} (version 1.3: {c13}), weights "
            f"{[round(float(x), 2) for x in w]}, inner {w_none:.3f}, with weights {w_score:.3f}")
    group, group_scores = best_group(train, inner)
    blend_inner = np.mean([inner[m] for m in group], axis=0)
    bw, bw_score, bw_none = best_weights(train, blend_inner)
    blend = {"group": list(group), "group_scores": group_scores, "weights": bw.tolist(), "inner_score": bw_none,
             "inner_score_weighted": bw_score, "proba": np.mean([members[m]["proba"] for m in group], axis=0)}
    log(f"Blend[{view}] {test} {outer_name}: {'+'.join(group)}, weights {[round(float(x), 2) for x in bw]} "
        f"({time.time() - t0:.0f} s)")
    return {"view": view, "test": test, "outer": outer_name, "te": te, "members": members, "blend": blend,
            "seconds": time.time() - t0}


def _task_file(work: Path, view: str, test: str, outer) -> Path:
    return work / f"{view}_{test}_{str(outer).replace(' ', '_')}.pkl"


def _task_to_file(view, test, outer, df, cands, inner13, chosen13, top_k, work: Path) -> str:
    r1._limit_threads()
    part = run_task(view, test, outer, df, cands, inner13, chosen13, top_k, log=lambda s: print(s, flush=True))
    tmp = _task_file(work, view, test, outer).with_suffix(".part")
    with open(tmp, "wb") as fh:
        pickle.dump(part, fh)
    os.replace(tmp, _task_file(work, view, test, outer))
    return f"{view} {test} {part['outer']} done in {part['seconds'] / 60:.1f} min"


def tasks_for(df: pd.DataFrame, views, folds: Optional[list] = None, districts: Optional[list] = None) -> list:
    folds = list(range(ev.N_BLOCK_FOLDS)) if folds is None else folds
    districts = sorted(df[ev.GROUP_COL].unique()) if districts is None else districts
    return [(v, "blockcv", f) for v in views for f in folds] + [(v, "lodo", d) for v in views for d in districts]


def run_tasks(df: pd.DataFrame, work: Path, tasks: list, cands: dict, inner13: pd.DataFrame, chosen13: pd.DataFrame,
              top_k: int, workers: int, log: Callable = print) -> list[dict]:
    work.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not _task_file(work, *t).exists()]
    if len(todo) < len(tasks):
        log(f"{len(tasks) - len(todo)} of {len(tasks)} tasks already done in {work}, kept")
    todo.sort(key=lambda t: (t[1] != "blockcv", t))             # the slow new-fields tasks first
    if workers <= 1:
        for t in todo:
            log(_task_to_file(*t, df, cands, inner13, chosen13, top_k, work))
    elif todo:
        import multiprocessing
        os.environ.update(r1.SINGLE_THREAD_ENV)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_task_to_file, *t, df, cands, inner13, chosen13, top_k, work): t for t in todo}
            for f in as_completed(futures):
                log(f.result())
    parts = []
    for t in tasks:
        with open(_task_file(work, *t), "rb") as fh:
            parts.append(pickle.load(fh))
    return parts


# ------------------------------------------------------------------ results
def _scores(name: str, test: str, df: pd.DataFrame, idx: np.ndarray, proba: np.ndarray) -> list[dict]:
    y, dist = df["label"].to_numpy()[idx], df[ev.GROUP_COL].to_numpy()[idx]
    pred = LABELS[np.argmax(proba, axis=1)]
    rows = []
    for d in sorted(set(dist)):
        m = dist == d
        rows.append({"test": test, "model": name, "district": d, "n_test": int(m.sum()),
                     **ev.scores(y[m], pred[m], proba[m], LABELS)})
    return rows


def collect(df: pd.DataFrame, parts: list[dict]) -> dict:
    """Outer probabilities of every variant, put together over the outer folds of each test, and their scores."""
    stacks: dict = {}
    for p in parts:
        key = (p["test"], p["view"])
        st = stacks.setdefault(key, {"idx": [], "probas": {}})
        st["idx"].append(p["te"])
        for m, r in p["members"].items():
            w = np.array(r["weights"])
            for variant, proba in (("v1.3", r["proba_v13"]), ("steady", r["proba"]),
                                   ("steady+weights", apply_weights(r["proba"], w))):
                st["probas"].setdefault(f"{m}[{p['view']}] {variant}", []).append(proba)
        b = p["blend"]
        st["probas"].setdefault(f"{BLEND}[{p['view']}] chosen inside", []).append(b["proba"])
        st["probas"].setdefault(f"{BLEND}[{p['view']}] chosen inside+weights", []).append(
            apply_weights(b["proba"], b["weights"]))
    rows, conf, oof = [], [], {}
    y, dist = df["label"].to_numpy(), df[ev.GROUP_COL].to_numpy()
    for (test, view), st in stacks.items():
        idx = np.concatenate(st["idx"])
        for name, chunks in st["probas"].items():
            proba = np.vstack(chunks)
            rows += _scores(name, test, df, idx, proba)
            pred = LABELS[np.argmax(proba, axis=1)]
            for d in sorted(set(dist[idx])):
                m = dist[idx] == d
                conf.append(ev._confusion("main", r1.TESTS[test], name, d, y[idx][m], pred[m]))
            full = np.zeros((len(df), len(CROPS)))
            full[idx] = proba
            oof[(test, name)] = full
    return {"results": pd.DataFrame(rows), "confusion": pd.concat(conf, ignore_index=True), "oof": oof}


def references() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The baselines of tuning_v2 on the same folds (B0, the district's most common crop, B1)."""
    rows = [pd.read_csv(ROUND2_DIR / f"main_{t}_results.csv").assign(test=t) for t in ("blockcv", "lodo")]
    rows = pd.concat(rows, ignore_index=True)
    rows = rows[rows["model"].isin(REFERENCES)][["test", "model", "district", "n_test"]
                                               + [c for c in rows.columns if c not in
                                                  ("test", "model", "district", "n_test", "version", "candidate",
                                                   "n_train")]]
    conf = pd.read_csv(ROUND2_DIR / "confusion_counts.csv")
    return rows.reset_index(drop=True), conf[conf["model"].isin(REFERENCES)].reset_index(drop=True)


def summarise(results: pd.DataFrame, b1: pd.Series) -> pd.DataFrame:
    out = []
    for (test, model), g in results.groupby(["test", "model"]):
        g = g.set_index("district")
        ahead = int((g["macro_f1"] > b1.reindex(g.index) + ev.ADOPT_MARGIN).sum()) if test == "lodo" else None
        out.append({"test": test, "model": model, "macro_f1_mean": g["macro_f1"].mean(),
                    "worst_district_macro_f1": g["macro_f1"].min(),
                    "top1_all_fields": float((g["top1"] * g["n_test"]).sum() / g["n_test"].sum()),
                    "districts_beating_B1": ahead,
                    "passes_rule": (None if test != "lodo" else
                                    bool(g["macro_f1"].mean() > b1.mean() and ahead >= 3))})
    return pd.DataFrame(out)


def builtin_check(results: pd.DataFrame) -> dict:
    """The version 1.3 choice of every fold, fitted again, must give exactly the tuning_v2 scores."""
    mism, n = [], 0
    for test, fname in (("blockcv", "main_blockcv_results.csv"), ("lodo", "main_lodo_results.csv")):
        ref = pd.read_csv(ROUND2_DIR / fname).set_index(["model", "district"])["macro_f1"]
        ours = results[(results["test"] == test) & results["model"].str.endswith(" v1.3")]
        for r in ours.itertuples():
            key = (r.model.replace(" v1.3", TUNED), r.district)
            n += 1
            if key not in ref.index or not np.isclose(ref[key], r.macro_f1, rtol=0, atol=1e-9):
                mism.append({"test": test, "model": r.model, "district": r.district, "ours": r.macro_f1,
                             "tuning_v2": ref.get(key)})
    return {"compared_with": "01-crop-recommendation/reports/tuning_v2", "rows_compared": n, "mismatches": mism,
            "passed": n > 0 and not mism}


def decisions(summ: pd.DataFrame) -> dict:
    r1sum = pd.read_csv(ROUND1_DIR / "main_blockcv_summary.csv").set_index("model")
    today, today_top1 = float(r1sum.loc[MAP_MODEL, "macro_f1_mean"]), float(r1sum.loc[MAP_MODEL, "top1_all_fields"])
    ours = ~summ["model"].str.endswith(" v1.3") & ~summ["model"].isin(REFERENCES)
    new = summ[(summ["test"] == "blockcv") & ours].set_index("model")
    beating = new[(new["macro_f1_mean"] > today + MAP_MARGIN) & (new["top1_all_fields"] >= today_top1 - TOP1_MARGIN)]
    lodo = summ[(summ["test"] == "lodo") & ours].set_index("model")
    best = new["macro_f1_mean"].idxmax()
    return {"map": {"today": MAP_MODEL, "today_score": today, "today_top1": today_top1, "margin": MAP_MARGIN,
                    "top1_margin": TOP1_MARGIN, "best": best, "best_score": float(new.loc[best, "macro_f1_mean"]),
                    "best_top1": float(new.loc[best, "top1_all_fields"]),
                    "beating": {m: float(v) for m, v in beating["macro_f1_mean"].items()},
                    "changes": not beating.empty,
                    "serve": beating["macro_f1_mean"].idxmax() if not beating.empty else MAP_MODEL},
            "adoption": {"passing": sorted(lodo.index[lodo["passes_rule"] == True]),     # noqa: E712
                         "best": lodo["macro_f1_mean"].idxmax(),
                         "best_score": float(lodo["macro_f1_mean"].max())}}


# ------------------------------------------------------------------ report
def label(model: str) -> str:
    """'ET[all_sat] steady+weights' becomes 'Extra Trees, all_sat, steadier choice + crop weights'."""
    core, _, variant = model.partition(" ")
    method, view = core.rstrip("]").split("[")
    names = {"v1.3": "version 1.3 tuned", "steady": "steadier choice", "steady+weights": "steadier choice + crop weights",
             "chosen inside": "group chosen inside", "chosen inside+weights": "group chosen inside + crop weights"}
    return f"{t2.METHOD_NAMES.get(method, 'Blend')}, {view}, {names.get(variant, variant)}"


def _md(frame: pd.DataFrame, index: bool = False) -> str:
    return r1._md(frame, index=index)


def write_report(out: Path, info: dict, res: dict, parts: list[dict]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    results, summ, check, dec = res["results"], res["summary"], res["check"], res["decisions"]
    for test, fname in (("blockcv", "main_blockcv"), ("lodo", "main_lodo")):
        results[results["test"] == test].to_csv(out / f"{fname}_results.csv", index=False)
        summ[summ["test"] == test].to_csv(out / f"{fname}_summary.csv", index=False)
    choices = []
    for p in parts:
        for m, r in p["members"].items():
            choices.append({"test": p["test"], "model": f"{m}[{p['view']}]", "outer": p["outer"],
                            "candidate_v13": r["candidate_v13"], "candidate": r["candidate"],
                            "weights": json.dumps([round(x, 4) for x in r["weights"]]),
                            "inner_score": r["inner_score"], "inner_score_weighted": r["inner_score_weighted"],
                            "top": json.dumps(r["top"]), "means": json.dumps({str(k): v for k, v in r["means"].items()})})
        b = p["blend"]
        choices.append({"test": p["test"], "model": f"{BLEND}[{p['view']}]", "outer": p["outer"],
                        "group": "+".join(b["group"]), "weights": json.dumps([round(x, 4) for x in b["weights"]]),
                        "inner_score": b["inner_score"], "inner_score_weighted": b["inner_score_weighted"]})
    ch = pd.DataFrame(choices)
    ch.to_csv(out / "choices.csv", index=False)
    res["confusion"].to_csv(out / "confusion_counts.csv", index=False)
    res["analyses"]["bands"].to_csv(out / "confidence_bands.csv", index=False)
    res["analyses"]["sure"].to_csv(out / "answer_when_sure.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")

    def table(test: str) -> pd.DataFrame:
        s = summ[(summ["test"] == test) & ~summ["model"].isin(REFERENCES)].sort_values("macro_f1_mean", ascending=False)
        cols = {"model": s["model"].map(label), "main score": s["macro_f1_mean"],
                "worst district": s["worst_district_macro_f1"], "first crop right, all fields": s["top1_all_fields"]}
        if test == "lodo":
            cols.update({"better than B1 in": s["districts_beating_B1"].map(lambda v: f"{int(v)} of 4"),
                         "passes the adoption rule": s["passes_rule"].map({True: "yes", False: "no"})})
        return pd.DataFrame(cols)

    mp = dec["map"]
    blk = summ[summ["test"] == "blockcv"].set_index("model")

    def gain(view, method, a, b):
        return blk.loc[f"{method}[{view}] {b}", "macro_f1_mean"] - blk.loc[f"{method}[{view}] {a}", "macro_f1_mean"]

    steady_gain = [gain(v, m, "v1.3", "steady") for v in VIEWS for m in METHODS if f"{m}[{v}] steady" in blk.index]
    w_gain = [gain(v, m, "steady", "steady+weights") for v in VIEWS for m in METHODS if f"{m}[{v}] steady" in blk.index]
    w_top1 = [blk.loc[f"{m}[{v}] steady+weights", "top1_all_fields"] - blk.loc[f"{m}[{v}] steady", "top1_all_fields"]
              for v in VIEWS for m in METHODS if f"{m}[{v}] steady" in blk.index]
    groups = ch[ch["model"].str.startswith(BLEND) & (ch["test"] == "blockcv")]["group"].value_counts()
    weights_kept = ch[~ch["model"].str.startswith(BLEND)]["weights"].map(lambda w: json.loads(w) == [1, 1, 1, 1]).mean()
    lines = [
        "# Crop recommendation: third round of tuning (tuning_v3)", "",
        f"Protocol: `docs/training_protocol.md`, version 1.4 (section 13). Run at {info['run_at_utc']} UTC: "
        f"{info['compute_hours']:.1f} hours of computing, {info['workers']} tasks at a time. Code commit: "
        f"`{(info['git_commit'] or 'none')[:10]}`." + (" Quick check only: these numbers are never reported."
                                                       if info.get("quick_check_only") else "")
        + (f" Report text rebuilt from the saved tasks at commit `{info['report_rebuilt_at_commit'][:10]}`, with the "
           "same tasks." if info.get("report_rebuilt_at_commit") else ""), "",
        f"**Built-in check: {'passed' if check['passed'] else 'FAILED'}.** Fitting the version 1.3 choice of every "
        f"fold again gives the `tuning_v2` main score in {check['rows_compared'] - len(check['mismatches'])} of "
        f"{check['rows_compared']} rows (model and district).", "",
        "## Decisions fixed before the run", "",
        f"1. **Adoption (main test).** " + (f"Passing: {', '.join(map(label, dec['adoption']['passing']))}."
                                           if dec["adoption"]["passing"] else
                                           f"No model of this round passes the adoption rule, so B1 stays the default "
                                           f"answer of the API. Best: {label(dec['adoption']['best'])}, "
                                           f"{dec['adoption']['best_score']:.3f} against 0.378 for B1."),
        f"2. **Map.** " + (f"The map moves to {label(mp['serve'])}." if mp["changes"] else
                           f"The map keeps Random Forest, all_sat, tuned of version 1.2 ({mp['today_score']:.3f} on new "
                           f"fields, first crop right {mp['today_top1']:.3f}): the best of this round, "
                           f"{label(mp['best'])}, scores {mp['best_score']:.3f} with first crop right "
                           f"{mp['best_top1']:.3f}, which is not more than 0.01 ahead with first crops kept."), "",
        "Expected before the run: (a) the steadier choice recovers part of the loss of version 1.3, to about the level "
        "of version 1.2; (b) crop weights raise the main score by about 0.01 and lower the share of first crops right; "
        "(c) nothing passes decision 2, and no model passes the adoption rule.",
        f"(a) Steadier choice minus version 1.3, same model and view: "
        + ", ".join(f"{x:+.3f}" for x in steady_gain) + f" (mean {np.mean(steady_gain):+.3f}).",
        f"(b) Crop weights minus the steadier choice: main score " + ", ".join(f"{x:+.3f}" for x in w_gain)
        + f" (mean {np.mean(w_gain):+.3f}); first crop right " + ", ".join(f"{x:+.3f}" for x in w_top1)
        + f" (mean {np.mean(w_top1):+.3f}). The weights stayed at no change in {weights_kept:.0%} of the folds.",
        f"(c) {'Nothing passes decision 2' if not mp['changes'] else 'Decision 2 moves the map'}; "
        + ("no model passes the adoption rule." if not dec["adoption"]["passing"] else "a model passes the adoption rule."),
        "",
        "## New fields in the known districts (decides the map)", "",
        "Everything chosen inside the training part of each block fold, then scored once on the test fold.", "",
        _md(table("blockcv")), "",
        "References on the same folds (from `tuning_v2`): "
        + "; ".join(f"{ {'B0_majority': 'B0, most common crop', 'B0d_district_majority': 'most common crop of the district', ev.BASELINE: 'B1, expert rules'}[m]} "
                    f"{blk.loc[m, 'macro_f1_mean']:.3f} (first crop right {blk.loc[m, 'top1_all_fields']:.3f})"
                    for m in REFERENCES if m in blk.index) + ".", "",
        "Groups the blend chose inside the training part (new fields, 2 views x 5 folds): "
        + ", ".join(f"{g} {n}" for g, n in groups.items()) + ".", "",
        f"## When the model the map serves is sure ({label(res['analyses']['model'])}, new fields)", "",
        "If it answered only when its probability for the first crop is at least this high (out-of-fold, crop weights "
        "applied):", "",
        _md(res["analyses"]["sure"][res["analyses"]["sure"]["district"] == "all"].drop(columns=["district", "model"])
            .rename(columns=r1.PLAIN)), "",
        "## Main test: leave one district out (decides adoption)", "",
        _md(table("lodo")), "",
        "## How to read this", "",
        "- Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over the "
        "four districts (section 5). 'First crop right' is the share of all test fields whose first crop was the crop "
        "planted.",
        "- 'Version 1.3 tuned' is the model of `tuning_v2`, fitted again (built-in check). 'Steadier choice' compared "
        "the 8 best candidates of version 1.3 on two more inner splits. 'Crop weights' multiply the probabilities by one "
        "weight per crop chosen inside the training part. The blend averages the group of tuned models chosen inside.",
        "- Every choice of every fold is in `choices.csv`. Settings, versions and timing: `run_info.json`.",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def assemble(df: pd.DataFrame, parts: list[dict]) -> dict:
    res = collect(df, parts)
    ref_rows, ref_conf = references()
    res["results"] = pd.concat([res["results"], ref_rows], ignore_index=True)
    res["confusion"] = pd.concat([res["confusion"], ref_conf], ignore_index=True)
    b1 = ref_rows[(ref_rows["test"] == "lodo") & (ref_rows["model"] == ev.BASELINE)].set_index("district")["macro_f1"]
    res["summary"] = summarise(res["results"], b1)
    res["check"] = builtin_check(res["results"])
    res["decisions"] = decisions(res["summary"])
    # How often the first crop is right at each level of the model's probability, for the model the map serves
    # (or the best of this round when the map does not change), on the new-fields test
    shown = res["decisions"]["map"]["serve"] if res["decisions"]["map"]["changes"] else res["decisions"]["map"]["best"]
    proba = res["oof"][("blockcv", shown)]
    res["analyses"] = {"model": shown,
                       "bands": r1.confidence_bands(df, proba).assign(model=shown),
                       "sure": r1.answer_when_sure(df, proba).assign(model=shown)}
    return res


# ------------------------------------------------------------------ the model the map serves
def final_model(df: pd.DataFrame, out: Path = DEFAULT_OUT, log: Callable = print) -> dict:
    """Decision 2 of section 13 (with section 12): when the map moves, its settings are chosen on all 1,904 fields
    with the same comparison as inside the nested test. Every candidate is scored on 3 block folds (seed 0), the 8
    best again on seeds 1 and 2, the best mean wins (ties: the candidate drawn first), and the crop weights are chosen
    on its out-of-fold probabilities (mean over the three splits). Writes <out>/final_model.json."""
    info = json.loads((Path(out) / "run_info.json").read_text(encoding="utf-8"))
    mp = info["decisions"]["map"]
    import sklearn
    result = {"model": mp["serve"], "changes": mp["changes"], "decided_by": "decision 2 of docs/training_protocol.md, "
              "section 13", "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
              "git_commit": ev._git_commit(), "uncommitted_changes": ev._git_has_uncommitted_changes(),
              "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                           "scikit-learn": sklearn.__version__}}
    if not mp["changes"]:
        result["note"] = "The map keeps the model of version 1.2: 01-crop-recommendation/reports/tuning_v1/final_model.json"
    else:
        core, _, variant = mp["serve"].partition(" ")
        method, view = core.rstrip("]").split("[")
        if method == BLEND:
            raise SystemExit("A blend on the map is not supported yet: add it to map_model.py first.")
        cands = t2.candidates(method)

        def make(c):
            return lambda: t2.Tunable(method, cands[c], view)

        t0 = time.time()
        folds = {seed: ev.block_folds(df, n_splits=r1.INNER_BLOCK_FOLDS, seed=seed) for seed in (SEED,) + EXTRA_SEEDS}
        oof0 = {c: r1.oof_proba(df, folds[SEED], make(c)) for c in range(len(cands))}
        s0 = [r1.block_score(df, oof0[c]) for c in range(len(cands))]
        top = sorted(range(len(cands)), key=lambda c: (-s0[c], c))[:TOP_K]
        oofs, means = {}, {}
        for c in top:
            oofs[c] = [oof0[c]] + [r1.oof_proba(df, folds[seed], make(c)) for seed in EXTRA_SEEDS]
            means[c] = float(np.mean([r1.block_score(df, p) for p in oofs[c]]))
        chosen = min(c for c in top if means[c] == max(means.values()))
        inner_p = np.mean(oofs[chosen], axis=0)
        weights, w_score, w_none = best_weights(df, inner_p) if variant.endswith("+weights") else (
            np.ones(len(CROPS)), r1.block_score(df, inner_p), r1.block_score(df, inner_p))
        log(f"{core}: candidate {chosen} chosen on all {len(df)} fields (mean of three splits {means[chosen]:.3f}), "
            f"crop weights {[round(float(w), 3) for w in weights]} ({w_none:.3f} -> {w_score:.3f}), "
            f"{time.time() - t0:.0f} s")
        result.update({
            "method": method, "view": view, "tuned": True, "candidate": chosen, "settings": cands[chosen],
            "crops": CROPS, "crop_weights": [round(float(w), 6) for w in weights],
            "chosen_by": (f"same comparison as inside the nested test of section 13, on all {len(df)} fields: every "
                          f"candidate of version 1.3 on 3 folds over 2 km blocks (seed {SEED}), the {TOP_K} best "
                          f"again on seeds {', '.join(map(str, EXTRA_SEEDS))}, best mean; then one weight per crop "
                          "(2/3, 1 or 3/2) chosen on its out-of-fold probabilities"),
            "comparison_scores_seed0": [round(x, 6) for x in s0], "top": top,
            "means": {str(k): round(v, 6) for k, v in means.items()},
            "inner_score": round(w_none, 6), "inner_score_weighted": round(w_score, 6),
            "importance_from": {"folder": "01-crop-recommendation/reports/tuning_v2", "model": f"{core}{TUNED}",
                                "note": "input importance measured in version 1.3 for the tuned model of the same "
                                        "method and inputs; this round did not measure it again"},
        })
    path = Path(out) / FINAL_FILE
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"Written to {path}")
    return result


# ------------------------------------------------------------------ command line
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="Tiny run to check the code. Never reported.")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--final", action="store_true", help="After the run: the settings and crop weights of the model "
                                                         "the map serves (decision 2), chosen on all fields.")
    args = ap.parse_args(argv)
    table = Path(args.table) if args.table else TABLE
    df = ev.prepare(pd.read_csv(table, low_memory=False))
    if args.final:
        final_model(df, Path(args.out) if args.out else DEFAULT_OUT, log=lambda s: print(s, flush=True))
        return 0
    cands = {m: json.loads((ROUND2_DIR / "candidates.json").read_text(encoding="utf-8"))[m] for m in METHODS}
    assert all(cands[m] == t2.candidates(m) for m in METHODS), "the candidates of version 1.3 changed"
    inner13 = pd.read_csv(ROUND2_DIR / "inner_scores.csv")
    chosen13 = pd.read_csv(ROUND2_DIR / "chosen_settings.csv")
    if args.quick:
        out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "tuning_v3_quick"
        tasks, top_k = tasks_for(df, ("field_sat",), folds=[0], districts=["Musanze"]), 2
    else:
        out = Path(args.out) if args.out else DEFAULT_OUT
        tasks, top_k = tasks_for(df, VIEWS), TOP_K
    work = out / "_work"
    print(f"Protocol {PROTOCOL_VERSION}. Output: {out}" + (" (quick check)" if args.quick else ""), flush=True)
    t0 = time.time()
    if args.assemble_only and not all(_task_file(work, *t).exists() for t in tasks):
        raise SystemExit(f"--assemble-only needs every task in {work}")
    parts = run_tasks(df, work, tasks, cands, inner13, chosen13, top_k, 1 if args.assemble_only else args.workers,
                      log=lambda s: print(s, flush=True))
    res = assemble(df, parts)
    import sklearn
    import xgboost
    previous = json.loads((out / "run_info.json").read_text(encoding="utf-8")) if (
        args.assemble_only and (out / "run_info.json").exists()) else {}
    info = {"protocol_version": PROTOCOL_VERSION, "quick_check_only": args.quick,
            "run_at_utc": previous.get("run_at_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "compute_hours": sum(p["seconds"] for p in parts) / 3600, "wall_hours_this_call": (time.time() - t0) / 3600,
            "workers": previous.get("workers") or args.workers,
            "git_commit": previous.get("git_commit") or ev._git_commit(),
            "uncommitted_changes": previous.get("uncommitted_changes", ev._git_has_uncommitted_changes()),
            **({"report_rebuilt_at_commit": ev._git_commit(),
                "report_rebuilt_with_uncommitted_changes": ev._git_has_uncommitted_changes()} if previous else {}),
            "table_sha256": ev.file_sha256(table), "methods": list(METHODS),
            "views": sorted({t[0] for t in tasks}), "top_k": top_k, "extra_seeds": list(EXTRA_SEEDS),
            "weight_steps": [round(w, 4) for w in WEIGHT_STEPS], "groups": ["+".join(g) for g in GROUPS],
            "builtin_check": res["check"], "decisions": res["decisions"],
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}}
    path = write_report(out, info, res, parts)
    blk = res["summary"][res["summary"]["test"] == "blockcv"].set_index("model")["macro_f1_mean"]
    print("\n== new fields\n" + blk.sort_values(ascending=False).round(3).to_string())
    print(f"\nBuilt-in check: {'passed' if res['check']['passed'] else 'FAILED'} ({res['check']['rows_compared']} rows, "
          f"{len(res['check']['mismatches'])} mismatches)")
    print(f"Map: {'moves to ' + res['decisions']['map']['serve'] if res['decisions']['map']['changes'] else 'unchanged'}. "
          f"Adoption: {res['decisions']['adoption']['passing'] or 'none'}")
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
