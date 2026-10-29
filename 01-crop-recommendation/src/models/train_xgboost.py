"""XGBoost training and scores for crop recommendation. It follows docs/training_protocol.md (version 1.0).

Agreed at the stand-up of 28 Oct 2026: train XGBoost and give its scores next to the other models.
What this script does, in order:
  1. Keeps the 1,904 training fields of feature_table_v2.csv (same filters as evaluate.py).
  2. Scores XGBoost on the three input views with the protocol's two tests, same folds and settings as eval_v1:
     a district the model never saw (main test, leave one district out) and new fields in the four known
     districts (second test, 5 folds by 2 km block). B0 and B1 go through the same folds as reference points.
  3. Saves the scores: the protocol's main score per district, overall scores (accuracy, balanced accuracy,
     macro-F1, weighted F1, Cohen's kappa, top-3), scores per crop (precision, recall, F1) and the confusion
     matrix.
  4. Checks that every score per district is the same as in reports/eval_v1 (same code, same folds).
  5. Trains the final XGBoost on all inputs and all 1,904 fields (the model behind the dashboard map) and saves
     it with its feature importance.
Nothing is tuned: the settings are the protocol's. Tuning (tracker T051) comes after the midterm.

Run from the repo root (under a minute on a laptop or on Colab):
    python 01-crop-recommendation/src/models/train_xgboost.py
Options: --table PATH, --out DIR
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

DEFAULT_OUT = ev.REPO / "01-crop-recommendation" / "reports" / "xgboost_v1"
EVAL_V1 = ev.REPO / "01-crop-recommendation" / "reports" / "eval_v1"
FINAL_MODEL = "XGB[all]"
MODELS = ["B0_majority", "B1_expert_rules", "XGB[district]", "XGB[field]", "XGB[all]"]
TESTS = {  # name: (plain description, eval_v1 file with the same test)
    "new_district": ("a district the model never saw (main test, decides adoption)", "main_lodo_results.csv"),
    "new_fields": ("new fields in the four known districts (second test)", "main_blockcv_results.csv"),
}
# District-level inputs have one value for every field in a district. Field-level inputs change from field to field.
DISTRICT_LEVEL = ev.CLIMATE + ev.NISR_SHARES
FIELD_LEVEL = ["elevation_m"] + ev.POINT_TEMP + ev.SOIL_TOP + ev.SOIL_SUB
CHECK_COLUMNS = ["n_test", "top1", "top3", "macro_f1", "macro_f1_all4"]


def pick_models(n_estimators: int = 300) -> list:
    """B0, B1 and the three XGBoost views from the protocol's own list, so the settings cannot drift."""
    by_name = {m.name: m for m in ev.default_models(n_estimators=n_estimators)}
    return [by_name[n] for n in MODELS]


def _hits(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray, k: int = 3) -> np.ndarray:
    order = np.argsort(-proba, axis=1)[:, :k]
    return np.array([t in set(classes[o]) for t, o in zip(y_true, order)], dtype=bool)


def run_test(df: pd.DataFrame, model, test: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One model through one test. Returns the scores per district (computed exactly as in evaluate.py), the
    prediction for every field (each made by a model that did not see that field) and the top-3 hits."""
    labels = ev.CROPS
    y = df["label"].to_numpy()
    groups = df[ev.GROUP_COL].to_numpy()
    pred = np.empty(len(df), dtype=object)
    hit3 = np.zeros(len(df), dtype=bool)
    rows = []
    if test == "new_district":                                   # same steps as ev.run_lodo
        for held_out in sorted(set(groups)):
            tr, te = np.where(groups != held_out)[0], np.where(groups == held_out)[0]
            train, test_df = df.iloc[tr], df.iloc[te]
            m = model.fit(train, train["label"])
            p = m.predict_proba(test_df)
            pr = m.classes_[np.argmax(p, axis=1)]
            pred[te], hit3[te] = pr, _hits(p, m.classes_, y[te])
            rows.append({"model": model.name, "district": held_out, "n_train": int(len(train)),
                         **ev.scores(y[te], pr, p, m.classes_, labels)})
    elif test == "new_fields":                                   # same steps as ev.run_block_cv
        labels_arr = np.array(labels)
        proba = np.zeros((len(df), len(labels)))
        for tr, te in ev.block_folds(df):
            train, test_df = df.iloc[tr], df.iloc[te]
            m = model.fit(train, train["label"])
            proba[np.ix_(te, [labels.index(c) for c in m.classes_])] = m.predict_proba(test_df)
        pred[:] = labels_arr[np.argmax(proba, axis=1)]
        hit3[:] = _hits(proba, labels_arr, y)
        for d in sorted(set(groups)):
            mask = groups == d
            rows.append({"model": model.name, "district": d, "n_train": float("nan"),
                         **ev.scores(y[mask], pred[mask], proba[mask], labels_arr, labels)})
    else:
        raise ValueError(f"unknown test {test}")
    return pd.DataFrame(rows), pred.astype(str), hit3


def overall_scores(y: np.ndarray, pred: np.ndarray, hit3: np.ndarray) -> dict:
    """Scores over all 1,904 test predictions together. For information: the protocol's main score is the mean of
    the four district scores, because one district (Nyagatare, 1,084 fields) would otherwise dominate."""
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=ev.CROPS, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, labels=ev.CROPS, average="weighted", zero_division=0),
            "cohen_kappa": cohen_kappa_score(y, pred),
            "top3": float(hit3.mean())}


def per_crop_scores(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=ev.CROPS, zero_division=0)
    return pd.DataFrame({"crop": ev.CROPS, "precision": p, "recall": r, "f1": f, "fields": s})


def check_against_eval_v1(by_district: pd.DataFrame, eval_dir: Path = EVAL_V1) -> pd.DataFrame:
    """Largest difference per test and model between this run and the committed eval_v1 results."""
    rows = []
    for test, (_, fname) in TESTS.items():
        path = Path(eval_dir) / fname
        mine = by_district[by_district["test"] == test]
        if not path.exists():
            rows += [{"test": test, "model": m, "largest_difference": float("nan"), "same_as_eval_v1": "no file"}
                     for m in mine["model"].unique()]
            continue
        official = pd.read_csv(path)
        for model, g in mine.groupby("model", sort=False):
            o = official[official["model"] == model].set_index("district")
            a = g.set_index("district")
            if o.empty or set(o.index) != set(a.index):
                rows.append({"test": test, "model": model, "largest_difference": float("nan"),
                             "same_as_eval_v1": "not in eval_v1"})
                continue
            gap = max(float((a[c] - o.loc[a.index, c]).abs().max()) for c in CHECK_COLUMNS)
            rows.append({"test": test, "model": model, "largest_difference": gap,
                         "same_as_eval_v1": "yes" if gap <= 1e-9 else "no"})
    return pd.DataFrame(rows)


def feature_importance(model) -> pd.DataFrame:
    """Gain importance of the fitted XGBoost, with each input marked as district-level or field-level."""
    gain = model.estimator.get_booster().get_score(importance_type="gain")
    names = model.used_
    rows = []
    for k, v in gain.items():
        name = names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() and k not in names else k
        rows.append({"feature": name, "gain": float(v),
                     "level": "district" if name in DISTRICT_LEVEL else "field"})
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    df["share"] = df["gain"] / df["gain"].sum()
    return df


def train(raw: pd.DataFrame, n_estimators: int = 300, eval_dir: Optional[Path] = EVAL_V1,
          log=print) -> dict:
    """Everything except writing files. Returns a dict of tables and the final model."""
    df = ev.prepare(raw)
    out = {"by_district": [], "overall": [], "by_crop": [], "confusion": []}
    y = df["label"].to_numpy()
    for test in TESTS:
        for model in pick_models(n_estimators):
            t0 = time.time()
            res, pred, hit3 = run_test(df, model, test)
            out["by_district"].append(res.assign(test=test))
            out["overall"].append({"test": test, "model": model.name, "fields": int(len(y)),
                                   "main_score": float(res["macro_f1"].mean()), **overall_scores(y, pred, hit3)})
            out["by_crop"].append(per_crop_scores(y, pred).assign(test=test, model=model.name))
            cm = confusion_matrix(y, pred, labels=ev.CROPS)
            out["confusion"].append(pd.DataFrame(cm, index=ev.CROPS, columns=ev.CROPS).rename_axis("true")
                                    .reset_index().melt(id_vars="true", var_name="predicted", value_name="count")
                                    .assign(test=test, model=model.name))
            log(f"  {test:12s} {model.name:16s} main score {res['macro_f1'].mean():.3f}  ({time.time() - t0:.1f} s)")
    by_district = pd.concat(out["by_district"], ignore_index=True)
    first = ["test", "model", "district"]
    by_district = by_district[first + [c for c in by_district.columns if c not in first]]
    summary = pd.concat([ev.summarise(by_district[by_district["test"] == t]).reset_index().assign(test=t)
                         for t in TESTS], ignore_index=True)
    final = next(m for m in pick_models(n_estimators) if m.name == FINAL_MODEL)
    final.fit(df, df["label"])
    log(f"  final {FINAL_MODEL} trained on all {len(df)} fields")
    return {
        "data": df,
        "by_district": by_district,
        "summary": summary[["test", "model", "macro_f1_mean", "worst_district_macro_f1", "top1_mean",
                            "districts_beating_B1", "passes_rule"]],
        "overall": pd.DataFrame(out["overall"]),
        "by_crop": pd.concat(out["by_crop"], ignore_index=True)[["test", "model", "crop", "precision", "recall",
                                                                  "f1", "fields"]],
        "confusion": pd.concat(out["confusion"], ignore_index=True)[["test", "model", "true", "predicted",
                                                                      "count"]],
        "check": check_against_eval_v1(by_district, eval_dir) if eval_dir else pd.DataFrame(),
        "final_model": final,
        "importance": feature_importance(final),
    }


SHOW = {  # column names for people, in report.md
    "macro_f1_mean": "main score", "worst_district_macro_f1": "worst district", "top1_mean": "accuracy (mean of districts)",
    "districts_beating_B1": "districts better than B1", "passes_rule": "passes adoption rule",
    "main_score": "main score", "accuracy": "accuracy (all fields)", "balanced_accuracy": "balanced accuracy",
    "macro_f1": "macro-F1 (all fields)", "weighted_f1": "weighted F1", "cohen_kappa": "Cohen's kappa",
    "top3": "top-3 (all fields)",
}


def _md(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    frame = frame.copy()
    if "districts_beating_B1" in frame:
        frame["districts_beating_B1"] = [("" if pd.isna(v) else f"{int(v)} of 4") for v in frame["districts_beating_B1"]]
    frame = frame.rename(columns=SHOW)
    try:
        return frame.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def _versions() -> dict:
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


def write_report(out: Path, res: dict, info: dict) -> Path:
    """CSV files, the final model, run_info.json and report.md."""
    out = Path(out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    for name in ("by_district", "summary", "overall", "by_crop", "confusion", "check"):
        if len(res[name]):
            res[name].to_csv(out / f"{name}.csv", index=False)
    res["importance"].to_csv(out / "feature_importance.csv", index=False)
    final = res["final_model"]
    final.estimator.save_model(str(out / "model" / "xgboost_all_inputs.json"))
    (out / "model" / "meta.json").write_text(json.dumps({
        "model": FINAL_MODEL, "trained_on_fields": int(len(res["data"])), "season": "B (CIAT, 2025)",
        "features": final.used_, "classes": [str(c) for c in final.classes_],
        "fill_values": {k: float(v) for k, v in final.fill_.items()},
        "settings": repr(final.estimator), "versions": info["versions"]}, indent=2), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    s, ov, bd = res["summary"], res["overall"], res["by_district"]
    main = s[s["test"] == "new_district"].set_index("model")
    second = s[s["test"] == "new_fields"].set_index("model")
    b1 = bd[bd["model"] == ev.BASELINE].set_index(["test", "district"])["macro_f1"]
    xgb = bd[bd["model"] == FINAL_MODEL].set_index(["test", "district"])["macro_f1"]
    side = pd.DataFrame({"XGBoost (all inputs)": xgb, "Expert rules (B1)": b1})
    side["difference"] = side["XGBoost (all inputs)"] - side["Expert rules (B1)"]
    side = side.reset_index()
    side["test"] = side["test"].map({k: v[0] for k, v in TESTS.items()})
    imp = res["importance"]
    level = imp.groupby("level")["share"].sum()
    check = res["check"]
    same = (not check.empty) and (check["same_as_eval_v1"] == "yes").all()

    lines = [
        "# XGBoost training and scores (protocol 1.0)", "",
        f"Made by `src/models/train_xgboost.py` on {info['finished_utc']} (UTC). Code commit `{info['git_commit']}`"
        + (" with uncommitted changes" if info.get("git_uncommitted_changes") else "") + ". "
        + f"Table `{info['table']}`, fingerprint `{info['table_sha256'][:16]}...`, {info['training_fields']} "
        "training fields (single crops, GPS checks passed). Settings: the protocol's, nothing tuned.", "",
        "## In short", "",
        f"- A district it never saw (main test): XGBoost on all inputs scores {main.loc[FINAL_MODEL, 'macro_f1_mean']:.3f} "
        f"against {main.loc[ev.BASELINE, 'macro_f1_mean']:.3f} for the expert rules (B1). "
        f"It does not pass the adoption rule, so the rules stay the default.",
        f"- New fields in the four known districts (second test): XGBoost on all inputs scores "
        f"{second.loc[FINAL_MODEL, 'macro_f1_mean']:.3f} against {second.loc[ev.BASELINE, 'macro_f1_mean']:.3f} for B1, "
        f"better in {int(second.loc[FINAL_MODEL, 'districts_beating_B1'])} of the 4 districts.",
        f"- District-level inputs (season weather and NISR shares, one value per district) carry "
        f"{level.get('district', 0):.0%} of the model's gain: "
        + ("most" if level.get("district", 0) > 0.6 else "about half" if level.get("district", 0) >= 0.4 else "part")
        + " of what the model uses tells it which district a field is in. That helps inside a known district and "
        "does not carry over to a new one.",
        "- Every score per district is " + ("the same as in `eval_v1`." if same else
                                            "NOT the same as in `eval_v1`. Check the library versions in run_info.json."),
        "", "Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over "
        "the four districts (protocol, rule 5). It goes from 0 to 1.", "",
        "## 1. Main score per model", "",
    ]
    for test, (desc, _) in TESTS.items():
        t = s[s["test"] == test].drop(columns="test")
        lines += [f"Test: {desc}", "", _md(t), ""]
    lines += ["## 2. XGBoost and the rules, district by district", "", _md(side), "",
              "## 3. Overall scores (all test predictions together, for information)", "",
              "Accuracy counts every field the same, so it mostly reflects Nyagatare (1,084 of the 1,904 fields). "
              "Balanced accuracy and macro-F1 give each crop the same weight. Cohen's kappa is 0 for chance "
              "agreement and 1 for perfect. Top-3 is near 1 for every method because there are only four crops.", ""]
    for test, (desc, _) in TESTS.items():
        lines += [f"Test: {desc}", "", _md(ov[ov["test"] == test].drop(columns="test")), ""]
    lines += ["## 4. Scores per crop, XGBoost on all inputs", ""]
    bc = res["by_crop"]
    for test, (desc, _) in TESTS.items():
        lines += [f"Test: {desc}", "", _md(bc[(bc["test"] == test) & (bc["model"] == FINAL_MODEL)]
                                           .drop(columns=["test", "model"])), ""]
    lines += ["## 5. Confusion matrix, XGBoost on all inputs (rows: true crop, columns: predicted crop)", ""]
    cf = res["confusion"]
    for test, (desc, _) in TESTS.items():
        m = (cf[(cf["test"] == test) & (cf["model"] == FINAL_MODEL)]
             .pivot(index="true", columns="predicted", values="count").reindex(index=ev.CROPS, columns=ev.CROPS))
        lines += [f"Test: {desc}", "", _md(m.reset_index(), floatfmt=".0f"), ""]
    lines += ["## 6. Final model", "",
              f"XGBoost on all inputs ({len(final.used_)} inputs), trained on all {len(res['data'])} fields with the "
              "protocol's settings. It is the model behind the dashboard map page "
              "(`src/recommendation/map_model.py` trains the same model with the same data and settings). "
              "Saved in `model/` (not committed: it takes seconds to retrain).", "",
              "Inputs by gain importance (top 10):", "",
              _md(imp.head(10)[["feature", "level", "share"]]), "",
              "## 7. Same as eval_v1?", "",
              _md(check) if not check.empty else "No eval_v1 results found.", "",
              "## 8. Limits", "",
              "- One season: CIAT season B 2025. Seasons A and C are not in the data yet.",
              "- Weather is one value per district (NASA POWER), and Musanze and Nyabihu share one weather cell. "
              "Per-field rainfall (CHIRPS) is the next input to add.",
              "- Settings are not tuned yet (tracker T051, after the midterm, nested inside the training districts).",
              ""]
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="XGBoost training and scores (protocol 1.0)")
    ap.add_argument("--table", type=Path, default=None, help="feature table (default: feature_table_v2.csv)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    args = ap.parse_args(argv)
    table = args.table or ev.default_table()
    t0 = time.time()
    print(f"Table: {table}")
    raw = pd.read_csv(table, low_memory=False)
    res = train(raw)
    info = {"protocol_version": ev.PROTOCOL_VERSION, "script": "src/models/train_xgboost.py",
            "table": str(Path(table).resolve().relative_to(ev.REPO)) if Path(table).resolve().is_relative_to(ev.REPO)
            else str(table), "table_sha256": ev.file_sha256(table), "table_rows": int(len(raw)),
            "training_fields": int(len(res["data"])), "git_commit": ev._git_commit(),
            "git_uncommitted_changes": ev._git_has_uncommitted_changes(), "versions": _versions(),
            "platform": platform.platform(), "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "seconds": round(time.time() - t0, 1)}
    path = write_report(args.out, res, info)
    print(f"\nReport: {path}")
    print(res["summary"].to_string(index=False))
    ok = (not res["check"].empty) and (res["check"]["same_as_eval_v1"] == "yes").all()
    print("\nEvery score per district is the same as in eval_v1." if ok else
          "\nSome scores differ from eval_v1: check the library versions (requirements-eval.txt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
