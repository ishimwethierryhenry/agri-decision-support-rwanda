"""Builds the table behind the confidence of the map's answer: 01-crop-recommendation/reports/map_confidence/.

The confidence of an answer on the map is how often answers like it were right in testing. "Like it" means the same
district, the same kind of answer, and field inputs taken from about as far from the point:

- Kind of answer. One crop, when the model's first two crops are not close (0.10 apart or more): how often its first
  crop was the crop planted. Two crops, when they are close: how often the crop planted was one of the two. The table
  also keeps the two-crop rate over all fields, which the map uses when a one-crop answer would be right less than
  half the time.
- Distance. The map takes soil, elevation and the satellite values from the CIAT fields of the clicked 2 km block, or
  of the nearest block in the district. Within 5 km the rates come from the new-fields test of the model on the map
  (tuning_v1), where every field kept its own inputs. Farther away, every test field is predicted again with the
  inputs of the nearest training block 5 to 10 km away, or 10 to 20 km away, in its own district, as the map does for
  a point that far from any surveyed field. Beyond 20 km nothing was tested, so the map gives no answer.

Every number comes from the new-fields test of tuning_v1 (5 folds over 2 km blocks, seed 0). Each fold's Random Forest
is fitted again with the settings that fold chose (tuning_v1/chosen_settings.csv), and the built-in check makes sure
its first crops give exactly the tuning_v1 hit rate in every district. No field is ever predicted by a model that saw
its block. A cell with fewer than 30 fields takes the value of the four districts together.

The report also replays the answers on the tested fields and on a grid of points over the four districts (season B,
about 1 km apart), with the confidence rules before and after this table. Files hold no coordinates.

Run from the repo root (about a minute):  python 01-crop-recommendation/src/recommendation/build_map_confidence.py
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate as ev  # noqa: E402
import expert_baseline as b1  # noqa: E402
import map_model as mm  # noqa: E402
import tune  # noqa: E402

OUT = mm.CONFIDENCE_TABLE_PATH.parent
OWN = "within 5 km"                        # the test fields keep their own inputs
BORROWED = {"5 to 10 km": (5.0, 10.0), "10 to 20 km": (10.0, 20.0)}
GRID_STEP = 0.01                           # degrees, about 1.1 km
PILOTS = ["Musanze", "Nyabihu", "Nyagatare", "Ruhango"]


# ------------------------------------------------------------------ predictions of the new-fields test
def fold_predictions(df: pd.DataFrame, rec: mm.MapRecommender, log=print) -> pd.DataFrame:
    """One row per test field and distance band: the first two crops of that fold's model and whether they were
    close, with the field's own inputs (within 5 km) or the inputs of a training block 5 to 10 or 10 to 20 km away."""
    chosen = pd.read_csv(mm.EVAL_DIR / "chosen_settings.csv")
    chosen = chosen[(chosen["test"] == "blockcv") & (chosen["model"] == mm.MODEL_NAME.replace(tune.TUNED, ""))]
    chosen = chosen.reset_index(drop=True)
    blocks = rec.blocks.assign(bid=rec.blocks["block_id"].astype(str))
    y, dist = df["label"].to_numpy(), df["district"].to_numpy()
    rows = []
    for f, (tr, te) in enumerate(ev.block_folds(df)):
        t0 = time.time()
        train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        model = tune.Tunable(mm.FINAL["method"], json.loads(chosen.loc[f, "settings"]), mm.FINAL["view"])
        model.fit(train, train["label"])
        test_blocks = set(test[ev.BLOCK_COL].astype(str))
        inputs, meta = [], []
        for k, i in enumerate(te):
            d, bid = dist[i], str(df.loc[i, ev.BLOCK_COL])
            bd = blocks[blocks["District"] == d]
            own = bd[bd["bid"] == bid].iloc[0]
            km = bd.apply(lambda b: mm.haversine_km(own["lat"], own["lon"], b["lat"], b["lon"]), axis=1)
            for band, (lo, hi) in BORROWED.items():
                cand = km[(km >= lo) & (km < hi) & ~bd["bid"].isin(test_blocks)]
                if len(cand):
                    inputs.append(rec.block_row(bd.loc[cand.idxmin()], d))
                    meta.append((i, band, float(cand.min())))
        for band, proba, idx, kms in (
                (OWN, model.predict_proba(test), list(te), [0.0] * len(te)),
                (None, model.predict_proba(pd.DataFrame(inputs)), [m[0] for m in meta], [m[2] for m in meta])):
            order = np.argsort(-proba, axis=1)
            for j, i in enumerate(idx):
                p = proba[j]
                rows.append({"fold": f + 1, "field": int(i), "district": dist[i], "band": band or meta[j][1],
                             "km": kms[j],
                             "first": model.classes_[order[j, 0]], "second": model.classes_[order[j, 1]],
                             "gap": float(p[order[j, 0]] - p[order[j, 1]]), "crop": y[i]})
        log(f"fold {f + 1}: {len(te)} test fields, {len(inputs)} predictions with borrowed inputs, "
            f"{time.time() - t0:.0f} s")
    r = pd.DataFrame(rows)
    r["close"] = r["gap"] < mm.CLOSE_CALL
    r["first_right"] = r["first"] == r["crop"]
    r["two_right"] = r["first_right"] | (r["second"] == r["crop"])
    return r


def builtin_check(pred: pd.DataFrame) -> dict:
    """The refitted folds must give exactly the tuning_v1 share of first crops right in every district."""
    res = pd.read_csv(mm.EVAL_DIR / "main_blockcv_results.csv")
    ref = res[res["model"] == mm.MODEL_NAME].set_index("district")["top1"]
    own = pred[pred["band"] == OWN].groupby("district")["first_right"].mean()
    rows = {d: {"tuning_v1": float(ref[d]), "replayed": float(own[d])} for d in ref.index}
    return {"compared_with": "01-crop-recommendation/reports/tuning_v1/main_blockcv_results.csv", "districts": rows,
            "passed": all(abs(v["tuning_v1"] - v["replayed"]) < 1e-9 for v in rows.values())}


# ------------------------------------------------------------------ the table
def _rate(g: pd.DataFrame, kind: str) -> tuple[int, int]:
    s = (g.loc[~g["close"], "first_right"] if kind == mm.ONE_CROP else
         g.loc[g["close"], "two_right"] if kind == mm.TWO_CLOSE else g["two_right"])
    return int(len(s)), int(s.sum())


def confidence_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, band in mm.DATA_BANDS:
        g = pred[pred["band"] == band]
        for kind in (mm.ONE_CROP, mm.TWO_CLOSE, mm.TWO_ANY):
            pn, pr = _rate(g, kind)
            for d in PILOTS:
                n, right = _rate(g[g["district"] == d], kind)
                pooled = n < mm.MIN_TESTED_FIELDS
                fields, ok = (pn, pr) if pooled else (n, right)
                rows.append({"district": d, "band": band, "kind": kind, "fields": fields, "right": ok,
                             "rate": round(ok / fields, 6) if fields else float("nan"), "pooled": pooled,
                             "district_fields": n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ replays: before and after
def level_of(rate: float) -> str:
    return "high" if rate >= mm.HIGH_HIT else "medium" if rate >= mm.LOW_HIT else "low"


def replay_tested_fields(pred: pd.DataFrame, rec: mm.MapRecommender, table: dict, df: pd.DataFrame) -> pd.DataFrame:
    """What each tested field would have been shown (within 5 km, season B) under three rules, and whether it was
    right: until 25 Nov (method by main score, level from the district hit rate, one step lower for a close call),
    the morning of 25 Nov (method by first crop right) and now (this table)."""
    own = pred[pred["band"] == OWN].reset_index(drop=True)
    # The rules' first crop and close calls, for the districts where the rules answered until 25 Nov
    rules = b1.ExpertRulesBaseline().fit(df, df["label"])
    key = rules.predict_proba(df)
    loaded = b1.load_rules()
    scores = np.array([[np.nan if (s := b1.explain_row(r, loaded)[c]["score"]) is None else s for c in ev.CROPS]
                       for r in df.to_dict("records")])
    srt = -np.sort(-np.nan_to_num(scores, nan=-1.0), axis=1)
    rules_first = pd.Series(rules.classes_[np.argmax(key, axis=1)], index=df.index)
    rules_close = pd.Series(srt[:, 0] - srt[:, 1] < mm.CLOSE_CALL, index=df.index)
    by_main = {d: "model" if rec.tested[d]["model"]["new_fields"] > rec.tested[d]["rules"]["new_fields"]
               + mm.CHOICE_MARGIN else "rules" for d in PILOTS}
    out = []
    for r in own.itertuples():
        d = r.district
        for rule, method in (("until 25 Nov", by_main[d]), ("25 Nov, first crop right", rec.method_for(d))):
            first = r.first if method == "model" else rules_first[r.field]
            close = r.close if method == "model" else bool(rules_close[r.field])
            lvl = mm.confidence_level(rec.hit_rates[d][method], True, close, False)
            out.append({"rule": rule, "district": d, "shown": f"one crop, {lvl}", "right": bool(first == r.crop)})
        if not r.close:
            c = table[(d, OWN, mm.ONE_CROP)]
            kind, right = (("one crop", r.first_right) if c["rate"] >= mm.LOW_HIT else
                           ("two crops", r.two_right))
            if kind == "two crops":
                c = table[(d, OWN, mm.TWO_ANY)]
        else:
            c, kind, right = table[(d, OWN, mm.TWO_CLOSE)], "two crops", r.two_right
        shown = "no answer" if c["rate"] < mm.LOW_HIT else f"{kind}, {level_of(c['rate'])}"
        out.append({"rule": "now, by situation", "district": d, "shown": shown, "right": bool(right)})
    return pd.DataFrame(out)


def scan_grid(rec: mm.MapRecommender, step: float = GRID_STEP) -> pd.DataFrame:
    """The map's answer for season B on a grid of points over the four districts (no coordinates are kept)."""
    rows = []
    for name, polys, (x0, y0, x1, y1) in rec.districts.shapes:
        if name not in PILOTS:
            continue
        for lat in np.arange(y0 + step / 2, y1, step):
            for lon in np.arange(x0 + step / 2, x1, step):
                if rec.districts.at(lat, lon) != name:
                    continue
                r = rec.recommend(float(lat), float(lon), mm.MODEL_SEASON)
                if r["status"] != "ok":
                    continue
                a, rank = r["answer"], r["model"]["ranking"]
                km = r["inputs"]["block"]["distance_km"]
                before = mm.confidence_level(rec.hit_rates[r["district"]][a["method"]], True,
                                             rank[0]["probability"] - rank[1]["probability"] < mm.CLOSE_CALL,
                                             km > mm.NEAR_KM)
                rows.append({"district": r["district"], "band": mm.data_band(km) or "beyond 20 km",
                             "before": f"one crop, {before}",
                             "now": "no answer" if a["kind"] == "none" else
                             f"{a['kind'].replace('_', ' ')}, {a['confidence']}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def _md(frame: pd.DataFrame, index: bool = True) -> str:
    return frame.to_markdown(index=index, disable_numparse=True)


def _pct(x: float) -> str:
    return "" if pd.isna(x) else "under 1%" if 0 < x < 0.005 else f"{x:.0%}"


def write_report(table: pd.DataFrame, replay: pd.DataFrame, grid: pd.DataFrame, info: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "confidence_table.csv", index=False)
    rp = (replay.groupby(["rule", "shown"]).agg(fields=("right", "size"), right=("right", "mean")).reset_index())
    rp["share"] = rp["fields"] / rp.groupby("rule")["fields"].transform("sum")
    rp.round(6).to_csv(OUT / "replay_tested_fields.csv", index=False)
    gs = pd.concat([grid.groupby("district")[c].value_counts(normalize=True).rename("share").reset_index()
                    .rename(columns={c: "shown"}).assign(rule=c) for c in ("before", "now")], ignore_index=True)
    gs = gs.merge(grid.groupby("district").size().rename("points").reset_index(), on="district")
    gs.round(6).to_csv(OUT / "map_grid_summary.csv", index=False)
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")

    show = table.assign(cell=[f"{_pct(r.rate)} of {r.fields}" + (" (four districts)" if r.pooled else "")
                              for r in table.itertuples()])
    pv = show.pivot_table(index=["band", "kind"], columns="district", values="cell", aggfunc="first")
    pv = pv.reindex([(b, k) for _, b in mm.DATA_BANDS for k in (mm.ONE_CROP, mm.TWO_CLOSE, mm.TWO_ANY)])
    pv = pv.reset_index().rename(columns={"band": "field data from", "kind": "answer"})
    rpv = rp.assign(cell=[f"{_pct(r.share)} of fields, right {_pct(r.right)}" for r in rp.itertuples()])
    rpv = rpv.pivot_table(index="shown", columns="rule", values="cell", aggfunc="first").fillna("")
    rpv = rpv[["until 25 Nov", "25 Nov, first crop right", "now, by situation"]]
    gpv = gs.assign(cell=gs["share"].map(_pct)).pivot_table(index=["rule", "shown"], columns="district",
                                                             values="cell", aggfunc="first").fillna("")
    gpv = gpv.reset_index().rename(columns={"rule": "rule", "shown": "shown on the map"})
    check = info["builtin_check"]
    lines = [
        "# Confidence of the map's answer (map_confidence)", "",
        f"Made by `src/recommendation/build_map_confidence.py` on {info['run_at_utc']} UTC (code commit "
        f"`{(info['git_commit'] or 'none')[:10]}`), from the new-fields test of the model on the map "
        f"({mm.MODEL_NAME}, `reports/tuning_v1`). Decided on 25 Nov 2026 after looking at the map: see "
        "`docs/training_protocol.md`, section 9.", "",
        f"**Built-in check: {'passed' if check['passed'] else 'FAILED'}.** The five refitted fold models give exactly "
        "the tuning_v1 share of first crops right in every district: "
        + ", ".join(f"{d} {v['replayed']:.3f}" for d, v in check["districts"].items()) + ".", "",
        "## How the map answers", "",
        "1. Season. Only season B was tested, so another season gets no answer.",
        "2. Distance. The field inputs come from the clicked 2 km block or the nearest CIAT block in the district. "
        "Beyond 20 km from any surveyed field there is no answer.",
        "3. One crop or two. If the model's first two crops are 0.10 apart or more, the answer is one crop and its "
        "confidence is the share of tested fields like it (same district, same distance band, not close) where the "
        "first crop was the crop planted. If they are closer, the answer is the two crops and its confidence is the "
        "share of close calls in that district and band where the crop planted was one of the two. A one-crop rate "
        "below 50 percent turns into a two-crop answer with the two-crop rate of all fields.",
        "4. Level. High from 75 percent, medium from 50 percent, as before. Below 50 percent the map gives no answer "
        "instead of a low one, so a farmer never gets one crop with a low confidence.", "",
        "## The table", "",
        "Share right, of how many tested fields. Within 5 km: the fields' own inputs (the tuning_v1 test). 5 to 10 km "
        "and 10 to 20 km: the same fields predicted with the inputs of a training block that far away in their own "
        f"district. A cell with fewer than {mm.MIN_TESTED_FIELDS} fields takes the value of the four districts.", "",
        _md(pv, index=False), "",
        "## Replay on the tested fields (season B, within 5 km)", "",
        "What each of the 1,904 tested fields would have been shown, and how often it was right (first crop for a "
        "one-crop answer, one of the two for a two-crop answer).", "",
        _md(rpv), "",
        "## Replay on the map (season B, a point about every 1 km)", "",
        f"{len(grid):,} points. Before: the rule of the morning of 25 Nov (district hit rate, one step lower for a close "
        "call or data more than 5 km away). Now: this table.", "",
        _md(gpv, index=False), "",
        "## How to read this", "",
        "- These are tested rates, not promises: about 1 in 4 high-confidence answers was still wrong in testing.",
        "- A two-crop answer means both crops fit the field about equally. The farmer can choose between them, for "
        "example by the seed and market they have.",
        "- The rates within 5 km are conservative for a point inside a surveyed block: the model on the map has seen "
        "the block's fields, the tested models had not.",
    ]
    path = OUT / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid-step", type=float, default=GRID_STEP, help="Grid step in degrees (default 0.01).")
    args = ap.parse_args(argv)
    t0 = time.time()
    raw = pd.read_csv(mm.TABLE_PATH, low_memory=False)
    df = ev.prepare(raw).reset_index(drop=True)
    base = mm.MapRecommender(raw, mm.Districts.load(), confidence_table=None)
    pred = fold_predictions(df, base)
    check = builtin_check(pred)
    if not check["passed"]:
        raise SystemExit(f"Built-in check failed: {check['districts']}")
    table = confidence_table(pred)
    lookup = {(r.district, r.band, r.kind): {"rate": r.rate, "fields": r.fields, "pooled": r.pooled}
              for r in table.itertuples()}
    replay = replay_tested_fields(pred, base, lookup, df)
    base.confidence = lookup                 # the map with the new table, for the grid
    grid = scan_grid(base, args.grid_step)
    import sklearn
    info = {"run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "git_commit": ev._git_commit(),
            "uncommitted_changes": ev._git_has_uncommitted_changes(), "table": "01-crop-recommendation/data/"
            "processed/feature_table_v6b.csv", "table_sha256": ev.file_sha256(mm.TABLE_PATH),
            "model": mm.MODEL_NAME, "folds": "5 folds over block_id (StratifiedGroupKFold, seed 0), settings of "
            "tuning_v1/chosen_settings.csv", "close_call": mm.CLOSE_CALL, "bands": [b for _, b in mm.DATA_BANDS],
            "min_tested_fields": mm.MIN_TESTED_FIELDS, "levels": {"high": mm.HIGH_HIT, "medium": mm.LOW_HIT},
            "grid_step_degrees": args.grid_step, "builtin_check": check, "minutes": round((time.time() - t0) / 60, 1),
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__}}
    path = write_report(table, replay, grid, info)
    print(f"Built-in check passed. Report written to {path} ({info['minutes']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
