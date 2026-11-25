"""How the map's one answer and its confidence level behave on the CIAT fields, with the method that answers in each
district chosen by main score (the rule until 25 Nov 2026) or by first crop right (the rule since then).

It replays the map's choice on the out-of-fold predictions of the new-fields test of the model on the map
(reports/tuning_v1): no field is predicted by a model that saw its 2 km block. For each field it takes the method
that answers in its district, that method's first crop, whether the first two crops are a close call (less than
0.10 apart, a tie included), and the confidence level the map would give (season B, data in the same block). It
prints how often the first crop is right at each level and how the levels are spread over the districts.

It needs the saved jobs of the tuning run (reports/tuning_v1/_work/RF_all_sat.pkl, not committed). Without them,
run `python 01-crop-recommendation/src/models/tune.py` first (about an hour on two cores; finished jobs are kept).
Run from the repo root:  python 01-crop-recommendation/src/recommendation/check_map_confidence.py
"""
from __future__ import annotations

import pickle
import sys
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

JOB = mm.EVAL_DIR / "_work" / "RF_all_sat.pkl"          # the model on the map, tuning_v1


def replay(choice: dict, df: pd.DataFrame, proba: np.ndarray, hit_rates: dict) -> pd.DataFrame:
    """One row per field: district, confidence level the map would give, and whether its first crop was right."""
    crops = ev.CROPS
    y, dist = df["label"].to_numpy(), df["district"].to_numpy()
    rules = b1.ExpertRulesBaseline().fit(df, df["label"])
    rules_first = rules.classes_[np.argmax(rules.predict_proba(df), axis=1)]
    model_first = np.array(crops)[np.argmax(proba, axis=1)]
    loaded = b1.load_rules()
    scores = np.array([[np.nan if (s := b1.explain_row(r, loaded)[c]["score"]) is None else s for c in crops]
                       for r in df.to_dict("records")])

    def gap(m: np.ndarray) -> np.ndarray:
        s = -np.sort(-np.nan_to_num(m, nan=-1.0), axis=1)
        return s[:, 0] - s[:, 1]

    gap_model, gap_rules = gap(proba), gap(scores)
    rows = []
    for i in range(len(df)):
        method = choice[dist[i]]
        first = model_first[i] if method == "model" else rules_first[i]
        close = bool((gap_model[i] if method == "model" else gap_rules[i]) < mm.CLOSE_CALL)
        level = mm.confidence_level(hit_rates[dist[i]][method], True, close, False)
        rows.append({"district": dist[i], "method": method, "level": level, "close_call": close,
                     "first_crop_right": bool(first == y[i])})
    return pd.DataFrame(rows)


def main() -> int:
    if not JOB.exists():
        raise SystemExit(f"Missing {JOB}. Run 01-crop-recommendation/src/models/tune.py first (see the docstring).")
    df = ev.prepare(pd.read_csv(mm.TABLE_PATH, low_memory=False))
    with open(JOB, "rb") as fh:
        proba = pickle.load(fh)["blockcv_proba"]["tuned"]
    rec = mm.default_recommender()
    tested, hits = rec.tested, rec.hit_rates
    by_main_score = {d: "model" if tested[d]["model"]["new_fields"] > tested[d]["rules"]["new_fields"] + mm.CHOICE_MARGIN
                     else "rules" for d in tested}
    by_first_crop = {d: "model" if hits[d]["model"] > hits[d]["rules"] + mm.CHOICE_MARGIN else "rules" for d in hits}
    assert by_first_crop == {d: rec.method_for(d) for d in hits}, "the map no longer uses first crop right"
    pd.set_option("display.width", 160)
    for name, choice in (("Until 25 Nov: method chosen by main score", by_main_score),
                         ("Since 25 Nov: method chosen by first crop right", by_first_crop)):
        t = replay(choice, df, proba, hits)
        print(f"\n== {name}: {choice}")
        print(f"First crop right, all fields: {t['first_crop_right'].mean():.3f}")
        print(t.groupby("level").agg(fields=("level", "size"), first_crop_right=("first_crop_right", "mean")).round(3))
        print("Share of each level in each district:")
        print(pd.crosstab(t["district"], t["level"], normalize="index").round(3))
        print("Close calls:", t.groupby("district")["close_call"].mean().round(3).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
