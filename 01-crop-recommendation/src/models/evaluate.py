"""Evaluation harness: leave-one-district-out (LODO) testing for crop recommendation.

Each run trains on three districts and tests on the fourth, which the model has never seen.
Every model (B0, B1, Random Forest, XGBoost) goes through exactly the same folds and metrics.

Run from the repo root:
    python 01-crop-recommendation/src/models/evaluate.py
    python 01-crop-recommendation/src/models/evaluate.py --table 01-crop-recommendation/data/processed/feature_table.csv --out 01-crop-recommendation/reports/eval_v0

Metrics per held-out district: top-1 accuracy, top-3 accuracy, macro-F1, recall per crop.
Note: with only 4 crops, top-3 is easy (random guessing gets 75 percent), so read top-1 and macro-F1 first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_baseline import ExpertRulesBaseline, MajorityBaseline  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULT_TABLE = REPO / "01-crop-recommendation" / "data" / "processed" / "feature_table.csv"
DEFAULT_OUT = REPO / "01-crop-recommendation" / "reports" / "eval_v0"

# Standard atmosphere lapse rate: air cools about 6.5 C per 1000 m of height (ICAO standard atmosphere).
LAPSE_RATE_C_PER_M = 0.0065

LABEL_MAP = {"Maize": "maize", "Beans": "beans", "Irish potatoes": "irish_potato", "Rice": "rice"}
GROUP_COL = "district"

# Only these columns may be used as model inputs. Anything else is refused.
ALLOWED_FEATURES = [
    "elevation_m", "precip_mm", "tmax_c", "tmin_c", "tmean_c", "solar_mj_m2", "tmean_point_c", "tmin_point_c",
    "nitrogen_0_20cm", "phosphorous_extractable_0_20cm", "potassium_extractable_0_20cm", "ph_0_20cm",
    "sand_0_20cm", "clay_0_20cm", "silt_0_20cm", "carbon_organic_0_20cm", "cation_exchange_capacity_0_20cm",
]
# Columns that describe or are derived from the crop itself. Using them leaks the answer.
FORBIDDEN_PREFIXES = ("nisr_",)
FORBIDDEN_COLUMNS = {"label", "Crop_plant", "landcover", "Mixed_crop", "All_crops", "SA_crop", "SA_prod",
                     "SB_yield", "Crop_sys", "Crop_area", "Crop_propo", "Seed_var", "Var_reason", "Growth_stg",
                     "Plant_date", "Har_season", "Har_period", "Harvest", "Fert_type", "Base_fert", "Base_quant",
                     "Topdr_fert", "Topdr_quan", "Comment", "District", "district", "Sector", "Cell", "Village"}


class LeakageError(ValueError):
    pass


def check_features(features: Iterable[str]) -> list[str]:
    features = list(features)
    bad = [f for f in features if f in FORBIDDEN_COLUMNS or f.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise LeakageError(f"these columns leak the label or the district and cannot be features: {bad}")
    unknown = [f for f in features if f not in ALLOWED_FEATURES]
    if unknown:
        raise LeakageError(f"not in the allowed feature list (add them on purpose first): {unknown}")
    return features


def prepare(df: pd.DataFrame, label_col: str = "Crop_plant", min_elev: float = 900.0,
            max_lon: float = 31.0) -> pd.DataFrame:
    """Apply Part A's filters, keep pure crops, and add derived columns. Returns a new frame."""
    out = df.copy()
    n0 = len(out)
    out = out[(out["GPS_Elev"] > min_elev) & (out["longitude"] < max_lon)]
    out = out[out[label_col].isin(LABEL_MAP)]
    out["label"] = out[label_col].map(LABEL_MAP)
    out["elevation_m"] = out["GPS_Elev"].astype(float)
    out[GROUP_COL] = out["District"]
    if {"tmax_c", "tmin_c"} <= set(out.columns):
        out["tmean_c"] = (out["tmax_c"] + out["tmin_c"]) / 2
        # NASA POWER gives one temperature per district. Move it to each point's own height with the
        # standard lapse rate. Reference height = median CIAT elevation of the district (an assumption,
        # because the NASA POWER cell height is not stored in the table). Uses inputs only, never labels.
        ref = out.groupby(GROUP_COL)["elevation_m"].transform("median")
        shift = LAPSE_RATE_C_PER_M * (out["elevation_m"] - ref)
        out["tmean_point_c"] = out["tmean_c"] - shift
        out["tmin_point_c"] = out["tmin_c"] - shift
    out.attrs["rows_before_filters"] = n0
    return out.reset_index(drop=True)


class SklearnModel:
    """Wraps a scikit-learn style classifier with a fixed feature list and median filling of gaps."""

    def __init__(self, name: str, estimator, features: list[str]):
        self.name, self.estimator, self.features = name, estimator, check_features(features)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        cols = [c for c in self.features if c in X.columns and X[c].notna().any()]
        if not cols:
            raise LeakageError(f"{self.name}: none of its features are present with data")
        self.used_ = cols
        self.fill_ = X[cols].median()
        self.classes_ = np.array(sorted(y.unique()))
        self._index = {c: i for i, c in enumerate(self.classes_)}
        self.estimator.fit(X[cols].fillna(self.fill_), y.map(self._index))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(X[self.used_].fillna(self.fill_))
        full = np.zeros((len(X), len(self.classes_)))
        full[:, list(self.estimator.classes_)] = p
        return full


def default_models(features: list[str]) -> list:
    from sklearn.ensemble import RandomForestClassifier
    models = [MajorityBaseline(), ExpertRulesBaseline(),
              SklearnModel("RF_elevation_only", RandomForestClassifier(
                  n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0), ["elevation_m"]),
              SklearnModel("RF_all_allowed", RandomForestClassifier(
                  n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0), features)]
    try:
        from xgboost import XGBClassifier
        models.append(SklearnModel("XGB_all_allowed", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, eval_metric="mlogloss"), features))
    except ImportError:
        pass
    return models


def topk(proba: np.ndarray, classes: np.ndarray, y_true: Iterable[str], k: int) -> float:
    order = np.argsort(-proba, axis=1)[:, :k]
    hits = [t in set(classes[o]) for t, o in zip(y_true, order)]
    return float(np.mean(hits)) if hits else float("nan")


def run_lodo(df: pd.DataFrame, models: list, all_labels: Optional[list[str]] = None) -> pd.DataFrame:
    labels = all_labels or sorted(df["label"].unique())
    rows = []
    for held_out in sorted(df[GROUP_COL].unique()):
        train, test = df[df[GROUP_COL] != held_out], df[df[GROUP_COL] == held_out]
        assert set(train[GROUP_COL]).isdisjoint(set(test[GROUP_COL]))
        for model in models:
            m = model.fit(train, train["label"])
            proba = m.predict_proba(test)
            pred = m.classes_[np.argmax(proba, axis=1)]
            rec = recall_score(test["label"], pred, labels=labels, average=None, zero_division=0)
            row = {"model": model.name, "held_out": held_out, "n_test": len(test), "n_train": len(train),
                   "top1": float(np.mean(pred == test["label"].to_numpy())),
                   "top3": topk(proba, m.classes_, test["label"], 3),
                   "macro_f1": f1_score(test["label"], pred, labels=labels, average="macro", zero_division=0)}
            present = set(test["label"])
            for lab, r in zip(labels, rec):
                row[f"recall_{lab}"] = float(r) if lab in present else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby("model")
    summary = pd.DataFrame({
        "top1_mean": g["top1"].mean(), "top3_mean": g["top3"].mean(), "macro_f1_mean": g["macro_f1"].mean(),
        "top1_weighted": (results["top1"] * results["n_test"]).groupby(results["model"]).sum() / g["n_test"].sum(),
        "worst_district_top1": g["top1"].min(),
    })
    return summary.sort_values("macro_f1_mean", ascending=False)


def _md(frame: pd.DataFrame, index: bool = True) -> str:
    try:
        return frame.to_markdown(index=index)
    except ImportError:  # tabulate not installed
        return "```\n" + frame.to_string(index=index) + "\n```"


def write_report(df: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame, out: Path, features: list[str]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "lodo_results.csv", index=False)
    summary.to_csv(out / "lodo_summary.csv")
    counts = pd.crosstab(df[GROUP_COL], df["label"])
    lines = [
        "# Leave-one-district-out evaluation (v0)", "",
        f"Rows used: {len(df)} of {df.attrs.get('rows_before_filters', 'n/a')} "
        "(filters: elevation above 900 m, longitude below 31.0, pure crops only, mixed crops dropped).", "",
        f"Features allowed for ML models: {', '.join(features)}. NISR columns are excluded because they are "
        "looked up from the crop label.", "",
        "## Rows per district and crop", "", _md(counts), "",
        "## Summary (mean over the four held-out districts)", "", _md(summary.round(3)), "",
        "## Per held-out district", "", _md(results.round(3), index=False), "",
        "## How to read this", "",
        "- Top-3 is weak with four crops: random guessing already gets 75 percent. Read top-1 and macro-F1 first.",
        "- B0 always gives the training share of each crop. B1 uses published ranges only and is not trained.",
        "- ML is adopted only if it beats B1 on macro-F1 in most held-out districts.",
        "- Climate is one value per district in this table, so climate features mostly tell the model which district it is.",
    ]
    path = out / "lodo_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"rows": len(df), "features": features}, indent=2))
    return path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--label-col", default="Crop_plant")
    args = ap.parse_args(argv)
    df = prepare(pd.read_csv(args.table), label_col=args.label_col)
    features = check_features([c for c in ALLOWED_FEATURES if c in df.columns and df[c].notna().any()])
    results = run_lodo(df, default_models(features))
    summary = summarise(results)
    path = write_report(df, results, summary, Path(args.out), features)
    print(summary.round(3).to_string())
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
