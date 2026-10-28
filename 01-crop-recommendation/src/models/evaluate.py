"""Evaluation harness v1 for crop recommendation. It follows docs/training_protocol.md.

Main test: leave one district out. Train on three districts, test on the fourth, which the model has never seen.
Second test: new fields inside the four known districts. 5 folds, and every 2 km block (block_id) stays on one
side, so neighbouring fields are never split between training and test.
Second check: both tests again on the fields CIAT marked Status == "Used" in its own satellite crop model.

Every model goes through the same folds and the same scores. ML replaces B1 only if it passes the adoption rule
on the main test (see ADOPT_* below and the protocol, rule 7).

Run from the repo root:
    python 01-crop-recommendation/src/models/evaluate.py
Options: --table PATH, --out DIR, --skip-second-check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_baseline import ExpertRulesBaseline, MajorityBaseline, adjust_to_height  # noqa: E402

PROTOCOL_VERSION = "1.0"
REPO = Path(__file__).resolve().parents[3]
PROCESSED = REPO / "01-crop-recommendation" / "data" / "processed"
# v2 is the training table. v1 is only a fallback for old checkouts; its crop-specific NISR columns are refused.
TABLE_CANDIDATES = [PROCESSED / "feature_table_v2.csv", PROCESSED / "feature_table.csv"]
DEFAULT_OUT = REPO / "01-crop-recommendation" / "reports" / "eval_v1"

LABEL_MAP = {"Maize": "maize", "Beans": "beans", "Irish potatoes": "irish_potato", "Rice": "rice"}
CROPS = sorted(LABEL_MAP.values())
GROUP_COL = "district"
BLOCK_COL = "block_id"
SEED = 0
N_BLOCK_FOLDS = 5

# Main score: macro-F1 over the crops with at least this many fields in the test district (protocol, rule 5).
MIN_CROP_FIELDS = 5
# Adoption rule (protocol, rule 7): higher mean main score than B1 AND better than B1 by more than
# ADOPT_MARGIN in at least ADOPT_MIN_WINS of the four districts.
BASELINE = "B1_expert_rules"
ADOPT_MARGIN = 0.01
ADOPT_MIN_WINS = 3

CLIMATE = ["precip_mm", "tmax_c", "tmin_c", "tmean_c", "solar_mj_m2"]
POINT_TEMP = ["tmean_point_c", "tmin_point_c"]
NISR_SHARES = ["nisr_share_beans", "nisr_share_irish_potato", "nisr_share_maize", "nisr_share_paddy_rice"]
SOIL_PROPS = ["nitrogen", "phosphorous_extractable", "potassium_extractable", "ph", "sand", "clay", "silt",
              "carbon_organic", "cation_exchange_capacity"]
SOIL_TOP = [f"{p}_0_20cm" for p in SOIL_PROPS]
SOIL_SUB = [f"{p}_20_50cm" for p in SOIL_PROPS]

# Feature views (protocol, rule 6). Elevation is in every view because it is known for any field.
VIEWS = {
    "district": ["elevation_m"] + CLIMATE + POINT_TEMP + NISR_SHARES,  # what is known without a soil map
    "field": ["elevation_m"] + SOIL_TOP,                               # measured at the field itself
    "all": ["elevation_m"] + CLIMATE + POINT_TEMP + NISR_SHARES + SOIL_TOP,
}
EXTRA_VIEWS = {"all_subsoil": VIEWS["all"] + SOIL_SUB}  # extra test: soil at 20 to 50 cm (Random Forest only)

# Only these columns may be used as model inputs. Anything else is refused.
ALLOWED_FEATURES = list(dict.fromkeys(f for feats in {**VIEWS, **EXTRA_VIEWS}.values() for f in feats))
# Columns that describe the crop, identify the field or the district, or come from CIAT's own selection.
# Using them leaks the answer. NISR statistics looked up with a field's own crop (table v1) are refused;
# district crop shares are allowed.
FORBIDDEN_PREFIXES = ("nisr_",)
ALLOWED_PREFIX_EXCEPTIONS = ("nisr_share_",)
FORBIDDEN_COLUMNS = {"label", "Crop_plant", "landcover", "Mixed_crop", "All_crops", "SA_crop", "SA_prod",
                     "SB_yield", "Crop_sys", "Crop_area", "Crop_propo", "Seed_var", "Var_reason", "Growth_stg",
                     "Plant_date", "Har_season", "Har_period", "Harvest", "Fert_type", "Base_fert", "Base_quant",
                     "Topdr_fert", "Topdr_quan", "Comment", "District", "district", "Sector", "Cell", "Village",
                     "Status", "qc_ok", "block_id", "F_id", "F_index", "Water_sour", "Land_conso", "GPS_Prec"}


class LeakageError(ValueError):
    pass


def check_features(features: Iterable[str]) -> list[str]:
    features = list(features)
    bad = [f for f in features if f in FORBIDDEN_COLUMNS
           or (f.startswith(FORBIDDEN_PREFIXES) and not f.startswith(ALLOWED_PREFIX_EXCEPTIONS))]
    if bad:
        raise LeakageError(f"these columns leak the label, the field or the district and cannot be features: {bad}")
    unknown = [f for f in features if f not in ALLOWED_FEATURES]
    if unknown:
        raise LeakageError(f"not in the allowed feature list (add them on purpose first): {unknown}")
    return features


def _first_present(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of these columns are in the table: {names}")


def default_table() -> Path:
    for p in TABLE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No feature table found. Expected one of: "
                            + ", ".join(str(p) for p in TABLE_CANDIDATES)
                            + ". Get the latest from Part A (git pull) or pass --table.")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _qc_mask(df: pd.DataFrame, min_elev: float, max_lon: float) -> pd.Series:
    """Part A's GPS checks. Use qc_ok when the table has it (coordinates are no longer published)."""
    if "qc_ok" in df.columns:
        return df["qc_ok"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    lon_col = _first_present(df, ["GPS_Long", "longitude", "longitude_x"])
    return (df["GPS_Elev"] > min_elev) & (df[lon_col] < max_lon)


def prepare(df: pd.DataFrame, label_col: str = "Crop_plant", status: Optional[str] = None,
            min_elev: float = 900.0, max_lon: float = 31.0) -> pd.DataFrame:
    """Apply the GPS checks, keep single-crop labels (and one CIAT Status if asked), add derived columns."""
    out = df.copy()
    n0 = len(out)
    out = out[_qc_mask(out, min_elev, max_lon)].copy()
    n_qc = len(out)
    out["elevation_m"] = out["GPS_Elev"].astype(float)
    out[GROUP_COL] = out["District"]
    # Reference height for the temperature adjustment: median elevation of all CIAT fields in the district
    # that pass the GPS checks (labelled or not, any Status). The service uses the same reference.
    ref = out.groupby(GROUP_COL)["elevation_m"].transform("median")
    keep = out[label_col].isin(LABEL_MAP)
    if status is not None:
        if "Status" not in out.columns:
            raise KeyError("this table has no Status column")
        keep &= out["Status"].eq(status)
    out, ref = out[keep].copy(), ref[keep]
    out["label"] = out[label_col].map(LABEL_MAP)
    if {"tmax_c", "tmin_c"} <= set(out.columns):
        out["tmean_c"] = (out["tmax_c"] + out["tmin_c"]) / 2
        # NASA POWER gives one temperature per district. Move it to each field's own height with the
        # standard lapse rate (an approximation, because the NASA POWER cell height is not stored).
        out["tmean_point_c"] = adjust_to_height(out["tmean_c"], out["elevation_m"], ref)
        out["tmin_point_c"] = adjust_to_height(out["tmin_c"], out["elevation_m"], ref)
    out = out.reset_index(drop=True)
    out.attrs.update(rows_before_filters=n0, rows_after_qc=n_qc, status_filter=status)
    return out


class SklearnModel:
    """Wraps a scikit-learn style classifier with a fixed feature list and median filling of gaps.

    balance=True passes class-balanced sample weights to fit (used for XGBoost, which has no class_weight).
    """

    def __init__(self, name: str, estimator, features: list[str], balance: bool = False):
        self.name, self.estimator, self.features = name, estimator, check_features(features)
        self.balance = balance

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        cols = [c for c in self.features if c in X.columns and X[c].notna().any()]
        if not cols:
            raise LeakageError(f"{self.name}: none of its features are present with data")
        self.used_ = cols
        self.fill_ = X[cols].median()
        self.classes_ = np.array(sorted(y.unique()))
        self._index = {c: i for i, c in enumerate(self.classes_)}
        yi = y.map(self._index).to_numpy()
        kwargs = {}
        if self.balance:
            counts = np.bincount(yi, minlength=len(self.classes_))
            kwargs["sample_weight"] = (len(yi) / (len(self.classes_) * counts))[yi]
        self.estimator.fit(X[cols].fillna(self.fill_), yi, **kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(X[self.used_].fillna(self.fill_))
        full = np.zeros((len(X), len(self.classes_)))
        full[:, list(self.estimator.classes_)] = p
        return full


def _require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as e:  # the protocol includes XGBoost, so a run without it would not be comparable
        raise ImportError("XGBoost is part of the training protocol. Install it once with: "
                          "python -m pip install xgboost") from e
    return XGBClassifier


def default_models(views: dict = VIEWS, extra_views: dict = EXTRA_VIEWS, n_estimators: int = 300) -> list:
    """B0, B1, then Logistic Regression, Random Forest and XGBoost on each view (fixed settings, seed 0).
    n_estimators stays 300 for every reported run; tests lower it only to run faster."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    XGBClassifier = _require_xgboost()

    def rf():
        return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)

    models = [MajorityBaseline(), ExpertRulesBaseline()]
    for view, feats in views.items():
        models.append(SklearnModel(f"LR[{view}]", make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")), feats))
        models.append(SklearnModel(f"RF[{view}]", rf(), feats))
        models.append(SklearnModel(f"XGB[{view}]", XGBClassifier(
            n_estimators=n_estimators, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss",
            n_jobs=4), feats, balance=True))
    for view, feats in extra_views.items():
        models.append(SklearnModel(f"RF[{view}]", rf(), feats))
    return models


def topk(proba: np.ndarray, classes: np.ndarray, y_true: Iterable[str], k: int) -> float:
    order = np.argsort(-proba, axis=1)[:, :k]
    hits = [t in set(classes[o]) for t, o in zip(y_true, order)]
    return float(np.mean(hits)) if hits else float("nan")


def scores(y_true, pred, proba: np.ndarray, classes: np.ndarray, labels: list[str] = CROPS,
           min_fields: int = MIN_CROP_FIELDS) -> dict:
    """All scores for one test district. The main score is macro_f1 (crops with at least min_fields fields)."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    counts = pd.Series(y_true).value_counts()
    scored = sorted(counts[counts >= min_fields].index)
    out = {
        "n_test": int(len(y_true)),
        "top1": float(np.mean(pred == y_true)) if len(y_true) else float("nan"),
        "top3": topk(proba, classes, y_true, 3),
        "macro_f1": (float(f1_score(y_true, pred, labels=scored, average="macro", zero_division=0))
                     if scored else float("nan")),
        "macro_f1_all4": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "crops_scored": "+".join(scored),
    }
    rec = recall_score(y_true, pred, labels=labels, average=None, zero_division=0)
    present = set(y_true)
    for lab, r in zip(labels, rec):
        out[f"recall_{lab}"] = float(r) if lab in present else float("nan")
    return out


def _confusion(analysis: str, test: str, model: str, district: str, y_true, pred,
               labels: list[str] = CROPS) -> pd.DataFrame:
    tab = pd.crosstab(np.asarray(y_true), np.asarray(pred)).reindex(index=labels, columns=labels, fill_value=0)
    tab.index.name, tab.columns.name = "true", "predicted"
    long = tab.reset_index().melt(id_vars="true", var_name="predicted", value_name="count")
    long = long[long["count"] > 0]
    return long.assign(analysis=analysis, test=test, model=model, district=district)[
        ["analysis", "test", "model", "district", "true", "predicted", "count"]]


def run_lodo(df: pd.DataFrame, models: list, labels: list[str] = CROPS,
             analysis: str = "main") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main test. Returns one row per (model, held-out district) and the confusion counts."""
    rows, conf = [], []
    for held_out in sorted(df[GROUP_COL].unique()):
        train, test = df[df[GROUP_COL] != held_out], df[df[GROUP_COL] == held_out]
        assert set(train[GROUP_COL]).isdisjoint(set(test[GROUP_COL]))
        for model in models:
            m = model.fit(train, train["label"])
            proba = m.predict_proba(test)
            pred = m.classes_[np.argmax(proba, axis=1)]
            rows.append({"model": model.name, "district": held_out, "n_train": int(len(train)),
                         **scores(test["label"], pred, proba, m.classes_, labels)})
            conf.append(_confusion(analysis, "leave_one_district_out", model.name, held_out, test["label"], pred,
                                   labels))
    return pd.DataFrame(rows), pd.concat(conf, ignore_index=True)


def block_folds(df: pd.DataFrame, n_splits: int = N_BLOCK_FOLDS, seed: int = SEED) -> list:
    """5 folds over 2 km blocks, stratified by crop. A block is never on both sides of a fold."""
    if BLOCK_COL not in df.columns or df[BLOCK_COL].isna().any():
        raise KeyError(f"the second test needs a complete {BLOCK_COL} column from Part A")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df[BLOCK_COL].astype(str).to_numpy()
    folds = list(cv.split(df, df["label"], groups=groups))
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))
    return folds


def run_block_cv(df: pd.DataFrame, models: list, labels: list[str] = CROPS, analysis: str = "main",
                 n_splits: int = N_BLOCK_FOLDS, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Second test. Every field is predicted once, by a model that never saw its 2 km block.
    Scores are then computed per district, so the table has the same shape as the main test."""
    folds = block_folds(df, n_splits, seed)
    labels_arr = np.array(labels)
    rows, conf = [], []
    for model in models:
        proba = np.zeros((len(df), len(labels)))
        for tr, te in folds:
            train, test = df.iloc[tr], df.iloc[te]
            m = model.fit(train, train["label"])
            cols = [labels.index(c) for c in m.classes_]
            proba[np.ix_(te, cols)] = m.predict_proba(test)
        pred = labels_arr[np.argmax(proba, axis=1)]
        for d in sorted(df[GROUP_COL].unique()):
            mask = (df[GROUP_COL] == d).to_numpy()
            y = df.loc[mask, "label"]
            rows.append({"model": model.name, "district": d, "n_train": float("nan"),
                         **scores(y, pred[mask], proba[mask], labels_arr, labels)})
            conf.append(_confusion(analysis, "new_fields_known_districts", model.name, d, y, pred[mask], labels))
    return pd.DataFrame(rows), pd.concat(conf, ignore_index=True)


def summarise(results: pd.DataFrame, baseline: str = BASELINE, margin: float = ADOPT_MARGIN,
              min_wins: int = ADOPT_MIN_WINS) -> pd.DataFrame:
    """Mean scores per model and the adoption decision against B1 (protocol, rule 7)."""
    b = results[results["model"] == baseline].set_index("district")["macro_f1"]
    rows = []
    for name, g in results.groupby("model", sort=False):
        s = g.set_index("district")["macro_f1"]
        wins = int(((s - b.reindex(s.index)) > margin).sum())
        is_ml = name not in {baseline, MajorityBaseline.name}
        passes = ("yes" if (s.mean() > b.mean() and wins >= min_wins) else "no") if is_ml else ""
        rows.append({"model": name, "macro_f1_mean": s.mean(), "macro_f1_all4_mean": g["macro_f1_all4"].mean(),
                     "top1_mean": g["top1"].mean(), "top3_mean": g["top3"].mean(),
                     "worst_district_macro_f1": s.min(),
                     "districts_beating_B1": wins if name != baseline else float("nan"), "passes_rule": passes})
    return pd.DataFrame(rows).set_index("model").sort_values("macro_f1_mean", ascending=False)


def _md(frame: pd.DataFrame, index: bool = True) -> str:
    try:
        return frame.to_markdown(index=index)
    except ImportError:  # tabulate not installed
        return "```\n" + frame.to_string(index=index) + "\n```"


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    return _git("rev-parse", "HEAD") or None


def _git_has_uncommitted_changes() -> Optional[bool]:
    """True when tracked files differ from the commit, so the recorded commit is not the exact code that ran."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def _decision(summary: pd.DataFrame, deciding: bool = True) -> str:
    """One line for the report. Only the main test (deciding=True) can change which method is served."""
    passing = summary.index[summary["passes_rule"] == "yes"].tolist()
    ml = summary[summary["passes_rule"].isin(["yes", "no"])]
    if ml.empty:
        return "No ML model was evaluated."
    best = ml.index[0]
    detail = (f"Best ML model: {best}, mean macro-F1 {ml['macro_f1_mean'].iloc[0]:.3f} vs B1 "
              f"{summary.loc[BASELINE, 'macro_f1_mean']:.3f}, better than B1 in "
              f"{int(ml['districts_beating_B1'].iloc[0])} of 4 districts.")
    if deciding:
        if passing:
            return f"Adopted: {', '.join(passing)} pass the adoption rule on the main test. {detail}"
        return f"No ML model passes the adoption rule, so B1 stays. {detail}"
    if passing:
        return (f"For information only (this test does not decide): {', '.join(passing)} would pass the same "
                f"rule here. {detail}")
    return f"For information only (this test does not decide): no ML model would pass the same rule here. {detail}"


def _section(title: str, df: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame,
             deciding: bool = False) -> list[str]:
    per_district = results.pivot(index="model", columns="district", values="macro_f1").reindex(summary.index)
    recall = results.groupby("model")[[f"recall_{c}" for c in CROPS]].mean().reindex(summary.index)
    recall.columns = [c.replace("recall_", "") for c in recall.columns]
    return [f"### {title}", "", f"**{_decision(summary, deciding)}**", "",
            "Mean over the four districts:", "", _md(summary.round(3)), "",
            "Main score (macro-F1) per district:", "", _md(per_district.round(3)), "",
            "Recall per crop (mean over the districts where the crop is grown):", "", _md(recall.round(2)), ""]


def write_report(out: Path, info: dict, analyses: dict[str, pd.DataFrame], results: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    conf = []
    for name, res in results.items():
        for test in ("lodo", "blockcv"):
            if test in res:
                res[test].to_csv(out / f"{name}_{test}_results.csv", index=False)
                res[f"{test}_summary"].to_csv(out / f"{name}_{test}_summary.csv")
                conf.append(res[f"{test}_confusion"])
    pd.concat(conf, ignore_index=True).to_csv(out / "confusion_counts.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    main = analyses["main"]
    lines = [
        "# Crop recommendation evaluation (eval_v1)", "",
        f"Protocol: `docs/training_protocol.md` (version {PROTOCOL_VERSION}). Run at {info['run_at_utc']} UTC.",
        f"Table: `{info['table']}` (SHA-256 `{info['table_sha256'][:16]}...`). Code commit: "
        f"`{(info['git_commit'] or 'unknown')[:10]}`"
        + (" (with uncommitted changes, so rerun after committing)." if info.get("uncommitted_changes") else "."),
        "",
        f"Rows: {info['rows']['before_filters']} in the table, {info['rows']['after_qc_ok']} pass qc_ok, "
        f"{len(main)} single-crop fields in the main set"
        + (f", {len(analyses['used_only'])} in the CIAT 'Used' second check." if "used_only" in analyses else "."),
        "", "## Fields per district and crop (main set)", "", _md(pd.crosstab(main[GROUP_COL], main["label"],
                                                                              margins=True)), "",
        "## Main test: leave one district out", "",
        "Train on three districts, test on the fourth. This decides adoption.", "",
    ]
    lines += _section("Main set: leave one district out", main, results["main"]["lodo"],
                      results["main"]["lodo_summary"], deciding=True)
    if "blockcv" in results["main"]:
        lines += ["## Second test: new fields inside the four known districts", "",
                  "5 folds over 2 km blocks (block_id). Reported, but it does not decide adoption.", ""]
        lines += _section("Main set: new fields inside the known districts", main, results["main"]["blockcv"],
                          results["main"]["blockcv_summary"])
    if "used_only" in results:
        u = analyses["used_only"]
        lines += ["## Second check: CIAT 'Used' fields only", "",
                  "Same tests on the fields CIAT used in its own satellite crop model (Status == 'Used').", "",
                  _md(pd.crosstab(u[GROUP_COL], u["label"], margins=True)), ""]
        lines += _section("Used only: leave one district out", u, results["used_only"]["lodo"],
                          results["used_only"]["lodo_summary"])
        if "blockcv" in results["used_only"]:
            lines += _section("Used only: new fields inside the known districts", u,
                              results["used_only"]["blockcv"], results["used_only"]["blockcv_summary"])
    lines += [
        "## How to read this", "",
        f"- Main score: macro-F1 per test district, over the crops with at least {MIN_CROP_FIELDS} fields there, "
        "then the mean over the four districts. `macro_f1_all4` is the old definition (all four crops, so a crop "
        "a district does not grow counts as zero).",
        f"- Adoption: ML replaces B1 only if its mean main score is higher AND it beats B1 by more than "
        f"{ADOPT_MARGIN} in at least {ADOPT_MIN_WINS} of the 4 districts, on the main test.",
        "- Top-3 cannot decide anything with four crops: a fixed ranking that never looks at the field "
        "already scores close to 100 percent.",
        "- B0 gives the training share of each crop. B1 uses published crop ranges only and is not trained.",
        "- Views: district = elevation, season climate and NISR district shares; field = elevation and soil "
        "0 to 20 cm; all = both; all_subsoil = all plus soil 20 to 50 cm.",
        "- Climate and NISR shares are one value per district, so in the main test they mostly tell the model "
        "which district it is in, and the held-out district has values the model has never seen.",
        "- Full numbers: the CSV files in this folder. Settings and versions: run_info.json.",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate(raw: pd.DataFrame, label_col: str = "Crop_plant", second_check: bool = True,
             models: Optional[list] = None) -> tuple[dict, dict]:
    analyses = {"main": prepare(raw, label_col=label_col)}
    if second_check and "Status" in raw.columns:
        analyses["used_only"] = prepare(raw, label_col=label_col, status="Used")
    missing = [f for f in ALLOWED_FEATURES if f not in analyses["main"].columns]
    if missing:
        raise ValueError(f"the table is missing protocol features: {missing}")
    models = models or default_models()
    results = {}
    for name, df in analyses.items():
        lodo, lodo_conf = run_lodo(df, models, analysis=name)
        res = {"lodo": lodo, "lodo_summary": summarise(lodo), "lodo_confusion": lodo_conf}
        if BLOCK_COL in df.columns:
            blk, blk_conf = run_block_cv(df, models, analysis=name)
            res.update(blockcv=blk, blockcv_summary=summarise(blk), blockcv_confusion=blk_conf)
        results[name] = res
    return analyses, results


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=None, help="Feature table CSV. Default: data/processed/feature_table_v2.csv.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--label-col", default="Crop_plant")
    ap.add_argument("--skip-second-check", action="store_true", help="Do not rerun on CIAT 'Used' fields only.")
    args = ap.parse_args(argv)
    table = Path(args.table) if args.table else default_table()
    print(f"Feature table: {table}")
    raw = pd.read_csv(table, low_memory=False)
    models = default_models()
    analyses, results = evaluate(raw, args.label_col, not args.skip_second_check, models)

    import sklearn
    import xgboost
    try:
        table_rel = table.resolve().relative_to(REPO).as_posix()
    except ValueError:
        table_rel = str(table)
    main_df = analyses["main"]
    info = {
        "protocol_version": PROTOCOL_VERSION,
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "table": table_rel, "table_sha256": file_sha256(table), "git_commit": _git_commit(),
        "uncommitted_changes": _git_has_uncommitted_changes(),
        "rows": {"before_filters": main_df.attrs.get("rows_before_filters"),
                 "after_qc_ok": main_df.attrs.get("rows_after_qc"),
                 **{name: len(df) for name, df in analyses.items()}},
        "label": args.label_col, "crops": CROPS, "min_crop_fields": MIN_CROP_FIELDS,
        "adoption": {"baseline": BASELINE, "margin": ADOPT_MARGIN, "min_districts": ADOPT_MIN_WINS},
        "tests": {"main": "leave one district out",
                  "second": f"{N_BLOCK_FOLDS} folds over {BLOCK_COL} (StratifiedGroupKFold, seed {SEED})"},
        "views": {**VIEWS, **EXTRA_VIEWS},
        "models": {m.name: (repr(m.estimator) if hasattr(m, "estimator") else type(m).__name__) for m in models},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    path = write_report(Path(args.out), info, analyses, results)
    for name, res in results.items():
        for test in ("lodo", "blockcv"):
            if test in res:
                title = {"lodo": "leave one district out", "blockcv": "new fields in known districts"}[test]
                print(f"\n== {name}: {title}")
                print(res[f"{test}_summary"][["macro_f1_mean", "top1_mean", "districts_beating_B1",
                                              "passes_rule"]].round(3).to_string())
                print(_decision(res[f"{test}_summary"], deciding=(name == "main" and test == "lodo")))
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
