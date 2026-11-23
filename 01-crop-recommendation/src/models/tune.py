"""Tuning of Logistic Regression, Random Forest and XGBoost with nested tests (training protocol 1.2, section 11).

What it does, in order:
  1. Keeps the 1,904 training fields of feature_table_v6b.csv (the rows and inputs of version 1.1).
  2. Builds 61 candidate settings per method: the protocol settings of section 6 first, then 60 random draws
     (scikit-learn ParameterSampler, seed 0). The same candidates go through every fold and every view.
  3. Main test (leave one district out). Inside the three training districts, every candidate is scored by leaving
     out each training district in turn. The best is trained on the three districts and scored once on the
     held-out district.
  4. Second test (new fields). The 5 block folds of eval_v1 and eval_v2. Inside each training part, every candidate
     is scored on 3 folds over 2 km blocks. The best is trained on the training part and scored once on the test
     fold. The test fields never help choose the settings.
  5. The protocol settings are also scored on the same outer folds. They must give the numbers of eval_v2
     (built-in check).
  6. Also reported, without deciding anything: input importance (shuffling one input or a group of inputs among
     the test fields), how often the first crop is right at each level of the model's own probability, simple
     alternatives (B0, the district's most common crop, B1) and how mixed the 2 km blocks are.
  7. Applies the two decisions the protocol fixed before the run: the adoption rule on the main test, and the rule
     for the model on the dashboard map.

Every (method, view) pair is one job. A finished job is saved in <out>/_work, so a stopped run starts again where it
stopped. Run from the repo root (a few hours on two cores):
    python 01-crop-recommendation/src/models/tune.py
Options: --workers N (parallel jobs, default 2), --quick (a tiny run into a temporary folder, to check the code; its
numbers are never reported), --assemble-only (rebuild the report from the saved jobs), --table, --out
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import tempfile
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from expert_baseline import ExpertRulesBaseline, MajorityBaseline  # noqa: E402

PROTOCOL_VERSION = "1.2"
TABLE = ev.PROCESSED / "feature_table_v6b.csv"
DEFAULT_OUT = ev.REPORTS / "tuning_v1"
UNTUNED_DIR = ev.REPORTS / "eval_v2"     # the built-in check compares the protocol settings with these numbers
SEED = ev.SEED
METHODS = ("LR", "RF", "XGB")
VIEWS = ("all", "field_sat", "all_sat")
JOB_ORDER = ("RF", "XGB", "LR")           # slowest first, so two workers finish at about the same time
N_DRAWS = 60
INNER_BLOCK_FOLDS = 3
N_SHUFFLES = 5
TUNED = " tuned"                          # tuned models are reported as e.g. "XGB[all_sat] tuned"
MAP_MODEL = "XGB[all]"                    # the model on the map before this version (section 6 settings)
MAP_MARGIN = 0.01
CROPS = ev.CROPS
LABELS = np.array(CROPS)
BAND_EDGES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # bands of 0.1 for the probability of the first crop
SURE_FROM = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TESTS = {"lodo": "leave_one_district_out", "blockcv": "new_fields_known_districts"}

# Inputs measured at the field: they change from field to field. All the others are one value per district.
FIELD_INPUTS = (["elevation_m", "srtm_elevation_m"] + ev.POINT_TEMP + ev.SOIL_TOP + ev.SOIL_SUB
                + ["chirps_precip_mm", "ndvi_annual", "swir_ratio"])
# Inputs that carry the same information are also shuffled together (group importance).
GROUPS = {
    "elevation and field temperature": ["elevation_m", "srtm_elevation_m"] + ev.POINT_TEMP,
    "district climate": ev.CLIMATE,
    "NISR crop shares": ev.NISR_SHARES,
    "soil 0 to 20 cm": ev.SOIL_TOP,
    "satellite rainfall and vegetation": ["chirps_precip_mm", "ndvi_annual", "swir_ratio"],
}

# Section 6 settings, written out for every setting that is searched. Anything not listed keeps the library default,
# exactly as in evaluate.default_models.
PROTOCOL_SETTINGS = {
    "LR": {"C": 1.0, "shape": "straight", "weight_power": 1.0},
    "RF": {"criterion": "gini", "max_depth": None, "max_features": "sqrt", "max_samples": None,
           "min_samples_leaf": 5, "weight_power": 1.0},
    "XGB": {"colsample_bytree": 1.0, "gamma": 0.0, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1,
            "n_estimators": 300, "reg_lambda": 1.0, "subsample": 1.0, "weight_power": 1.0},
}


def search_spaces() -> dict:
    """The ranges of section 11. Lists are drawn evenly; loguniform and uniform are continuous."""
    from scipy.stats import loguniform, uniform
    return {
        "LR": {"C": loguniform(1e-3, 1e2), "shape": ["straight", "curved"], "weight_power": [0.0, 0.5, 1.0]},
        "RF": {"max_depth": [None, 6, 10, 16], "min_samples_leaf": [1, 2, 3, 5, 8, 12, 20],
               "max_features": ["sqrt", 0.2, 0.33, 0.5], "max_samples": [None, 0.7, 0.5],
               "criterion": ["gini", "entropy"], "weight_power": [0.0, 0.5, 1.0]},
        "XGB": {"n_estimators": [100, 200, 300, 500, 800], "learning_rate": loguniform(0.01, 0.3),
                "max_depth": [2, 3, 4, 5, 6, 8], "min_child_weight": [1, 2, 5, 10],
                "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                "reg_lambda": loguniform(0.1, 10), "gamma": [0.0, 0.1, 0.5, 1.0],
                "weight_power": [0.0, 0.5, 1.0]},
    }


def candidates(method: str, n_draws: int = N_DRAWS, seed: int = SEED) -> list[dict]:
    """Candidate 0 is the protocol's own settings, then n_draws random draws. Continuous values are rounded to six
    decimals so the report can show them; the rounded value is the one that is used."""
    from sklearn.model_selection import ParameterSampler
    out = [dict(sorted(PROTOCOL_SETTINGS[method].items()))]
    for draw in ParameterSampler(search_spaces()[method], n_iter=n_draws, random_state=seed):
        out.append({k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                    for k, v in sorted(draw.items())})
    return out


def class_weights(yi: np.ndarray, k: int, power: float) -> np.ndarray:
    """Weight of each crop: (n / (k x n_crop)) to the power p. p = 1 is scikit-learn's 'balanced' (section 6) and
    p = 0 gives every field the same weight."""
    counts = np.bincount(yi, minlength=k)
    return (len(yi) / (k * counts)) ** power


def build_estimator(method: str, params: dict, Xf: pd.DataFrame, yi: np.ndarray, k: int, n_jobs: int = 1):
    """The estimator of one candidate and the extra arguments of its fit (sample weights for XGBoost)."""
    power = float(params["weight_power"])

    def sk_class_weight():
        if power == 1.0:
            return "balanced"
        if power == 0.0:
            return None
        return dict(enumerate(class_weights(yi, k, power)))

    if method == "LR":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        lr = LogisticRegression(C=float(params["C"]), max_iter=5000, class_weight=sk_class_weight())
        if params["shape"] == "straight":
            return make_pipeline(StandardScaler(), lr), {}
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import SplineTransformer
        # Curves only for inputs measured at the field. An input that is the same for a whole district has two or
        # three values in the training fields, too few for a curve, so it stays straight, like a constant column.
        curved = [c for c in Xf.columns if c in FIELD_INPUTS and Xf[c].nunique() > 1]
        straight = [c for c in Xf.columns if c not in curved]
        parts = [("curved", SplineTransformer(n_knots=5, degree=3, knots="uniform", extrapolation="constant"),
                  curved)]
        if straight:
            parts.append(("straight", "passthrough", straight))
        return make_pipeline(ColumnTransformer(parts), StandardScaler(), lr), {}
    if method == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, criterion=params["criterion"], max_depth=params["max_depth"],
            max_features=params["max_features"], max_samples=params["max_samples"],
            min_samples_leaf=int(params["min_samples_leaf"]), class_weight=sk_class_weight(), random_state=SEED,
            n_jobs=n_jobs), {}
    if method == "XGB":
        XGBClassifier = ev._require_xgboost()
        est = XGBClassifier(
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]), min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]), gamma=float(params["gamma"]), random_state=SEED,
            eval_metric="mlogloss", n_jobs=n_jobs)
        return est, ({} if power == 0.0 else {"sample_weight": class_weights(yi, k, power)[yi]})
    raise ValueError(f"unknown method {method}")


class Tunable(ev.SklearnModel):
    """One method with one candidate's settings. Same inputs, median filling and leakage checks as the protocol
    models (evaluate.SklearnModel)."""

    def __init__(self, method: str, params: dict, view: str, n_jobs: int = 1):
        super().__init__(f"{method}[{view}]", None, ev.VIEWS_V11[view])
        self.method, self.params, self.view, self.n_jobs = method, dict(params), view, n_jobs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Tunable":
        cols = [c for c in self.features if c in X.columns and X[c].notna().any()]
        if not cols:
            raise ev.LeakageError(f"{self.name}: none of its features are present with data")
        self.used_ = cols
        self.fill_ = X[cols].median()
        self.classes_ = np.array(sorted(y.unique()))
        self._index = {c: i for i, c in enumerate(self.classes_)}
        yi = y.map(self._index).to_numpy()
        Xf = X[cols].fillna(self.fill_)
        self.estimator, kwargs = build_estimator(self.method, self.params, Xf, yi, len(self.classes_), self.n_jobs)
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            # A candidate that stops at 5,000 iterations is still a valid candidate: the inner test judges it.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.estimator.fit(Xf, yi, **kwargs)
        return self


class DistrictMajority:
    """Reference only, not a protocol model: the share of each crop in the field's own district among the training
    fields, so its first crop is the district's most common one. It reads the district name, which no model may use
    as an input. A district with no training fields gets the overall shares (B0)."""
    name = "B0d_district_majority"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DistrictMajority":
        self.classes_ = np.array(sorted(y.unique()))
        tab = pd.crosstab(X[ev.GROUP_COL].to_numpy(), y.to_numpy()).reindex(columns=self.classes_, fill_value=0)
        self.share_ = {d: (row / row.sum()).to_numpy(dtype=float) for d, row in tab.iterrows()}
        counts = y.value_counts()
        self.prior_ = np.array([counts.get(c, 0) / len(y) for c in self.classes_])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.share_.get(d, self.prior_) for d in X[ev.GROUP_COL].to_numpy()])


# ------------------------------------------------------------------ scores
def macro_f1(y_true, pred) -> float:
    """The main score of section 5 for one district: macro-F1 over the crops with at least 5 fields there. Same code
    as evaluate.scores, without the other scores (it runs thousands of times)."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    counts = pd.Series(y_true).value_counts()
    scored = sorted(counts[counts >= ev.MIN_CROP_FIELDS].index)
    if not scored:
        return float("nan")
    return float(f1_score(y_true, pred, labels=scored, average="macro", zero_division=0))


def main_score(y_true, pred, districts) -> float:
    """Main score per district, then the mean over the districts (section 5)."""
    y_true, pred, districts = np.asarray(y_true), np.asarray(pred), np.asarray(districts)
    vals = np.array([macro_f1(y_true[districts == d], pred[districts == d]) for d in sorted(set(districts))])
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def pick(scores: list[float]) -> int:
    """The best candidate. A tie goes to the one drawn first, so the protocol settings win every tie."""
    arr = np.array(scores, dtype=float)
    return int(np.argmax(np.where(np.isnan(arr), -np.inf, arr)))


def lodo_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fold per district: train on the others, test on it."""
    d = df[ev.GROUP_COL].to_numpy()
    return [(np.flatnonzero(d != x), np.flatnonzero(d == x)) for x in sorted(set(d))]


def lodo_score(df: pd.DataFrame, folds: list, make: Callable) -> float:
    """Mean main score over the districts left out in these folds."""
    y = df["label"].to_numpy()
    vals = []
    for tr, te in folds:
        m = make().fit(df.iloc[tr], df["label"].iloc[tr])
        vals.append(macro_f1(y[te], m.classes_[np.argmax(m.predict_proba(df.iloc[te]), axis=1)]))
    return float(np.nanmean(vals))


def oof_proba(df: pd.DataFrame, folds: list, make: Callable) -> np.ndarray:
    """Every field predicted once, by a model trained without its fold. Columns follow CROPS."""
    proba = np.zeros((len(df), len(CROPS)))
    for tr, te in folds:
        m = make().fit(df.iloc[tr], df["label"].iloc[tr])
        proba[np.ix_(te, [CROPS.index(c) for c in m.classes_])] = m.predict_proba(df.iloc[te])
    return proba


def block_score(df: pd.DataFrame, proba: np.ndarray) -> float:
    return main_score(df["label"].to_numpy(), LABELS[np.argmax(proba, axis=1)], df[ev.GROUP_COL].to_numpy())


# ------------------------------------------------------------------ importance
def importance_keys(used: list[str]) -> dict[str, list[str]]:
    keys = {f"input:{c}": [c] for c in used}
    for g, cols in GROUPS.items():
        inside = [c for c in cols if c in used]
        if inside:
            keys[f"group:{g}"] = inside
    return keys


def _rng(*parts) -> np.random.Generator:
    """A random generator fixed by seed 0 and the names given, so the shuffles do not depend on the job order."""
    return np.random.default_rng([SEED] + [zlib.crc32(str(p).encode()) for p in parts])


def shuffled_preds(model, test: pd.DataFrame, n_shuffles: int, tag: tuple) -> dict[tuple[str, int], np.ndarray]:
    """First crop (index into CROPS) of every test field after shuffling one input, or one group of inputs together,
    among the test fields."""
    base = test[model.used_].reset_index(drop=True)
    idx = np.array([CROPS.index(c) for c in model.classes_])
    out = {}
    for key, cols in importance_keys(model.used_).items():
        for r in range(n_shuffles):
            perm = _rng(*tag, key, r).permutation(len(base))
            X = base.copy()
            X[cols] = base[cols].to_numpy()[perm]
            out[(key, r)] = idx[np.argmax(model.predict_proba(X), axis=1)]
    return out


def _importance_rows(test: str, name: str, base: float, perm_scores: dict[str, list[float]]) -> list[dict]:
    rows = []
    for key, vals in perm_scores.items():
        drops = base - np.array(vals)
        kind, label = key.split(":", 1)
        rows.append({"test": test, "model": name + TUNED, "kind": kind, "name": label, "main_score": base,
                     "drop_mean": float(drops.mean()), "drop_sd": float(drops.std()), "shuffles": len(vals)})
    return rows


# ------------------------------------------------------------------ one job: one method on one view
def run_job(method: str, view: str, df: pd.DataFrame, cands: list[dict], n_shuffles: int = N_SHUFFLES,
            log: Callable = print) -> dict:
    """Both nested tests for one method on one view. Returns plain tables and the out-of-fold probabilities."""
    t0 = time.time()
    name = f"{method}[{view}]"

    def make(c):
        return lambda: Tunable(method, c, view)

    rows, inner_rows, chosen_rows, imp_rows, conf = [], [], [], [], []

    def choose(test: str, outer: str, scores: list[float]) -> int:
        best = pick(scores)
        inner_rows.extend({"test": test, "model": name, "outer": outer, "candidate": i, "inner_score": s}
                          for i, s in enumerate(scores))
        chosen_rows.append({"test": test, "model": name, "outer": outer, "candidate": best,
                            "inner_score": scores[best], "inner_score_protocol": scores[0],
                            "settings": json.dumps(cands[best], sort_keys=True)})
        return best

    # Main test: leave one district out, with the settings chosen inside the three training districts.
    base_by_district, perm_by_key = {}, {}
    for held_out in sorted(df[ev.GROUP_COL].unique()):
        t1 = time.time()
        train = df[df[ev.GROUP_COL] != held_out].reset_index(drop=True)
        test = df[df[ev.GROUP_COL] == held_out].reset_index(drop=True)
        assert held_out not in set(train[ev.GROUP_COL])
        inner = lodo_folds(train)
        scores = [lodo_score(train, inner, make(c)) for c in cands]
        best = choose("lodo", held_out, scores)
        fitted = {}
        for version, i in (("untuned", 0), ("tuned", best)):
            if i not in fitted:
                fitted[i] = Tunable(method, cands[i], view).fit(train, train["label"])
            m = fitted[i]
            proba = m.predict_proba(test)
            pred = m.classes_[np.argmax(proba, axis=1)]
            model_name = name + (TUNED if version == "tuned" else "")
            rows.append({"test": "lodo", "model": model_name, "version": version, "candidate": i,
                         "district": held_out, "n_train": int(len(train)),
                         **ev.scores(test["label"], pred, proba, m.classes_)})
            conf.append(ev._confusion("main", TESTS["lodo"], model_name, held_out, test["label"], pred))
        m, y = fitted[best], test["label"].to_numpy()
        base_by_district[held_out] = macro_f1(y, m.classes_[np.argmax(m.predict_proba(test), axis=1)])
        for (key, r), p in shuffled_preds(m, test, n_shuffles, (name, "lodo", held_out)).items():
            perm_by_key.setdefault(key, {}).setdefault(r, {})[held_out] = macro_f1(y, LABELS[p])
        log(f"{name} main test, {held_out} left out: candidate {best} chosen (inner score {scores[best]:.3f}, "
            f"protocol settings {scores[0]:.3f}), {time.time() - t1:.0f} s")
    lodo_base = float(np.nanmean(list(base_by_district.values())))
    complete = {k: [float(np.nanmean([v[r][d] for d in base_by_district])) for r in sorted(v)]
                for k, v in perm_by_key.items() if all(set(v[r]) == set(base_by_district) for r in v)}
    imp_rows += _importance_rows("lodo", name, lodo_base, complete)

    # Second test: new fields in the known districts, same 5 block folds as eval_v2.
    folds = ev.block_folds(df)
    proba_u, proba_t = np.zeros((len(df), len(CROPS))), np.zeros((len(df), len(CROPS)))
    perm_pred: dict[tuple[str, int], np.ndarray] = {}
    for f, (tr, te) in enumerate(folds):
        t1 = time.time()
        train = df.iloc[tr].reset_index(drop=True)
        test = df.iloc[te].reset_index(drop=True)
        inner = ev.block_folds(train, n_splits=INNER_BLOCK_FOLDS, seed=SEED)
        scores = [block_score(train, oof_proba(train, inner, make(c))) for c in cands]
        best = choose("blockcv", f"fold {f + 1}", scores)
        fitted = {}
        for target, i in ((proba_u, 0), (proba_t, best)):
            if i not in fitted:
                fitted[i] = Tunable(method, cands[i], view).fit(train, train["label"])
            target[np.ix_(te, [CROPS.index(c) for c in fitted[i].classes_])] = fitted[i].predict_proba(test)
        for (key, r), p in shuffled_preds(fitted[best], test, n_shuffles, (name, "blockcv", f)).items():
            perm_pred.setdefault((key, r), np.full(len(df), -1))[te] = p
        log(f"{name} new fields, fold {f + 1} of {len(folds)}: candidate {best} chosen (inner score "
            f"{scores[best]:.3f}, protocol settings {scores[0]:.3f}), {time.time() - t1:.0f} s")
    y, dist = df["label"].to_numpy(), df[ev.GROUP_COL].to_numpy()
    for version, proba in (("untuned", proba_u), ("tuned", proba_t)):
        pred = LABELS[np.argmax(proba, axis=1)]
        model_name = name + (TUNED if version == "tuned" else "")
        for d in sorted(set(dist)):
            mask = dist == d
            rows.append({"test": "blockcv", "model": model_name, "version": version, "candidate": float("nan"),
                         "district": d, "n_train": float("nan"),
                         **ev.scores(y[mask], pred[mask], proba[mask], LABELS)})
            conf.append(ev._confusion("main", TESTS["blockcv"], model_name, d, y[mask], pred[mask]))
    block_base = block_score(df, proba_t)
    keys = sorted({k for k, _ in perm_pred})
    complete = {k: [main_score(y, LABELS[perm_pred[(k, r)]], dist) for r in range(n_shuffles)]
                for k in keys if all((perm_pred[(k, r)] >= 0).all() for r in range(n_shuffles))}
    imp_rows += _importance_rows("blockcv", name, block_base, complete)

    return {"method": method, "view": view, "results": pd.DataFrame(rows), "inner": pd.DataFrame(inner_rows),
            "chosen": pd.DataFrame(chosen_rows), "importance": pd.DataFrame(imp_rows),
            "confusion": pd.concat(conf, ignore_index=True),
            "blockcv_proba": {"untuned": proba_u, "tuned": proba_t}, "seconds": time.time() - t0}


_THREAD_LIMIT = None
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _limit_threads() -> None:
    """One thread per job: two jobs on two cores, no oversubscription. Models use n_jobs=1 as well."""
    global _THREAD_LIMIT
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMIT = threadpool_limits(1)
    except Exception:
        pass


def _job_to_file(method, view, df, cands, n_shuffles, work: Path) -> str:
    _limit_threads()
    part = run_job(method, view, df, cands, n_shuffles, log=lambda s: print(s, flush=True))
    tmp = work / f"{method}_{view}.pkl.part"
    with open(tmp, "wb") as fh:
        pickle.dump(part, fh)
    os.replace(tmp, work / f"{method}_{view}.pkl")
    return f"{method}[{view}] done in {part['seconds'] / 60:.1f} min"


def run_jobs(df: pd.DataFrame, work: Path, methods, views, n_draws: int, n_shuffles: int, workers: int,
             log: Callable = print) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    cands = {m: candidates(m, n_draws) for m in methods}
    jobs = [(m, v) for m in JOB_ORDER if m in methods for v in views]
    todo = [(m, v) for m, v in jobs if not (work / f"{m}_{v}.pkl").exists()]
    if len(todo) < len(jobs):
        log(f"{len(jobs) - len(todo)} of {len(jobs)} jobs already done in {work}, kept")
    if workers <= 1:
        for m, v in todo:
            log(_job_to_file(m, v, df, cands[m], n_shuffles, work))
    elif todo:
        # Fresh processes ("spawn", as on Windows): forking after the maths libraries started threads can hang.
        import multiprocessing
        os.environ.update(SINGLE_THREAD_ENV)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_job_to_file, m, v, df, cands[m], n_shuffles, work): (m, v) for m, v in todo}
            for fut in as_completed(futures):
                log(fut.result())
    parts = {}
    for m, v in jobs:
        with open(work / f"{m}_{v}.pkl", "rb") as fh:
            parts[(m, v)] = pickle.load(fh)
    return {"candidates": cands, "parts": parts, "jobs": jobs}


# ------------------------------------------------------------------ analyses that decide nothing
def confidence_bands(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Share of fields at each level of the model's probability for its first crop, and how often that first crop
    was right there. One row per band, for all fields and for each district."""
    top = proba.max(axis=1)
    hit = LABELS[np.argmax(proba, axis=1)] == df["label"].to_numpy()
    dist = df[ev.GROUP_COL].to_numpy()
    rows = []
    for d in ["all"] + sorted(set(dist)):
        mask = np.ones(len(df), bool) if d == "all" else dist == d
        for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
            inside = mask & (top >= lo) & ((top < hi) if hi < 1.0 else (top <= hi))
            rows.append({"district": d, "band": f"{lo:.1f} to {hi:.1f}", "fields": int(inside.sum()),
                         "share_of_fields": float(inside.sum() / mask.sum()),
                         "first_crop_right": float(hit[inside].mean()) if inside.any() else float("nan")})
    return pd.DataFrame(rows)


def answer_when_sure(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """If the model only answered when its probability for the first crop is at least t: the share of fields that
    get an answer and how often that answer is right."""
    top = proba.max(axis=1)
    hit = LABELS[np.argmax(proba, axis=1)] == df["label"].to_numpy()
    dist = df[ev.GROUP_COL].to_numpy()
    rows = []
    for d in ["all"] + sorted(set(dist)):
        mask = np.ones(len(df), bool) if d == "all" else dist == d
        for t in SURE_FROM:
            sure = mask & (top >= t)
            rows.append({"district": d, "probability_at_least": t, "fields_answered": int(sure.sum()),
                         "share_answered": float(sure.sum() / mask.sum()),
                         "first_crop_right": float(hit[sure].mean()) if sure.any() else float("nan")})
    return pd.DataFrame(rows)


def mixed_land(df: pd.DataFrame) -> pd.DataFrame:
    """How mixed the 2 km blocks are. 'Block's most common crop' gives every field the most common crop of its own
    block, counting the field itself (ties: alphabetical). No model can know this in advance."""
    labels = df["label"].to_numpy()
    blocks = df[ev.BLOCK_COL].astype(str).to_numpy()
    majority = {}
    for b, g in pd.Series(labels).groupby(blocks):
        counts = g.value_counts()
        majority[b] = sorted(counts[counts == counts.max()].index)[0]
    guess = np.array([majority[b] for b in blocks])
    n_crops = pd.Series(labels).groupby(blocks).nunique()
    dist = df[ev.GROUP_COL].to_numpy()
    rows = []
    for d in ["all"] + sorted(set(dist)):
        mask = np.ones(len(df), bool) if d == "all" else dist == d
        bl = sorted(set(blocks[mask]))
        mixed = [b for b in bl if n_crops[b] > 1]
        rows.append({"district": d, "fields": int(mask.sum()), "blocks": len(bl),
                     "blocks_with_more_than_one_crop": len(mixed) / len(bl),
                     "fields_in_those_blocks": float(np.isin(blocks[mask], mixed).mean()),
                     "block_most_common_crop_right": float((guess[mask] == labels[mask]).mean()),
                     "block_most_common_crop_main_score": (main_score(labels, guess, dist) if d == "all"
                                                           else macro_f1(labels[mask], guess[mask]))})
    return pd.DataFrame(rows)


def pooled_top1(results: pd.DataFrame) -> pd.Series:
    """Share of all fields whose first crop was right (districts weighted by their number of fields)."""
    r = results.assign(hits=results["top1"] * results["n_test"])
    g = r.groupby("model", sort=False)
    return g["hits"].sum() / g["n_test"].sum()


# ------------------------------------------------------------------ putting it together
def builtin_check(results: dict[str, pd.DataFrame], untuned_dir: Path = UNTUNED_DIR) -> dict:
    """The protocol settings and the baselines, scored on the same outer folds, must give the eval_v2 numbers.
    Probabilities can differ in the seventh decimal with the number of threads, so scores are compared to 1e-9."""
    cols = ["n_test", "top1", "top3", "macro_f1", "macro_f1_all4"] + [f"recall_{c}" for c in CROPS]
    out = {"compared_with": str(untuned_dir.relative_to(ev.REPO)) if untuned_dir.is_relative_to(ev.REPO)
           else str(untuned_dir), "rows_compared": 0, "mismatches": []}
    for test, res in results.items():
        path = untuned_dir / f"main_{test}_results.csv"
        if not path.exists():
            out["mismatches"].append(f"{path.name} not found")
            continue
        ref = pd.read_csv(path).set_index(["model", "district"])
        mine = res[res["version"].isin(["untuned", ""]) & (res["model"] != DistrictMajority.name)]
        mine = mine.set_index(["model", "district"])
        for key, row in mine.iterrows():
            if key not in ref.index:
                out["mismatches"].append(f"{test} {key}: not in eval_v2")
                continue
            a = row[cols].to_numpy(dtype=float)
            b = ref.loc[key, cols].to_numpy(dtype=float)
            out["rows_compared"] += 1
            if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
                bad = [c for c, x, z in zip(cols, a, b) if not np.isclose(x, z, rtol=0, atol=1e-9, equal_nan=True)]
                out["mismatches"].append(f"{test} {key}: {', '.join(bad)}")
    out["passed"] = out["rows_compared"] > 0 and not out["mismatches"]
    return out


def baselines(df: pd.DataFrame) -> dict:
    lodo, lodo_conf = ev.run_lodo(df, [MajorityBaseline(), ExpertRulesBaseline()])
    blk, blk_conf = ev.run_block_cv(df, [MajorityBaseline(), DistrictMajority(), ExpertRulesBaseline()])
    return {"lodo": lodo.assign(version="", candidate=float("nan")),
            "blockcv": blk.assign(version="", candidate=float("nan")),
            "confusion": pd.concat([lodo_conf, blk_conf], ignore_index=True)}


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    s = ev.summarise(results)
    s.loc[s.index == DistrictMajority.name, ["districts_beating_B1", "passes_rule"]] = [float("nan"), ""]
    s["top1_all_fields"] = pooled_top1(results).reindex(s.index)
    return s


def decisions(summ: dict[str, pd.DataFrame]) -> dict:
    """The two decisions section 11 fixed before the run."""
    lodo = summ["lodo"]
    tuned = [i for i in lodo.index if i.endswith(TUNED)]
    passing = [i for i in tuned if lodo.loc[i, "passes_rule"] == "yes"]
    best_lodo = lodo.loc[tuned, "macro_f1_mean"].idxmax()
    blk = summ["blockcv"]
    today = float(blk.loc[MAP_MODEL, "macro_f1_mean"])
    beating = blk.loc[tuned][blk.loc[tuned, "macro_f1_mean"] > today + MAP_MARGIN]
    winner = beating["macro_f1_mean"].idxmax() if len(beating) else None
    return {"adoption": {"passing": passing, "best_tuned": best_lodo,
                         "best_tuned_score": float(lodo.loc[best_lodo, "macro_f1_mean"]),
                         "best_tuned_wins": int(lodo.loc[best_lodo, "districts_beating_B1"]),
                         "b1_score": float(lodo.loc[ev.BASELINE, "macro_f1_mean"])},
            "map": {"today": MAP_MODEL, "today_score": today, "margin": MAP_MARGIN,
                    "beating": {i: float(beating.loc[i, "macro_f1_mean"]) for i in beating.index},
                    "serve": winner or MAP_MODEL, "changes": winner is not None}}


def assemble(df: pd.DataFrame, run: dict, untuned_dir: Path = UNTUNED_DIR) -> dict:
    parts = run["parts"]
    base = baselines(df)
    results, summ = {}, {}
    for test in TESTS:
        frames = [base[test]] + [p["results"][p["results"]["test"] == test].drop(columns="test")
                                 for p in parts.values()]
        results[test] = pd.concat(frames, ignore_index=True)
        summ[test] = summarise(results[test])
    check = builtin_check(results, untuned_dir)
    dec = decisions(summ)
    serve = dec["map"]["serve"]
    method, view = serve.replace(TUNED, "").rstrip("]").split("[")
    version = "tuned" if serve.endswith(TUNED) else "untuned"
    proba = parts[(method, view)]["blockcv_proba"][version]
    analyses = {
        "bands": pd.concat([confidence_bands(df, p["blockcv_proba"][v]).assign(
            model=f"{m}[{w}]" + (TUNED if v == "tuned" else "")) for (m, w), p in parts.items()
            for v in ("untuned", "tuned")], ignore_index=True),
        "sure": pd.concat([answer_when_sure(df, p["blockcv_proba"][v]).assign(
            model=f"{m}[{w}]" + (TUNED if v == "tuned" else "")) for (m, w), p in parts.items()
            for v in ("untuned", "tuned")], ignore_index=True),
        "map_bands": confidence_bands(df, proba), "map_sure": answer_when_sure(df, proba),
        "mixed": mixed_land(df),
        "importance": pd.concat([p["importance"] for p in parts.values()], ignore_index=True),
        "chosen": pd.concat([p["chosen"] for p in parts.values()], ignore_index=True),
        "inner": pd.concat([p["inner"] for p in parts.values()], ignore_index=True),
        "confusion": pd.concat([base["confusion"]] + [p["confusion"] for p in parts.values()], ignore_index=True),
    }
    return {"results": results, "summary": summ, "check": check, "decisions": dec, "analyses": analyses}


# ------------------------------------------------------------------ report
COUNT_COLUMNS = {"fields", "blocks", "fields answered", "better than B1 in, tuned"}
TEST_NAMES = {"lodo": "main", "blockcv": "new fields"}
PLAIN = {"share_of_fields": "share of fields", "first_crop_right": "first crop right", "band": "probability of the first crop",
         "probability_at_least": "probability at least", "fields_answered": "fields answered",
         "share_answered": "share answered", "blocks_with_more_than_one_crop": "blocks with more than one crop",
         "fields_in_those_blocks": "fields in those blocks",
         "block_most_common_crop_right": "block's most common crop is right",
         "block_most_common_crop_main_score": "main score of the block's most common crop"}


def _fmt(frame: pd.DataFrame) -> pd.DataFrame:
    """Numbers for the report: three decimals, whole numbers for counts, blanks for missing values, no -0.000."""
    out = frame.copy().astype(object)
    for col in out.columns:
        def cell(v, count=str(col) in COUNT_COLUMNS):
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                return ""
            if isinstance(v, (int, np.integer, float, np.floating)) and not isinstance(v, bool):
                if count:
                    return f"{int(round(float(v)))}"
                s = f"{float(v):.3f}"
                return "0.000" if s == "-0.000" else s
            return v
        out[col] = out[col].map(cell)
    return out


def _md(frame: pd.DataFrame, index: bool = True) -> str:
    """Markdown table that keeps the three decimals (tabulate would turn '0.430' back into 0.43)."""
    try:
        return _fmt(frame).to_markdown(index=index, disable_numparse=True)
    except ImportError:  # tabulate not installed
        return "```\n" + _fmt(frame).to_string(index=index) + "\n```"


def _gain_table(summ: pd.DataFrame, methods, views, deciding: bool = True) -> pd.DataFrame:
    """Protocol settings against tuned settings. Only the main test decides adoption (section 7), so on the other
    test the rule column is labelled as information only."""
    rule = "passes the adoption rule, tuned" if deciding else "would pass the same rule, tuned (information only)"
    rows = []
    for m in methods:
        for v in views:
            name = f"{m}[{v}]"
            if name not in summ.index or name + TUNED not in summ.index:
                continue
            u, t = summ.loc[name], summ.loc[name + TUNED]
            rows.append({"model": name, "main score, protocol settings": u["macro_f1_mean"],
                         "main score, tuned": t["macro_f1_mean"], "change": t["macro_f1_mean"] - u["macro_f1_mean"],
                         "worst district, tuned": t["worst_district_macro_f1"],
                         "better than B1 in, tuned": t["districts_beating_B1"],
                         rule: t["passes_rule"],
                         "first crop right, all fields, tuned": t["top1_all_fields"]})
    return pd.DataFrame(rows).set_index("model")


def _per_district(results: pd.DataFrame, methods, views) -> pd.DataFrame:
    piv = results.pivot(index="model", columns="district", values="macro_f1")
    rows = {}
    for name in [ev.BASELINE] + [f"{m}[{v}]" for m in methods for v in views]:
        if name not in piv.index:
            continue
        row = {}
        for d in piv.columns:
            row[f"{d}, protocol"] = piv.loc[name, d]
            if name + TUNED in piv.index:
                row[f"{d}, tuned"] = piv.loc[name + TUNED, d]
        rows[name] = row
    return pd.DataFrame(rows).T


def _model_order(methods, views) -> list[str]:
    return [f"{m}[{v}]" for m in methods for v in views]


def _settings_table(chosen: pd.DataFrame, methods, views) -> pd.DataFrame:
    rows = []
    for test in TESTS:
        for model in _model_order(methods, views):
            g = chosen[(chosen["test"] == test) & (chosen["model"] == model)]
            if g.empty:
                continue
            rows.append({"test": TEST_NAMES[test], "model": model,
                         "candidate chosen in each fold": ", ".join(f"{o}: {c}" for o, c in
                                                                   zip(g["outer"], g["candidate"])),
                         "folds that kept the protocol settings": f"{int((g['candidate'] == 0).sum())} of {len(g)}"})
    return pd.DataFrame(rows)


def _importance_tables(imp: pd.DataFrame, methods, views) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = [(TEST_NAMES[t], m + TUNED) for t in TESTS for m in _model_order(methods, views)]
    imp = imp.assign(test=imp["test"].map(TEST_NAMES))
    groups = imp[imp["kind"] == "group"].pivot_table(index=["test", "model"], columns="name", values="drop_mean")
    groups = groups.reindex([o for o in order if o in groups.index])[[g for g in GROUPS if g in groups.columns]]
    top = (imp[imp["kind"] == "input"].sort_values("drop_mean", ascending=False).groupby(["test", "model"])
           .head(5).assign(item=lambda f: f["name"] + " (" + f["drop_mean"].map("{:.3f}".format) + ")")
           .groupby(["test", "model"])["item"].apply(", ".join).rename("five inputs that matter most"))
    top = top.reindex([o for o in order if o in top.index])
    return groups.reset_index(), top.reset_index()


def write_report(out: Path, info: dict, df: pd.DataFrame, res: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    a, summ, dec, check = res["analyses"], res["summary"], res["decisions"], res["check"]
    for test in TESTS:
        res["results"][test].to_csv(out / f"main_{test}_results.csv", index=False)
        summ[test].to_csv(out / f"main_{test}_summary.csv")
    a["confusion"].to_csv(out / "confusion_counts.csv", index=False)
    a["chosen"].to_csv(out / "chosen_settings.csv", index=False)
    a["inner"].to_csv(out / "inner_scores.csv", index=False)
    a["importance"].to_csv(out / "importance.csv", index=False)
    a["bands"].to_csv(out / "confidence_bands.csv", index=False)
    a["sure"].to_csv(out / "answer_when_sure.csv", index=False)
    a["mixed"].to_csv(out / "mixed_land.csv", index=False)
    (out / "candidates.json").write_text(json.dumps(info["candidates"], indent=1), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({k: v for k, v in info.items() if k != "candidates"}, indent=2,
                                                  default=str), encoding="utf-8")

    methods, views = info["methods"], info["views"]
    ad, mp = dec["adoption"], dec["map"]
    serve = mp["serve"]
    if ad["passing"]:
        adoption = (f"{', '.join(ad['passing'])} pass the adoption rule on the main test, so the best of them "
                    "replaces B1 as the default answer of the API (section 7).")
    else:
        adoption = (f"No tuned model passes the adoption rule, so B1 stays the default answer of the API. Best tuned "
                    f"model on the main test: {ad['best_tuned']}, {ad['best_tuned_score']:.3f} against "
                    f"{ad['b1_score']:.3f} for B1, better than B1 in {ad['best_tuned_wins']} of 4 districts.")
    if mp["changes"]:
        others = ", ".join(f"{k} ({v:.3f})" for k, v in sorted(mp["beating"].items(), key=lambda kv: -kv[1]))
        map_line = (f"The map moves to {serve}: on new fields it scores {mp['beating'][serve]:.3f} against "
                    f"{mp['today_score']:.3f} for {mp['today']} with the protocol settings, more than "
                    f"{mp['margin']} ahead. Tuned models ahead by more than {mp['margin']}: {others}.")
    else:
        map_line = (f"The map keeps {mp['today']} with the protocol settings ({mp['today_score']:.3f} on new "
                    f"fields): no tuned model is ahead of it by more than {mp['margin']}.")
    gains = {t: _gain_table(summ[t], methods, views, deciding=(t == "lodo")) for t in TESTS}
    biggest = max(((t, i, r["change"]) for t in TESTS for i, r in gains[t].iterrows()), key=lambda x: x[2])
    by_method = {t: {m: gains[t].loc[[i for i in gains[t].index if i.startswith(m + "[")], "change"].mean()
                     for m in methods} for t in TESTS}
    lr_more = [t for t in TESTS if "LR" in methods and len(methods) > 1
               and all(by_method[t]["LR"] > by_method[t][m] for m in methods if m != "LR")]
    pooled = pooled_top1(res["results"]["blockcv"])
    mixed = a["mixed"].rename(columns=PLAIN)
    bands = a["map_bands"]
    bands_all = bands[bands["district"] == "all"].drop(columns="district").rename(columns=PLAIN)
    sure = a["map_sure"]
    sure_all = sure[sure["district"] == "all"].drop(columns="district").rename(columns=PLAIN)
    sure_d = (sure[(sure["district"] != "all") & (sure["probability_at_least"] == 0.7)]
              .drop(columns="probability_at_least").rename(columns=PLAIN))
    blk = summ["blockcv"]
    alt_names = {"B0_majority": "B0: the most common crop of all training fields",
                 DistrictMajority.name: "the most common crop of the field's district",
                 ev.BASELINE: "B1: expert rules", MAP_MODEL: f"{MAP_MODEL}, protocol settings (map before this run)"}
    if serve != MAP_MODEL:
        alt_names[serve] = f"{serve} (map after this run)"
    names = list(alt_names)
    alternatives = pd.DataFrame({
        "": [alt_names[n] for n in names],
        "main score": blk.loc[names, "macro_f1_mean"].to_numpy(),
        "first crop right, all fields": pooled.reindex(names).to_numpy(),
        "first crop right, mean of the districts": blk.loc[names, "top1_mean"].to_numpy()})
    groups, top = _importance_tables(a["importance"], methods, views)

    lines = [
        "# Crop recommendation: tuned settings (tuning_v1)", "",
        f"Protocol: `docs/training_protocol.md`, version {info['protocol_version']} (section 11). Run at "
        f"{info['run_at_utc']} UTC: {info['compute_hours']:.1f} hours of computing, {info['workers']} jobs at a "
        "time." + (" **Quick check only: these numbers are not reported.**" if info.get("quick_check_only") else ""),
        f"Table: `{info['table']}` (SHA-256 `{info['table_sha256'][:16]}...`). Code commit: "
        f"`{(info['git_commit'] or 'unknown')[:10]}`"
        + (" (with uncommitted changes, so rerun after committing)." if info.get("uncommitted_changes") else ".")
        + (f" Report text rebuilt from the saved jobs at commit `{info['report_rebuilt_at_commit'][:10]}`, with the "
           "same numbers." if info.get("report_rebuilt_at_commit") else ""),
        f"{info['n_candidates']} candidates per method (the protocol settings, then {info['n_draws']} random draws), "
        f"{info['inner_block_folds']} inner block folds, {info['n_shuffles']} shuffles per input.",
        "",
        f"**Built-in check: {'passed' if check['passed'] else 'FAILED'}.** The protocol settings and the baselines "
        f"give the `eval_v2` numbers in {check['rows_compared'] - len(check['mismatches'])} of "
        f"{check['rows_compared']} rows (district and model)."
        + ("" if check["passed"] else " Mismatches: " + "; ".join(check["mismatches"][:10])),
        "",
        "## Decisions fixed before the run", "",
        f"1. **Adoption (main test).** {adoption}",
        f"2. **Map.** {map_line}", "",
        "Expected before the run: (a) small gains, a few hundredths of main score at most; (b) Logistic Regression "
        "gains more than the tree models.",
        f"(a) Largest gain: {biggest[2]:+.3f} ({biggest[1]}, {'main test' if biggest[0] == 'lodo' else 'new fields'}).",
        "(b) Mean gain per method: " + "; ".join(
            f"{'main test' if t == 'lodo' else 'new fields'}: "
            + ", ".join(f"{m} {by_method[t][m]:+.3f}" for m in methods) for t in TESTS)
        + (". Logistic Regression gained the most in " + " and ".join(
            "the main test" if t == "lodo" else "the new-fields test" for t in lr_more) + "." if lr_more
           else ". Logistic Regression did not gain the most in either test."),
        "",
        "## Main test: leave one district out (decides adoption)", "",
        "Settings chosen inside the three training districts only, then scored once on the district left out.", "",
        _md(gains["lodo"]), "",
        f"B1 (expert rules): {summ['lodo'].loc[ev.BASELINE, 'macro_f1_mean']:.3f}. "
        f"B0 (most common crop): {summ['lodo'].loc['B0_majority', 'macro_f1_mean']:.3f}.", "",
        "Main score per district:", "", _md(_per_district(res["results"]["lodo"], methods, views)), "",
        "## Second test: new fields in the known districts", "",
        "Settings chosen inside the training part of each block fold only, then scored once on the test fold.", "",
        _md(gains["blockcv"]), "",
        f"B1 (expert rules): {blk.loc[ev.BASELINE, 'macro_f1_mean']:.3f}. This test does not decide adoption "
        "(section 7): the rule column only says whether a model would pass the same rule here, as in `eval_v2`.", "",
        "Main score per district:", "", _md(_per_district(res["results"]["blockcv"], methods, views)), "",
        "## Settings chosen in each fold", "",
        "Candidate 0 is the protocol settings. The settings of every candidate are in `candidates.json`; the inner "
        "score of every candidate in every fold is in `inner_scores.csv`.", "",
        _md(_settings_table(a["chosen"], methods, views), index=False), "",
        f"## When the model is sure ({serve}, new fields)", "",
        "Probability the model gives its first crop, and how often that first crop was the crop actually planted. "
        "Out-of-fold predictions of the new-fields test: no field was predicted by a model that saw its 2 km block.",
        "", _md(bands_all, index=False), "",
        "If the model answered only when its probability for the first crop is at least this high:", "",
        _md(sure_all, index=False), "",
        "Per district, answering only from 0.7:", "", _md(sure_d, index=False), "",
        "## Simple alternatives on the same folds (new fields)", "",
        "B0 gives every field the most common crop of all training fields. The district's most common crop reads "
        "the district name, which no model may use as an input. B1 uses published crop ranges only.", "",
        _md(alternatives, index=False), "",
        "## How mixed the land is", "",
        "If every field got the most common crop of its own 2 km block (which no model can know in advance):", "",
        _md(mixed, index=False), "",
        "## Input importance (tuned models)", "",
        "How much the main score drops when the values of one input, or of a group of inputs that carry the same "
        "information, are shuffled among the test fields (mean of the shuffles; 0 means the model does not use it). "
        "In the main test, an input that is the same for the whole district cannot change inside the district left "
        "out, so its drop there is zero by construction. Every input and its spread: `importance.csv`.", "",
        "Groups of inputs:", "", _md(groups, index=False), "",
        "Single inputs:", "", _md(top, index=False), "",
        "## How to read this", "",
        "- Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over "
        "the four districts (section 5). 'First crop right' is the share of fields whose first crop was the crop "
        "actually planted.",
        "- 'Protocol settings' is the model of `eval_v2` (section 6 settings). 'Tuned' chose its settings again in "
        "every fold, inside the training fields of that fold.",
        "- The adoption rule (section 7) and the map rule (section 11) were written before this run.",
        "- Full numbers: the CSV files in this folder. Settings, versions and timing: `run_info.json`.",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ the model the map serves
FINAL_FILE = "final_model.json"


def final_model(df: pd.DataFrame, out: Path = DEFAULT_OUT, log: Callable = print) -> dict:
    """The model the map serves under decision 2 of section 11, written to <out>/final_model.json.

    Its settings come from the same comparison as inside the nested test, run once on all training fields: every
    candidate is scored on 3 folds over 2 km blocks (seed 0), main score per district then the mean, and a tie goes
    to the candidate drawn first. The nested test measured this exact procedure, so its scores are the tested scores
    of the model trained here. If decision 2 kept the untuned model, its protocol settings are used."""
    info = json.loads((Path(out) / "run_info.json").read_text(encoding="utf-8"))
    serve = info["decisions"]["map"]["serve"]
    method, view = serve.replace(TUNED, "").rstrip("]").split("[")
    cands = candidates(method, info["n_draws"])
    scores: list[float] = []
    best = 0
    if serve.endswith(TUNED):
        t0 = time.time()
        folds = ev.block_folds(df, n_splits=INNER_BLOCK_FOLDS, seed=SEED)
        scores = [block_score(df, oof_proba(df, folds, (lambda c=c: Tunable(method, c, view)))) for c in cands]
        best = pick(scores)
        log(f"{serve}: candidate {best} chosen on all {len(df)} fields (comparison score {scores[best]:.3f}, "
            f"protocol settings {scores[0]:.3f}), {time.time() - t0:.0f} s")
    import sklearn
    result = {
        "model": serve, "method": method, "view": view, "tuned": serve.endswith(TUNED),
        "features": list(ev.VIEWS_V11[view]), "candidate": best, "settings": cands[best],
        "chosen_by": (f"same comparison as inside the nested test, on all {len(df)} fields: {INNER_BLOCK_FOLDS} folds "
                      f"over {ev.BLOCK_COL} (seed {SEED}), main score per district then the mean; ties go to the "
                      "candidate drawn first" if serve.endswith(TUNED) else "protocol settings (section 6)"),
        "comparison_scores": [round(s, 6) for s in scores],
        "tested_main_score_new_fields": info["decisions"]["map"]["beating"].get(serve,
                                                                                 info["decisions"]["map"]["today_score"]),
        "trained_on_fields": int(len(df)), "decided_by": "decision 2 of docs/training_protocol.md, section 11",
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "git_commit": ev._git_commit(),
        "uncommitted_changes": ev._git_has_uncommitted_changes(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    path = Path(out) / FINAL_FILE
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"Written to {path}")
    return result


# ------------------------------------------------------------------ command line
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=None, help="Feature table CSV. Default: feature_table_v6b.csv.")
    ap.add_argument("--out", default=None, help="Output folder. Default: reports/tuning_v1 (quick: a temp folder).")
    ap.add_argument("--workers", type=int, default=2, help="Jobs run at the same time (default 2).")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny run (2 random draws, 1 shuffle, two views) to check the code. Never reported.")
    ap.add_argument("--assemble-only", action="store_true", help="Rebuild the report from the saved jobs.")
    ap.add_argument("--final", action="store_true",
                    help="After the run: choose the settings of the model the map serves (decision 2) on all fields "
                         "and write final_model.json next to the report.")
    args = ap.parse_args(argv)

    table = Path(args.table) if args.table else TABLE
    if args.final:
        out = Path(args.out) if args.out else DEFAULT_OUT
        final_model(ev.prepare(pd.read_csv(table, low_memory=False)), out, log=lambda s: print(s, flush=True))
        return 0
    if args.quick:
        methods, views, n_draws, n_shuffles = METHODS, ("all", "field_sat"), 2, 1
        out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "tuning_quick"
    else:
        methods, views, n_draws, n_shuffles = METHODS, VIEWS, N_DRAWS, N_SHUFFLES
        out = Path(args.out) if args.out else DEFAULT_OUT
    work = out / "_work"
    print(f"Protocol {PROTOCOL_VERSION}. Table: {table}. Output: {out}" + (" (quick check)" if args.quick else ""),
          flush=True)
    raw = pd.read_csv(table, low_memory=False)
    df = ev.prepare(raw)
    t0 = time.time()
    if args.assemble_only and not all((work / f"{m}_{v}.pkl").exists() for m in methods for v in views):
        raise SystemExit(f"--assemble-only needs every job in {work}")
    run = run_jobs(df, work, methods, views, n_draws, n_shuffles, 1 if args.assemble_only else args.workers,
                   log=lambda s: print(s, flush=True))
    res = assemble(df, run)

    import sklearn
    import xgboost
    try:
        table_rel = table.resolve().relative_to(ev.REPO).as_posix()
    except ValueError:
        table_rel = str(table)
    job_minutes = {f"{m}[{v}]": round(p["seconds"] / 60, 1) for (m, v), p in run["parts"].items()}
    previous = {}
    if (out / "run_info.json").exists() and args.assemble_only:
        previous = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    info = {
        "protocol_version": PROTOCOL_VERSION, "quick_check_only": args.quick,
        "run_at_utc": previous.get("run_at_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "compute_hours": sum(p["seconds"] for p in run["parts"].values()) / 3600,
        "wall_hours_this_call": (time.time() - t0) / 3600,
        "workers": previous.get("workers") or args.workers, "job_minutes": job_minutes,
        "table": table_rel, "table_sha256": ev.file_sha256(table),
        # The commit the jobs ran at. Rebuilding only the report keeps it and records the rebuild separately.
        "git_commit": previous.get("git_commit") or ev._git_commit(),
        "uncommitted_changes": (previous["uncommitted_changes"] if "uncommitted_changes" in previous
                                else ev._git_has_uncommitted_changes()),
        **({"report_rebuilt_at_commit": ev._git_commit(),
            "report_rebuilt_with_uncommitted_changes": ev._git_has_uncommitted_changes()} if previous else {}),
        "rows": {"before_filters": df.attrs.get("rows_before_filters"), "after_qc_ok": df.attrs.get("rows_after_qc"),
                 "main": len(df)},
        "methods": list(methods), "views": list(views), "n_draws": n_draws, "n_candidates": n_draws + 1,
        "inner_block_folds": INNER_BLOCK_FOLDS, "n_shuffles": n_shuffles, "seed": SEED,
        "tests": {"main": "leave one district out; inside: leave each training district out in turn",
                  "second": f"{ev.N_BLOCK_FOLDS} folds over {ev.BLOCK_COL} (StratifiedGroupKFold, seed {SEED}); "
                            f"inside: {INNER_BLOCK_FOLDS} folds over {ev.BLOCK_COL}, seed {SEED}"},
        "builtin_check": res["check"], "decisions": res["decisions"],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "candidates": run["candidates"],
    }
    path = write_report(out, info, df, res)
    for test, title in (("lodo", "main test (leave one district out)"), ("blockcv", "new fields")):
        print(f"\n== {title}")
        print(_gain_table(res["summary"][test], methods, views).round(3).to_string())
    print(f"\nBuilt-in check: {'passed' if res['check']['passed'] else 'FAILED'} "
          f"({res['check']['rows_compared']} rows, {len(res['check']['mismatches'])} mismatches)")
    print(f"Map: serve {res['decisions']['map']['serve']}. Adoption: {res['decisions']['adoption']['passing'] or 'none'}")
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
