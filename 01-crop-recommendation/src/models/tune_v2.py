"""Second round of tuning (training protocol 1.3, section 12): a wider search, Extra Trees, a blend of the three tree
methods, and nearby-crop inputs.

What it adds to tune.py (version 1.2), which it reuses for everything else (folds, scores, tie rule, analyses):
  1. Random Forest, Extra Trees and XGBoost. Each gets the section 6 settings first, then 100 random draws over wider
     ranges (scikit-learn ParameterSampler, seed 0). Random Forest also gets the map model of version 1.2
     (candidate 30 of tuning_v1) as its second candidate.
  2. Views field_sat and all_sat on both nested tests, as in version 1.2, and field_sat_near and all_sat_near on the
     new-fields test only. The _near views add five nearby-crop inputs (NearCrops): the weighted crop shares of the
     training fields in the other 2 km blocks around a field, and how much nearby data there is. A field's own block
     never counts, and the inputs are rebuilt inside every step from the training fields of that step only.
  3. A blend per view: the mean of the probabilities of the tuned Random Forest, Extra Trees and XGBoost, with equal
     weights fixed before the run. It has nothing to tune.
  4. A reference that decides nothing: on new fields, the crop with the highest nearby share (h = 4 km).
  5. The decisions section 12 fixed before the run: the adoption rule on the main test, and the map rule against the
     model on the map since 23 Nov (RF[all_sat] tuned of tuning_v1).

Every (method, view) pair is one job. A finished job is saved in <out>/_work, so a stopped run starts again where it
stopped. Run from the repo root (about three hours on two cores):
    python 01-crop-recommendation/src/models/tune_v2.py
Options: --workers N (parallel jobs, default 2), --quick (a tiny run into a temporary folder, to check the code; its
numbers are never reported), --assemble-only (rebuild the report from the saved jobs), --final (after the run: the
settings of the model the map serves, chosen on all fields), --table, --out
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import tune as r1  # noqa: E402  (version 1.2: folds, scores, tie rule and analyses are reused as they are)

PROTOCOL_VERSION = "1.3"
TABLE = r1.TABLE
DEFAULT_OUT = ev.REPORTS / "tuning_v2"
ROUND1_DIR = ev.REPORTS / "tuning_v1"
UNTUNED_DIR = ev.REPORTS / "eval_v2"     # the built-in check compares the section 6 settings with these numbers
SEED = ev.SEED
CROPS, LABELS, TUNED = r1.CROPS, r1.LABELS, r1.TUNED
TESTS, TEST_NAMES = r1.TESTS, r1.TEST_NAMES
METHODS = ("RF", "ET", "XGB")
BASE_VIEWS = ("field_sat", "all_sat")                 # both tests
NEAR_VIEWS = ("field_sat_near", "all_sat_near")       # new-fields test only: a district never seen has no neighbours
VIEWS = BASE_VIEWS + NEAR_VIEWS
JOB_ORDER = ("RF", "ET", "XGB")                       # slowest first
N_DRAWS = 100
INNER_BLOCK_FOLDS = r1.INNER_BLOCK_FOLDS
N_SHUFFLES = r1.N_SHUFFLES
BLEND = "Blend"
NEAR_REF = "near_fields_4km"                          # reference only: the crop with the highest nearby share
MAP_MODEL = "RF[all_sat]" + TUNED                     # the model on the map since 23 Nov (tuning_v1)
MAP_MARGIN = 0.01
METHOD_NAMES = {"RF": "Random Forest", "ET": "Extra Trees", "XGB": "XGBoost", BLEND: "Blend of the three"}

# Nearby-crop inputs (section 12)
BLOCK_DEG = 0.018               # Part A's block_id is round(lat / 0.018) and round(lon / 0.018)
NEAR_KM_PROTOCOL = 4.0          # distance h of the section 6 settings and of the reference
NEAR_CUTOFF = 3.0               # blocks farther than 3h do not count
NEAR_COLS = [f"near_share_{c}" for c in CROPS] + ["near_weight"]
NEAR_GROUP = "crops grown nearby"

# Section 6 settings for every searched setting. Extra Trees takes the Random Forest settings of section 6.
PROTOCOL_SETTINGS = {
    "RF": {**r1.PROTOCOL_SETTINGS["RF"], "near_km": NEAR_KM_PROTOCOL},
    "ET": {**r1.PROTOCOL_SETTINGS["RF"], "near_km": NEAR_KM_PROTOCOL},
    "XGB": {**r1.PROTOCOL_SETTINGS["XGB"], "reg_alpha": 0.0, "near_km": NEAR_KM_PROTOCOL},
}


def search_spaces() -> dict:
    """The ranges of section 12. Lists are drawn evenly; loguniform and uniform are continuous."""
    from scipy.stats import loguniform, uniform
    trees = {"max_depth": [None, 6, 10, 16, 24], "min_samples_leaf": [1, 2, 3, 5, 8, 12],
             "max_features": ["sqrt", 0.2, 0.33, 0.5, 0.7], "max_samples": [None, 0.85, 0.7, 0.5],
             "criterion": ["gini", "entropy"], "weight_power": [0.0, 0.5, 1.0, 1.5], "near_km": [2.0, 4.0, 8.0]}
    return {
        "RF": dict(trees), "ET": dict(trees),
        "XGB": {"n_estimators": [100, 200, 300, 500, 800, 1200, 1500], "learning_rate": loguniform(0.005, 0.3),
                "max_depth": [2, 3, 4, 5, 6, 8, 10], "min_child_weight": [1, 2, 5, 10, 20],
                "subsample": uniform(0.5, 0.5), "colsample_bytree": uniform(0.3, 0.7),
                "reg_lambda": loguniform(0.01, 100), "reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "gamma": [0.0, 0.1, 0.5, 1.0, 2.0], "weight_power": [0.0, 0.5, 1.0, 1.5],
                "near_km": [2.0, 4.0, 8.0]},
    }


def round1_map_settings() -> dict:
    """The settings of the map model of version 1.2 (candidate 30 of tuning_v1), with the protocol distance h."""
    final = json.loads((ROUND1_DIR / "final_model.json").read_text(encoding="utf-8"))
    return {**final["settings"], "near_km": NEAR_KM_PROTOCOL}


def candidates(method: str, n_draws: int = N_DRAWS, seed: int = SEED) -> list[dict]:
    """Candidate 0 is the section 6 settings; for Random Forest candidate 1 is the map model of version 1.2; then
    n_draws random draws. Continuous values are rounded to six decimals, and the rounded value is the one used."""
    from sklearn.model_selection import ParameterSampler
    out = [dict(sorted(PROTOCOL_SETTINGS[method].items()))]
    if method == "RF":
        out.append(dict(sorted(round1_map_settings().items())))
    for draw in ParameterSampler(search_spaces()[method], n_iter=n_draws, random_state=seed):
        out.append({k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                    for k, v in sorted(draw.items())})
    return out


# ------------------------------------------------------------------ nearby-crop inputs
def block_centres(blocks) -> np.ndarray:
    """Latitude and longitude (degrees) of the centre of each 2 km block, from Part A's block_id 'i_j'."""
    return np.array([str(b).split("_") for b in blocks], dtype=float).reshape(-1, 2) * BLOCK_DEG


def distance_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between every point of a (n x 2) and every point of b (m x 2), in degrees."""
    lat1, lon1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    lat2, lon2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


class NearCrops:
    """Nearby-crop inputs of section 12, learnt from training fields only.

    For a field in block b, every training field in another block counts with the weight exp(-0.5 (d / h)^2), d being
    the distance between the block centres in km, up to 3h. The inputs are the weighted share of each crop and the sum
    of the weights. The field's own block never counts, for training fields too, so no field sees its own crop or the
    crops of its own block. With nothing counted, the shares are those of all training fields and the weight is 0."""

    def __init__(self, km: float = NEAR_KM_PROTOCOL):
        self.km = float(km)

    def fit(self, blocks, labels) -> "NearCrops":
        tab = pd.crosstab(np.asarray(blocks).astype(str), np.asarray(labels)).reindex(columns=CROPS, fill_value=0)
        self.blocks_ = tab.index.to_numpy().astype(str)
        self.counts_ = tab.to_numpy(dtype=float)             # training fields per block and crop
        self.centres_ = block_centres(self.blocks_)
        total = self.counts_.sum(axis=0)
        self.prior_ = total / total.sum()
        return self

    def transform(self, blocks) -> pd.DataFrame:
        blocks = np.asarray(blocks).astype(str)
        uniq, inv = np.unique(blocks, return_inverse=True)
        d = distance_km(block_centres(uniq), self.centres_)
        w = np.exp(-0.5 * (d / self.km) ** 2)
        w[d > NEAR_CUTOFF * self.km] = 0.0
        w[uniq[:, None] == self.blocks_[None, :]] = 0.0     # never the field's own block
        sums = w @ self.counts_                              # weighted training fields per crop
        weight = sums.sum(axis=1)
        safe = np.where(weight > 0, weight, 1.0)
        shares = np.where(weight[:, None] > 0, sums / safe[:, None], self.prior_[None, :])
        return pd.DataFrame(np.column_stack([shares, weight])[inv], columns=NEAR_COLS)

    def first_crop_proba(self, blocks) -> np.ndarray:
        """The shares as probabilities (columns follow CROPS): the 'nearest surveyed fields' reference."""
        return self.transform(blocks)[NEAR_COLS[:-1]].to_numpy()


# ------------------------------------------------------------------ models
def _sk_class_weight(power: float, yi: np.ndarray, k: int):
    if power == 1.0:
        return "balanced"
    if power == 0.0:
        return None
    return dict(enumerate(r1.class_weights(yi, k, power)))


def build_estimator(method: str, params: dict, Xf: pd.DataFrame, yi: np.ndarray, k: int, n_jobs: int = 1):
    """The estimator of one candidate and the extra arguments of its fit. Random Forest is built exactly as in
    version 1.2; XGBoost also gets the L1 penalty (0 in the section 6 settings, the library default)."""
    power = float(params["weight_power"])
    if method == "RF":
        return r1.build_estimator("RF", params, Xf, yi, k, n_jobs)
    if method == "ET":
        from sklearn.ensemble import ExtraTreesClassifier
        ms = params["max_samples"]
        return ExtraTreesClassifier(
            n_estimators=300, criterion=params["criterion"], max_depth=params["max_depth"],
            max_features=params["max_features"], min_samples_leaf=int(params["min_samples_leaf"]),
            bootstrap=ms is not None, max_samples=ms, class_weight=_sk_class_weight(power, yi, k),
            random_state=SEED, n_jobs=n_jobs), {}
    if method == "XGB":
        XGBClassifier = ev._require_xgboost()
        est = XGBClassifier(
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]), min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]), reg_alpha=float(params.get("reg_alpha", 0.0)),
            gamma=float(params["gamma"]), random_state=SEED, eval_metric="mlogloss", n_jobs=n_jobs)
        return est, ({} if power == 0.0 else {"sample_weight": r1.class_weights(yi, k, power)[yi]})
    raise ValueError(f"unknown method {method}")


def base_view(view: str) -> str:
    return view[:-len("_near")] if view.endswith("_near") else view


class Tunable(ev.SklearnModel):
    """One method with one candidate's settings on one view. Same inputs, median filling and leakage checks as the
    protocol models; a _near view adds the nearby-crop inputs, rebuilt from the training fields given to fit."""

    def __init__(self, method: str, params: dict, view: str, n_jobs: int = 1):
        super().__init__(f"{method}[{view}]", None, ev.VIEWS_V11[base_view(view)])
        self.method, self.params, self.view, self.n_jobs = method, dict(params), view, n_jobs
        self.near = view.endswith("_near")

    def _inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.used_].fillna(self.fill_).reset_index(drop=True)
        if self.near:
            Xf = pd.concat([Xf, self.near_.transform(X[ev.BLOCK_COL].astype(str).to_numpy())], axis=1)
        return Xf

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Tunable":
        cols = [c for c in self.features if c in X.columns and X[c].notna().any()]
        if not cols:
            raise ev.LeakageError(f"{self.name}: none of its features are present with data")
        self.used_ = cols
        self.fill_ = X[cols].median()
        self.classes_ = np.array(sorted(y.unique()))
        self._index = {c: i for i, c in enumerate(self.classes_)}
        yi = y.map(self._index).to_numpy()
        if self.near:
            self.near_ = NearCrops(self.params.get("near_km", NEAR_KM_PROTOCOL)).fit(
                X[ev.BLOCK_COL].astype(str).to_numpy(), y.to_numpy())
        Xf = self._inputs(X)
        self.estimator, kwargs = build_estimator(self.method, self.params, Xf, yi, len(self.classes_), self.n_jobs)
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.estimator.fit(Xf, yi, **kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(self._inputs(X))
        full = np.zeros((len(X), len(self.classes_)))
        full[:, list(self.estimator.classes_)] = p
        return full


def shuffled_preds(model: Tunable, test: pd.DataFrame, n_shuffles: int, tag: tuple) -> dict:
    """As in version 1.2, plus one group for the nearby-crop inputs: they are shuffled together by shuffling the
    blocks they are computed from among the test fields."""
    base = test.reset_index(drop=True)
    idx = np.array([CROPS.index(c) for c in model.classes_])
    keys = r1.importance_keys(model.used_)
    if model.near:
        keys[f"group:{NEAR_GROUP}"] = [ev.BLOCK_COL]
    out = {}
    for key, cols in keys.items():
        for r in range(n_shuffles):
            perm = r1._rng(*tag, key, r).permutation(len(base))
            X = base.copy()
            X[cols] = base[cols].to_numpy()[perm]
            out[(key, r)] = idx[np.argmax(model.predict_proba(X), axis=1)]
    return out


# ------------------------------------------------------------------ one job: one method on one view
def run_job(method: str, view: str, df: pd.DataFrame, cands: list[dict], n_shuffles: int = N_SHUFFLES,
            log: Callable = print) -> dict:
    """The nested tests of one method on one view (both tests for field_sat and all_sat, new fields only for the
    _near views). Same steps as tune.run_job; it also keeps the out-of-fold probabilities of both tests for the blend."""
    t0 = time.time()
    name = f"{method}[{view}]"
    near = view in NEAR_VIEWS

    def make(c):
        return lambda: Tunable(method, c, view)

    rows, inner_rows, chosen_rows, imp_rows, conf = [], [], [], [], []

    def choose(test: str, outer: str, scores: list[float]) -> int:
        best = r1.pick(scores)
        inner_rows.extend({"test": test, "model": name, "outer": outer, "candidate": i, "inner_score": s}
                          for i, s in enumerate(scores))
        chosen_rows.append({"test": test, "model": name, "outer": outer, "candidate": best,
                            "inner_score": scores[best], "inner_score_protocol": scores[0],
                            "settings": json.dumps(cands[best], sort_keys=True)})
        return best

    y, dist = df["label"].to_numpy(), df[ev.GROUP_COL].to_numpy()
    lodo_proba = None
    if not near:
        # Main test: leave one district out, settings chosen inside the three training districts.
        lodo_proba = {"untuned": np.zeros((len(df), len(CROPS))), "tuned": np.zeros((len(df), len(CROPS)))}
        base_by_district, perm_by_key = {}, {}
        for held_out in sorted(set(dist)):
            t1 = time.time()
            tr_idx, te_idx = np.flatnonzero(dist != held_out), np.flatnonzero(dist == held_out)
            train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
            assert held_out not in set(train[ev.GROUP_COL])
            inner = r1.lodo_folds(train)
            scores = [r1.lodo_score(train, inner, make(c)) for c in cands]
            best = choose("lodo", held_out, scores)
            fitted = {}
            for version, i in (("untuned", 0), ("tuned", best)):
                if i not in fitted:
                    fitted[i] = Tunable(method, cands[i], view).fit(train, train["label"])
                m = fitted[i]
                proba = m.predict_proba(test)
                pred = m.classes_[np.argmax(proba, axis=1)]
                lodo_proba[version][np.ix_(te_idx, [CROPS.index(c) for c in m.classes_])] = proba
                model_name = name + (TUNED if version == "tuned" else "")
                rows.append({"test": "lodo", "model": model_name, "version": version, "candidate": i,
                             "district": held_out, "n_train": int(len(train)),
                             **ev.scores(test["label"], pred, proba, m.classes_)})
                conf.append(ev._confusion("main", TESTS["lodo"], model_name, held_out, test["label"], pred))
            m, yt = fitted[best], test["label"].to_numpy()
            base_by_district[held_out] = r1.macro_f1(yt, m.classes_[np.argmax(m.predict_proba(test), axis=1)])
            for (key, r), p in shuffled_preds(m, test, n_shuffles, (name, "lodo", held_out)).items():
                perm_by_key.setdefault(key, {}).setdefault(r, {})[held_out] = r1.macro_f1(yt, LABELS[p])
            log(f"{name} main test, {held_out} left out: candidate {best} chosen (inner score {scores[best]:.3f}, "
                f"protocol settings {scores[0]:.3f}), {time.time() - t1:.0f} s")
        lodo_base = float(np.nanmean(list(base_by_district.values())))
        complete = {k: [float(np.nanmean([v[r][d] for d in base_by_district])) for r in sorted(v)]
                    for k, v in perm_by_key.items() if all(set(v[r]) == set(base_by_district) for r in v)}
        imp_rows += r1._importance_rows("lodo", name, lodo_base, complete)

    # Second test: new fields in the known districts, the same 5 block folds as eval_v2 and tuning_v1.
    folds = ev.block_folds(df)
    proba_u, proba_t = np.zeros((len(df), len(CROPS))), np.zeros((len(df), len(CROPS)))
    perm_pred: dict[tuple[str, int], np.ndarray] = {}
    for f, (tr, te) in enumerate(folds):
        t1 = time.time()
        train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        inner = ev.block_folds(train, n_splits=INNER_BLOCK_FOLDS, seed=SEED)
        scores = [r1.block_score(train, r1.oof_proba(train, inner, make(c))) for c in cands]
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
    for version, proba in (("untuned", proba_u), ("tuned", proba_t)):
        pred = LABELS[np.argmax(proba, axis=1)]
        model_name = name + (TUNED if version == "tuned" else "")
        for d in sorted(set(dist)):
            mask = dist == d
            rows.append({"test": "blockcv", "model": model_name, "version": version, "candidate": float("nan"),
                         "district": d, "n_train": float("nan"),
                         **ev.scores(y[mask], pred[mask], proba[mask], LABELS)})
            conf.append(ev._confusion("main", TESTS["blockcv"], model_name, d, y[mask], pred[mask]))
    block_base = r1.block_score(df, proba_t)
    keys = sorted({k for k, _ in perm_pred})
    complete = {k: [r1.main_score(y, LABELS[perm_pred[(k, r)]], dist) for r in range(n_shuffles)]
                for k in keys if all((perm_pred[(k, r)] >= 0).all() for r in range(n_shuffles))}
    imp_rows += r1._importance_rows("blockcv", name, block_base, complete)

    return {"method": method, "view": view, "results": pd.DataFrame(rows), "inner": pd.DataFrame(inner_rows),
            "chosen": pd.DataFrame(chosen_rows), "importance": pd.DataFrame(imp_rows),
            "confusion": pd.concat(conf, ignore_index=True),
            "lodo_proba": lodo_proba, "blockcv_proba": {"untuned": proba_u, "tuned": proba_t},
            "seconds": time.time() - t0}


def _job_to_file(method, view, df, cands, n_shuffles, work: Path) -> str:
    r1._limit_threads()
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
        os.environ.update(r1.SINGLE_THREAD_ENV)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_job_to_file, m, v, df, cands[m], n_shuffles, work): (m, v) for m, v in todo}
            for fut in as_completed(futures):
                log(fut.result())
    parts = {}
    for m, v in jobs:
        with open(work / f"{m}_{v}.pkl", "rb") as fh:
            parts[(m, v)] = pickle.load(fh)
    return {"candidates": cands, "parts": parts, "jobs": jobs}


# ------------------------------------------------------------------ blend and reference
def _rows(test: str, name: str, version: str, df: pd.DataFrame, proba: np.ndarray) -> tuple[list[dict], list]:
    y, dist = df["label"].to_numpy(), df[ev.GROUP_COL].to_numpy()
    pred = LABELS[np.argmax(proba, axis=1)]
    rows, conf = [], []
    for d in sorted(set(dist)):
        mask = dist == d
        rows.append({"test": test, "model": name, "version": version, "candidate": float("nan"), "district": d,
                     "n_train": int((dist != d).sum()) if test == "lodo" else float("nan"),
                     **ev.scores(y[mask], pred[mask], proba[mask], LABELS)})
        conf.append(ev._confusion("main", TESTS[test], name, d, y[mask], pred[mask]))
    return rows, conf


def blend(df: pd.DataFrame, parts: dict, view: str) -> Optional[dict]:
    """Equal-weight mean of the tuned members' out-of-fold probabilities on one view, in every test they ran."""
    members = [parts.get((m, view)) for m in METHODS]
    if any(p is None for p in members):
        return None
    name = f"{BLEND}[{view}]" + TUNED
    out = {"name": name, "rows": [], "confusion": [], "proba": {}}
    for test, key in (("lodo", "lodo_proba"), ("blockcv", "blockcv_proba")):
        if any(p.get(key) is None for p in members):
            continue
        proba = np.mean([p[key]["tuned"] for p in members], axis=0)
        rows, conf = _rows(test, name, "tuned", df, proba)
        out["rows"] += rows
        out["confusion"] += conf
        out["proba"][test] = proba
    return out


def near_reference(df: pd.DataFrame) -> tuple[list[dict], list, np.ndarray]:
    """Nearest surveyed fields (h = 4 km) on the 5 block folds of the new-fields test. Decides nothing."""
    proba = np.zeros((len(df), len(CROPS)))
    blocks, labels = df[ev.BLOCK_COL].astype(str).to_numpy(), df["label"].to_numpy()
    for tr, te in ev.block_folds(df):
        proba[te] = NearCrops(NEAR_KM_PROTOCOL).fit(blocks[tr], labels[tr]).first_crop_proba(blocks[te])
    rows, conf = _rows("blockcv", NEAR_REF, "", df, proba)
    return rows, conf, proba


# ------------------------------------------------------------------ putting it together
CHECKED = {"B0_majority", ev.BASELINE} | {f"{m}[{v}]" for m in ("RF", "XGB") for v in BASE_VIEWS}


def builtin_check(results: dict[str, pd.DataFrame], untuned_dir: Path = UNTUNED_DIR) -> dict:
    """Random Forest and XGBoost with the section 6 settings on field_sat and all_sat, and the baselines, must give the
    eval_v2 numbers in every district of both tests. Extra Trees and the _near views have no earlier numbers."""
    return r1.builtin_check({t: r[r["model"].isin(CHECKED)] for t, r in results.items()}, untuned_dir)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    s = r1.summarise(results)
    s.loc[s.index == NEAR_REF, ["districts_beating_B1", "passes_rule"]] = [float("nan"), ""]
    return s


def round1_summary() -> dict[str, pd.DataFrame]:
    return {t: pd.read_csv(ROUND1_DIR / f"main_{t}_summary.csv", index_col=0) for t in TESTS}


def decisions(summ: dict[str, pd.DataFrame], round1: dict[str, pd.DataFrame]) -> dict:
    """The two decisions section 12 fixed before the run."""
    lodo = summ["lodo"]
    contenders = [i for i in lodo.index if i.endswith(TUNED)]
    passing = [i for i in contenders if lodo.loc[i, "passes_rule"] == "yes"]
    best_lodo = lodo.loc[contenders, "macro_f1_mean"].idxmax()
    blk = summ["blockcv"]
    tuned = [i for i in blk.index if i.endswith(TUNED)]
    today = float(round1["blockcv"].loc[MAP_MODEL, "macro_f1_mean"])
    beating = blk.loc[tuned][blk.loc[tuned, "macro_f1_mean"] > today + MAP_MARGIN]
    winner = beating["macro_f1_mean"].idxmax() if len(beating) else None
    best_new = blk.loc[tuned, "macro_f1_mean"].idxmax()
    return {"adoption": {"passing": passing, "best_tuned": best_lodo,
                         "best_tuned_score": float(lodo.loc[best_lodo, "macro_f1_mean"]),
                         "best_tuned_wins": int(lodo.loc[best_lodo, "districts_beating_B1"]),
                         "b1_score": float(lodo.loc[ev.BASELINE, "macro_f1_mean"])},
            "map": {"today": MAP_MODEL, "today_score": today, "margin": MAP_MARGIN,
                    "beating": {i: float(beating.loc[i, "macro_f1_mean"]) for i in beating.index},
                    "serve": winner or MAP_MODEL, "changes": winner is not None},
            "new_fields": {"best": best_new, "best_score": float(blk.loc[best_new, "macro_f1_mean"])}}


def assemble(df: pd.DataFrame, run: dict, untuned_dir: Path = UNTUNED_DIR) -> dict:
    parts = run["parts"]
    base = r1.baselines(df)
    ref_rows, ref_conf, ref_proba = near_reference(df)
    blends = [b for b in (blend(df, parts, v) for v in VIEWS) if b is not None]
    results, summ = {}, {}
    for test in TESTS:
        frames = [base[test]] + [p["results"][p["results"]["test"] == test].drop(columns="test")
                                 for p in parts.values()]
        extra = [r for b in blends for r in b["rows"] if r["test"] == test]
        if test == "blockcv":
            extra += ref_rows
        if extra:
            frames.append(pd.DataFrame(extra).drop(columns="test"))
        results[test] = pd.concat(frames, ignore_index=True)
        summ[test] = summarise(results[test])
    check = builtin_check(results, untuned_dir)
    round1 = round1_summary()
    dec = decisions(summ, round1)
    probas = {f"{m}[{v}]" + TUNED: p["blockcv_proba"]["tuned"] for (m, v), p in parts.items()}
    probas.update({b["name"]: b["proba"]["blockcv"] for b in blends if "blockcv" in b["proba"]})
    best = dec["new_fields"]["best"]
    analyses = {
        "bands": pd.concat([r1.confidence_bands(df, p).assign(model=k) for k, p in probas.items()], ignore_index=True),
        "sure": pd.concat([r1.answer_when_sure(df, p).assign(model=k) for k, p in probas.items()], ignore_index=True),
        "best_bands": r1.confidence_bands(df, probas[best]), "best_sure": r1.answer_when_sure(df, probas[best]),
        "mixed": r1.mixed_land(df),
        "importance": pd.concat([p["importance"] for p in parts.values()], ignore_index=True),
        "chosen": pd.concat([p["chosen"] for p in parts.values()], ignore_index=True),
        "inner": pd.concat([p["inner"] for p in parts.values()], ignore_index=True),
        "confusion": pd.concat([base["confusion"]] + ref_conf + [p["confusion"] for p in parts.values()]
                               + [c for b in blends for c in b["confusion"]], ignore_index=True),
        "reference_proba": ref_proba,
    }
    return {"results": results, "summary": summ, "check": check, "decisions": dec, "analyses": analyses,
            "round1": round1, "blends": [b["name"] for b in blends]}


# ------------------------------------------------------------------ report
def label(model: str) -> str:
    """'ET[all_sat_near] tuned' becomes 'Extra Trees, all_sat_near, tuned'."""
    core = model.replace(TUNED, "")
    if "[" not in core:
        return model
    method, view = core.rstrip("]").split("[")
    return f"{METHOD_NAMES.get(method, method)}, {view}" + (", tuned" if model.endswith(TUNED) else ", section 6")


def _new_fields_table(summ: pd.DataFrame, round1: pd.DataFrame) -> pd.DataFrame:
    tuned = [i for i in summ.index if i.endswith(TUNED)]
    order = summ.loc[tuned, "macro_f1_mean"].sort_values(ascending=False).index
    rows = []
    for i in order:
        r = summ.loc[i]
        untuned = i.replace(TUNED, "")
        rows.append({"model": label(i), "main score": r["macro_f1_mean"],
                     "worst district": r["worst_district_macro_f1"],
                     "first crop right, all fields": r["top1_all_fields"],
                     "section 6 settings": summ.loc[untuned, "macro_f1_mean"] if untuned in summ.index else np.nan,
                     "version 1.2, tuned": round1.loc[i, "macro_f1_mean"] if i in round1.index else np.nan})
    return pd.DataFrame(rows)


def _lodo_table(summ: pd.DataFrame, round1: pd.DataFrame) -> pd.DataFrame:
    tuned = [i for i in summ.index if i.endswith(TUNED)]
    order = summ.loc[tuned, "macro_f1_mean"].sort_values(ascending=False).index
    rows = []
    for i in order:
        r = summ.loc[i]
        untuned = i.replace(TUNED, "")
        rows.append({"model": label(i), "main score": r["macro_f1_mean"],
                     "worst district": r["worst_district_macro_f1"],
                     "better than B1 in": f"{int(r['districts_beating_B1'])} of 4", "passes the adoption rule": r["passes_rule"],
                     "section 6 settings": summ.loc[untuned, "macro_f1_mean"] if untuned in summ.index else np.nan,
                     "version 1.2, tuned": round1.loc[i, "macro_f1_mean"] if i in round1.index else np.nan})
    return pd.DataFrame(rows)


def _per_district(results: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    piv = results.pivot(index="model", columns="district", values="macro_f1")
    return piv.reindex([m for m in models if m in piv.index]).rename(index=label)


def _near_gain(summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in list(METHODS) + [BLEND]:
        for v in BASE_VIEWS:
            a, b = f"{m}[{v}]" + TUNED, f"{m}[{v}_near]" + TUNED
            if a in summ.index and b in summ.index:
                rows.append({"model": f"{METHOD_NAMES[m]}, {v}", "without nearby crops": summ.loc[a, "macro_f1_mean"],
                             "with nearby crops": summ.loc[b, "macro_f1_mean"],
                             "change": summ.loc[b, "macro_f1_mean"] - summ.loc[a, "macro_f1_mean"],
                             "first crop right, with": summ.loc[b, "top1_all_fields"]})
    return pd.DataFrame(rows)


def _settings_table(chosen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test in TESTS:
        for (model,), g in chosen[chosen["test"] == test].groupby(["model"], sort=False):
            rows.append({"test": TEST_NAMES[test], "model": label(model + TUNED),
                         "candidate chosen in each fold": ", ".join(f"{o}: {c}" for o, c in zip(g["outer"], g["candidate"])),
                         "folds that kept the section 6 settings": f"{int((g['candidate'] == 0).sum())} of {len(g)}"})
    return pd.DataFrame(rows)


def _importance_groups(imp: pd.DataFrame) -> pd.DataFrame:
    g = imp[(imp["kind"] == "group") & (imp["test"] == "blockcv")]
    if g.empty:
        return pd.DataFrame()
    piv = g.pivot_table(index="model", columns="name", values="drop_mean")
    cols = [c for c in list(r1.GROUPS) + [NEAR_GROUP] if c in piv.columns]
    return piv[cols].rename(index=label).reset_index()


def write_report(out: Path, info: dict, df: pd.DataFrame, res: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    a, summ, dec, check, round1 = res["analyses"], res["summary"], res["decisions"], res["check"], res["round1"]
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

    ad, mp, nf = dec["adoption"], dec["map"], dec["new_fields"]
    blk, lodo = summ["blockcv"], summ["lodo"]
    if ad["passing"]:
        adoption = (f"{', '.join(label(i) for i in ad['passing'])} pass the adoption rule on the main test, so the best "
                    "of them replaces B1 as the default answer of the API (section 7).")
    else:
        adoption = (f"No tuned model or blend passes the adoption rule, so B1 stays the default answer of the API. Best "
                    f"on the main test: {label(ad['best_tuned'])}, {ad['best_tuned_score']:.3f} against "
                    f"{ad['b1_score']:.3f} for B1, better than B1 in {ad['best_tuned_wins']} of 4 districts.")
    if mp["changes"]:
        map_line = (f"The map moves to {label(mp['serve'])}: on new fields it scores {mp['beating'][mp['serve']]:.3f} "
                    f"against {mp['today_score']:.3f} for the model on the map since 23 Nov ({label(mp['today'])}), "
                    f"more than {mp['margin']} ahead. Its final settings are chosen next, on all fields (`--final`).")
    else:
        map_line = (f"The map keeps {label(mp['today'])} ({mp['today_score']:.3f} on new fields): the best of this "
                    f"round, {label(nf['best'])}, scores {nf['best_score']:.3f}, not more than {mp['margin']} ahead.")
    gain = _near_gain(blk)
    near_best = max((i for i in blk.index if i.endswith(TUNED) and "_near]" in i),
                    key=lambda i: blk.loc[i, "macro_f1_mean"], default=None)
    base_best = max((i for i in blk.index if i.endswith(TUNED) and "_near]" not in i),
                    key=lambda i: blk.loc[i, "macro_f1_mean"], default=None)
    r1_best = round1["blockcv"].loc[[i for i in round1["blockcv"].index if i.endswith(TUNED)], "macro_f1_mean"]
    expect_a = (f"(a) Best with nearby crops: {label(near_best)}, {blk.loc[near_best, 'macro_f1_mean']:.3f}; best "
                f"without: {label(base_best)}, {blk.loc[base_best, 'macro_f1_mean']:.3f}."
                if near_best and base_best else "(a) Not measured in this run.")
    et_rf = [(v, blk.loc[f"ET[{v}]" + TUNED, "macro_f1_mean"] - blk.loc[f"RF[{v}]" + TUNED, "macro_f1_mean"])
             for v in VIEWS if f"ET[{v}]" + TUNED in blk.index and f"RF[{v}]" + TUNED in blk.index]
    expect_b = (f"(b) Best without nearby crops in this round {blk.loc[base_best, 'macro_f1_mean']:.3f}, against "
                f"{r1_best.max():.3f} for the best of version 1.2 ({label(r1_best.idxmax())}). "
                "Extra Trees minus Random Forest on the same view: "
                + ", ".join(f"{v} {d:+.3f}" for v, d in et_rf) + "." if base_best else "(b) Not measured in this run.")
    expect_c = f"(c) {'Some models pass' if ad['passing'] else 'No model passes'} the adoption rule on the main test."

    ref_names = {"B0_majority": "B0: the most common crop of all training fields",
                 r1.DistrictMajority.name: "the most common crop of the field's district",
                 ev.BASELINE: "B1: expert rules", NEAR_REF: "nearest surveyed fields (h = 4 km)"}
    pooled = r1.pooled_top1(res["results"]["blockcv"])
    refs = pd.DataFrame({"": list(ref_names.values()) + [f"{label(MAP_MODEL)} (map since 23 Nov, version 1.2)"],
                         "main score": [blk.loc[n, "macro_f1_mean"] for n in ref_names] + [mp["today_score"]],
                         "first crop right, all fields": [pooled.get(n, np.nan) for n in ref_names]
                         + [round1["blockcv"].loc[MAP_MODEL, "top1_all_fields"]]})
    top_models = list(blk.loc[[i for i in blk.index if i.endswith(TUNED)], "macro_f1_mean"]
                      .sort_values(ascending=False).index[:6]) + [ev.BASELINE, NEAR_REF]
    bands = a["best_bands"]
    bands_all = bands[bands["district"] == "all"].drop(columns="district").rename(columns=r1.PLAIN)
    sure = a["best_sure"]
    sure_all = sure[sure["district"] == "all"].drop(columns="district").rename(columns=r1.PLAIN)
    sure_d = (sure[(sure["district"] != "all") & (sure["probability_at_least"] == 0.7)]
              .drop(columns="probability_at_least").rename(columns=r1.PLAIN))
    lines = [
        "# Crop recommendation: second round of tuning (tuning_v2)", "",
        f"Protocol: `docs/training_protocol.md`, version {info['protocol_version']} (section 12). Run at "
        f"{info['run_at_utc']} UTC: {info['compute_hours']:.1f} hours of computing, {info['workers']} jobs at a "
        "time." + (" **Quick check only: these numbers are not reported.**" if info.get("quick_check_only") else ""),
        f"Table: `{info['table']}` (SHA-256 `{info['table_sha256'][:16]}...`). Code commit: "
        f"`{(info['git_commit'] or 'unknown')[:10]}`"
        + (" (with uncommitted changes, so rerun after committing)." if info.get("uncommitted_changes") else ".")
        + (f" Report text rebuilt from the saved jobs at commit `{info['report_rebuilt_at_commit'][:10]}`, with the "
           "same numbers." if info.get("report_rebuilt_at_commit") else ""),
        f"Candidates per method: {', '.join(f'{m} {len(c)}' for m, c in info['candidates'].items())} (the section 6 "
        f"settings, then {info['n_draws']} random draws; Random Forest also has the map model of version 1.2 as "
        f"candidate 1). {info['inner_block_folds']} inner block folds, {info['n_shuffles']} shuffles per input.",
        "",
        f"**Built-in check: {'passed' if check['passed'] else 'FAILED'}.** Random Forest and XGBoost with the section 6 "
        f"settings and the baselines give the `eval_v2` numbers in {check['rows_compared'] - len(check['mismatches'])} "
        f"of {check['rows_compared']} rows (district and model)."
        + ("" if check["passed"] else " Mismatches: " + "; ".join(check["mismatches"][:10])),
        "",
        "## Decisions fixed before the run", "",
        f"1. **Adoption (main test).** {adoption}",
        f"2. **Map.** {map_line}", "",
        "Expected before the run: (a) the nearby-crop inputs give the largest gain on new fields; (b) the wider search "
        "and Extra Trees add little, up to about 0.01; (c) no model passes the adoption rule.",
        expect_a, expect_b, expect_c, "",
        "## New fields in the known districts (decides the map)", "",
        "Settings chosen inside the training part of each block fold only, then scored once on the test fold. The "
        "nearby-crop inputs of a test field come from the training fields of other blocks only.", "",
        r1._md(_new_fields_table(blk, round1["blockcv"]), index=False), "",
        "Simple choices and the map model of version 1.2 on the same folds:", "", r1._md(refs, index=False), "",
        "Main score per district (the six best and two references):", "",
        r1._md(_per_district(res["results"]["blockcv"], top_models)), "",
        "## What the nearby-crop inputs add (new fields)", "",
        r1._md(gain, index=False) if len(gain) else "Not measured in this run.", "",
        "## Main test: leave one district out (decides adoption)", "",
        "Settings chosen inside the three training districts only, then scored once on the district left out. The "
        "nearby-crop views do not run here: a district never seen has no surveyed field nearby.", "",
        r1._md(_lodo_table(lodo, round1["lodo"]), index=False), "",
        f"B1 (expert rules): {lodo.loc[ev.BASELINE, 'macro_f1_mean']:.3f}. "
        f"B0 (most common crop): {lodo.loc['B0_majority', 'macro_f1_mean']:.3f}.", "",
        "## Settings chosen in each fold", "",
        "Candidate 0 is the section 6 settings. Every candidate is in `candidates.json`; the inner score of every "
        "candidate in every fold is in `inner_scores.csv`.", "",
        r1._md(_settings_table(a["chosen"]), index=False), "",
        f"## When the best model of this round is sure ({label(nf['best'])}, new fields)", "",
        "Probability the model gives its first crop, and how often that first crop was the crop actually planted. "
        "Out-of-fold predictions: no field was predicted by a model that saw its 2 km block.", "",
        r1._md(bands_all, index=False), "",
        "If it answered only when its probability for the first crop is at least this high:", "",
        r1._md(sure_all, index=False), "",
        "Per district, answering only from 0.7:", "", r1._md(sure_d, index=False), "",
        "## Input importance (tuned models, new fields)", "",
        "How much the main score drops when a group of inputs is shuffled among the test fields (mean of the "
        "shuffles). The nearby-crop inputs are shuffled together by shuffling the blocks they come from. Every input "
        "and its spread: `importance.csv`.", "",
        r1._md(_importance_groups(a["importance"]), index=False), "",
        "## How mixed the land is", "",
        "If every field got the most common crop of its own 2 km block (which no model can know in advance):", "",
        r1._md(a["mixed"].rename(columns=r1.PLAIN), index=False), "",
        "## How to read this", "",
        "- Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over the "
        "four districts (section 5). 'First crop right' is the share of fields whose first crop was the crop actually "
        "planted.",
        "- 'Section 6 settings' is the model with the fixed settings of the protocol. 'Tuned' chose its settings again "
        "in every fold, inside the training fields of that fold. 'Version 1.2, tuned' is the same model in `tuning_v1`.",
        "- The blend averages the probabilities of the tuned Random Forest, Extra Trees and XGBoost of the same view.",
        "- The adoption rule (section 7) and the map rule (section 12) were written before this run.",
        "- Full numbers: the CSV files in this folder. Settings, versions and timing: `run_info.json`.",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ the model the map serves
FINAL_FILE = "final_model.json"


def final_model(df: pd.DataFrame, out: Path = DEFAULT_OUT, log: Callable = print) -> dict:
    """The model the map serves under decision 2 of section 12, written to <out>/final_model.json.

    Each member's settings (one member, or the three of a blend) come from the same comparison as inside the nested
    test, run once on all fields: every candidate scored on 3 folds over 2 km blocks (seed 0), main score per district
    then the mean, ties to the candidate drawn first. If the map does not change, the file says so."""
    info = json.loads((Path(out) / "run_info.json").read_text(encoding="utf-8"))
    mp = info["decisions"]["map"]
    import sklearn
    result = {"model": mp["serve"], "changes": mp["changes"], "decided_by": "decision 2 of docs/training_protocol.md, "
              "section 12", "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
              "git_commit": ev._git_commit(), "uncommitted_changes": ev._git_has_uncommitted_changes(),
              "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                           "scikit-learn": sklearn.__version__}}
    if mp["changes"]:
        method, view = mp["serve"].replace(TUNED, "").rstrip("]").split("[")
        folds = ev.block_folds(df, n_splits=INNER_BLOCK_FOLDS, seed=SEED)
        members = {}
        for m in (METHODS if method == BLEND else (method,)):
            t0 = time.time()
            cands = candidates(m, info["n_draws"])
            scores = [r1.block_score(df, r1.oof_proba(df, folds, (lambda c=c, m=m: Tunable(m, c, view))))
                      for c in cands]
            best = r1.pick(scores)
            members[m] = {"candidate": best, "settings": cands[best], "comparison_scores": [round(s, 6) for s in scores]}
            log(f"{m}[{view}]: candidate {best} chosen on all {len(df)} fields (comparison score {scores[best]:.3f}, "
                f"section 6 settings {scores[0]:.3f}), {time.time() - t0:.0f} s")
        result.update({
            "method": method, "view": view, "members": members,
            "features": list(ev.VIEWS_V11[base_view(view)]) + (NEAR_COLS if view.endswith("_near") else []),
            "chosen_by": (f"same comparison as inside the nested test, on all {len(df)} fields: {INNER_BLOCK_FOLDS} "
                          f"folds over {ev.BLOCK_COL} (seed {SEED}), main score per district then the mean; ties go to "
                          "the candidate drawn first"),
            "tested_main_score_new_fields": mp["beating"][mp["serve"]], "trained_on_fields": int(len(df))})
    else:
        result["note"] = (f"The map keeps the model of version 1.2: {ROUND1_DIR.relative_to(ev.REPO).as_posix()}/"
                          "final_model.json")
    path = Path(out) / FINAL_FILE
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"Written to {path}")
    return result


# ------------------------------------------------------------------ command line
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=None, help="Feature table CSV. Default: feature_table_v6b.csv.")
    ap.add_argument("--out", default=None, help="Output folder. Default: reports/tuning_v2 (quick: a temp folder).")
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
        methods, views, n_draws, n_shuffles = METHODS, ("field_sat", "field_sat_near"), 2, 1
        out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "tuning_v2_quick"
    else:
        methods, views, n_draws, n_shuffles = METHODS, VIEWS, N_DRAWS, N_SHUFFLES
        out = Path(args.out) if args.out else DEFAULT_OUT
    work = out / "_work"
    print(f"Protocol {PROTOCOL_VERSION}. Table: {table}. Output: {out}" + (" (quick check)" if args.quick else ""),
          flush=True)
    df = ev.prepare(pd.read_csv(table, low_memory=False))
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
    previous = {}
    if (out / "run_info.json").exists() and args.assemble_only:
        previous = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    info = {
        "protocol_version": PROTOCOL_VERSION, "quick_check_only": args.quick,
        "run_at_utc": previous.get("run_at_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "compute_hours": sum(p["seconds"] for p in run["parts"].values()) / 3600,
        "wall_hours_this_call": (time.time() - t0) / 3600,
        "workers": previous.get("workers") or args.workers,
        "job_minutes": {f"{m}[{v}]": round(p["seconds"] / 60, 1) for (m, v), p in run["parts"].items()},
        "table": table_rel, "table_sha256": ev.file_sha256(table),
        "git_commit": previous.get("git_commit") or ev._git_commit(),
        "uncommitted_changes": (previous["uncommitted_changes"] if "uncommitted_changes" in previous
                                else ev._git_has_uncommitted_changes()),
        **({"report_rebuilt_at_commit": ev._git_commit(),
            "report_rebuilt_with_uncommitted_changes": ev._git_has_uncommitted_changes()} if previous else {}),
        "rows": {"before_filters": df.attrs.get("rows_before_filters"), "after_qc_ok": df.attrs.get("rows_after_qc"),
                 "main": len(df)},
        "methods": list(methods), "views": list(views), "near_views_tests": "new fields only",
        "n_draws": n_draws, "inner_block_folds": INNER_BLOCK_FOLDS, "n_shuffles": n_shuffles, "seed": SEED,
        "near": {"block_deg": BLOCK_DEG, "cutoff_times_h": NEAR_CUTOFF, "h_of_section_6_settings": NEAR_KM_PROTOCOL},
        "tests": {"main": "leave one district out; inside: leave each training district out in turn",
                  "second": f"{ev.N_BLOCK_FOLDS} folds over {ev.BLOCK_COL} (StratifiedGroupKFold, seed {SEED}); "
                            f"inside: {INNER_BLOCK_FOLDS} folds over {ev.BLOCK_COL}, seed {SEED}"},
        "builtin_check": res["check"], "decisions": res["decisions"], "blends": res["blends"],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "candidates": run["candidates"],
    }
    path = write_report(out, info, df, res)
    blk = res["summary"]["blockcv"]
    tuned = blk.loc[[i for i in blk.index if i.endswith(TUNED)], "macro_f1_mean"].sort_values(ascending=False)
    print("\n== new fields (tuned models and blends)")
    print(tuned.round(3).to_string())
    print(f"\nBuilt-in check: {'passed' if res['check']['passed'] else 'FAILED'} "
          f"({res['check']['rows_compared']} rows, {len(res['check']['mismatches'])} mismatches)")
    print(f"Map: serve {res['decisions']['map']['serve']}. Adoption: {res['decisions']['adoption']['passing'] or 'none'}")
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
