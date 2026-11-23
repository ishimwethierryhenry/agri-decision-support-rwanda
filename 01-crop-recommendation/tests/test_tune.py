"""Tests for the tuning harness (src/models/tune.py, training protocol 1.2, section 11).

They use the real feature table with tiny budgets, so they run in about a minute. The full run is
`python 01-crop-recommendation/src/models/tune.py`.
Run from the repo root:  python -m pytest 01-crop-recommendation/tests/test_tune.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-crop-recommendation" / "src" / "models"))
pytest.importorskip("xgboost")
import evaluate as ev  # noqa: E402
import tune  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return ev.prepare(pd.read_csv(tune.TABLE, low_memory=False))


# ------------------------------------------------------------------ candidates and settings
def test_candidates_start_with_the_protocol_settings_and_never_change():
    for m in tune.METHODS:
        c = tune.candidates(m)
        assert len(c) == tune.N_DRAWS + 1 and c[0] == tune.PROTOCOL_SETTINGS[m]
        assert c == tune.candidates(m)                      # seed 0: the same 61 candidates every time
        assert len({json.dumps(x, sort_keys=True) for x in c}) > 50


def test_candidates_stay_inside_the_ranges_of_section_11():
    spaces = tune.search_spaces()
    for m in tune.METHODS:
        for c in tune.candidates(m)[1:]:
            assert set(c) == set(spaces[m])
            for k, v in c.items():
                space = spaces[m][k]
                if isinstance(space, list):
                    assert v in space, (m, k, v)
                else:
                    lo, hi = space.support()
                    assert lo - 1e-6 <= v <= hi + 1e-6, (m, k, v)


def test_protocol_settings_give_exactly_the_protocol_models(df):
    train, test = df[df["district"] != "Ruhango"], df[df["district"] == "Ruhango"]
    ref = {m.name: m for m in ev.default_models(ev.VIEWS_V11, {})}
    for m in tune.METHODS:
        a = tune.Tunable(m, tune.PROTOCOL_SETTINGS[m], "all_sat").fit(train, train["label"]).predict_proba(test)
        b = ref[f"{m}[all_sat]"].fit(train, train["label"]).predict_proba(test)
        assert np.allclose(a, b, rtol=0, atol=1e-9), m


def test_class_weight_strength():
    from sklearn.utils.class_weight import compute_class_weight
    y = np.array([0] * 10 + [1] * 30 + [2] * 60)
    assert np.allclose(tune.class_weights(y, 3, 1.0), compute_class_weight("balanced", classes=np.arange(3), y=y))
    assert np.allclose(tune.class_weights(y, 3, 0.0), 1.0)
    assert np.allclose(tune.class_weights(y, 3, 0.5) ** 2, tune.class_weights(y, 3, 1.0))


def test_curved_logistic_regression_bends_only_field_inputs_and_handles_gaps(df):
    train = df[df["district"].isin(["Musanze", "Nyabihu"])].copy()   # one weather cell: their climate is the same
    train.loc[train.index[:20], "ph_0_20cm"] = np.nan
    m = tune.Tunable("LR", {"C": 10.0, "shape": "curved", "weight_power": 0.5}, "all_sat").fit(train, train["label"])
    columns = {name: cols for name, _, cols in m.estimator.steps[0][1].transformers}
    assert {"elevation_m", "ph_0_20cm", "chirps_precip_mm"} <= set(columns["curved"])
    assert not set(columns["curved"]) & set(ev.CLIMATE + ev.NISR_SHARES)
    p = m.predict_proba(df[df["district"] == "Nyagatare"])           # lower and drier than anything in training
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)


# ------------------------------------------------------------------ nested tests
def test_inner_folds_only_use_the_training_part(df):
    for held_out in sorted(df["district"].unique()):
        train = df[df["district"] != held_out].reset_index(drop=True)
        inner = tune.lodo_folds(train)
        assert len(inner) == 3
        for tr, te in inner:
            seen = set(train["district"].iloc[tr]) | set(train["district"].iloc[te])
            assert held_out not in seen and len(set(train["district"].iloc[te])) == 1
            assert set(train["district"].iloc[tr]).isdisjoint(train["district"].iloc[te])
    outer = ev.block_folds(df)
    assert len(outer) == ev.N_BLOCK_FOLDS
    for tr, te in outer:
        train = df.iloc[tr].reset_index(drop=True)
        assert set(df["block_id"].iloc[te].astype(str)).isdisjoint(train["block_id"].astype(str))
        for itr, ite in ev.block_folds(train, n_splits=tune.INNER_BLOCK_FOLDS, seed=tune.SEED):
            assert set(train["block_id"].iloc[itr].astype(str)).isdisjoint(train["block_id"].iloc[ite].astype(str))


def test_a_tie_goes_to_the_candidate_drawn_first():
    assert tune.pick([0.5, 0.7, 0.7]) == 1
    assert tune.pick([0.7, 0.7]) == 0
    assert tune.pick([float("nan"), 0.1]) == 1


def test_main_score_is_the_one_of_evaluate():
    rng = np.random.default_rng(0)
    y, pred = rng.choice(ev.CROPS, 300), rng.choice(ev.CROPS, 300)
    dist = rng.choice(["A", "B", "C"], 300)
    full = np.full((300, 4), 0.25)
    assert tune.macro_f1(y, pred) == ev.scores(y, pred, full, np.array(ev.CROPS))["macro_f1"]
    per = [ev.scores(y[dist == d], pred[dist == d], full[dist == d], np.array(ev.CROPS))["macro_f1"] for d in "ABC"]
    assert np.isclose(tune.main_score(y, pred, dist), np.mean(per))


# ------------------------------------------------------------------ analyses that decide nothing
def test_district_majority_reference(df):
    m = tune.DistrictMajority().fit(df, df["label"])
    first = m.classes_[m.predict_proba(df).argmax(axis=1)]
    expected = df.groupby("district")["label"].agg(lambda s: s.value_counts().idxmax())
    assert (first == df["district"].map(expected).to_numpy()).all()
    unseen = df.head(2).assign(district="Kigali")
    assert np.allclose(m.predict_proba(unseen), m.prior_)


def test_bands_and_answer_when_sure_on_a_small_table():
    toy = pd.DataFrame({"label": ["maize", "beans", "maize", "rice"], "district": ["A", "A", "B", "B"]})
    # columns follow CROPS: beans, irish_potato, maize, rice
    proba = np.array([[0.10, 0.0, 0.85, 0.05], [0.6, 0.1, 0.2, 0.1], [0.3, 0.0, 0.35, 0.35], [0.0, 0.0, 1.0, 0.0]])
    bands = tune.confidence_bands(toy, proba)
    every = bands[bands["district"] == "all"].set_index("band")
    assert every["fields"].sum() == 4
    assert every.loc["0.8 to 0.9", "fields"] == 1 and every.loc["0.8 to 0.9", "first_crop_right"] == 1.0
    assert every.loc["0.9 to 1.0", "fields"] == 1 and every.loc["0.9 to 1.0", "first_crop_right"] == 0.0
    sure = tune.answer_when_sure(toy, proba)
    row = sure[(sure["district"] == "all") & (sure["probability_at_least"] == 0.7)].iloc[0]
    assert row["fields_answered"] == 2 and row["first_crop_right"] == 0.5


def test_mixed_land_on_a_small_table():
    toy = pd.DataFrame({"label": ["maize", "maize", "beans", "rice", "rice"], "district": ["A", "A", "A", "B", "B"],
                        "block_id": ["1", "1", "1", "2", "2"]})
    mix = tune.mixed_land(toy).set_index("district")
    assert mix.loc["all", "blocks"] == 2 and mix.loc["all", "blocks_with_more_than_one_crop"] == 0.5
    assert mix.loc["all", "fields_in_those_blocks"] == pytest.approx(0.6)
    assert mix.loc["all", "block_most_common_crop_right"] == pytest.approx(0.8)


def test_shuffling_an_input_the_model_ignores_changes_nothing(df):
    class OnlyElevation:
        used_ = ["elevation_m", "ph_0_20cm"]
        classes_ = np.array(ev.CROPS)

        def predict_proba(self, X):
            high = X["elevation_m"].to_numpy() > 2000
            p = np.zeros((len(X), 4))
            p[high, 1], p[~high, 2] = 1.0, 1.0
            return p

    test = df.sample(300, random_state=0)
    out = tune.shuffled_preds(OnlyElevation(), test, 2, ("t",))
    base = OnlyElevation().predict_proba(test).argmax(axis=1)
    assert all((out[("input:ph_0_20cm", r)] == base).all() for r in range(2))
    assert any((out[("input:elevation_m", r)] != base).any() for r in range(2))
    assert ("group:soil 0 to 20 cm", 0) in out and ("group:elevation and field temperature", 0) in out
    again = tune.shuffled_preds(OnlyElevation(), test, 2, ("t",))
    assert all((again[k] == v).all() for k, v in out.items())


# ------------------------------------------------------------------ one small job, end to end
def test_a_small_job_end_to_end_with_the_builtin_check(df, tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "MAP_MODEL", "LR[field_sat]")      # the only model of this small run
    cands = tune.candidates("LR", 1)
    part = tune.run_job("LR", "field_sat", df, cands, n_shuffles=1, log=lambda *a: None)
    run = {"candidates": {"LR": cands}, "parts": {("LR", "field_sat"): part}, "jobs": [("LR", "field_sat")]}
    res = tune.assemble(df, run)
    assert res["check"]["passed"], res["check"]["mismatches"]       # protocol settings give the eval_v2 numbers
    assert res["check"]["rows_compared"] == 2 * 3 * 4               # B0, B1 and LR, 4 districts, both tests
    for test in tune.TESTS:
        r = res["results"][test]
        assert {"LR[field_sat]", "LR[field_sat] tuned", ev.BASELINE} <= set(r["model"])
        assert r.loc[r["model"] == "LR[field_sat] tuned", "n_test"].sum() == len(df)
    chosen = res["analyses"]["chosen"]
    assert len(chosen) == 4 + ev.N_BLOCK_FOLDS and set(chosen["candidate"]) <= {0, 1}
    assert len(res["analyses"]["inner"]) == (4 + ev.N_BLOCK_FOLDS) * len(cands)
    imp = res["analyses"]["importance"]
    lodo_groups = imp[(imp["test"] == "lodo") & (imp["kind"] == "group")]   # field_sat has no district inputs
    assert set(lodo_groups["name"]) == {"elevation and field temperature", "soil 0 to 20 cm",
                                        "satellite rainfall and vegetation"}
    info = {"candidates": run["candidates"], "methods": ["LR"], "views": ["field_sat"], "protocol_version": "1.2",
            "run_at_utc": "test", "compute_hours": 0.0, "workers": 1, "table": "t", "table_sha256": "0" * 64,
            "git_commit": None, "n_candidates": 2, "n_draws": 1, "inner_block_folds": 3, "n_shuffles": 1,
            "quick_check_only": True}
    text = tune.write_report(tmp_path, info, df, res).read_text(encoding="utf-8")
    assert "**Built-in check: passed.**" in text and "## When the model is sure (LR[field_sat]" in text
    for name in ("main_lodo_results.csv", "main_blockcv_results.csv", "confusion_counts.csv", "importance.csv",
                 "confidence_bands.csv", "answer_when_sure.csv", "mixed_land.csv", "chosen_settings.csv",
                 "inner_scores.csv", "candidates.json", "run_info.json"):
        assert (tmp_path / name).exists(), name
