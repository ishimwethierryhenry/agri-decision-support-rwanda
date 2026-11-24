"""Tests for the second round of tuning (src/models/tune_v2.py, training protocol 1.3, section 12).

They use the real feature table with tiny budgets, so they run in about a minute. The full run is
`python 01-crop-recommendation/src/models/tune_v2.py`.
Run from the repo root:  python -m pytest 01-crop-recommendation/tests/test_tune_v2.py -q
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
import tune_v2 as t2  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return ev.prepare(pd.read_csv(t2.TABLE, low_memory=False))


# ------------------------------------------------------------------ candidates and settings
def test_candidates_start_with_section_6_and_the_map_model_of_version_1_2():
    for m in t2.METHODS:
        c = t2.candidates(m)
        assert c[0] == dict(sorted(t2.PROTOCOL_SETTINGS[m].items()))
        assert c == t2.candidates(m)                                   # seed 0: the same candidates every time
        assert len(c) == t2.N_DRAWS + (2 if m == "RF" else 1)
    final = json.loads((t2.ROUND1_DIR / "final_model.json").read_text(encoding="utf-8"))
    assert t2.candidates("RF")[1] == dict(sorted({**final["settings"], "near_km": 4.0}.items()))
    assert final["candidate"] == 30 and tune.candidates("RF")[30] == final["settings"]


def test_candidates_stay_inside_the_ranges_of_section_12():
    spaces = t2.search_spaces()
    for m in t2.METHODS:
        for c in t2.candidates(m)[(2 if m == "RF" else 1):]:
            assert set(c) == set(spaces[m])
            for k, v in c.items():
                space = spaces[m][k]
                if isinstance(space, list):
                    assert v in space, (m, k, v)
                else:
                    lo, hi = space.support()
                    assert lo - 1e-6 <= v <= hi + 1e-6, (m, k, v)


def test_section_6_settings_give_exactly_the_models_of_version_1_2(df):
    """What the built-in check relies on: Random Forest and XGBoost with the section 6 settings are the same models."""
    train, test = df[df["district"] != "Ruhango"], df[df["district"] == "Ruhango"]
    for m in ("RF", "XGB"):
        a = t2.Tunable(m, t2.PROTOCOL_SETTINGS[m], "all_sat").fit(train, train["label"]).predict_proba(test)
        b = tune.Tunable(m, tune.PROTOCOL_SETTINGS[m], "all_sat").fit(train, train["label"]).predict_proba(test)
        assert np.allclose(a, b, rtol=0, atol=1e-9), m


def test_extra_trees_draw_fields_only_below_all(df):
    small = df.sample(300, random_state=0)
    for ms, boot in ((None, False), (0.7, True)):
        p = {**t2.PROTOCOL_SETTINGS["ET"], "max_samples": ms}
        m = t2.Tunable("ET", p, "field_sat").fit(small, small["label"])
        assert m.estimator.bootstrap is boot and m.estimator.max_samples == ms


# ------------------------------------------------------------------ nearby-crop inputs
def test_blocks_one_step_apart_are_about_2_km():
    d = t2.distance_km(t2.block_centres(["0_0"]), t2.block_centres(["1_0", "0_1", "3_4"]))[0]
    assert d[0] == pytest.approx(2.0, abs=0.01) and d[1] == pytest.approx(2.0, abs=0.01)
    assert d[2] == pytest.approx(10.0, abs=0.05)


def test_nearby_inputs_never_count_the_fields_own_block():
    blocks = ["0_0", "0_0", "0_1", "0_1", "0_1", "9_9"]
    labels = ["maize", "maize", "beans", "beans", "beans", "rice"]
    near = t2.NearCrops(4.0).fit(blocks, labels)
    out = near.transform(["0_0", "0_1"])
    assert out.loc[0, "near_share_beans"] == 1.0 and out.loc[0, "near_share_maize"] == 0.0   # only block 0_1 counts
    assert out.loc[1, "near_share_maize"] == 1.0 and out.loc[1, "near_share_beans"] == 0.0   # only block 0_0 counts
    assert out.loc[0, "near_share_rice"] == 0.0           # block 9_9 is 25 km away, beyond 3h = 12 km
    assert out.loc[0, "near_weight"] == pytest.approx(3 * np.exp(-0.5 * (2.0 / 4.0) ** 2), rel=1e-3)
    # The crops of a field's own block never change its inputs
    other = t2.NearCrops(4.0).fit(blocks, ["rice", "rice"] + labels[2:]).transform(["0_0"])
    assert np.allclose(other.to_numpy(), out.iloc[[0]].to_numpy())


def test_nearby_inputs_with_nothing_around_use_all_training_fields():
    near = t2.NearCrops(2.0).fit(["0_0", "0_0", "0_1", "0_2"], ["maize", "beans", "maize", "rice"])
    out = near.transform(["50_50"])
    assert out.loc[0, "near_weight"] == 0.0
    assert out.loc[0, "near_share_maize"] == 0.5 and out.loc[0, "near_share_beans"] == 0.25
    assert np.isclose(out[t2.NEAR_COLS[:-1]].sum(axis=1), 1.0).all()


def test_a_near_model_learns_its_inputs_from_other_blocks_only(df):
    """Changing the crops of one training block must not change the nearby inputs that block's fields trained with."""
    train = df[df["district"] == "Musanze"].reset_index(drop=True)
    block = train[ev.BLOCK_COL].value_counts().index[0]
    p = {**t2.PROTOCOL_SETTINGS["ET"], "near_km": 4.0}
    a = t2.Tunable("ET", p, "field_sat_near").fit(train, train["label"])
    flipped = train["label"].where(train[ev.BLOCK_COL] != block, "rice")
    b = t2.Tunable("ET", p, "field_sat_near").fit(train, flipped)
    rows = train[ev.BLOCK_COL] == block
    xa, xb = a._inputs(train[rows]), b._inputs(train[rows])
    assert list(xa.columns[-5:]) == t2.NEAR_COLS
    assert np.allclose(xa[t2.NEAR_COLS].to_numpy(), xb[t2.NEAR_COLS].to_numpy())
    others = ~rows & (train[ev.BLOCK_COL] != block)
    assert not np.allclose(a._inputs(train[others])[t2.NEAR_COLS].to_numpy(),
                           b._inputs(train[others])[t2.NEAR_COLS].to_numpy())   # but its neighbours do see them


# ------------------------------------------------------------------ blend
def test_the_blend_is_the_mean_of_its_members(df):
    rng = np.random.default_rng(0)
    parts = {}
    for m in t2.METHODS:
        p = rng.random((len(df), 4))
        parts[(m, "field_sat_near")] = {"blockcv_proba": {"tuned": p / p.sum(axis=1, keepdims=True)}, "lodo_proba": None}
    b = t2.blend(df, parts, "field_sat_near")
    mean = np.mean([parts[(m, "field_sat_near")]["blockcv_proba"]["tuned"] for m in t2.METHODS], axis=0)
    assert np.allclose(b["proba"]["blockcv"], mean) and "lodo" not in b["proba"]
    assert {r["test"] for r in b["rows"]} == {"blockcv"} and len(b["rows"]) == 4
    del parts[("ET", "field_sat_near")]
    assert t2.blend(df, parts, "field_sat_near") is None                  # never a blend with a member missing


# ------------------------------------------------------------------ one small run, end to end
def test_a_small_run_end_to_end(df, tmp_path, monkeypatch):
    monkeypatch.setattr(t2, "METHODS", ("ET",))                           # one method keeps the test short
    cands = t2.candidates("ET", 1)
    parts = {("ET", v): t2.run_job("ET", v, df, cands, n_shuffles=1, log=lambda *a: None)
             for v in ("field_sat", "field_sat_near")}
    assert parts[("ET", "field_sat_near")]["lodo_proba"] is None          # nearby views: new fields only
    run = {"candidates": {"ET": cands}, "parts": parts, "jobs": list(parts)}
    res = t2.assemble(df, run)
    assert res["check"]["passed"], res["check"]["mismatches"]            # baselines give the eval_v2 numbers
    assert res["check"]["rows_compared"] == 2 * 4 * 2                    # B0 and B1, 4 districts, both tests
    blk, lodo = res["results"]["blockcv"], res["results"]["lodo"]
    assert {"ET[field_sat_near] tuned", "Blend[field_sat_near] tuned", t2.NEAR_REF} <= set(blk["model"])
    assert "ET[field_sat_near] tuned" not in set(lodo["model"]) and "Blend[field_sat] tuned" in set(lodo["model"])
    assert blk.loc[blk["model"] == "ET[field_sat_near] tuned", "n_test"].sum() == len(df)
    imp = res["analyses"]["importance"]
    assert t2.NEAR_GROUP in set(imp.loc[imp["model"] == "ET[field_sat_near] tuned", "name"])
    dec = res["decisions"]
    assert dec["map"]["today"] == "RF[all_sat] tuned" and dec["map"]["today_score"] == pytest.approx(0.514, abs=5e-4)
    info = {"candidates": run["candidates"], "methods": ["ET"], "views": ["field_sat", "field_sat_near"],
            "protocol_version": "1.3", "run_at_utc": "test", "compute_hours": 0.0, "workers": 1, "table": "t",
            "table_sha256": "0" * 64, "git_commit": None, "n_draws": 1, "inner_block_folds": 3, "n_shuffles": 1,
            "quick_check_only": True}
    text = t2.write_report(tmp_path, info, df, res).read_text(encoding="utf-8")
    assert "**Built-in check: passed.**" in text and "## What the nearby-crop inputs add (new fields)" in text
    assert "nearest surveyed fields (h = 4 km)" in text
    for name in ("main_lodo_results.csv", "main_blockcv_results.csv", "confusion_counts.csv", "importance.csv",
                 "confidence_bands.csv", "answer_when_sure.csv", "mixed_land.csv", "chosen_settings.csv",
                 "inner_scores.csv", "candidates.json", "run_info.json", "report.md"):
        assert (tmp_path / name).exists(), name

    # --final: the map stays when nothing beats it, and a blend gets the settings of each member
    (tmp_path / "run_info.json").write_text(json.dumps({"n_draws": 1, "decisions": {"map": {
        "serve": "RF[all_sat] tuned", "changes": False, "beating": {}}}}), encoding="utf-8")
    assert "keeps the model of version 1.2" in t2.final_model(df, tmp_path, log=lambda *a: None)["note"]
    (tmp_path / "run_info.json").write_text(json.dumps({"n_draws": 1, "decisions": {"map": {
        "serve": "Blend[field_sat_near] tuned", "changes": True,
        "beating": {"Blend[field_sat_near] tuned": 0.6}}}}), encoding="utf-8")
    final = t2.final_model(df, tmp_path, log=lambda *a: None)
    assert set(final["members"]) == {"ET"} and final["features"][-5:] == t2.NEAR_COLS
    assert len(final["members"]["ET"]["comparison_scores"]) == len(cands)
