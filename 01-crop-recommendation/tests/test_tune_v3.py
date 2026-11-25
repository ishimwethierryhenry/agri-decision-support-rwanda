"""Tests for the third round of tuning (src/models/tune_v3.py, training protocol 1.4, section 13).

Run from the repo root:  python -m pytest 01-crop-recommendation/tests/test_tune_v3.py -q
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
import tune as r1  # noqa: E402
import tune_v2 as t2  # noqa: E402
import tune_v3 as t3  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return ev.prepare(pd.read_csv(t3.TABLE, low_memory=False))


def test_weights_start_with_no_change_and_cover_the_81_combinations():
    grid = t3.weight_grid()
    assert len(grid) == 81 and np.allclose(grid[0], 1.0)
    assert {round(x, 4) for w in grid for x in w} == {0.6667, 1.0, 1.5}


def test_a_tie_keeps_the_probabilities_as_they_are(df):
    one = np.zeros((len(df), len(t3.CROPS)))
    one[:, t3.CROPS.index("maize")] = 1.0                      # every weight gives the same answers
    w, score, none = t3.best_weights(df, one)
    assert np.allclose(w, 1.0) and score == none


def test_weights_can_only_be_chosen_on_the_training_part(df):
    rng = np.random.default_rng(0)
    p = rng.random((len(df), 4))
    w, score, none = t3.best_weights(df, p)
    assert score >= none and score == pytest.approx(r1.block_score(df, p * w))


def test_blend_groups_go_from_small_to_large_in_a_fixed_order():
    assert t3.GROUPS == [("RF",), ("ET",), ("XGB",), ("RF", "ET"), ("RF", "XGB"), ("ET", "XGB"), ("RF", "ET", "XGB")]


def test_candidates_are_those_of_version_1_3():
    saved = json.loads((t3.ROUND2_DIR / "candidates.json").read_text(encoding="utf-8"))
    assert all(saved[m] == t2.candidates(m) for m in t3.METHODS)


def test_one_task_of_the_main_test_gives_the_version_1_3_numbers(df):
    """The built-in check on one district: the version 1.3 choice fitted again gives the tuning_v2 score."""
    cands = {m: t2.candidates(m) for m in t3.METHODS}
    inner13 = pd.read_csv(t3.ROUND2_DIR / "inner_scores.csv")
    chosen13 = pd.read_csv(t3.ROUND2_DIR / "chosen_settings.csv")
    part = t3.run_task("field_sat", "lodo", "Musanze", df, cands, inner13, chosen13, log=lambda *a: None)
    res = t3.collect(df, [part])["results"]
    ref = pd.read_csv(t3.ROUND2_DIR / "main_lodo_results.csv").set_index(["model", "district"])["macro_f1"]
    for m in t3.METHODS:
        ours = res[(res.model == f"{m}[field_sat] v1.3") & (res.district == "Musanze")]["macro_f1"].iloc[0]
        assert ours == pytest.approx(ref[(f"{m}[field_sat] tuned", "Musanze")], abs=1e-9)
        assert part["members"][m]["candidate"] == part["members"][m]["candidate_v13"]   # no random split there
    assert tuple(part["blend"]["group"]) in t3.GROUPS
