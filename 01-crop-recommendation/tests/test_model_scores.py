"""Tests for model_scores.py, which recomputes every score shown in the dashboard from reports/eval_v1.

Run from the repo root:  python -m pytest 01-crop-recommendation/tests/test_model_scores.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-crop-recommendation" / "src" / "models"))
import model_scores as ms  # noqa: E402


@pytest.fixture(scope="module")
def scores():
    return ms.Scores()


@pytest.mark.parametrize("analysis", ["main", "used_only"])
def test_main_score_recomputed_from_confusion_counts_matches_the_results(analysis):
    s = ms.Scores(analysis=analysis)
    checked = 0
    for test in ms.TESTS:
        for row in s.results[test].itertuples():
            mine = s.protocol_main_score(test, row.model, row.district)
            if math.isnan(row.macro_f1):
                assert math.isnan(mine)
            else:
                assert abs(mine - row.macro_f1) < 1e-12, (analysis, test, row.model, row.district)
            checked += 1
    assert checked == 2 * 12 * 4


def test_every_field_is_counted_once_per_model_and_test(scores):
    for test in ms.TESTS:
        for m in scores.models:
            assert scores.matrix(test, m).values.sum() == 1904


def test_main_scores_table_and_adoption_decision(scores):
    main = scores.main_scores("new_district").set_index("model")
    assert len(main) == 12 and main.loc["B1_expert_rules", "Passes rule"] == ""
    assert (main.drop(index=["B0_majority", "B1_expert_rules"])["Passes rule"] == "no").all()
    fields = scores.main_scores("new_fields").set_index("model")
    assert fields["Main score"].idxmax() == "XGB[all]"
    assert round(fields.loc["XGB[all]", "Main score"], 3) == 0.451
    assert fields.loc["XGB[all]", "Beats B1 in"] == "2 of 4 districts"


def test_overall_scores_match_the_xgboost_report(scores):
    o = scores.overall("new_fields").set_index("model")
    assert round(o.loc["XGB[all]", "Accuracy"], 3) == 0.656
    assert round(o.loc["XGB[all]", "Cohen's kappa"], 3) == 0.437
    assert round(o.loc["B1_expert_rules", "Accuracy"], 3) == 0.388
    assert round(o.loc["RF[all]", "Accuracy"], 3) == 0.689      # pooled, Random Forest is slightly ahead
    for c in ["Accuracy", "Balanced accuracy", "Macro-F1", "Weighted F1"]:
        assert o[c].between(0, 1).all()


def test_per_crop_scores_in_a_district(scores):
    t = scores.per_crop("new_fields", "XGB[all]", ["Nyagatare"]).set_index("crop")
    assert t.loc["maize", "fields"] == 792 and round(t.loc["maize", "f1"], 2) == 0.80
    rules = scores.per_crop("new_fields", "B1_expert_rules", ["Nyagatare"]).set_index("crop")
    assert rules.loc["maize", "f1"] == 0.0                            # the rules never pick maize there
    musanze = scores.per_crop("new_fields", "XGB[all]", ["Musanze"]).set_index("crop")
    assert musanze.loc["rice", "fields"] == 1 and not musanze.loc["rice", "scored"]


def test_crop_table_hides_crops_with_too_few_fields(scores):
    t = scores.crop_table("new_fields", "Musanze")
    assert list(t.columns[:1]) == ["Fields"] and "XGBoost F1" in t.columns and "Rules F1" in t.columns
    assert np.isnan(t.loc["Rice", "XGBoost F1"]) and round(t.loc["Irish potato", "XGBoost F1"], 2) == 0.77


def test_second_check_on_used_fields():
    s = ms.Scores(analysis="used_only").main_scores("new_fields").set_index("model")
    assert round(s.loc["XGB[all]", "Main score"], 3) == 0.483 and s.loc["XGB[all]", "Beats B1 in"] == "3 of 4 districts"


def test_labels():
    assert ms.model_label("XGB[all]") == ("XGBoost", "all (21)")
    assert ms.short_label("RF[all_subsoil]") == "Random Forest, all + subsoil inputs"
    assert ms.short_label("B1_expert_rules") == "B1, expert rules"
