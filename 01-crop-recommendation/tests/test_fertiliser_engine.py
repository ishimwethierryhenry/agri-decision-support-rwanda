"""Tests for the fertiliser engine. The rules below are TEST FIXTURES, not agronomic advice."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-crop-recommendation" / "src" / "recommendation"))
sys.path.insert(0, str(ROOT / "03-shared" / "apis"))

from fertiliser_engine import RuleConflict, RuleError, load_rules, parse_rules, recommend  # noqa: E402
from recommend_schemas import FertiliserAdvice  # noqa: E402

SRC = {"document": "TEST FIXTURE ONLY, not a real guideline", "section": "0", "page": "0"}


def rs(*rules, version="test"):
    return parse_rules({"version": version, "rules": list(rules)})


def rule(id="r1", crop="maize", when=None, advice=None, status="verified", **extra):
    base = {
        "id": id, "crop": crop, "status": status, "source": dict(SRC),
        "when": when or {},
        "advice": [{"input_type": "lime", "rate": 1.0, "rate_unit": "t/ha"}] if advice is None else advice,
    }
    base.update(extra)
    return base


def test_shipped_rule_file_loads_and_is_empty():
    path = ROOT / "01-crop-recommendation" / "src" / "recommendation" / "rules" / "fertiliser_rules.yaml"
    loaded = load_rules(path)
    assert loaded.rules == []
    out = recommend(loaded, "maize", {"ph_water": 4.5})
    assert out["status"] == "no_matching_rule" and out["items"] == []


def test_match_and_output_fits_the_api_contract():
    out = recommend(rs(rule(when={"ph_water": {"lt": 5.5}})), "maize", {"ph_water": 5.0})
    assert out["status"] == "ok"
    advice = FertiliserAdvice(**out)  # raises if the shape drifts from the contract
    assert advice.items[0].source.document.startswith("TEST FIXTURE")
    assert "r1" in advice.items[0].condition_matched


def test_no_match_returns_no_matching_rule():
    out = recommend(rs(rule(when={"ph_water": {"lt": 5.5}})), "maize", {"ph_water": 6.5})
    assert out["status"] == "no_matching_rule" and out["items"] == []
    FertiliserAdvice(**out)


def test_missing_soil_value_means_no_match():
    out = recommend(rs(rule(when={"ph_water": {"lt": 5.5}})), "maize", {})
    assert out["status"] == "no_matching_rule"


def test_none_values_are_treated_as_missing():
    out = recommend(rs(rule(when={"ph_water": {"lt": 5.5}})), "maize", {"ph_water": None})
    assert out["status"] == "no_matching_rule"


def test_crop_must_match_and_is_case_insensitive():
    r = rs(rule(crop="maize"))
    assert recommend(r, "Maize")["status"] == "ok"
    assert recommend(r, "beans")["status"] == "no_matching_rule"


def test_rule_without_condition_applies_in_general():
    assert recommend(rs(rule()), "maize", {})["status"] == "ok"


def test_draft_rules_are_not_served_by_default():
    r = rs(rule(status="draft"))
    assert recommend(r, "maize")["status"] == "no_matching_rule"
    assert recommend(r, "maize", allow_draft=True)["status"] == "ok"


@pytest.mark.parametrize("op,val,x,expected", [
    ("lt", 5.5, 5.4, True), ("lt", 5.5, 5.5, False),
    ("lte", 5.5, 5.5, True), ("gt", 5.5, 5.6, True), ("gt", 5.5, 5.5, False),
    ("gte", 5.5, 5.5, True), ("between", [5.0, 6.0], 5.0, True),
    ("between", [5.0, 6.0], 6.0, True), ("between", [5.0, 6.0], 6.1, False),
])
def test_operators(op, val, x, expected):
    out = recommend(rs(rule(when={"ph_water": {op: val}})), "maize", {"ph_water": x})
    assert (out["status"] == "ok") is expected


def test_texture_eq_and_in():
    r = rs(rule(when={"texture": {"in": ["clay", "clay_loam"]}}))
    assert recommend(r, "maize", {"texture": "clay"})["status"] == "ok"
    assert recommend(r, "maize", {"texture": "sand"})["status"] == "no_matching_rule"


def test_more_specific_rule_wins_for_same_input_type():
    general = rule(id="general", advice=[{"input_type": "lime", "rate": 1.0, "rate_unit": "t/ha"}])
    specific = rule(id="specific", when={"ph_water": {"lt": 5.0}},
                    advice=[{"input_type": "lime", "rate": 2.0, "rate_unit": "t/ha"}])
    out = recommend(rs(general, specific), "maize", {"ph_water": 4.5})
    assert len(out["items"]) == 1 and out["items"][0]["rate"] == 2.0


def test_priority_beats_specificity():
    a = rule(id="a", when={"ph_water": {"lt": 5.0}}, priority=0)
    b = rule(id="b", priority=5, advice=[{"input_type": "lime", "rate": 9.0, "rate_unit": "t/ha"}])
    out = recommend(rs(a, b), "maize", {"ph_water": 4.5})
    assert out["items"][0]["rate"] == 9.0


def test_equal_rules_conflict_loudly():
    a = rule(id="a", when={"ph_water": {"lt": 5.0}})
    b = rule(id="b", when={"organic_carbon_pct": {"lt": 2.0}})
    with pytest.raises(RuleConflict):
        recommend(rs(a, b), "maize", {"ph_water": 4.5, "organic_carbon_pct": 1.0})


def test_different_input_types_are_all_returned():
    r = rs(rule(id="a", advice=[{"input_type": "lime", "rate": 1.0, "rate_unit": "t/ha"}]),
           rule(id="b", advice=[{"input_type": "nitrogen", "rate": 50, "rate_unit": "kg/ha"}]))
    types = [i["input_type"] for i in recommend(r, "maize")["items"]]
    assert types == ["lime", "nitrogen"]


@pytest.mark.parametrize("bad_source", [
    {}, {"document": ""}, {"document": "TODO"}, {"document": "<guideline title>"},
    {"document": "TBD later"}, {"document": "placeholder"},
])
def test_rule_without_a_real_source_is_rejected(bad_source):
    r = rule()
    r["source"] = bad_source
    with pytest.raises(RuleError):
        rs(r)


def test_missing_source_key_is_rejected():
    r = rule()
    del r["source"]
    with pytest.raises(RuleError):
        rs(r)


def test_verified_rule_needs_page_or_section():
    r = rule()
    r["source"] = {"document": "Some real title"}
    with pytest.raises(RuleError):
        rs(r)
    r["status"] = "draft"
    rs(r)  # a draft may lack the page


@pytest.mark.parametrize("when", [
    {"soil_colour": {"eq": "red"}}, {"ph_water": {"near": 5}}, {"ph_water": 5.0},
    {"ph_water": {"between": [6, 5]}}, {"texture": {"lt": 3}}, {"texture": {"in": []}},
])
def test_bad_conditions_are_rejected(when):
    with pytest.raises(RuleError):
        rs(rule(when=when))


@pytest.mark.parametrize("advice", [
    [], [{"rate": 1, "rate_unit": "t/ha"}],
    [{"input_type": "lime", "rate": 1}], [{"input_type": "lime", "rate": 1, "rate_unit": "bags"}],
    [{"input_type": "lime", "rate": -1, "rate_unit": "t/ha"}],
])
def test_bad_advice_is_rejected(advice):
    with pytest.raises(RuleError):
        rs(rule(advice=advice))


def test_duplicate_ids_and_missing_version_are_rejected():
    with pytest.raises(RuleError):
        rs(rule(id="x"), rule(id="x"))
    with pytest.raises(RuleError):
        parse_rules({"rules": []})


def test_engine_version_includes_rule_set_version():
    out = recommend(rs(rule(), version="2026-10-a"), "maize")
    assert out["engine_version"].endswith("rules-2026-10-a")
