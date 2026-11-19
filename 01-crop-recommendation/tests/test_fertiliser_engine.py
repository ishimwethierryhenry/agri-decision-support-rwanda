"""Tests for the fertiliser engine. The rules below are TEST FIXTURES, not agronomic advice, except the checks on
the shipped rule file (interim national rates copied from published sources)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-crop-recommendation" / "src" / "recommendation"))
sys.path.insert(0, str(ROOT / "03-shared" / "apis"))

from fertiliser_engine import (INTERIM_NOTE, RuleConflict, RuleError, interim_rules_used, load_rules,  # noqa: E402
                               parse_rules, recommend)
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


SHIPPED = ROOT / "01-crop-recommendation" / "src" / "recommendation" / "rules" / "fertiliser_rules.yaml"
LINK = "https://example.org/published-guideline.pdf"


def test_shipped_rule_file_has_one_interim_rule_per_crop():
    loaded = load_rules(SHIPPED)
    assert sorted(r.crop for r in loaded.rules) == ["beans", "irish_potato", "maize", "rice"]
    for r in loaded.rules:
        assert r.status == "interim" and not r.when                  # the same for every field
        assert r.source["url"].startswith("https://") and r.source.get("page")
        out = recommend(loaded, r.crop, {"ph_water": 4.5})
        assert out["status"] == "ok" and interim_rules_used(loaded, out) == [r.id]
        FertiliserAdvice(**out)
        for item in out["items"]:
            assert item["explanation"].startswith(INTERIM_NOTE) and item["product"] and item["rate_unit"] == "kg/ha"
        assert recommend(loaded, r.crop, allow_interim=False)["status"] == "no_matching_rule"


def test_shipped_interim_rates_are_the_published_ones():
    """Guards against a rate being changed without its source (see the source of each rule in the YAML)."""
    got = {(r.crop, a["product"]): a["rate"] for r in load_rules(SHIPPED).rules for a in r.advice}
    assert got == {("maize", "DAP (18-46-0)"): 100, ("maize", "Urea (46% N)"): 100,     # Mudereri et al. 2025, p. 7
                   ("beans", "DAP (18-46-0)"): 100,                                      # 18 N + 46 P2O5, OFRA 2017
                   ("irish_potato", "NPK 17-17-17"): 300,                                # IFDC 2018, Table 2
                   ("rice", "NPK 17-17-17"): 200, ("rice", "Urea (46% N)"): 100}         # Kayumba et al. 2023


def test_interim_rule_needs_page_and_link():
    r = rule(status="interim")                     # the fixture source has a page but no link
    with pytest.raises(RuleError):
        rs(r)
    r["source"]["url"] = LINK
    rs(r)
    r["source"] = {"document": "Some real title", "url": LINK}
    with pytest.raises(RuleError):
        rs(r)


def test_interim_rules_are_only_a_fallback_and_always_labelled():
    interim = rule(id="interim", status="interim", source={**SRC, "url": LINK},
                   advice=[{"input_type": "nitrogen", "rate": 50, "rate_unit": "kg/ha", "explanation": "From X."},
                           {"input_type": "lime", "rate": 1.0, "rate_unit": "t/ha"}])
    verified = rule(id="verified", when={"ph_water": {"lt": 5.5}},
                    advice=[{"input_type": "lime", "rate": 2.0, "rate_unit": "t/ha"}])
    r = rs(interim, verified)
    out = recommend(r, "maize", {"ph_water": 5.0})            # a verified rule fits: no interim item at all
    assert [i["condition_matched"].split(":")[0] for i in out["items"]] == ["verified"]
    assert interim_rules_used(r, out) == []
    out = recommend(r, "maize", {"ph_water": 6.0})            # no verified rule fits: the interim rule answers
    assert interim_rules_used(r, out) == ["interim"]
    assert [i["explanation"] for i in out["items"]] == [INTERIM_NOTE, f"{INTERIM_NOTE} From X."]
    assert recommend(r, "maize", {"ph_water": 6.0}, allow_interim=False)["status"] == "no_matching_rule"


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
