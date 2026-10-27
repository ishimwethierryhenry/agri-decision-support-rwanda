"""Tests for the recommendation service (B1 baseline behind POST /v1/recommend).

They use only district_profiles.json and the rule files, so no raw data is needed.
Run from the repo root:  python -m pytest 01-crop-recommendation/tests -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-crop-recommendation" / "src" / "recommendation"))
sys.path.insert(0, str(ROOT / "03-shared" / "apis"))

from fastapi.testclient import TestClient  # noqa: E402

import service  # noqa: E402
from recommend_schemas import ErrorEnvelope, MetadataResponse, RecommendResponse  # noqa: E402

client = TestClient(service.app)


def post(body):
    return client.post("/v1/recommend", json=body)


def ok(body):
    r = post(body)
    assert r.status_code == 200, r.text
    return RecommendResponse(**r.json()), r


def codes(resp):
    return [w.code for w in resp.warnings]


def test_health_says_rules_only():
    j = client.get("/v1/health").json()
    assert j["status"] == "degraded" and j["rules_loaded"] and not j["model_loaded"]


def test_metadata_lists_the_pilot_districts_and_supported_crops():
    m = MetadataResponse(**client.get("/v1/metadata").json())
    assert {d.name for d in m.districts} == {"Musanze", "Nyabihu", "Nyagatare", "Ruhango"}
    assert m.crops == ["beans", "irish_potato", "maize", "rice"]
    assert all(d.in_training_data for d in m.districts)


def test_district_request_returns_ranked_crops_with_reasons_and_sources():
    resp, r = ok({"district": "Musanze", "season": "B"})
    assert r.headers["X-Request-ID"] == resp.request_id
    assert resp.model_version is None and resp.rules_version.startswith("crops-")
    assert [c.rank for c in resp.crop_recommendations] == [1, 2, 3]
    for c in resp.crop_recommendations:
        assert c.method == "expert_rules" and 0 <= c.score <= 1
        assert c.reliability.level.value in {"medium", "low"}          # never high without ML
        assert "ML_NOT_ADOPTED" in c.reliability.reason_codes
        assert any("[" in reason.text for reason in c.reasons)          # every check cites a source
    assert "ELEVATION_FROM_DISTRICT_MEDIAN" in codes(resp)
    assert resp.inputs_used.site["elevation_m"].source == "district_median_ciat"


def test_district_names_are_case_insensitive():
    resp, _ = ok({"district": "musanze", "season": "B"})
    assert resp.location.district == "Musanze" and resp.location.province == "North"


def test_elevation_changes_the_answer():
    low, _ = ok({"district": "Musanze", "season": "B", "elevation_m": 1700})
    mid, _ = ok({"district": "Musanze", "season": "B"})
    assert low.crop_recommendations[0].crop == "beans"
    assert mid.crop_recommendations[0].crop == "irish_potato"
    assert low.inputs_used.site["elevation_m"].source == "user"
    t_low = low.inputs_used.site["tmean_point_c"].value
    t_mid = mid.inputs_used.site["tmean_point_c"].value
    assert t_low > t_mid                                                 # lower field, warmer air


def test_rice_is_always_low_reliability_and_says_why():
    resp, _ = ok({"district": "Nyagatare", "season": "B", "top_k": 4})
    rice = [c for c in resp.crop_recommendations if c.crop == "rice"]
    assert rice and rice[0].reliability.level.value == "low"
    assert "CROP_RULES_WEAK" in rice[0].reliability.reason_codes
    assert any(r.factor == "water_access" for r in rice[0].reasons)


def test_top_k_and_include_are_respected():
    resp, _ = ok({"district": "Nyagatare", "season": "B", "top_k": 1, "include": ["crops"]})
    assert len(resp.crop_recommendations) == 1 and resp.fertiliser is None
    resp, _ = ok({"district": "Nyagatare", "season": "B", "include": ["fertiliser"]})
    assert resp.crop_recommendations == [] and resp.fertiliser is not None


def test_fertiliser_is_for_rank_1_or_the_named_crop_and_never_invented():
    resp, _ = ok({"district": "Musanze", "season": "B"})
    assert resp.fertiliser.crop == resp.crop_recommendations[0].crop
    assert resp.fertiliser.status == "no_matching_rule" and resp.fertiliser.items == []
    assert "NO_GUIDELINE_RULE_FOUND" in codes(resp)
    resp, _ = ok({"district": "Musanze", "season": "B", "crop": "maize"})
    assert resp.fertiliser.crop == "maize"


def test_soil_values_are_recorded_with_their_source():
    resp, _ = ok({"district": "Musanze", "season": "B", "soil": {"ph_water": 5.1, "organic_carbon_pct": 2.0}})
    assert resp.inputs_used.soil.values["ph_water"].source == "user"
    assert resp.inputs_used.soil.missing == ["total_nitrogen_pct"]
    assert "SOIL_NOT_USED_BY_METHOD" in codes(resp)


def test_seasons_other_than_b_are_low_reliability():
    resp, _ = ok({"district": "Nyabihu", "season": "A"})
    assert "SEASON_NOT_EVALUATED" in codes(resp)
    assert all(c.reliability.level.value == "low" for c in resp.crop_recommendations)


def test_point_near_a_district_is_matched_to_it():
    resp, _ = ok({"lat": -1.50, "lon": 29.60, "season": "B"})
    assert resp.location.resolution == "point" and resp.location.district == "Musanze"
    assert "LOCATION_MATCHED_TO_NEAREST_DISTRICT" in codes(resp)


def test_point_far_from_pilot_districts_is_low_reliability_without_invented_climate():
    resp, _ = ok({"lat": -2.6, "lon": 29.0, "season": "B", "elevation_m": 1600})
    assert resp.location.district is None and "DISTRICT_NOT_IN_TRAINING_DATA" in codes(resp)
    assert "precip_mm" not in resp.inputs_used.site
    for c in resp.crop_recommendations:
        assert c.reliability.level.value == "low" and "DISTRICT_NOT_IN_TRAINING" in c.reliability.reason_codes


def test_point_far_away_without_elevation_abstains():
    resp, _ = ok({"lat": -2.6, "lon": 29.0, "season": "B"})
    assert resp.abstain and resp.abstain_message and resp.crop_recommendations == []
    assert "NO_CROP_ABOVE_MIN_SCORE" in codes(resp)


def test_coordinates_win_over_a_different_district_name():
    resp, _ = ok({"district": "Nyagatare", "lat": -1.50, "lon": 29.60, "season": "B"})
    assert resp.location.district == "Musanze" and "DISTRICT_MISMATCH" in codes(resp)


def test_abstains_when_no_crop_reaches_the_minimum_score(monkeypatch):
    monkeypatch.setattr(service, "MIN_SCORE", 1.01)
    resp, _ = ok({"district": "Musanze", "season": "B"})
    assert resp.abstain and resp.crop_recommendations == []
    assert resp.fertiliser is None and "FERTILISER_SKIPPED" in codes(resp)
    resp, _ = ok({"district": "Musanze", "season": "B", "crop": "beans"})
    assert resp.abstain and resp.fertiliser.crop == "beans"


@pytest.mark.parametrize("body,status,code", [
    ({"district": "Kigali", "season": "B"}, 422, "UNKNOWN_DISTRICT"),
    ({"district": "Musanze", "season": "B", "crop": "cassava"}, 422, "UNSUPPORTED_CROP"),
    ({"season": "B"}, 422, "VALIDATION_ERROR"),
    ({"lat": -5.0, "lon": 29.6, "season": "B"}, 422, "OUT_OF_BOUNDS"),
    ({"district": "Musanze", "season": "B", "elevation_m": 200}, 422, "VALIDATION_ERROR"),
    ({"district": "Musanze", "season": "B", "colour": "red"}, 422, "VALIDATION_ERROR"),
])
def test_errors_use_the_contract_envelope(body, status, code):
    r = post(body)
    assert r.status_code == status
    env = ErrorEnvelope(**r.json())
    assert env.error.code == code and env.error.request_id == r.headers["X-Request-ID"]


def test_invalid_json_is_400():
    r = client.post("/v1/recommend", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and ErrorEnvelope(**r.json()).error.code == "INVALID_JSON"


def test_service_and_harness_rank_the_same_way():
    """The service must use the very same B1 ranking as the evaluation."""
    import expert_baseline as b1
    row = {"elevation_m": 2300.0, "precip_mm": 700.0, "tmean_point_c": 17.0, "tmin_point_c": 13.0}
    prior = service.PROFILES["crop_prior"]
    ranked = [i["crop"] for i in b1.rank_crops(row, service.CROP_RULES, prior)]
    model = b1.ExpertRulesBaseline(service.CROP_RULES)
    import numpy as np
    import pandas as pd
    model.classes_ = np.array(sorted(prior))
    model.prior_ = np.array([prior[c] for c in model.classes_])
    proba = model.predict_proba(pd.DataFrame([row]))[0]
    assert [model.classes_[i] for i in np.argsort(-proba)] == ranked


def test_district_profiles_hold_no_farm_level_data():
    text = (Path(service.PROFILES_PATH)).read_text()
    for forbidden in ("Village", "F_uuid", "GPS_Lat", "Sector"):
        assert forbidden not in text
