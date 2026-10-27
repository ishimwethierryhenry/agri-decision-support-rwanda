"""Contract tests for the Recommendation API (draft 0.2.0).

Run from the repository root:  python -m pytest 03-shared/tests -q
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apis"))

import recommend_examples as ex  # noqa: E402
from recommend_schemas import (ErrorEnvelope, FertiliserAdvice, RecommendRequest,  # noqa: E402
                               RecommendResponse)


# ------------------------------------------------------------ examples parse
def test_example_requests_are_valid():
    RecommendRequest(**ex.REQUEST_DISTRICT)
    RecommendRequest(**ex.REQUEST_POINT_FERTILISER_ONLY)


def test_example_responses_are_valid():
    RecommendResponse(**ex.RESPONSE_OK)
    RecommendResponse(**ex.RESPONSE_ABSTAIN)
    ErrorEnvelope(**ex.ERROR_VALIDATION)


# ---------------------------------------------------------- request checks
def test_needs_a_location():
    with pytest.raises(ValidationError):
        RecommendRequest(season="B")


def test_lat_without_lon_rejected():
    with pytest.raises(ValidationError):
        RecommendRequest(season="B", lat=-1.5)


@pytest.mark.parametrize("lat,lon", [(10.0, 29.6), (-1.5, 40.0), (-5.0, 29.6)])
def test_outside_rwanda_rejected(lat, lon):
    with pytest.raises(ValidationError):
        RecommendRequest(season="B", lat=lat, lon=lon)


@pytest.mark.parametrize("soil", [{"ph_water": 11}, {"ph_water": 2.5}, {"organic_carbon_pct": -1},
                                  {"organic_carbon_pct": 25}, {"texture": "mud"}])
def test_bad_soil_values_rejected(soil):
    with pytest.raises(ValidationError):
        RecommendRequest(district="Musanze", season="B", soil=soil)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        RecommendRequest(district="Musanze", season="B", fertilizer="urea")


def test_season_must_be_a_b_or_c():
    with pytest.raises(ValidationError):
        RecommendRequest(district="Musanze", season="D")


def test_top_k_limits():
    for k in (0, 6):
        with pytest.raises(ValidationError):
            RecommendRequest(district="Musanze", season="B", top_k=k)


def test_defaults():
    r = RecommendRequest(district="Musanze", season="B")
    assert r.top_k == 3 and r.include == ["crops", "fertiliser"]


# --------------------------------------------------------- response checks
def test_abstain_needs_message_and_no_crops():
    bad = dict(ex.RESPONSE_ABSTAIN, abstain_message=None)
    with pytest.raises(ValidationError):
        RecommendResponse(**bad)
    bad2 = dict(ex.RESPONSE_OK, abstain=True, abstain_message="Uncertain.")
    with pytest.raises(ValidationError):
        RecommendResponse(**bad2)


def test_ranks_must_be_in_order():
    bad = dict(ex.RESPONSE_OK)
    crops = [dict(c) for c in bad["crop_recommendations"]]
    crops[0]["rank"], crops[1]["rank"] = 2, 1
    bad["crop_recommendations"] = crops
    with pytest.raises(ValidationError):
        RecommendResponse(**bad)


def test_fertiliser_item_must_cite_a_source():
    item = dict(ex.RESPONSE_OK["fertiliser"]["items"][0])
    item.pop("source")
    with pytest.raises(ValidationError):
        FertiliserAdvice(crop="maize", status="ok", engine_version="x", items=[item])


def test_fertiliser_status_matches_items():
    with pytest.raises(ValidationError):
        FertiliserAdvice(crop="maize", status="ok", engine_version="x", items=[])
    with pytest.raises(ValidationError):
        FertiliserAdvice(crop="maize", status="no_matching_rule", engine_version="x",
                         items=ex.RESPONSE_OK["fertiliser"]["items"])


def test_score_range():
    bad = dict(ex.RESPONSE_OK)
    crops = [dict(c) for c in bad["crop_recommendations"]]
    crops[0]["score"] = 1.4
    bad["crop_recommendations"] = crops
    with pytest.raises(ValidationError):
        RecommendResponse(**bad)


# -------------------------------------------------------------- stub server
def test_stub_server_follows_contract():
    from fastapi.testclient import TestClient
    from stub_app import app
    c = TestClient(app)
    assert c.get("/v1/health").status_code == 200
    assert c.get("/v1/metadata").json()["seasons"] == ["A", "B", "C"]

    ok = c.post("/v1/recommend", json=ex.REQUEST_DISTRICT)
    assert ok.status_code == 200
    RecommendResponse(**ok.json())
    assert any(w["code"] == "STUB_RESPONSE" for w in ok.json()["warnings"])

    only_fert = c.post("/v1/recommend", json=ex.REQUEST_POINT_FERTILISER_ONLY).json()
    assert only_fert["crop_recommendations"] == [] and only_fert["fertiliser"] is not None

    bad = c.post("/v1/recommend", json={"district": "Musanze", "season": "B", "soil": {"ph_water": 11}})
    assert bad.status_code == 422
    env = ErrorEnvelope(**bad.json())
    assert env.error.code == "VALIDATION_ERROR"

    oob = c.post("/v1/recommend", json={"lat": 10, "lon": 29.6, "season": "B"})
    assert ErrorEnvelope(**oob.json()).error.code == "OUT_OF_BOUNDS"


# ------------------------------------------------------------ added in 0.2.0
@pytest.mark.parametrize("elev,ok", [(950, True), (2300, True), (4507, True), (500, False), (5000, False)])
def test_elevation_is_optional_and_bounded(elev, ok):
    if ok:
        assert RecommendRequest(district="Musanze", season="B", elevation_m=elev).elevation_m == elev
    else:
        with pytest.raises(ValidationError):
            RecommendRequest(district="Musanze", season="B", elevation_m=elev)
    assert RecommendRequest(district="Musanze", season="B").elevation_m is None


def test_site_inputs_are_optional_and_carry_a_source():
    body = dict(ex.RESPONSE_OK)
    body["inputs_used"] = {**ex.RESPONSE_OK["inputs_used"],
                           "site": {"elevation_m": {"value": 2107.2, "source": "district_median_ciat"}}}
    assert RecommendResponse(**body).inputs_used.site["elevation_m"].source == "district_median_ciat"
    bad = dict(body)
    bad["inputs_used"] = {**ex.RESPONSE_OK["inputs_used"], "site": {"elevation_m": {"value": 2107.2}}}
    with pytest.raises(ValidationError):
        RecommendResponse(**bad)
