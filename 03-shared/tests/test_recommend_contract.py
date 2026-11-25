"""Contract tests for the Recommendation API (version 1.0).

Run from the repository root:  python -m pytest 03-shared/tests -q
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apis"))

import recommend_examples as ex  # noqa: E402
from recommend_schemas import (ErrorEnvelope, FertiliserAdvice, PointRequest, PointResponse,  # noqa: E402
                               RecommendRequest, RecommendResponse)


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


# ------------------------------------------------------------ added in 1.0: one answer for a point
def test_point_examples_are_valid():
    PointRequest(**ex.REQUEST_POINT)
    ok = PointResponse(**ex.RESPONSE_POINT_OK)
    assert not ok.abstain and ok.recommendation.crop == ok.fertiliser.crop
    assert ok.recommendation.confidence.level.value in {"high", "medium", "low"}
    out = PointResponse(**ex.RESPONSE_POINT_OUTSIDE)
    assert out.abstain and out.recommendation is None and out.abstain_message
    # Added in 1.3: two crops that fit about equally, each with its fertiliser, and the confidence of such answers
    PointRequest(**ex.REQUEST_POINT_TWO_CROPS)
    two = PointResponse(**ex.RESPONSE_POINT_TWO_CROPS)
    r = two.recommendation
    assert r.kind == "two_crops" and r.crops == [r.crop, r.close_second] and r.confidence.basis == "first_two_crops"
    assert two.fertiliser.crop == r.crop and two.fertiliser_second.crop == r.close_second
    assert ok.recommendation.kind == "one_crop" and ok.recommendation.confidence.tested_fields >= 30


def test_two_crop_answers_and_their_second_fertiliser_must_agree():
    two = ex.RESPONSE_POINT_TWO_CROPS
    rec = two["recommendation"]
    with pytest.raises(ValidationError):                       # a two-crop answer needs both crops
        PointResponse(**{**two, "recommendation": {**rec, "crops": [rec["crop"]]}})
    with pytest.raises(ValidationError):                       # the second crop is close_second
        PointResponse(**{**two, "recommendation": {**rec, "close_second": "rice"}})
    with pytest.raises(ValidationError):                       # the second fertiliser is for the second crop
        PointResponse(**{**two, "fertiliser_second": {**two["fertiliser_second"], "crop": "rice"}})
    with pytest.raises(ValidationError):                       # at most two crops
        PointResponse(**{**two, "recommendation": {**rec, "crops": ["beans", "maize", "rice"]}})


@pytest.mark.parametrize("body", [{"lat": -1.5, "season": "B"}, {"lon": 29.6, "season": "B"},
                                  {"lat": -1.5, "lon": 29.6}, {"lat": 10.0, "lon": 29.6, "season": "B"},
                                  {"lat": -1.5, "lon": 29.6, "season": "D"},
                                  {"lat": -1.5, "lon": 29.6, "season": "B", "district": "Musanze"},
                                  {"lat": -1.5, "lon": 29.6, "season": "B", "elevation_m": 500}])
def test_point_request_rules(body):
    with pytest.raises(ValidationError):
        PointRequest(**body)


def test_point_response_is_one_answer_or_an_abstain():
    ok = dict(ex.RESPONSE_POINT_OK)
    with pytest.raises(ValidationError):                       # no answer without abstaining
        PointResponse(**{**ok, "recommendation": None})
    with pytest.raises(ValidationError):                       # an abstain has no answer
        PointResponse(**{**ok, "abstain": True, "abstain_message": "Uncertain."})
    with pytest.raises(ValidationError):                       # an abstain needs a message
        PointResponse(**{**ex.RESPONSE_POINT_OUTSIDE, "abstain_message": None})
    other = {**ok["fertiliser"], "crop": "beans"}
    with pytest.raises(ValidationError):                       # fertiliser is for the recommended crop
        PointResponse(**{**ok, "fertiliser": other})
    bad = {**ok["recommendation"], "confidence": {**ok["recommendation"]["confidence"], "tested_hit_rate": 1.2}}
    with pytest.raises(ValidationError):
        PointResponse(**{**ok, "recommendation": bad})


def test_stub_point_route_follows_contract():
    from fastapi.testclient import TestClient
    from stub_app import app
    c = TestClient(app)
    r = c.post("/v1/recommend/point", json={"lat": -1.5, "lon": 29.6, "season": "A"})
    assert r.status_code == 200
    body = PointResponse(**r.json())
    assert body.season.asked == "A" and body.season.used == "B"
    assert any(w.code == "STUB_RESPONSE" for w in body.warnings)
    assert ErrorEnvelope(**c.post("/v1/recommend/point", json={"lat": 10, "lon": 29.6, "season": "B"}).json()
                         ).error.code == "OUT_OF_BOUNDS"
