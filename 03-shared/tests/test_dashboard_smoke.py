"""Smoke tests for the dashboard skeleton. Run from the repo root: python -m pytest 03-shared/tests -q"""
import sys
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASH))
import client  # noqa: E402

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_build_payload_drops_empty_soil_values():
    body = client.build_payload("Musanze", "B", 3, ["crops"], None, {"ph_water": 5.1, "texture": "", "x": None})
    assert body["soil"] == {"ph_water": 5.1} and "crop" not in body


def test_build_payload_adds_elevation_only_when_given():
    assert "elevation_m" not in client.build_payload("Musanze", "B", 3, ["crops"], None, {})
    assert client.build_payload("Musanze", "B", 3, ["crops"], None, {}, elevation_m=2300.0)["elevation_m"] == 2300.0


def test_build_payload_without_soil_has_no_soil_key():
    assert "soil" not in client.build_payload("Musanze", "B", 3, ["crops"], None, {"ph_water": None})


def test_mock_metadata_and_recommend_round_trip():
    meta = client.get_metadata("mock")
    assert "Musanze" in [d["name"] for d in meta["districts"]]
    out = client.recommend(client.build_payload("Musanze", "B", 1, ["crops", "fertiliser"], None, {}), "mock")
    assert len(out["crop_recommendations"]) == 1
    assert any(w["code"] == "STUB_RESPONSE" for w in out["warnings"])


def test_invalid_request_raises_contract_error():
    with pytest.raises(client.ApiError) as exc:
        client.recommend({"season": "B"}, "mock")  # no district and no coordinates
    assert exc.value.code == "VALIDATION_ERROR"


def test_unreachable_live_service_gives_a_clear_error():
    with pytest.raises(client.ApiError) as exc:
        client.get_health("live", "http://127.0.0.1:9")
    assert "Could not reach" in str(exc.value)


def test_recommendation_page_renders_and_submits_in_mock_mode():
    at = st_testing.AppTest.from_file(str(DASH / "pages" / "recommendation.py"), default_timeout=30)
    at.session_state["mode"] = "mock"
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Suggested crops" in s.value for s in at.subheader)
    assert any("placeholder" in w.value.lower() for w in at.warning)


def test_diagnosis_page_is_a_placeholder():
    at = st_testing.AppTest.from_file(str(DASH / "pages" / "diagnosis.py"))
    at.run()
    assert not at.exception and "CV sub-team" in at.info[0].value


def test_baseline_mode_calls_the_real_service():
    out = client.recommend(client.build_payload("Musanze", "B", 3, ["crops", "fertiliser"], None, {}), "baseline")
    assert out["crop_recommendations"][0]["method"] == "expert_rules"
    assert not any(w["code"] == "STUB_RESPONSE" for w in out["warnings"])
    assert client.get_metadata("baseline")["crops"] == ["beans", "irish_potato", "maize", "rice"]


def test_recommendation_page_renders_and_submits_in_baseline_mode():
    at = st_testing.AppTest.from_file(str(DASH / "pages" / "recommendation.py"), default_timeout=30)
    at.session_state["mode"] = "baseline"
    at.run()
    assert not at.exception
    assert any("Baseline mode" in i.value for i in at.info)
    at.button[0].click().run()
    assert not at.exception
    assert any("Suggested crops for" in s.value for s in at.subheader)
    assert not any("placeholder" in w.value.lower() for w in at.warning)


def test_baseline_service_errors_reach_the_page_as_api_errors():
    with pytest.raises(client.ApiError) as exc:
        client.recommend({"district": "Kigali", "season": "B"}, "baseline")
    assert exc.value.code == "UNKNOWN_DISTRICT"
