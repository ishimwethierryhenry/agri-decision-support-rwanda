"""Smoke tests for the map page and the app navigation. Run from the repo root: python -m pytest 03-shared/tests -q

The map itself is a browser component, so these tests set the point with the coordinate boxes below the map,
which go through the same code as a click.
"""
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "dashboard"
st_testing = pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("xgboost")


def _page():
    at = st_testing.AppTest.from_file(str(DASH / "pages" / "map.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _picks(at):
    return [m.value for m in at.markdown if m.value.startswith("### ")]


def _type_point(at, lat, lon):
    at.number_input[0].set_value(lat)
    at.number_input[1].set_value(lon)
    at.button[0].click().run()
    assert not at.exception
    return at


def test_app_opens_on_the_map_page():
    at = st_testing.AppTest.from_file(str(DASH / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert [h.value for h in at.header] == ["Click a field on the map"]
    at.switch_page("pages/recommendation.py").run()
    assert not at.exception and "Crop and fertiliser" in at.header[0].value


def test_default_point_shows_both_methods_with_inputs():
    at = _page()
    assert at.subheader[0].value == "Musanze (Northern Province)"
    captions = [c.value for c in at.caption]
    assert "XGBoost suggests" in captions and "Expert rules suggest" in captions
    picks = [m.value for m in at.markdown if m.value.startswith("### ")]
    assert len(picks) == 2 and all(v[4:] in {"Beans", "Irish potato", "Maize", "Rice"} for v in picks)
    text = " ".join(m.value for m in at.markdown)
    assert "Soil, 0 to 20 cm (iSDAsoil)" in text and "NISR" in text
    # Musanze: the rules did better than XGBoost on new fields in testing, and the page says so
    assert any("the rules did better" in w.value for w in at.warning)


def test_season_follows_today_and_other_seasons_get_a_note():
    at = _page()
    at.selectbox[0].set_value("A").run()
    assert any("not in the training data yet" in i.value for i in at.info)
    at.selectbox[0].set_value("B").run()
    assert not any("not in the training data yet" in i.value for i in at.info)


def test_typed_point_in_nyagatare():
    at = _type_point(_page(), -1.35, 30.30)
    assert at.subheader[0].value == "Nyagatare (Eastern Province)"
    assert any("XGBoost did better than the rules" in s.value for s in at.success)


def test_points_outside_the_pilot_districts_or_rwanda_get_no_crop():
    at = _type_point(_page(), -1.95, 30.06)       # Kigali
    assert at.subheader[0].value == "Nyarugenge" and not _picks(at)
    assert "not one of the four pilot districts" in at.warning[0].value
    at = _type_point(at, 10.0, 200.0)             # nowhere near Rwanda, and no crash on odd numbers
    assert at.subheader[0].value == "Outside Rwanda" and not _picks(at)
