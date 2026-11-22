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
    """The two side-by-side picks of the evaluation view."""
    return [m.value for m in at.markdown if m.value.startswith("### ")]


def _answer(at):
    """The one recommended crop shown to the user."""
    return [m.value[3:] for m in at.markdown if m.value.startswith("## ")]


def _evaluation_view(at):
    at.toggle(key="eval_view").set_value(True).run()
    assert not at.exception
    return at


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


def test_default_point_shows_one_answer_with_its_confidence():
    at = _page()
    assert at.subheader[0].value == "Musanze (Northern Province)"
    assert _answer(at) == ["Irish potato"] and not _picks(at)       # one answer, no comparison for the user
    assert "XGBoost suggests" not in [c.value for c in at.caption]
    assert any(m.value.startswith("Confidence: ") for m in at.markdown)
    assert any(c.value.startswith("In testing on new fields in Musanze, the first crop was right about")
               for c in at.caption)
    text = " ".join(m.value for m in at.markdown)
    assert "Soil, 0 to 20 cm (iSDAsoil)" in text and "NISR" in text     # inputs stay available, folded


def test_evaluation_view_shows_both_methods():
    at = _evaluation_view(_page())
    captions = [c.value for c in at.caption]
    assert "XGBoost suggests" in captions and "Expert rules suggest" in captions
    picks = _picks(at)
    assert len(picks) == 2 and all(v[4:] in {"Beans", "Irish potato", "Maize", "Rice"} for v in picks)
    # Musanze: the rules did better than XGBoost on new fields in testing, and the page says so
    assert any("the rules did better" in w.value for w in at.warning)


def test_fertiliser_for_the_recommended_crop_is_interim_and_cited():
    at = _page()
    pick = _answer(at)[0]
    text = " ".join(m.value for m in at.markdown)
    assert f"**Fertiliser for {pick}**" in text and "kg/ha" in text
    assert any(w.value.startswith("Interim rates") for w in at.warning)
    assert any(c.value.startswith("Source: ") and "[link](https://" in c.value for c in at.caption)


def test_season_follows_today_and_other_seasons_get_a_note():
    at = _page()
    at.selectbox[0].set_value("A").run()
    assert any("not in the training data yet" in i.value for i in at.info)
    assert "Confidence: :red[**Low**]" in [m.value for m in at.markdown]
    at.selectbox[0].set_value("B").run()
    assert not any("not in the training data yet" in i.value for i in at.info)
    assert "Confidence: :orange[**Medium**]" in [m.value for m in at.markdown]


def test_typed_point_in_nyagatare():
    at = _type_point(_page(), -1.35, 30.30)
    assert at.subheader[0].value == "Nyagatare (Eastern Province)"
    assert _answer(at) == ["Maize"]
    assert any("so the answer comes from XGBoost" in m.value for m in at.markdown)
    at = _evaluation_view(at)
    assert any("XGBoost did better than the rules" in s.value for s in at.success)


def test_points_outside_the_pilot_districts_or_rwanda_get_no_crop():
    at = _type_point(_page(), -1.95, 30.06)       # Kigali
    assert at.subheader[0].value == "Nyarugenge" and not _picks(at) and not _answer(at)
    assert "not one of the four pilot districts" in at.warning[0].value
    at = _type_point(at, 10.0, 200.0)             # nowhere near Rwanda, and no crash on odd numbers
    assert at.subheader[0].value == "Outside Rwanda" and not _picks(at)


def test_crop_by_crop_table_for_the_clicked_district():
    at = _evaluation_view(_page())                   # default point: Musanze
    table = at.dataframe[-1].value
    assert list(table.index) == ["Beans", "Irish potato", "Maize", "Rice"]
    assert table.loc["Rice", "XGBoost F1"] == "not scored"            # 1 rice field in Musanze
    assert table.loc["Irish potato", "XGBoost F1"] == "0.77"
    at.radio(key="crop_test").set_value("new_district").run()
    assert not at.exception


def test_scores_page_shows_every_model():
    at = st_testing.AppTest.from_file(str(DASH / "pages" / "scores.py"), default_timeout=60)
    at.run()
    assert not at.exception
    info = at.info[0].value
    assert "No trained model passes the adoption rule" in info
    assert "Random Forest, field + satellite inputs: 0.368 against 0.378" in info
    main = at.dataframe[0].value
    assert len(main) == 18 and "Main score" in main.columns
    effect = at.dataframe[1].value
    assert list(effect.index) == ["Logistic Regression", "Random Forest", "XGBoost"]
    at.radio(key="scores_test").set_value("new_fields").run()
    assert not at.exception
    assert any("within 0.003 of the best model" in c.value for c in at.caption)
    at.selectbox(key="scores_district").set_value("Nyagatare").run()
    crops = at.dataframe[4].value
    assert crops.loc["Maize", "XGBoost F1"] == "0.80"
    at.selectbox(key="scores_other").set_value("XGB[all_sat]").run()
    assert not at.exception and "XGBoost, all + satellite inputs F1" in at.dataframe[4].value.columns
    assert any("(+0.005)" in m.value for m in at.markdown)
