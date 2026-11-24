"""Smoke tests for the map page and the app navigation. Run from the repo root: python -m pytest 03-shared/tests -q

The map itself is a browser component, so these tests set the point with the coordinate boxes below the map,
which go through the same code as a click.
"""
import re
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "dashboard"
st_testing = pytest.importorskip("streamlit.testing.v1")


def _page():
    at = st_testing.AppTest.from_file(str(DASH / "pages" / "map.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _picks(at):
    """The two side-by-side picks of the evaluation view."""
    return [m.value for m in at.markdown if m.value.startswith("### ")]


def _html(at, pattern):
    """Text pieces of the result card (rec_ui.py writes it as HTML through st.markdown)."""
    return [x for m in at.markdown for x in re.findall(pattern, m.value)]


def _answer(at):
    """The one recommended crop shown to the user: the title of the result card."""
    return _html(at, r'class="card-title"[^>]*>([^<]+)<')


def _confidence(at):
    """The confidence badge of the answer: High, Medium or Low."""
    return _html(at, r'class="result-badge \w+">(\w+) confidence<')


def _place(at):
    """District and point line at the top of the result card."""
    return _html(at, r'class="place">([^<]+)<')


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
    assert any("Which <em>crop</em> fits this field?" in m.value for m in at.markdown)       # hero of the map page
    at.switch_page("pages/recommendation.py").run()
    assert not at.exception
    assert any("What should I <em>plant</em> in my district?" in m.value for m in at.markdown)


def test_default_point_shows_one_answer_with_its_confidence():
    at = _page()
    assert _place(at) == ["Musanze (Northern Province), point -1.458, 29.556"]
    assert _answer(at) == ["Irish potato"] and not _picks(at)       # one answer, no comparison for the user
    assert "Random Forest suggests" not in [c.value for c in at.caption]
    assert len(_confidence(at)) == 1 and _confidence(at)[0] in {"High", "Medium", "Low"}
    text = " ".join(m.value for m in at.markdown)
    assert "In testing on new fields in Musanze, the first crop was right about" in text
    assert "Soil, 0 to 20 cm (iSDAsoil)" in text and "NISR" in text     # inputs stay available, folded
    assert "Satellite at the field: SRTM elevation" in text


def test_evaluation_view_shows_both_methods():
    at = _evaluation_view(_page())
    captions = [c.value for c in at.caption]
    assert "Random Forest suggests" in captions and "Expert rules suggest" in captions
    picks = _picks(at)
    assert len(picks) == 2 and all(v[4:] in {"Beans", "Irish potato", "Maize", "Rice"} for v in picks)
    # Musanze: the tuned Random Forest did better than the rules on new fields in testing (0.66 against 0.59)
    assert any("the Random Forest did better than the rules" in s.value for s in at.success)
    assert "**Tested scores (macro-F1, 0 to 1, protocol 1.2)**" in [m.value for m in at.markdown]


def test_fertiliser_for_the_recommended_crop_is_interim_and_cited():
    at = _page()
    pick = _answer(at)[0]
    text = " ".join(m.value for m in at.markdown)
    assert f"Fertiliser for {pick} <b>" in text and "kg/ha" in text
    assert any(w.value.startswith("Interim rates") for w in at.warning)
    assert any(c.value.startswith("Source: ") and "[link](https://" in c.value for c in at.caption)


def test_season_follows_today_and_other_seasons_get_a_note():
    at = _page()
    at.radio(key="pills_season").set_value("A").run()
    assert any("not in the training data yet" in i.value for i in at.info)
    assert _confidence(at) == ["Low"]
    at.radio(key="pills_season").set_value("B").run()
    assert not any("not in the training data yet" in i.value for i in at.info)
    assert _confidence(at) == ["High"]                       # Musanze: tested hit rate 0.75


def test_typed_point_in_nyagatare():
    at = _type_point(_page(), -1.35, 30.30)
    assert _place(at)[0].startswith("Nyagatare (Eastern Province)")
    assert _answer(at) == ["Maize"]
    assert any("so the answer comes from Random Forest" in m.value for m in at.markdown)
    at = _evaluation_view(at)
    assert any("the Random Forest did better than the rules" in s.value for s in at.success)


def test_points_outside_the_pilot_districts_or_rwanda_get_no_crop():
    at = _type_point(_page(), -1.95, 30.06)       # Kigali
    assert _place(at) == ["Nyarugenge"] and not _picks(at) and not _answer(at)
    assert "not one of the four pilot districts" in at.warning[0].value
    at = _type_point(at, 10.0, 200.0)             # nowhere near Rwanda, and no crash on odd numbers
    assert _place(at) == ["Outside Rwanda"] and not _picks(at)


def test_crop_by_crop_table_for_the_clicked_district():
    at = _evaluation_view(_page())                   # default point: Musanze
    table = at.dataframe[-1].value
    assert list(table.index) == ["Beans", "Irish potato", "Maize", "Rice"]
    assert table.loc["Rice", "Random Forest F1"] == "not scored"            # 1 rice field in Musanze
    assert table.loc["Irish potato", "Random Forest F1"] == "0.83"          # tuning_v1, new fields
    assert table.loc["Irish potato", "Rules F1"] == "0.69"
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
    assert len(main) == 18 and "Main score (macro-F1, 0 to 1)" in main.columns
    assert "Worst district (main score)" in main.columns and "Top-3 (mean)" not in main.columns
    assert "Passes the adoption rule" in main.columns and "Better than the rules in" in main.columns
    assert any("Random Forest in Musanze, the rules in Nyabihu, Random Forest in Nyagatare, Random Forest in Ruhango"
               in m.value for m in at.markdown)                         # same choice as the map
    why = " ".join(m.value for m in at.markdown)
    assert "The tuned Random Forest, all + satellite inputs scores 0.514 against 0.451" in why
    assert "that crop is right 80% of the time" in why and "for only 80% of the fields" in why
    gains = at.dataframe[5].value
    assert len(gains) == 9 and {"Protocol settings", "Tuned", "Change"} <= set(gains.columns)
    assert any("tuning lowered 5 of the 9 models" in c.value for c in at.caption)
    effect = at.dataframe[1].value
    assert list(effect.index) == ["Logistic Regression", "Random Forest", "XGBoost"]
    at.radio(key="pills_scores_test").set_value("new_fields").run()
    assert not at.exception
    assert any("within 0.003 of the best model" in c.value for c in at.caption)
    at.selectbox(key="scores_district").set_value("Nyagatare").run()
    crops = at.dataframe[4].value
    assert crops.loc["Maize", "XGBoost F1"] == "0.80"
    at.selectbox(key="scores_other").set_value("XGB[all_sat]").run()
    assert not at.exception and "XGBoost, all + satellite inputs F1" in at.dataframe[4].value.columns
    assert any("Logistic Regression the most on average" in c.value for c in at.caption)
