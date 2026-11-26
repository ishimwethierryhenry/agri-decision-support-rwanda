"""Tests for the map recommender (dashboard map page and Colab notebook).

They use the committed feature table (v6b), district outlines and tuning_v3 results, and train the model on the map,
the Random Forest of the third round of tuning with its crop weights, once (a few seconds).
Run from the repo root:  python -m pytest 01-crop-recommendation/tests/test_map_model.py -q
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-crop-recommendation" / "src" / "recommendation"))
import map_model as mm  # noqa: E402

PILOTS = {"Musanze", "Nyabihu", "Nyagatare", "Ruhango"}


@pytest.fixture(scope="module")
def rec():
    return mm.MapRecommender.load()


def test_seasons_follow_the_calendar():
    assert [mm.current_season(date(2026, m, 15)) for m in range(1, 13)] == \
        ["A", "B", "B", "B", "B", "B", "C", "C", "A", "A", "A", "A"]


def test_block_id_uses_part_a_grid():
    assert mm.block_id(-2.124, 29.646) == "-118_1647"
    assert mm.block_id(-1.4855, 29.6280) == f"{round(-1.4855 / 0.018)}_{round(29.6280 / 0.018)}"


@pytest.mark.parametrize("sand,silt,clay,expected", [
    (40, 40, 20, "loam"), (20, 30, 50, "clay"), (60, 25, 15, "sandy loam"), (45, 20, 35, "clay loam"),
    (48, 19, 33, "sandy clay loam"), (90, 5, 5, "sand"), (10, 75, 15, "silt loam"), (5, 50, 45, "silty clay"),
])
def test_usda_texture_classes(sand, silt, clay, expected):
    assert mm.texture_class(sand, silt, clay) == expected


def test_district_outlines_cover_rwanda(rec):
    assert len(rec.districts.shapes) == 30
    assert rec.districts.at(-1.95, 30.06) == "Nyarugenge"      # Kigali city centre
    assert rec.districts.at(-1.5, 31.2) is None                # Tanzania


def test_lookup_holds_block_centres_only_not_farm_points(rec):
    for col in ("lat", "lon"):
        steps = rec.blocks[col] / mm.BLOCK_DEG
        assert np.allclose(steps, np.round(steps), atol=1e-6)  # multiples of 0.018 degrees, so 2 km cells
    assert set(rec.blocks["District"]) == PILOTS
    assert rec.blocks["n_fields"].sum() == 2728                # every field that passes qc_ok


def test_model_is_the_tuned_model_the_protocol_chose_for_the_map(rec):
    import evaluate as ev
    run = json.loads((mm.EVAL_DIR / "run_info.json").read_text(encoding="utf-8"))
    assert rec.final["model"] == mm.MODEL_NAME == run["decisions"]["map"]["serve"] == "RF[field_sat] steady+weights"
    assert rec.model.used_ == ev.VIEWS_V11["field_sat"] and rec.model.params == rec.final["settings"]
    assert rec.final["crop_weights"] == pytest.approx([1.5, 2 / 3, 1.5, 1.5], abs=1e-5)
    rows = rec.main.head(5)
    raw = rec.model.predict_proba(rows) * rec.crop_weights           # the weights, then normalised again
    assert np.allclose(rec.predict_proba(rows), raw / raw.sum(axis=1, keepdims=True))
    est = rec.model.estimator.get_params()
    assert est["n_estimators"] == 300 and est["max_depth"] == rec.final["settings"]["max_depth"]
    assert len(rec.main) == 1904 and list(rec.model.classes_) == ["beans", "irish_potato", "maize", "rice"]
    assert mm.API_METHOD == "random_forest" and mm.METHOD_NAMES["model"] == "Random Forest"


def test_every_ciat_block_gets_an_answer_for_its_own_district(rec):
    for b in rec.blocks.itertuples():
        r = rec.recommend(b.lat, b.lon, "B")
        assert r["status"] == "ok" and r["district"] == b.District


def test_answer_has_both_methods_with_tested_scores(rec):
    r = rec.recommend(-1.4580, 29.5560, "B", today=date(2026, 3, 10))   # a Musanze block
    assert r["status"] == "ok" and r["district"] == "Musanze"
    probs = [x["probability"] for x in r["model"]["ranking"]]
    assert abs(sum(probs) - 1) < 0.01 and probs == sorted(probs, reverse=True)
    assert {x["crop"] for x in r["model"]["ranking"]} == {"beans", "irish_potato", "maize", "rice"}
    assert r["rules"]["ranking"][0]["reasons"]
    blockcv = pd.read_csv(mm.EVAL_DIR / "main_blockcv_results.csv")
    tested = blockcv[(blockcv.model == mm.MODEL_NAME) & (blockcv.district == "Musanze")]["macro_f1"].iloc[0]
    assert abs(r["model"]["tested"]["new_fields"] - tested) < 0.0005
    sat = r["inputs"]["satellite_at_field"]
    assert set(sat) == {"srtm_elevation_m", "chirps_precip_mm", "ndvi_annual", "swir_ratio"}
    assert abs(sat["srtm_elevation_m"] - r["inputs"]["elevation_m"]) < 100   # SRTM and GPS agree on CIAT fields
    assert r["better_in_testing"] in {"model", "rules", "same"}
    assert r["fertiliser"]["status"] == "ok" and r["fertiliser"]["crop"] == r["answer"]["crop"]
    assert any(w.startswith("Fertiliser rates are interim") for w in r["warnings"])
    assert len(r["warning_codes"]) == len(r["warnings"]) and "FERTILISER_INTERIM" in r["warning_codes"]
    assert not any("season" in w.lower() and "not in the training" in w for w in r["warnings"])


def test_one_answer_comes_from_the_method_that_names_the_right_crop_first_more_often(rec):
    # tuning_v3, new fields, first crop right: Random Forest 0.756, 0.720, 0.756, 0.637 against the rules 0.622, 0.661,
    # 0.219, 0.592. Until 25 Nov the choice used the main score, where the rules led in Nyabihu.
    assert {d: rec.method_for(d) for d in sorted(PILOTS)} == {
        "Musanze": "model", "Nyabihu": "model", "Nyagatare": "model", "Ruhango": "model"}
    blockcv = pd.read_csv(mm.EVAL_DIR / "main_blockcv_results.csv")
    for d in PILOTS:
        for method, model in (("model", mm.MODEL_NAME), ("rules", "B1_expert_rules")):
            top1 = blockcv[(blockcv.model == model) & (blockcv.district == d)]["top1"].iloc[0]
            assert rec.hit_rates[d][method] == round(top1, 3)
    r = rec.recommend(-1.4580, 29.5560, "B")                       # a Musanze block: Random Forest answers
    a = r["answer"]
    assert a["method"] == "model" and a["kind"] == "one_crop" and a["crops"] == [a["crop"]] == ["irish_potato"]
    assert a["crop"] == r["model"]["ranking"][0]["crop"]
    assert a["others"] == [x["crop"] for x in r["model"]["ranking"][1:]] and a["close_second"] is None
    assert a["why"][0] == ("In Musanze, Random Forest named the right crop first for 76% of the new fields in "
                           "testing, against 62% for the expert rules, so the answer comes from Random Forest.")
    r = rec.recommend(-1.35, 30.30, "B")                           # Nyagatare: Random Forest answers
    assert r["answer"]["method"] == "model" and r["answer"]["crop"] == r["model"]["ranking"][0]["crop"]
    assert r["fertiliser"]["crop"] == r["answer"]["crop"] and r["fertiliser_second"] is None
    assert r["answer"]["lowered_by"] == []
    for lat, lon in ((-1.62, 29.50), (-1.692, 29.538)):            # Nyabihu: the rules rate crops equally here
        r = rec.recommend(lat, lon, "B")
        a = r["answer"]
        assert r["district"] == "Nyabihu" and a["method"] == "model" and a["kind"] == "one_crop"
        assert a["crop"] == r["model"]["ranking"][0]["crop"] and a["close_second"] is None and a["lowered_by"] == []
        assert r["rules"]["ranking"][0]["score"] == r["rules"]["ranking"][1]["score"]   # a tie for the rules
        assert "CROPS_TIED" not in r["warning_codes"]                  # the tie matters only when the rules answer


def _cell(district, band, kind):
    t = pd.read_csv(mm.CONFIDENCE_TABLE_PATH)
    return t[(t.district == district) & (t.band == band) & (t.kind == kind)].iloc[0]


@pytest.mark.parametrize("lat,lon,district,kind,crops,band,table_kind", [
    (-1.4580, 29.5560, "Musanze", "one_crop", ["irish_potato"], "within 5 km", mm.ONE_CROP),
    (-1.692, 29.538, "Nyabihu", "one_crop", ["maize"], "within 5 km", mm.ONE_CROP),
    (-1.35, 30.30, "Nyagatare", "one_crop", ["maize"], "within 5 km", mm.ONE_CROP),
    (-2.196, 29.592, "Ruhango", "two_crops", ["beans", "maize"], "within 5 km", mm.TWO_CLOSE),
    (-1.554, 29.505, "Musanze", "one_crop", ["beans"], "5 to 10 km", mm.ONE_CROP),
    (-1.513, 30.165, "Nyagatare", "one_crop", ["maize"], "5 to 10 km", mm.ONE_CROP),
    (-1.463, 30.645, "Nyagatare", "one_crop", ["maize"], "10 to 20 km", mm.ONE_CROP),
])
def test_confidence_is_how_often_answers_like_this_one_were_right(rec, lat, lon, district, kind, crops, band,
                                                                    table_kind):
    """Since 25 Nov 2026 (reports/map_confidence): same district, same kind of answer, field data from as far."""
    r = rec.recommend(lat, lon, "B")
    a, cell = r["answer"], _cell(district, band, table_kind)
    assert r["district"] == district and a["kind"] == kind and a["crops"] == crops and a["crop"] == crops[0]
    assert mm.data_band(r["inputs"]["block"]["distance_km"]) == band
    assert a["hit_rate"] == pytest.approx(cell.rate) and a["tested_fields"] == cell.fields
    assert a["confidence"] == ("high" if cell.rate >= 0.75 else "medium")
    assert a["basis"] == ("first_two_crops" if kind == "two_crops" else "first_crop")
    assert [x["crop"] for x in r["model"]["ranking"][:len(crops)]] == crops
    assert r["fertiliser"]["crop"] == crops[0]
    if kind == "two_crops":                                    # both crops, each with its fertiliser
        assert a["close_second"] == crops[1] and r["fertiliser_second"]["crop"] == crops[1]
        assert any(f"the crop planted was one of the two for {cell.rate:.0%} of the fields" in w for w in a["why"])
    else:
        assert a["close_second"] is None and r["fertiliser_second"] is None
    if band != "within 5 km":
        assert any(w.startswith("The nearest surveyed field is") for w in a["why"])


def test_no_answer_beyond_20_km_or_for_a_season_never_tested(rec):
    r = rec.recommend(-1.453, 30.725, "B")                          # Nyagatare, nearest surveyed field 21.9 km away
    a = r["answer"]
    assert r["inputs"]["block"]["distance_km"] > 20 and a["kind"] == "none" and a["crops"] == []
    assert a["crop"] is None and a["confidence"] is None and r["fertiliser"] is None
    assert a["message"].startswith("The nearest surveyed field is 21.9 km away") and "NO_RELIABLE_ANSWER" in r["warning_codes"]
    a = rec.recommend(-1.35, 30.30, "A")["answer"]                   # a season never tested
    assert a["kind"] == "none" and a["lowered_by"] == ["this season was never tested"]
    assert a["why"] == [a["message"] + " Please ask the local agronomist."] and "season B" in a["message"]


def test_no_point_gets_one_crop_with_a_low_confidence(rec):
    """A grid of points about every 3 km over the four districts, season B: one crop or two with high or medium
    confidence, or no answer, never a low one (the complaint of 25 Nov 2026)."""
    seen = set()
    for name, polys, (x0, y0, x1, y1) in rec.districts.shapes:
        if name not in PILOTS:
            continue
        for lat in np.arange(y0 + 0.015, y1, 0.03):
            for lon in np.arange(x0 + 0.015, x1, 0.03):
                if rec.districts.at(lat, lon) != name:
                    continue
                a = rec.recommend(float(lat), float(lon), "B")["answer"]
                seen.add((a["kind"], a["confidence"]))
                assert a["kind"] == "none" or (a["confidence"] in {"high", "medium"} and a["hit_rate"] >= 0.5)
    assert ("one_crop", "high") in seen and ("two_crops", "high") in seen and ("one_crop", "medium") in seen


def test_the_confidence_table_is_complete_and_its_check_passed():
    t = pd.read_csv(mm.CONFIDENCE_TABLE_PATH)
    assert len(t) == 4 * 3 * 3 and set(t.district) == PILOTS
    assert set(t.band) == {b for _, b in mm.DATA_BANDS} and set(t.kind) == {mm.ONE_CROP, mm.TWO_CLOSE, mm.TWO_ANY}
    assert ((t.rate >= 0) & (t.rate <= 1)).all() and (t.fields >= mm.MIN_TESTED_FIELDS).all()
    assert (t.pooled == (t.district_fields < mm.MIN_TESTED_FIELDS)).all()
    assert np.allclose(t.rate, t.right / t.fields, atol=1e-6)
    info = json.loads((mm.CONFIDENCE_TABLE_PATH.parent / "run_info.json").read_text(encoding="utf-8"))
    check = info["builtin_check"]
    assert check["passed"] and all(abs(v["replayed"] - v["tested"]) < 1e-9 for v in check["districts"].values())
    assert check["model"] == mm.MODEL_NAME


def test_without_the_table_the_district_hit_rate_decides_and_low_means_no_answer():
    old = mm.MapRecommender.load(confidence_table=None)
    assert old.confidence == {}
    a = old.recommend(-1.4580, 29.5560, "B")["answer"]                  # Musanze: 0.756, no reason to lower it
    assert a["kind"] == "one_crop" and a["confidence"] == "high" and a["hit_rate"] == 0.756
    a = old.recommend(-2.196, 29.592, "B")["answer"]                    # Ruhango 0.63 and a close call: low
    assert a["kind"] == "none" and "two crops fit almost as well" in a["message"]


def test_the_rules_answer_when_the_model_is_not_ahead_by_more_than_the_margin(rec, monkeypatch):
    monkeypatch.setitem(rec.hit_rates["Nyabihu"], "model", 0.668)  # 0.007 above the rules: not enough
    assert rec.method_for("Nyabihu") == "rules"
    r = rec.recommend(-1.62, 29.50, "B")
    a = r["answer"]
    assert a["method"] == "rules" and a["kind"] == "none"            # a tie lowers 0.66 to low: no answer
    assert a["message"] == ("The confidence here would be low (two crops fit almost as well), so there is no "
                            "reliable answer.")
    assert "CROPS_TIED" in r["warning_codes"]


@pytest.mark.parametrize("hit,season,close,far,expected", [
    (0.80, True, False, False, "high"), (0.80, True, True, False, "medium"), (0.80, True, True, True, "low"),
    (0.62, True, False, False, "medium"), (0.62, True, True, False, "low"), (0.62, True, False, True, "low"),
    (0.62, False, False, False, "low"), (0.40, True, False, False, "low"), (None, True, False, False, "low"),
])
def test_confidence_levels(hit, season, close, far, expected):
    assert mm.confidence_level(hit, season, close, far) == expected


def test_other_seasons_get_no_answer_with_a_note(rec):
    r = rec.recommend(-1.4580, 29.5560, today=date(2026, 10, 29))      # 29 Oct is season A
    assert r["season"] == {"today": "A", "asked": "A", "used": "B"}
    assert any("not in the training data yet" in w for w in r["warnings"])
    assert r["answer"]["kind"] == "none" and r["warning_codes"][0] == "SEASON_NOT_EVALUATED"


def test_points_outside_the_pilot_districts_get_no_crop(rec):
    r = rec.recommend(-1.95, 30.06, "B")
    assert r["status"] == "outside_pilot" and "model" not in r
    r = rec.recommend(-1.5, 31.2, "B")
    assert r["status"] == "outside_rwanda" and "model" not in r


def test_far_from_ciat_data_is_flagged(rec):
    # find a point inside Nyagatare that is more than 5 km from every CIAT block
    blocks = rec.blocks[rec.blocks.District == "Nyagatare"]
    for lat in np.arange(-1.55, -1.05, 0.02):
        for lon in np.arange(30.0, 30.9, 0.02):
            if rec.districts.at(lat, lon) != "Nyagatare":
                continue
            if min(mm.haversine_km(lat, lon, b.lat, b.lon) for b in blocks.itertuples()) > 6:
                r = rec.recommend(lat, lon, "B")
                assert r["status"] == "ok" and r["inputs"]["block"]["distance_km"] > 5
                assert any("No CIAT field within" in w for w in r["warnings"])
                return
    pytest.skip("no point far from CIAT data found in Nyagatare")


def test_user_elevation_replaces_both_elevations(rec):
    r = rec.recommend(-1.4580, 29.5560, "B", elevation_m=1900)
    assert r["inputs"]["elevation_m"] == 1900 and r["inputs"]["satellite_at_field"]["srtm_elevation_m"] == 1900
    assert r["inputs"]["block"]["elevation_source"] == "user"


def test_feature_importance_and_export(rec, tmp_path):
    imp = rec.feature_importance()      # tested drop in main score on new fields, measured in tuning_v2 (see final)
    assert rec.final["importance_from"]["model"] == "RF[field_sat] tuned"
    assert set(imp["feature"]) == set(rec.model.used_) and abs(imp["share"].sum() - 1) < 1e-9
    assert set(imp["feature"][:3]) == {"chirps_precip_mm", "srtm_elevation_m", "elevation_m"}   # rain and height
    folder = rec.export(tmp_path / "model")
    assert (folder / "model.joblib").exists() and (folder / "meta.json").exists()
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == mm.MODEL_NAME and meta["features"] == rec.model.used_
    import joblib
    rows = rec.main[rec.model.used_].head(5).fillna(rec.model.fill_)
    assert np.allclose(joblib.load(folder / "model.joblib").predict_proba(rows), rec.model.estimator.predict_proba(rows))


def test_each_ranked_crop_carries_its_tested_scores_in_the_district(rec):
    r = rec.recommend(-1.350, 30.258, "B")                               # a Nyagatare block
    maize = next(x for x in r["model"]["ranking"] if x["crop"] == "maize")
    assert maize["tested_here"]["new_fields"]["fields"] == 792
    assert round(maize["tested_here"]["new_fields"]["f1"], 3) == 0.846               # tuning_v3 confusion counts
    rules_maize = next(x for x in r["rules"]["ranking"] if x["crop"] == "maize")
    assert rules_maize["tested_here"]["new_fields"]["f1"] == 0.0
    r = rec.recommend(-1.4580, 29.5560, "B")                             # Musanze: 1 rice field, not scored
    rice = next(x for x in r["model"]["ranking"] if x["crop"] == "rice")
    assert rice["tested_here"]["new_fields"]["scored"] is False
