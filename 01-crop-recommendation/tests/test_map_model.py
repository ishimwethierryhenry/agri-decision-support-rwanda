"""Tests for the map recommender (dashboard map page and Colab notebook).

They use the committed feature table (v6b), district outlines and tuning_v1 results, and train the model on the map,
the tuned Random Forest, once (a few seconds).
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
    assert rec.final["model"] == mm.MODEL_NAME == run["decisions"]["map"]["serve"] == "RF[all_sat] tuned"
    assert rec.model.used_ == ev.VIEWS_V11["all_sat"] and rec.model.params == rec.final["settings"]
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
    # tuning_v1, new fields, first crop right: Random Forest 0.750, 0.744, 0.759, 0.629 against the rules 0.622, 0.661,
    # 0.219, 0.592. Until 25 Nov the choice used the main score, where the rules led in Nyabihu (0.645 against 0.627).
    assert {d: rec.method_for(d) for d in sorted(PILOTS)} == {
        "Musanze": "model", "Nyabihu": "model", "Nyagatare": "model", "Ruhango": "model"}
    blockcv = pd.read_csv(mm.EVAL_DIR / "main_blockcv_results.csv")
    for d in PILOTS:
        for method, model in (("model", mm.MODEL_NAME), ("rules", "B1_expert_rules")):
            top1 = blockcv[(blockcv.model == model) & (blockcv.district == d)]["top1"].iloc[0]
            assert rec.hit_rates[d][method] == round(top1, 3)
    r = rec.recommend(-1.4580, 29.5560, "B")                       # a Musanze block: Random Forest answers
    a = r["answer"]
    assert a["method"] == "model" and a["crop"] == r["model"]["ranking"][0]["crop"] == "irish_potato"
    assert a["hit_rate"] == rec.hit_rates["Musanze"]["model"] == 0.75 and a["confidence"] == "high"
    assert a["others"] == [x["crop"] for x in r["model"]["ranking"][1:]] and a["close_second"] is None
    assert a["why"][0] == ("In Musanze, Random Forest named the right crop first for 75% of the new fields in "
                           "testing, against 62% for the expert rules, so the answer comes from Random Forest.")
    r = rec.recommend(-1.35, 30.30, "B")                           # Nyagatare: Random Forest answers
    assert r["answer"]["method"] == "model" and r["answer"]["crop"] == r["model"]["ranking"][0]["crop"]
    assert r["fertiliser"]["crop"] == r["answer"]["crop"]
    a = rec.recommend(-1.35, 30.30, "A")["answer"]                      # a season never tested
    assert a["confidence"] == "low" and a["lowered_by"] == ["this season was never tested"]
    assert r["answer"]["lowered_by"] == []
    for lat, lon in ((-1.62, 29.50), (-1.692, 29.538)):            # Nyabihu: the rules rate crops equally here
        r = rec.recommend(lat, lon, "B")
        a = r["answer"]
        assert r["district"] == "Nyabihu" and a["method"] == "model" and a["hit_rate"] == 0.744
        assert a["crop"] == r["model"]["ranking"][0]["crop"] and a["close_second"] is None
        assert a["confidence"] == "medium" and a["lowered_by"] == []
        assert r["rules"]["ranking"][0]["score"] == r["rules"]["ranking"][1]["score"]   # a tie for the rules
        assert "CROPS_TIED" not in r["warning_codes"]                  # the tie matters only when the rules answer


def test_the_rules_answer_when_the_model_is_not_ahead_by_more_than_the_margin(rec, monkeypatch):
    monkeypatch.setitem(rec.hit_rates["Nyabihu"], "model", 0.668)  # 0.007 above the rules: not enough
    assert rec.method_for("Nyabihu") == "rules"
    r = rec.recommend(-1.62, 29.50, "B")
    a = r["answer"]
    assert a["method"] == "rules" and a["close_second"] is not None and a["confidence"] == "low"
    assert a["why"][0] == ("In Nyabihu, Random Forest was not ahead of the expert rules in testing on new fields "
                           "(first crop right for 67% of the fields, against 66% for the rules), so the answer comes "
                           "from the rules, the default.")
    assert any("fits just as well" in w for w in a["why"]) and "CROPS_TIED" in r["warning_codes"]


@pytest.mark.parametrize("hit,season,close,far,expected", [
    (0.80, True, False, False, "high"), (0.80, True, True, False, "medium"), (0.80, True, True, True, "low"),
    (0.62, True, False, False, "medium"), (0.62, True, True, False, "low"), (0.62, True, False, True, "low"),
    (0.62, False, False, False, "low"), (0.40, True, False, False, "low"), (None, True, False, False, "low"),
])
def test_confidence_levels(hit, season, close, far, expected):
    assert mm.confidence_level(hit, season, close, far) == expected


def test_other_seasons_are_answered_for_season_b_with_a_note(rec):
    r = rec.recommend(-1.4580, 29.5560, today=date(2026, 10, 29))      # 29 Oct is season A
    assert r["season"] == {"today": "A", "asked": "A", "used": "B"}
    assert any("not in the training data yet" in w for w in r["warnings"])


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
    imp = rec.feature_importance()                      # tested drop in main score on new fields (tuning_v1)
    assert set(imp["feature"]) == set(rec.model.used_) and abs(imp["share"].sum() - 1) < 1e-9
    assert imp.iloc[0]["feature"] in {"srtm_elevation_m", "elevation_m"}       # elevation matters most
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
    assert round(maize["tested_here"]["new_fields"]["f1"], 3) == 0.852               # tuning_v1 confusion counts
    rules_maize = next(x for x in r["rules"]["ranking"] if x["crop"] == "maize")
    assert rules_maize["tested_here"]["new_fields"]["f1"] == 0.0
    r = rec.recommend(-1.4580, 29.5560, "B")                             # Musanze: 1 rice field, not scored
    rice = next(x for x in r["model"]["ranking"] if x["crop"] == "rice")
    assert rice["tested_here"]["new_fields"]["scored"] is False
