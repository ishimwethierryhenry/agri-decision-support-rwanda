"""Tests for the crop profiles of the map page (src/recommendation/crop_profiles.py, rules/crop_profiles.yaml).

The numbers checked here are copied from the cited tables: Mulambu et al. (2017), Table 10.3, for the iron beans,
the Rwanda Potato Variety Description Guide (CIP, 2025) for the potatoes, and RAB's Annual Report 2019-2020,
Tables 2 and 7, for the maize hybrids. Run from the repo root: python -m pytest 01-crop-recommendation/tests -q
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "recommendation"
sys.path.insert(0, str(SRC))
import crop_profiles as cp  # noqa: E402


def _varieties(crop):
    return {v["name"]: v for v in cp.load()["crops"][crop]["varieties"]}


def test_profiles_file_has_no_problems():
    assert cp.check() == []


def test_every_crop_the_map_can_recommend_has_a_profile():
    import map_model as mm
    assert set(mm.CROP_NAMES) == set(cp.load()["crops"]) == set(cp.CROPS)


def test_kinyarwanda_names_are_the_ones_nisr_uses():
    crops = cp.load()["crops"]
    assert {c: p["kinyarwanda"] for c, p in crops.items()} == {
        "beans": "Ibishyimbo", "irish_potato": "Ibirayi", "maize": "Ibigori", "rice": "Umuceri"}


def test_iron_beans_match_table_10_3():
    beans = _varieties("beans")
    assert len(beans) == 10
    assert sorted(v["released"] for v in beans.values()) == [2010] * 5 + [2012] * 5
    assert {n for n, v in beans.items() if v["habit"] == "bush"} == {"RWR 2245", "RWR 2154"}
    assert (beans["CAB 2"]["iron_ppm"], beans["CAB 2"]["yield_t_ha"], beans["CAB 2"]["zone"]) == (95, [4.0, 4.5], "high")
    assert (beans["RWR 2154"]["iron_ppm"], beans["RWR 2154"]["seed"], beans["RWR 2154"]["zone"]) == (71, "sugar",
                                                                                                      "low_mid")
    assert beans["RWV 3006"]["resistant"] == ["AB", "AC", "ALS", "BCMV"] and beans["RWV 3006"]["tolerant"] == []


def test_potatoes_match_the_cip_guide():
    potato = _varieties("irish_potato")
    assert len(potato) == 15
    assert (potato["Kinigi"]["released"], potato["Kinigi"]["maturity_days"], potato["Kinigi"]["skin"]) == (
        1983, [110, 120], "red")
    assert potato["Twihaze"]["yield_t_ha"] == [56] and potato["Seka"]["heat_tolerant"] is True
    # the guide gives two different maturities and yields for Izihirwe, so the profile shows neither
    assert "maturity_days" not in potato["Izihirwe"] and "yield_t_ha" not in potato["Izihirwe"]


def test_maize_hybrids_match_rab_tables_2_and_7():
    maize = _varieties("maize")
    assert {n: v.get("trial_yield_t_ha") for n, v in maize.items() if v["zone"] == "mid"} == {
        "RHM1409": 9.23, "RHM1402": 8.86, "RHM1407": 8.41, "RHM104": 7.73}
    assert {n for n, v in maize.items() if v["zone"] == "high"} == {"RHMH1520", "RHMH1521"}
    assert cp.load()["zones"]["mid"] == {"label": "Mid altitude", "min_m": 900, "max_m": 1700}


def test_high_field_puts_climbing_iron_beans_first():
    prof = cp.for_field("beans", 2284)                 # Musanze, the map's default point
    fits = [v["name"] for v in prof["varieties"] if v["fits"]]
    assert prof["fitting"] == 8 and "CAB 2" in fits and "RWR 2245" not in fits
    assert [v["fits"] for v in prof["varieties"]] == [True] * 8 + [False] * 2      # those that fit come first
    assert prof["varieties"][-1]["fit_note"] == "Better below 1,700 m"


def test_mid_field_puts_bush_and_mid_altitude_varieties_first():
    beans = cp.for_field("beans", 1620)                # the Ruhango point where beans or maize fit
    assert [v["name"] for v in beans["varieties"] if v["fits"]] == ["RWV 3006", "MAC 44", "RWV 1129", "RWR 2245",
                                                                     "RWR 2154"]
    assert {v["fit_note"] for v in beans["varieties"] if not v["fits"]} == {"Better above 1,700 m"}
    maize = cp.for_field("maize", 2210)                # the Nyabihu point
    assert [v["name"] for v in maize["varieties"] if v["fits"]] == ["RHMH1520", "RHMH1521"]
    assert maize["varieties"][-1]["fit_note"] == "Better at 900 to 1,700 m"


def test_without_elevation_and_for_rice_nothing_is_marked():
    assert {v["fits"] for v in cp.for_field("maize", None)["varieties"]} == {None}
    rice = cp.for_field("rice", 1400)
    assert {v["fits"] for v in rice["varieties"]} == {None} and {v["fit_note"] for v in rice["varieties"]} == {""}


def test_sources_are_numbered_in_order_of_use_with_the_names_source():
    prof = cp.for_field("irish_potato", 2200)
    keys = [s["key"] for s in prof["sources"]]
    assert keys[0] == "cip_potato_2025" and keys[-1] == "nisr_2025" and len(keys) == len(set(keys))
    assert "cgiar_sns_2025" in keys and all(s["url"].startswith("https://") for s in prof["sources"])


def test_check_catches_a_claim_without_source_and_a_long_dash():
    data = cp.load()
    data["crops"]["beans"]["facts"][0].pop("source")
    data["crops"]["maize"]["intro"] += " — and more"
    data["crops"]["beans"]["varieties"][0]["seed"] = "purple"
    problems = cp.check(data)
    assert any("no source" in p for p in problems)
    assert any("long dash" in p for p in problems)
    assert any("no label for seed purple" in p for p in problems)


def test_unknown_crop_is_refused():
    with pytest.raises(KeyError):
        cp.for_field("cassava", 1500)
