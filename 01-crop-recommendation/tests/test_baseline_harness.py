"""Tests for the baselines and the LODO harness. Synthetic data only, no real farm data needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
from evaluate import (LeakageError, SklearnModel, check_features, prepare, run_lodo, summarise,  # noqa: E402
                      topk)
from expert_baseline import (ExpertRulesBaseline, MajorityBaseline, RulesError, explain_row,  # noqa: E402
                             load_rules, score_factor)


def raw_table(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    spec = {"A": (2300, "Irish potatoes"), "B": (1500, "Beans"), "C": (1400, "Maize"), "D": (1350, "Rice")}
    for dist, (elev, main_crop) in spec.items():
        for i in range(n_per):
            crop = main_crop if i % 4 else rng.choice(["Maize", "Beans", "Irish potatoes", "Rice"])
            rows.append({"District": dist, "GPS_Elev": elev + rng.normal(0, 50), "longitude": 30.0,
                         "Crop_plant": crop, "precip_mm": 600.0,
                         "tmax_c": 22.0 if dist == "A" else 24.0, "tmin_c": 12.0 if dist == "A" else 15.0,
                         "solar_mj_m2": 17.0, "nisr_yield_kg_ha": 1.0})
    rows.append({"District": "A", "GPS_Elev": 0.0, "longitude": 30.0, "Crop_plant": "Maize"})      # GPS error
    rows.append({"District": "A", "GPS_Elev": 1500.0, "longitude": 36.0, "Crop_plant": "Maize"})   # outside Rwanda
    rows.append({"District": "A", "GPS_Elev": 1500.0, "longitude": 30.0, "Crop_plant": "Mixed crops"})
    return pd.DataFrame(rows)


def test_shipped_rules_load_and_every_factor_has_a_source():
    rules = load_rules()
    assert set(rules["crops"]) == {"beans", "irish_potato", "maize", "rice"}
    for spec in rules["crops"].values():
        assert all(f["source"]["ref"] and f["source"]["text"] for f in spec["factors"])


@pytest.mark.parametrize("bad", [
    {"version": "x", "crops": {"maize": {"factors": [{"variable": "tmean_c", "optimal": [1, 2]}]}}},
    {"version": "x", "crops": {"maize": {"factors": [{"variable": "tmean_c", "optimal": [5, 2],
                                                      "source": {"ref": "[1]", "text": "t"}}]}}},
    {"version": "x", "crops": {"maize": {"factors": []}}},
])
def test_bad_rules_are_rejected(tmp_path, bad):
    import yaml
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(RulesError):
        load_rules(p)


def test_score_factor_open_bounds_and_missing():
    f = {"optimal": [1800, None], "tolerated": None}
    assert score_factor(2000, f) == 1.0 and score_factor(1500, f) == 0.0
    assert score_factor(float("nan"), f) is None and score_factor(None, f) is None
    assert score_factor(12, {"optimal": [18, 33], "tolerated": [10, 47]}) == 0.5


def test_explain_row_gives_reasons_in_contract_shape():
    ex = explain_row({"elevation_m": 2300, "tmean_point_c": 17.0, "tmin_point_c": 12.0, "precip_mm": 600}, load_rules())
    assert ex["irish_potato"]["score"] == 1.0
    assert ex["rice"]["score"] == 0.0 and ex["rice"]["reliability_cap"] == "low"
    for r in ex["beans"]["reasons"]:
        assert r["direction"] in {"supports", "against", "neutral"} and r["text"]


def test_prepare_applies_filters_and_drops_mixed_crops():
    df = prepare(raw_table())
    assert len(df) == 160 and df.attrs["rows_before_filters"] == 163
    assert set(df["label"]) <= {"maize", "beans", "irish_potato", "rice"}
    assert "tmean_c" in df and (df["elevation_m"] > 900).all()


def test_point_temperature_follows_the_lapse_rate():
    df = prepare(raw_table())
    a = df[df["District"] == "A"]
    hi, lo = a.loc[a["elevation_m"].idxmax()], a.loc[a["elevation_m"].idxmin()]
    expected = 0.0065 * (hi["elevation_m"] - lo["elevation_m"])
    assert abs((lo["tmean_point_c"] - hi["tmean_point_c"]) - expected) < 1e-9


@pytest.mark.parametrize("cols", [["nisr_yield_kg_ha"], ["Crop_plant"], ["district"], ["landcover"],
                                  ["Fert_type"], ["made_up_column"]])
def test_leaky_or_unknown_features_are_refused(cols):
    with pytest.raises(LeakageError):
        check_features(cols)


def test_lodo_never_tests_on_a_training_district_and_reports_all_models():
    df = prepare(raw_table())
    from sklearn.ensemble import RandomForestClassifier
    models = [MajorityBaseline(), ExpertRulesBaseline(),
              SklearnModel("rf", RandomForestClassifier(n_estimators=20, random_state=0), ["elevation_m"])]
    res = run_lodo(df, models)
    assert len(res) == 4 * 3 and set(res["held_out"]) == {"A", "B", "C", "D"}
    assert (res["n_train"] + res["n_test"] == len(df)).all()
    assert res[["top1", "top3", "macro_f1"]].apply(lambda s: s.between(0, 1)).all().all()
    s = summarise(res)
    assert set(s.index) == {"B0_majority", "B1_expert_rules", "rf"}


def test_expert_rules_rank_potato_high_and_beans_low_at_high_altitude():
    df = prepare(raw_table())
    m = ExpertRulesBaseline().fit(df, df["label"])
    high = df[df["District"] == "A"]
    order = np.argsort(-m.predict_proba(high), axis=1)
    top2 = [set(m.classes_[o[:2]]) for o in order]
    assert all("irish_potato" in t for t in top2)
    assert all(m.classes_[o[0]] != "beans" for o in order)


def test_more_evidence_breaks_a_tie():
    rules = {"version": "t", "crops": {
        "a": {"factors": [{"variable": "x", "optimal": [0, 10], "source": {"ref": "[t]", "text": "t"}}]},
        "b": {"factors": [{"variable": "x", "optimal": [0, 10], "source": {"ref": "[t]", "text": "t"}},
                          {"variable": "z", "optimal": [0, 10], "source": {"ref": "[t]", "text": "t"}}]}}}
    X = pd.DataFrame({"x": [5.0], "z": [5.0]})
    m = ExpertRulesBaseline(rules).fit(X, pd.Series(["a"]))
    m.classes_, m.prior_ = np.array(["a", "b"]), np.array([0.9, 0.1])
    assert m.classes_[np.argmax(m.predict_proba(X))] == "b"


def test_majority_baseline_is_constant():
    df = prepare(raw_table())
    p = MajorityBaseline().fit(df, df["label"]).predict_proba(df)
    assert np.allclose(p, p[0]) and np.isclose(p[0].sum(), 1.0)


def test_topk():
    classes = np.array(["a", "b", "c"])
    proba = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    assert topk(proba, classes, ["b", "a"], 1) == 0.0
    assert topk(proba, classes, ["b", "a"], 2) == 0.5


def test_unknown_label_without_rules_is_refused():
    df = pd.DataFrame({"elevation_m": [1500.0]})
    with pytest.raises(RulesError):
        ExpertRulesBaseline().fit(df, pd.Series(["cassava"]))


def test_prepare_handles_table_v2_column_names():
    raw = raw_table().rename(columns={"longitude": "longitude_x"})
    raw["GPS_Long"] = raw["longitude_x"]
    assert len(prepare(raw)) == 160


def test_district_crop_shares_are_allowed_but_point_crop_stats_are_not():
    assert check_features(["nisr_share_maize", "elevation_m"]) == ["nisr_share_maize", "elevation_m"]
    with pytest.raises(LeakageError):
        check_features(["nisr_yield_kg_ha"])


def test_default_table_prefers_v2_and_explains_when_missing(tmp_path, monkeypatch):
    import evaluate
    v2, v1 = tmp_path / "feature_table_v2.csv", tmp_path / "feature_table.csv"
    monkeypatch.setattr(evaluate, "TABLE_CANDIDATES", [v2, v1])
    with pytest.raises(FileNotFoundError, match="Part A"):
        evaluate.default_table()
    v1.write_text("x\n1\n")
    assert evaluate.default_table() == v1
    v2.write_text("x\n1\n")
    assert evaluate.default_table() == v2
