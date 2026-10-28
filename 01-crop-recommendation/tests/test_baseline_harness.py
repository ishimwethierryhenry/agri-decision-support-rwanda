"""Tests for the baselines and the evaluation harness. Synthetic data only, no real farm data needed."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
from evaluate import (ALLOWED_FEATURES, CROPS, SOIL_SUB, SOIL_TOP, LeakageError, SklearnModel,  # noqa: E402
                      block_folds, check_features, file_sha256, prepare, run_block_cv, run_lodo, scores,
                      summarise, topk)
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


def table_v2(n_per=40, seed=0):
    """Shaped like Part A's current table: qc_ok and block_id instead of coordinates, Status, soil, NISR shares."""
    rng = np.random.default_rng(seed)
    df = raw_table(n_per, seed).drop(columns=["longitude", "nisr_yield_kg_ha"])
    df["qc_ok"] = df["GPS_Elev"] > 900
    df.loc[len(df) - 2, "qc_ok"] = False                 # the field outside Rwanda
    df["Status"] = np.where(np.arange(len(df)) % 3 == 0, "Used", "Not Used")
    df["block_id"] = [f"{d}_{i // 4}" for d, i in zip(df["District"], range(len(df)))]
    for c in ["nisr_share_beans", "nisr_share_irish_potato", "nisr_share_maize", "nisr_share_paddy_rice"]:
        df[c] = df["District"].map({"A": 0.2, "B": 0.5, "C": 0.3, "D": 0.1})
    for c in SOIL_TOP + SOIL_SUB:
        df[c] = rng.normal(10, 2, len(df)) + (df["District"] == "A") * 3
    df["F_id"] = np.arange(len(df)) + 1000
    return df


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


def test_prepare_uses_qc_ok_when_coordinates_are_gone():
    raw = table_v2()
    assert "longitude" not in raw.columns
    df = prepare(raw)
    assert len(df) == 160 and df.attrs["rows_after_qc"] == 161     # 163 rows, 2 fail qc_ok, 1 mixed crop
    raw.loc[0, "qc_ok"] = False                                     # qc_ok wins even when the elevation looks fine
    assert len(prepare(raw)) == 159
    raw["qc_ok"] = raw["qc_ok"].map({True: "True", False: "False"})  # text values after a CSV round trip
    assert len(prepare(raw)) == 159


def test_status_filter_keeps_only_used_and_keeps_the_same_reference_height():
    raw = table_v2()
    main, used = prepare(raw), prepare(raw, status="Used")
    assert len(used) < len(main) and used.attrs["status_filter"] == "Used"
    merged = used.merge(main, on="F_id", suffixes=("_u", "_m"))
    assert len(merged) == len(used)
    assert np.allclose(merged["tmean_point_c_u"], merged["tmean_point_c_m"])


def test_point_temperature_follows_the_lapse_rate():
    df = prepare(raw_table())
    a = df[df["District"] == "A"]
    hi, lo = a.loc[a["elevation_m"].idxmax()], a.loc[a["elevation_m"].idxmin()]
    expected = 0.0065 * (hi["elevation_m"] - lo["elevation_m"])
    assert abs((lo["tmean_point_c"] - hi["tmean_point_c"]) - expected) < 1e-9


@pytest.mark.parametrize("cols", [["nisr_yield_kg_ha"], ["Crop_plant"], ["district"], ["landcover"],
                                  ["Fert_type"], ["made_up_column"], ["Status"], ["qc_ok"], ["block_id"],
                                  ["F_id"], ["F_index"], ["Water_sour"]])
def test_leaky_or_unknown_features_are_refused(cols):
    with pytest.raises(LeakageError):
        check_features(cols)


def test_every_view_feature_is_allowed():
    assert check_features(ALLOWED_FEATURES) == ALLOWED_FEATURES
    assert all(c in ALLOWED_FEATURES for c in SOIL_TOP + SOIL_SUB + ["elevation_m", "nisr_share_maize"])


def test_lodo_never_tests_on_a_training_district_and_reports_all_models():
    df = prepare(raw_table())
    from sklearn.ensemble import RandomForestClassifier
    models = [MajorityBaseline(), ExpertRulesBaseline(),
              SklearnModel("rf", RandomForestClassifier(n_estimators=20, random_state=0), ["elevation_m"])]
    res, conf = run_lodo(df, models)
    assert len(res) == 4 * 3 and set(res["district"]) == {"A", "B", "C", "D"}
    assert (res["n_train"] + res["n_test"] == len(df)).all()
    assert res[["top1", "top3", "macro_f1", "macro_f1_all4"]].apply(lambda s: s.between(0, 1)).all().all()
    assert conf.groupby(["model", "district"])["count"].sum().eq(res.set_index(["model", "district"])["n_test"]).all()
    s = summarise(res)
    assert set(s.index) == {"B0_majority", "B1_expert_rules", "rf"}
    assert s.loc["B1_expert_rules", "passes_rule"] == "" and s.loc["rf", "passes_rule"] in {"yes", "no"}


def test_main_score_skips_crops_with_fewer_than_five_fields():
    y = np.array(["maize"] * 10 + ["beans"] * 10 + ["rice"])
    pred = y.copy()
    pred[-1] = "maize"                       # the single rice field is missed
    proba = np.zeros((len(y), 4))
    out = scores(y, pred, proba, np.array(CROPS))
    assert out["crops_scored"] == "beans+maize"
    assert out["macro_f1"] > 0.97                                 # rice (1 field) does not count
    assert out["macro_f1_all4"] < 0.5                             # old definition: rice and potato count as zero
    assert np.isnan(out["recall_irish_potato"]) and out["recall_rice"] == 0.0


def _fake_results(b1, other):
    rows = [{"model": "B1_expert_rules", "district": d, "macro_f1": v, "macro_f1_all4": v, "top1": v, "top3": 1.0}
            for d, v in zip("ABCD", b1)]
    for name, vals in other.items():
        rows += [{"model": name, "district": d, "macro_f1": v, "macro_f1_all4": v, "top1": v, "top3": 1.0}
                 for d, v in zip("ABCD", vals)]
    return pd.DataFrame(rows)


def test_adoption_rule_needs_a_higher_mean_and_three_clear_wins():
    b1 = [0.60, 0.60, 0.10, 0.20]
    s = summarise(_fake_results(b1, {
        "wins_three": [0.65, 0.62, 0.30, 0.10],     # mean higher, 3 wins by more than 0.01
        "tiny_win": [0.605, 0.62, 0.30, 0.10],      # one 'win' is only 0.005, so 2 wins
        "mean_lower": [0.62, 0.62, 0.12, 0.00],     # 3 wins but a lower mean
        "two_big_wins": [0.40, 0.40, 0.60, 0.60],   # higher mean, only 2 wins
    }))
    assert s.loc["wins_three", "passes_rule"] == "yes"
    assert s.loc["tiny_win", "passes_rule"] == "no" and s.loc["tiny_win", "districts_beating_B1"] == 2
    assert s.loc["mean_lower", "passes_rule"] == "no" and s.loc["mean_lower", "districts_beating_B1"] == 3
    assert s.loc["two_big_wins", "passes_rule"] == "no"


def test_only_the_main_test_can_adopt():
    from evaluate import _decision
    s = summarise(_fake_results([0.60, 0.60, 0.10, 0.20], {"wins_three": [0.65, 0.62, 0.30, 0.10]}))
    assert _decision(s, deciding=True).startswith("Adopted: wins_three")
    assert _decision(s, deciding=False).startswith("For information only")
    s = summarise(_fake_results([0.60, 0.60, 0.10, 0.20], {"loses": [0.50, 0.50, 0.20, 0.20]}))
    assert _decision(s, deciding=True).startswith("No ML model passes the adoption rule, so B1 stays")


def test_block_folds_never_split_a_block_and_cover_every_field():
    df = prepare(table_v2())
    folds = block_folds(df)
    assert len(folds) == 5
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen) == list(range(len(df)))
    for tr, te in folds:
        assert set(df.loc[tr, "block_id"]).isdisjoint(set(df.loc[te, "block_id"]))


def test_block_cv_predicts_every_field_once():
    df = prepare(table_v2())
    from sklearn.ensemble import RandomForestClassifier
    models = [ExpertRulesBaseline(),
              SklearnModel("rf", RandomForestClassifier(n_estimators=20, random_state=0), ["elevation_m"])]
    res, conf = run_block_cv(df, models)
    assert len(res) == 2 * 4
    assert res.groupby("model")["n_test"].sum().eq(len(df)).all()
    assert conf.groupby("model")["count"].sum().eq(len(df)).all()


def test_block_cv_needs_block_id():
    df = prepare(raw_table())
    with pytest.raises(KeyError, match="block_id"):
        block_folds(df)


def test_balanced_weights_reach_the_estimator():
    class Spy:
        def fit(self, X, y, sample_weight=None):
            self.w, self.classes_ = sample_weight, np.unique(y)
            return self

        def predict_proba(self, X):
            return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))

    X = pd.DataFrame({"elevation_m": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series(["maize", "maize", "maize", "rice"])
    m = SklearnModel("spy", Spy(), ["elevation_m"], balance=True).fit(X, y)
    assert np.allclose(m.estimator.w, [4 / 6, 4 / 6, 4 / 6, 2.0])


def test_file_fingerprint_matches_sha256_and_ignores_windows_line_endings(tmp_path):
    unix, windows = tmp_path / "unix.csv", tmp_path / "windows.csv"
    unix.write_bytes(b"a,b\n1,2\n")
    windows.write_bytes(b"a,b\r\n1,2\r\n")
    assert file_sha256(unix) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert file_sha256(windows) == file_sha256(unix)


def test_full_run_writes_report_and_run_info(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    import evaluate
    full = evaluate.default_models
    monkeypatch.setattr(evaluate, "default_models", lambda: full(n_estimators=10))  # same pipeline, faster
    table, out = tmp_path / "feature_table_v2.csv", tmp_path / "eval"
    table_v2(n_per=60).to_csv(table, index=False)
    assert evaluate.main(["--table", str(table), "--out", str(out)]) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Main test: leave one district out" in report and "Second check" in report
    import json
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["table_sha256"] == file_sha256(table) and info["rows"]["main"] > info["rows"]["used_only"]
    for name in ("main_lodo_results.csv", "main_blockcv_summary.csv", "used_only_lodo_summary.csv",
                 "confusion_counts.csv"):
        assert (out / name).exists()


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


def test_prepare_handles_older_column_names():
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
