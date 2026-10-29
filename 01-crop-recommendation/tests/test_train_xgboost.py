"""Tests for the XGBoost training and scores script (src/models/train_xgboost.py).

The quick tests use 10 trees. The last test uses the protocol's 300 trees and checks the per-district scores
against the committed eval_v1, but only with the pinned library versions (requirements-eval.txt).
Run from the repo root:  python -m pytest 01-crop-recommendation/tests/test_train_xgboost.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-crop-recommendation" / "src" / "models"))
pytest.importorskip("xgboost")
import evaluate as ev  # noqa: E402
import train_xgboost as tx  # noqa: E402

TABLE = ROOT / "01-crop-recommendation" / "data" / "processed" / "feature_table_v2.csv"


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(TABLE, low_memory=False)


@pytest.fixture(scope="module")
def quick(raw):
    return tx.train(raw, n_estimators=10, log=lambda *a: None)


def test_models_come_from_the_protocol_list():
    models = tx.pick_models(n_estimators=10)
    assert [m.name for m in models] == tx.MODELS
    xgb = {m.name: m for m in models}["XGB[all]"]
    assert xgb.features == ev.VIEWS["all"] and xgb.balance
    assert xgb.estimator.get_params()["max_depth"] == 4 and xgb.estimator.get_params()["learning_rate"] == 0.05


def test_every_field_is_predicted_once_per_test(quick):
    cf = quick["confusion"]
    per = cf.groupby(["test", "model"])["count"].sum()
    assert set(per) == {1904} and len(per) == 2 * len(tx.MODELS)
    bd = quick["by_district"]
    assert len(bd) == 2 * len(tx.MODELS) * 4
    assert (bd.groupby(["test", "model"])["n_test"].sum() == 1904).all()


def test_scores_are_in_range_and_main_score_is_the_mean_of_districts(quick):
    ov = quick["overall"]
    for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3", "main_score"]:
        assert ov[c].between(0, 1).all(), c
    bd = quick["by_district"]
    mean = bd.groupby(["test", "model"])["macro_f1"].mean()
    got = ov.set_index(["test", "model"])["main_score"]
    assert np.allclose(got.sort_index(), mean.sort_index())


def test_per_crop_scores_match_the_confusion_matrix(quick):
    bc, cf = quick["by_crop"], quick["confusion"]
    for (test, model), g in bc.groupby(["test", "model"]):
        c = cf[(cf["test"] == test) & (cf["model"] == model)]
        for row in g.itertuples():
            tp = c[(c["true"] == row.crop) & (c["predicted"] == row.crop)]["count"].sum()
            support = c[c["true"] == row.crop]["count"].sum()
            assert row.fields == support
            assert abs(row.recall - (tp / support if support else 0)) < 1e-12


def test_baselines_match_eval_v1_whatever_the_number_of_trees(quick):
    chk = quick["check"].set_index(["test", "model"])
    for test in tx.TESTS:
        for model in ("B0_majority", "B1_expert_rules"):
            assert chk.loc[(test, model), "same_as_eval_v1"] == "yes"


def test_final_model_and_feature_importance(quick):
    final = quick["final_model"]
    assert final.name == "XGB[all]" and final.used_ == ev.VIEWS["all"]
    imp = quick["importance"]
    assert abs(imp["share"].sum() - 1) < 1e-9 and set(imp["feature"]) <= set(final.used_)
    assert set(imp["level"]) <= {"district", "field"}
    assert set(imp.loc[imp["level"] == "district", "feature"]) <= set(tx.DISTRICT_LEVEL)


def test_report_files(quick, tmp_path):
    info = {"finished_utc": "2026-10-29 00:00", "git_commit": "test", "table": "t.csv", "table_sha256": "0" * 64,
            "training_fields": 1904, "versions": tx._versions()}
    path = tx.write_report(tmp_path, quick, info)
    for f in ["by_district.csv", "summary.csv", "overall.csv", "by_crop.csv", "confusion.csv", "check.csv",
              "feature_importance.csv", "run_info.json", "model/xgboost_all_inputs.json", "model/meta.json"]:
        assert (tmp_path / f).exists(), f
    text = path.read_text(encoding="utf-8")
    assert "## 1. Main score per model" in text and "## 7. Same as eval_v1?" in text
    meta = json.loads((tmp_path / "model" / "meta.json").read_text(encoding="utf-8"))
    assert meta["trained_on_fields"] == 1904 and meta["features"] == ev.VIEWS["all"]


def _pinned() -> bool:
    import sklearn
    import xgboost
    return (xgboost.__version__, sklearn.__version__, pd.__version__, np.__version__) == \
        ("3.4.1", "1.7.2", "3.1.0", "2.3.5")


@pytest.mark.skipif(not _pinned(), reason="needs the pinned versions of requirements-eval.txt")
def test_full_run_gives_the_eval_v1_scores(raw):
    res = tx.train(raw, log=lambda *a: None)
    assert (res["check"]["same_as_eval_v1"] == "yes").all()
    s = res["summary"].set_index(["test", "model"])
    assert round(s.loc[("new_fields", "XGB[all]"), "macro_f1_mean"], 3) == 0.451
    assert round(s.loc[("new_district", "XGB[all]"), "macro_f1_mean"], 3) == 0.294
    assert (res["summary"]["passes_rule"] != "yes").all()
