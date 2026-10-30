"""Scores of every model, read from the official evaluation (reports/eval_v1). Used by the dashboard and the notebooks.

Everything is recomputed from the committed result files, so the pages always show the official numbers:
  - the main score of each model, overall and per district (protocol 1.0, rule 5);
  - overall scores over all test fields: accuracy, balanced accuracy, macro-F1, weighted F1, Cohen's kappa;
  - precision, recall and F1 of each crop in each district, from the confusion counts.

Only pandas and numpy are needed, so it also runs in the Colab kernel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[2]          # 01-crop-recommendation
EVAL_DIR = MODULE / "reports" / "eval_v1"
CROPS = ["beans", "irish_potato", "maize", "rice"]
CROP_NAMES = {"beans": "Beans", "irish_potato": "Irish potato", "maize": "Maize", "rice": "Rice"}
DISTRICTS = ["Musanze", "Nyabihu", "Nyagatare", "Ruhango"]
MIN_CROP_FIELDS = 5                                    # protocol, rule 5: crops with fewer fields are not scored
BASELINE = "B1_expert_rules"
# test name used here: (name in confusion_counts.csv, file prefix)
TESTS = {"new_district": ("leave_one_district_out", "lodo"), "new_fields": ("new_fields_known_districts", "blockcv")}
TEST_NAMES = {"new_district": "A district the model never saw (main test)",
              "new_fields": "New fields in the four known districts"}
MODEL_ORDER = ["B0_majority", "B1_expert_rules", "LR[district]", "LR[field]", "LR[all]", "RF[district]",
               "RF[field]", "RF[all]", "RF[all_subsoil]", "XGB[district]", "XGB[field]", "XGB[all]"]
_ALGO = {"LR": "Logistic Regression", "RF": "Random Forest", "XGB": "XGBoost"}
_FIXED = {"B0_majority": ("B0, most common crop", "none"), "B1_expert_rules": ("B1, expert rules", "rules")}
_VIEWS = {"district": "district (12)", "field": "field (10)", "all": "all (21)", "all_subsoil": "all + subsoil (30)"}


def model_label(model: str) -> tuple[str, str]:
    """('XGBoost', 'all (21)') for 'XGB[all]'."""
    if model in _FIXED:
        return _FIXED[model]
    algo, _, view = model.partition("[")
    return _ALGO.get(algo, algo), _VIEWS.get(view.rstrip("]"), view.rstrip("]"))


def short_label(model: str) -> str:
    name, inputs = model_label(model)
    return name if model in _FIXED else f"{name}, {inputs.split(' (')[0]} inputs"


def _crop_scores(cm: np.ndarray) -> dict:
    """Per-crop precision, recall, F1 and number of fields from a 4 x 4 confusion matrix (rows: true crop)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fields = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(fields > 0, tp / fields, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"fields": fields, "precision": precision, "recall": recall, "f1": f1}


def _overall(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    s = _crop_scores(cm)
    accuracy = np.diag(cm).sum() / n
    expected = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / n ** 2
    present = s["fields"] > 0
    return {"accuracy": accuracy, "balanced_accuracy": float(s["recall"][present].mean()),
            "macro_f1": float(s["f1"].mean()), "weighted_f1": float((s["f1"] * s["fields"]).sum() / n),
            "cohen_kappa": float((accuracy - expected) / (1 - expected))}


class Scores:
    """The evaluation of one analysis: 'main' (all 1,904 fields) or 'used_only' (the 650 CIAT 'Used' fields)."""

    def __init__(self, folder: Path = EVAL_DIR, analysis: str = "main"):
        folder = Path(folder)
        self.analysis = analysis
        self.summary, self.results = {}, {}
        for test, (_, prefix) in TESTS.items():
            self.summary[test] = pd.read_csv(folder / f"{analysis}_{prefix}_summary.csv")
            self.results[test] = pd.read_csv(folder / f"{analysis}_{prefix}_results.csv")
        conf = pd.read_csv(folder / "confusion_counts.csv")
        conf = conf[conf["analysis"] == analysis].copy()
        conf["test"] = conf["test"].map({v[0]: k for k, v in TESTS.items()})
        self.confusion = conf
        self.models = [m for m in MODEL_ORDER if m in set(self.summary["new_district"]["model"])]

    # ------------------------------------------------------------------ matrices
    def matrix(self, test: str, model: str, district: Optional[str] = None) -> pd.DataFrame:
        c = self.confusion[(self.confusion["test"] == test) & (self.confusion["model"] == model)]
        if district is not None:
            c = c[c["district"] == district]
        m = c.pivot_table(index="true", columns="predicted", values="count", aggfunc="sum")
        return m.reindex(index=CROPS, columns=CROPS).fillna(0).astype(int)

    # ------------------------------------------------------------------ tables
    def main_scores(self, test: str) -> pd.DataFrame:
        """One row per model: main score, worst district, accuracy and top-3 (means over the four districts),
        number of districts where the model beats B1 by more than 0.01, and the adoption decision."""
        s = self.summary[test].set_index("model")
        rows = []
        for m in self.models:
            name, inputs = model_label(m)
            wins = s.loc[m, "districts_beating_B1"]
            rows.append({"model": m, "Model": name, "Inputs": inputs, "Main score": s.loc[m, "macro_f1_mean"],
                         "Worst district": s.loc[m, "worst_district_macro_f1"],
                         "Accuracy (mean)": s.loc[m, "top1_mean"], "Top-3 (mean)": s.loc[m, "top3_mean"],
                         "Beats B1 in": "" if pd.isna(wins) else f"{int(wins)} of 4 districts",
                         "Passes rule": "" if pd.isna(s.loc[m, "passes_rule"]) else s.loc[m, "passes_rule"]})
        return pd.DataFrame(rows)

    def per_district(self, test: str) -> pd.DataFrame:
        """Main score (macro-F1 over the crops with at least 5 fields) of every model in every district."""
        r = self.results[test].pivot(index="model", columns="district", values="macro_f1")
        r = r.reindex(index=self.models, columns=DISTRICTS)
        r.index = [short_label(m) for m in r.index]
        r.index.name = "Model"
        return r

    def overall(self, test: str) -> pd.DataFrame:
        """Scores over all test fields together (pooled over the four districts)."""
        rows = []
        for m in self.models:
            name, inputs = model_label(m)
            rows.append({"model": m, "Model": name, "Inputs": inputs, **_overall(self.matrix(test, m).values)})
        return pd.DataFrame(rows).rename(columns={"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
                                                  "macro_f1": "Macro-F1", "weighted_f1": "Weighted F1",
                                                  "cohen_kappa": "Cohen's kappa"})

    def per_crop(self, test: str, model: str, districts: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Precision, recall and F1 of each crop, per district (or over all districts with districts=['All'])."""
        rows = []
        for d in (districts or DISTRICTS):
            cm = self.matrix(test, model, None if d == "All" else d).values
            s = _crop_scores(cm)
            for i, c in enumerate(CROPS):
                n = int(s["fields"][i])
                rows.append({"district": d, "crop": c, "fields": n, "precision": float(s["precision"][i]),
                             "recall": float(s["recall"][i]), "f1": float(s["f1"][i]),
                             "scored": n >= MIN_CROP_FIELDS})
        return pd.DataFrame(rows)

    def protocol_main_score(self, test: str, model: str, district: str) -> float:
        """The protocol's main score recomputed from the confusion counts (mean F1 over the scored crops)."""
        t = self.per_crop(test, model, [district])
        t = t[t["scored"]]
        return float(t["f1"].mean()) if len(t) else float("nan")

    def crop_table(self, test: str, district: str, models: tuple = ("XGB[all]", BASELINE)) -> pd.DataFrame:
        """Side by side table for one district: fields, then precision, recall and F1 for each model."""
        base = None
        for m in models:
            t = self.per_crop(test, m, [district]).set_index("crop")
            label = "XGBoost" if m == "XGB[all]" else "Rules" if m == BASELINE else short_label(m)
            part = t[["precision", "recall", "f1"]].rename(columns=lambda c: f"{label} {c if c != 'f1' else 'F1'}")
            part[~t["scored"]] = np.nan
            base = t[["fields"]].join(part) if base is None else base.join(part)
        base.index = [CROP_NAMES[c] for c in base.index]
        base.index.name = "Crop"
        return base.rename(columns={"fields": "Fields"})
