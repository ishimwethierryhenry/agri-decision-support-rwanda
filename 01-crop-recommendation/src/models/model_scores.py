"""Scores of every model, read from the official evaluation. Used by the dashboard and the notebooks.

The default is reports/eval_v2 (protocol 1.1, 18 Nov 2026): the 12 combinations of the midterm evaluation
(reports/eval_v1, protocol 1.0) with exactly the same numbers, plus 6 combinations with the satellite inputs.
Everything is recomputed from the committed result files, so the pages always show the official numbers:
  - the main score of each model, overall and per district (protocol, rule 5);
  - overall scores over all test fields: accuracy, balanced accuracy, macro-F1, weighted F1, Cohen's kappa;
  - precision, recall and F1 of each crop in each district, from the confusion counts.

Only pandas and numpy are needed, so it also runs in the Colab kernel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[2]          # 01-crop-recommendation
EVAL_V1 = MODULE / "reports" / "eval_v1"            # midterm, protocol 1.0 (12 combinations)
EVAL_V2 = MODULE / "reports" / "eval_v2"            # protocol 1.1 (18 combinations, satellite inputs added)
EVAL_DIR = EVAL_V2
TUNING_V1 = MODULE / "reports" / "tuning_v1"        # protocol 1.2: the same models with tuned settings
TUNING_V2 = MODULE / "reports" / "tuning_v2"        # protocol 1.3: second round of tuning
TUNED = " tuned"
CROPS = ["beans", "irish_potato", "maize", "rice"]
CROP_NAMES = {"beans": "Beans", "irish_potato": "Irish potato", "maize": "Maize", "rice": "Rice"}
DISTRICTS = ["Musanze", "Nyabihu", "Nyagatare", "Ruhango"]
MIN_CROP_FIELDS = 5                                    # protocol, rule 5: crops with fewer fields are not scored
BASELINE = "B1_expert_rules"
# test name used here: (name in confusion_counts.csv, file prefix)
TESTS = {"new_district": ("leave_one_district_out", "lodo"), "new_fields": ("new_fields_known_districts", "blockcv")}
TEST_NAMES = {"new_district": "A district the model never saw (main test)",
              "new_fields": "New fields in the four known districts"}
MODEL_ORDER = ["B0_majority", "B1_expert_rules",
               "LR[district]", "LR[field]", "LR[field_sat]", "LR[all]", "LR[all_sat]",
               "RF[district]", "RF[field]", "RF[field_sat]", "RF[all]", "RF[all_sat]", "RF[all_subsoil]",
               "XGB[district]", "XGB[field]", "XGB[field_sat]", "XGB[all]", "XGB[all_sat]"]
_ALGO = {"LR": "Logistic Regression", "RF": "Random Forest", "ET": "Extra Trees", "XGB": "XGBoost",
         "Blend": "Blend of the three"}
_FIXED = {"B0_majority": ("B0, most common crop", "none"), "B1_expert_rules": ("B1, expert rules", "rules")}
_VIEWS = {"district": "district (12)", "field": "field (10)", "field_sat": "field + satellite (14)",
          "all": "all (21)", "all_sat": "all + satellite (25)", "all_subsoil": "all + subsoil (30)",
          "field_sat_near": "field + satellite + nearby crops (19)", "all_sat_near": "all + satellite + nearby crops (30)"}
# the same inputs without and with the four satellite inputs of protocol 1.1
SATELLITE_PAIRS = {"field": "field_sat", "all": "all_sat"}


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

    def satellite_effect(self, test: str) -> pd.DataFrame:
        """Main score of each method on the same inputs without and with the four satellite inputs (protocol 1.1).
        Scores are rounded to 3 decimals, as shown, before the change is taken, so the table adds up on screen.
        Empty for an evaluation without them (eval_v1)."""
        s = self.summary[test].set_index("model")["macro_f1_mean"].round(3)
        rows = []
        for algo, name in _ALGO.items():
            row = {}
            for base, sat in SATELLITE_PAIRS.items():
                a, b = f"{algo}[{base}]", f"{algo}[{sat}]"
                if a not in s.index or b not in s.index:
                    continue
                label = _VIEWS[base].split(" (")[0].capitalize()
                row.update({label: s[a], f"{label} + satellite": s[b],
                            f"Change ({label.lower()})": round(s[b] - s[a], 3)})
            if row:
                rows.append({"Model": name, **row})
        return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()

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


class Tuning:
    """The tuning run (protocol 1.2, reports/tuning_v1): every method with the protocol settings and with tuned
    settings on both tests, the model the map serves (decision 2), how often it is right at each level of its own
    probability, simple alternatives on the same folds, and how mixed the 2 km blocks are."""

    def __init__(self, folder: Path = TUNING_V1):
        folder = Path(folder)
        self.summary = {t: pd.read_csv(folder / f"main_{p}_summary.csv").set_index("model")
                        for t, (_, p) in TESTS.items()}
        self.results = {t: pd.read_csv(folder / f"main_{p}_results.csv") for t, (_, p) in TESTS.items()}
        self.sure = pd.read_csv(folder / "answer_when_sure.csv")
        self.bands = pd.read_csv(folder / "confidence_bands.csv")
        self.mixed = pd.read_csv(folder / "mixed_land.csv").set_index("district")
        self.info = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
        self.decisions = self.info["decisions"]
        self.map_model = self.decisions["map"]["serve"]

    def gains(self, test: str) -> pd.DataFrame:
        """One row per tuned model: main score with the protocol settings and tuned, the change, and the tuned
        model's districts ahead of the rules and share of fields with the right first crop."""
        s = self.summary[test]
        rows = []
        for m in MODEL_ORDER:
            if m + TUNED not in s.index:
                continue
            name, inputs = model_label(m)
            u, t = s.loc[m], s.loc[m + TUNED]
            wins = t["districts_beating_B1"]
            rows.append({"model": m + TUNED, "Model": name, "Inputs": inputs,
                         "Protocol settings": u["macro_f1_mean"], "Tuned": t["macro_f1_mean"],
                         "Change": round(t["macro_f1_mean"], 3) - round(u["macro_f1_mean"], 3),
                         "Better than the rules in (tuned)": "" if pd.isna(wins) else f"{int(wins)} of 4 districts",
                         "First crop right, all fields (tuned)": t["top1_all_fields"]})
        return pd.DataFrame(rows)

    def answered_by(self, margin: float = 0.01) -> dict:
        """In each district, 'model' if the model on the map named the right crop first more often than the rules
        on new fields, by more than the margin (the choice of map_model.method_for since 25 Nov 2026), otherwise
        'rules'."""
        r = self.results["new_fields"].pivot(index="model", columns="district", values="top1")
        return {d: "model" if r.loc[self.map_model, d] > r.loc[BASELINE, d] + margin else "rules" for d in DISTRICTS}

    def when_sure(self, model: Optional[str] = None, district: str = "all") -> pd.DataFrame:
        """If the model answered only from a probability for its first crop: share of fields answered, how often
        right (new fields)."""
        t = self.sure[(self.sure["model"] == (model or self.map_model)) & (self.sure["district"] == district)]
        return t.rename(columns={"probability_at_least": "Answers only from", "share_answered": "Fields answered",
                                 "first_crop_right": "First crop right"})[
            ["Answers only from", "Fields answered", "First crop right"]].reset_index(drop=True)

    def alternatives(self) -> pd.DataFrame:
        """Simple choices on the same folds of the new-fields test, next to the model on the map before and after."""
        s = self.summary["new_fields"]
        before = self.decisions["map"]["today"]
        names = {"B0_majority": "Most common crop of all training fields (B0)",
                 "B0d_district_majority": "Most common crop of the field's district",
                 BASELINE: "Expert rules (B1)",
                 before: f"{short_label(before)} (map before 23 Nov)"}
        if self.map_model != before:
            names[self.map_model] = f"{short_label(self.map_model.replace(TUNED, ''))}, tuned (map now)"
        return pd.DataFrame([{"Choice": label, "Main score": s.loc[m, "macro_f1_mean"],
                              "First crop right, all fields": s.loc[m, "top1_all_fields"]}
                             for m, label in names.items() if m in s.index])


def second_round(folder: Path = TUNING_V2) -> Optional[dict]:
    """The second round of tuning (protocol 1.3, reports/tuning_v2), or None if its results are not there: the best
    tuned model or blend on new fields and its main score, the model on the map before the round and its score, the
    margin of decision 2, whether the map changed, and how many settings each method tried."""
    folder = Path(folder)
    if not (folder / "run_info.json").exists():
        return None
    info = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
    cands = json.loads((folder / "candidates.json").read_text(encoding="utf-8"))
    d = info["decisions"]
    return {"best": d["new_fields"]["best"], "best_score": d["new_fields"]["best_score"],
            "map_model": d["map"]["today"], "map_score": d["map"]["today_score"], "margin": d["map"]["margin"],
            "changes": d["map"]["changes"], "serve": d["map"]["serve"],
            "settings": {m: len(c) for m, c in cands.items()}, "run_at_utc": info["run_at_utc"]}
