"""Baselines for crop recommendation.

B0  MajorityBaseline : always ranks crops by how common they are in the training districts.
B1  ExpertRulesBaseline : scores each crop with published agronomic ranges (rules/crop_suitability.yaml).
     It is not trained. The training data is used only to break ties between crops with equal scores.

Both expose fit(df, y) and predict_proba(df) so the evaluation harness can treat them like any model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

RULES_PATH = Path(__file__).resolve().parent / "rules" / "crop_suitability.yaml"
SCORE_OPTIMAL, SCORE_TOLERATED, SCORE_OUTSIDE = 1.0, 0.5, 0.0

# Standard atmosphere lapse rate: air cools about 6.5 C per 1000 m of height (ICAO standard atmosphere).
LAPSE_RATE_C_PER_M = 0.0065


def adjust_to_height(temp_c, elevation_m, ref_elevation_m):
    """Move a temperature measured at ref_elevation_m to elevation_m. Works on numbers and pandas Series."""
    return temp_c - LAPSE_RATE_C_PER_M * (elevation_m - ref_elevation_m)


class RulesError(ValueError):
    pass


def load_rules(path: str | Path = RULES_PATH) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not data or "crops" not in data or "version" not in data:
        raise RulesError("rules file needs 'version' and 'crops'")
    for crop, spec in data["crops"].items():
        factors = spec.get("factors") or []
        if not factors:
            raise RulesError(f"{crop}: no factors")
        for f in factors:
            if not f.get("variable"):
                raise RulesError(f"{crop}: a factor has no variable")
            src = f.get("source") or {}
            if not src.get("ref") or not src.get("text"):
                raise RulesError(f"{crop}/{f['variable']}: every factor needs source.ref and source.text")
            for key in ("optimal", "tolerated"):
                rng = f.get(key)
                if rng is None:
                    continue
                if len(rng) != 2:
                    raise RulesError(f"{crop}/{f['variable']}: {key} must be [low, high]")
                lo, hi = rng
                if lo is not None and hi is not None and lo > hi:
                    raise RulesError(f"{crop}/{f['variable']}: {key} low is above high")
            if f.get("optimal") is None:
                raise RulesError(f"{crop}/{f['variable']}: optimal range is required")
    return data


def _inside(x: float, rng: Optional[list]) -> bool:
    if rng is None:
        return False
    lo, hi = rng
    return (lo is None or x >= lo) and (hi is None or x <= hi)


def _fmt_range(rng: Optional[list]) -> Optional[str]:
    if rng is None:
        return None
    lo, hi = rng
    if lo is None:
        return f"up to {hi}"
    if hi is None:
        return f"{lo} or more"
    return f"{lo} to {hi}"


def score_factor(x: Any, factor: dict) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if _inside(x, factor["optimal"]):
        return SCORE_OPTIMAL
    if _inside(x, factor.get("tolerated")):
        return SCORE_TOLERATED
    return SCORE_OUTSIDE


# Plain-language names for the rule variables, used in the reasons shown to extension officers.
VARIABLE_LABELS = {
    "elevation_m": ("Elevation", "m", 0),
    "precip_mm": ("Season rainfall", "mm", 0),
    "tmean_point_c": ("Mean temperature", "C", 1),
    "tmin_point_c": ("Minimum temperature", "C", 1),
}


def _label(variable: str) -> tuple[str, str, int]:
    return VARIABLE_LABELS.get(variable, (variable, "", 1))


def _range_text(rng: Optional[list], unit: str) -> Optional[str]:
    if rng is None:
        return None
    lo, hi = rng
    u = f" {unit}" if unit else ""
    if lo is None:
        return f"up to {hi}{u}"
    if hi is None:
        return f"{lo}{u} or more"
    return f"{lo} to {hi}{u}"


def explain_row(row: dict, rules: dict) -> dict[str, dict]:
    """Score every crop for one location and return scores with reasons in the API contract shape."""
    out = {}
    for crop, spec in rules["crops"].items():
        scores, reasons = [], []
        for f in spec["factors"]:
            name, unit, digits = _label(f["variable"])
            x = row.get(f["variable"])
            s = score_factor(x, f)
            best = _range_text(f["optimal"], unit)
            if s is None:
                reasons.append({"factor": f["variable"], "direction": "neutral", "value": None,
                                "expected_range": _fmt_range(f["optimal"]),
                                "text": f"{name} is not available, so this check was skipped."})
                continue
            scores.append(s)
            val = f"{float(x):.{digits}f} {unit}".strip()
            if s == SCORE_OPTIMAL:
                direction, verdict = "supports", f"inside the expected range ({best})"
            elif s == SCORE_TOLERATED:
                direction = "neutral"
                verdict = (f"outside the best range ({best}) but within what the crop tolerates "
                           f"({_range_text(f.get('tolerated'), unit)})")
            else:
                direction, verdict = "against", f"outside the expected range ({best})"
            reasons.append({"factor": f["variable"], "direction": direction, "value": float(x),
                            "expected_range": _fmt_range(f["optimal"]),
                            "text": f"{name} is {val}, {verdict} {f['source']['ref']}. {f['source']['text']}"})
        if spec.get("note"):
            reasons.append({"factor": spec.get("note_factor", "note"), "direction": "neutral", "value": None,
                            "expected_range": None, "text": spec["note"]})
        out[crop] = {"score": float(np.mean(scores)) if scores else None, "n_evaluated": len(scores),
                     "reasons": reasons, "reliability_cap": spec.get("reliability_cap")}
    return out


class MajorityBaseline:
    """B0: rank crops by their share in the training data, the same for every location."""
    name = "B0_majority"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityBaseline":
        self.classes_ = np.array(sorted(y.unique()))
        counts = y.value_counts()
        self.prior_ = np.array([counts.get(c, 0) / len(y) for c in self.classes_])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.prior_, (len(X), 1))


# Tie-break weights. They only matter when two crops have the same rule score:
# first the crop backed by more evaluated factors wins, then the crop more common in the training data.
EVIDENCE_WEIGHT = 1e-2
PRIOR_WEIGHT = 1e-4


def ranking_key(score: Optional[float], n_evaluated: int, prior: float) -> float:
    return (0.0 if score is None else score) + EVIDENCE_WEIGHT * n_evaluated + PRIOR_WEIGHT * prior


def rank_crops(row: dict, rules: dict, prior: dict[str, float]) -> list[dict]:
    """Score and rank every crop in the rules for one location. Best first.

    Each item: crop, score (0 to 1, or None if no factor could be evaluated), n_evaluated, reasons,
    reliability_cap. The evaluation harness and the service both use this, so they cannot drift apart.
    """
    ex = explain_row(row, rules)
    items = [{"crop": c, **v} for c, v in ex.items()]
    items.sort(key=lambda d: ranking_key(d["score"], d["n_evaluated"], prior.get(d["crop"], 0.0)), reverse=True)
    return items


class ExpertRulesBaseline:
    """B1: published ranges only.

    Ties are broken in this order: first the crop backed by more evaluated factors (more evidence),
    then the crop that is more common in the training districts. Neither changes a clear score difference.
    """
    name = "B1_expert_rules"
    EVIDENCE_WEIGHT = EVIDENCE_WEIGHT
    PRIOR_WEIGHT = PRIOR_WEIGHT

    def __init__(self, rules: Optional[dict] = None):
        self.rules = rules or load_rules()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExpertRulesBaseline":
        unknown = set(y.unique()) - set(self.rules["crops"])
        if unknown:
            raise RulesError(f"labels without rules: {sorted(unknown)}")
        self.classes_ = np.array(sorted(y.unique()))
        counts = y.value_counts()
        self.prior_ = np.array([counts.get(c, 0) / len(y) for c in self.classes_])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rows = X.to_dict("records")
        out = np.zeros((len(rows), len(self.classes_)))
        for i, row in enumerate(rows):
            ex = explain_row(row, self.rules)
            for j, c in enumerate(self.classes_):
                out[i, j] = ranking_key(ex[c]["score"], ex[c]["n_evaluated"], self.prior_[j])
        return out
