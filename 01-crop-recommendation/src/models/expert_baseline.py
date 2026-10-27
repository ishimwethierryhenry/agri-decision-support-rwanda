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


def explain_row(row: dict, rules: dict) -> dict[str, dict]:
    """Score every crop for one location and return scores with reasons in the API contract shape."""
    out = {}
    for crop, spec in rules["crops"].items():
        scores, reasons = [], []
        for f in spec["factors"]:
            x = row.get(f["variable"])
            s = score_factor(x, f)
            if s is None:
                reasons.append({"factor": f["variable"], "direction": "neutral", "value": None,
                                "expected_range": _fmt_range(f["optimal"]),
                                "text": f"{f['variable']} not available, so this factor was skipped."})
                continue
            scores.append(s)
            direction = "supports" if s == SCORE_OPTIMAL else ("neutral" if s == SCORE_TOLERATED else "against")
            reasons.append({"factor": f["variable"], "direction": direction, "value": float(x),
                            "expected_range": _fmt_range(f["optimal"]),
                            "text": f"{f['variable']} is {float(x):.1f}; expected {_fmt_range(f['optimal'])} "
                                    f"{f['source']['ref']}. {f['source']['text']}"})
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


class ExpertRulesBaseline:
    """B1: published ranges only.

    Ties are broken in this order: first the crop backed by more evaluated factors (more evidence),
    then the crop that is more common in the training districts. Neither changes a clear score difference.
    """
    name = "B1_expert_rules"
    EVIDENCE_WEIGHT = 1e-2
    PRIOR_WEIGHT = 1e-4

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
                s = ex[c]["score"]
                out[i, j] = ((0.0 if s is None else s) + self.EVIDENCE_WEIGHT * ex[c]["n_evaluated"]
                             + self.PRIOR_WEIGHT * self.prior_[j])
        return out
