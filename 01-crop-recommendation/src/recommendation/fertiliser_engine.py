"""Fertiliser rule engine v0.

Rules live in data (YAML), not in code. The engine only loads, validates and matches them.
Design rules:
  1. Every rule MUST have a real source (document name at least). A rule without one is rejected at load time.
  2. A rule marked status "draft" is never served unless the caller asks for drafts explicitly.
  3. If a condition needs a soil value that was not given, the rule does not match. We never guess.
  4. If no rule matches, the answer is status "no_matching_rule". The engine never invents advice.
  5. RAG, if used later, may only write the explanation text. It cannot add or change an item.

The output is a plain dict that matches FertiliserAdvice in 03-shared/apis/recommend_schemas.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ENGINE_VERSION = "0.1.0"
SOIL_FIELDS = (
    "ph_water", "organic_carbon_pct", "total_nitrogen_pct",
    "phosphorus_mgkg", "potassium_cmolkg", "texture",
)
OPS = ("lt", "lte", "gt", "gte", "eq", "in", "between")
PLACEHOLDER_MARKERS = ("todo", "tbd", "<", "fill in", "placeholder", "xxx")
VALID_STATUS = ("verified", "draft")
VALID_RATE_UNITS = ("kg/ha", "t/ha")


class RuleError(ValueError):
    """A rule file is invalid. The message names the rule id and the problem."""


class RuleConflict(RuntimeError):
    """Two rules with the same priority and specificity match for one input type."""


@dataclass(frozen=True)
class Rule:
    id: str
    crop: str
    when: dict[str, dict[str, Any]]
    advice: list[dict[str, Any]]
    source: dict[str, Any]
    status: str = "draft"
    priority: int = 0
    note: Optional[str] = None


@dataclass
class RuleSet:
    version: str
    rules: list[Rule] = field(default_factory=list)


def _is_placeholder(text: Any) -> bool:
    t = str(text).strip().lower()
    return not t or any(m in t for m in PLACEHOLDER_MARKERS)


def _check_condition(rule_id: str, var: str, cond: Any) -> None:
    if var not in SOIL_FIELDS:
        raise RuleError(f"{rule_id}: unknown soil variable '{var}'")
    if not isinstance(cond, dict) or not cond:
        raise RuleError(f"{rule_id}: condition for '{var}' must be a mapping such as {{lt: 5.5}}")
    for op, val in cond.items():
        if op not in OPS:
            raise RuleError(f"{rule_id}: unknown operator '{op}' for '{var}'")
        if op == "between" and not (isinstance(val, (list, tuple)) and len(val) == 2 and val[0] <= val[1]):
            raise RuleError(f"{rule_id}: 'between' for '{var}' needs [low, high] with low <= high")
        if op == "in" and not (isinstance(val, (list, tuple)) and val):
            raise RuleError(f"{rule_id}: 'in' for '{var}' needs a non-empty list")
        if var == "texture" and op not in ("eq", "in"):
            raise RuleError(f"{rule_id}: texture only supports eq and in")


def _parse_rule(raw: dict[str, Any]) -> Rule:
    rid = str(raw.get("id", "")).strip()
    if not rid:
        raise RuleError("a rule has no id")
    for key in ("crop", "advice", "source"):
        if key not in raw:
            raise RuleError(f"{rid}: missing '{key}'")
    src = raw["source"]
    if not isinstance(src, dict) or _is_placeholder(src.get("document")):
        raise RuleError(f"{rid}: source.document is missing or a placeholder. Every rule needs a real source.")
    status = raw.get("status", "draft")
    if status not in VALID_STATUS:
        raise RuleError(f"{rid}: status must be one of {VALID_STATUS}")
    if status == "verified" and not (src.get("page") or src.get("section")):
        raise RuleError(f"{rid}: a verified rule needs source.page or source.section")
    when = raw.get("when") or {}
    for var, cond in when.items():
        _check_condition(rid, var, cond)
    advice = raw["advice"]
    if not isinstance(advice, list) or not advice:
        raise RuleError(f"{rid}: advice must be a non-empty list")
    for a in advice:
        if not a.get("input_type"):
            raise RuleError(f"{rid}: every advice item needs input_type")
        if (a.get("rate") is None) != (a.get("rate_unit") is None):
            raise RuleError(f"{rid}: rate and rate_unit must be given together")
        if a.get("rate_unit") is not None and a["rate_unit"] not in VALID_RATE_UNITS:
            raise RuleError(f"{rid}: rate_unit must be one of {VALID_RATE_UNITS}")
        if a.get("rate") is not None and a["rate"] < 0:
            raise RuleError(f"{rid}: rate cannot be negative")
    return Rule(
        id=rid, crop=str(raw["crop"]).strip().lower(), when=when, advice=advice,
        source=src, status=status, priority=int(raw.get("priority", 0)), note=raw.get("note"),
    )


def load_rules(path: str | Path) -> RuleSet:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return parse_rules(data)


def parse_rules(data: dict[str, Any]) -> RuleSet:
    if "version" not in data:
        raise RuleError("rule file needs a top-level 'version'")
    rules = [_parse_rule(r) for r in (data.get("rules") or [])]
    ids = [r.id for r in rules]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RuleError(f"duplicate rule ids: {sorted(dupes)}")
    return RuleSet(version=str(data["version"]), rules=rules)


def _holds(op: str, val: Any, x: Any) -> bool:
    if op == "lt":
        return x < val
    if op == "lte":
        return x <= val
    if op == "gt":
        return x > val
    if op == "gte":
        return x >= val
    if op == "eq":
        return x == val
    if op == "in":
        return x in val
    return val[0] <= x <= val[1]  # between, inclusive


def _matches(rule: Rule, soil: dict[str, Any]) -> bool:
    for var, cond in rule.when.items():
        x = soil.get(var)
        if x is None:
            return False  # missing value: do not guess
        if not all(_holds(op, val, x) for op, val in cond.items()):
            return False
    return True


def _describe(rule: Rule) -> str:
    if not rule.when:
        return "applies to this crop in general"
    parts = []
    for var, cond in rule.when.items():
        for op, val in cond.items():
            parts.append(f"{var} {op} {val}")
    return "; ".join(parts)


def recommend(
    ruleset: RuleSet,
    crop: str,
    soil: Optional[dict[str, Any]] = None,
    allow_draft: bool = False,
) -> dict[str, Any]:
    """Return fertiliser advice for one crop as a dict shaped like FertiliserAdvice."""
    crop = crop.strip().lower()
    soil = {k: v for k, v in (soil or {}).items() if v is not None}
    candidates = [
        r for r in ruleset.rules
        if r.crop == crop and (r.status == "verified" or allow_draft) and _matches(r, soil)
    ]
    # For each input type keep the best rule: higher priority, then more conditions.
    best: dict[str, list[tuple[Rule, dict[str, Any]]]] = {}
    for r in candidates:
        for a in r.advice:
            best.setdefault(a["input_type"], []).append((r, a))
    items: list[dict[str, Any]] = []
    for input_type in sorted(best):
        ranked = sorted(best[input_type], key=lambda ra: (ra[0].priority, len(ra[0].when)), reverse=True)
        top_key = (ranked[0][0].priority, len(ranked[0][0].when))
        tied = [ra for ra in ranked if (ra[0].priority, len(ra[0].when)) == top_key]
        if len(tied) > 1:
            ids = sorted(r.id for r, _ in tied)
            raise RuleConflict(f"rules {ids} all match for '{input_type}' with equal priority and specificity")
        rule, a = ranked[0]
        items.append({
            "input_type": a["input_type"],
            "product": a.get("product"),
            "rate": a.get("rate"),
            "rate_unit": a.get("rate_unit"),
            "timing": a.get("timing"),
            "condition_matched": f"{rule.id}: {_describe(rule)}",
            "source": {k: v for k, v in rule.source.items() if k in ("document", "section", "page", "url") and v is not None},
            "explanation": a.get("explanation"),
            "explanation_method": "rule_text",
        })
    return {
        "crop": crop,
        "status": "ok" if items else "no_matching_rule",
        "engine_version": f"{ENGINE_VERSION}+rules-{ruleset.version}",
        "items": items,
    }
