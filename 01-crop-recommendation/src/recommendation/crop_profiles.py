"""Crop profiles for the map page: names (English, Kinyarwanda, scientific), short facts and the varieties released
in Rwanda, each with its source. The data is rules/crop_profiles.yaml; nothing here comes from the farm data.

for_field() marks the varieties whose altitude zone fits the clicked field. The sources give the zones in words
(for example "High altitude"); the YAML turns them into metres with RAB's split for maize (Annual Report 2019-2020,
Table 7): mid altitude 900 to 1,700 m, high altitude above 1,700 m.

Quick check from the repository root:
    python 01-crop-recommendation/src/recommendation/crop_profiles.py --crop beans --elevation 2100
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import yaml

PROFILES_PATH = Path(__file__).resolve().parent / "rules" / "crop_profiles.yaml"
CROPS = ("beans", "irish_potato", "maize", "rice")
FORBIDDEN = ("—", "–", "→", "←", "⇒")   # long dashes and arrows: not in the app's style


@lru_cache(maxsize=4)
def _load_cached(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load(path: Path = PROFILES_PATH) -> dict:
    """The whole YAML file (a fresh copy, so callers can change it)."""
    return copy.deepcopy(_load_cached(str(path)))


def source_keys(value) -> list:
    """A `source` field holds one key or a list of keys."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def fits(zone: Optional[dict], elevation_m: Optional[float]) -> Optional[bool]:
    """True if the elevation is inside the zone, False if outside, None if unknown (no elevation or no limits)."""
    if not zone or elevation_m is None or ("min_m" not in zone and "max_m" not in zone):
        return None
    lo, hi = zone.get("min_m"), zone.get("max_m")
    return (lo is None or elevation_m >= lo) and (hi is None or elevation_m <= hi)


def zone_words(zone: dict) -> str:
    """The zone in metres, in plain words: 'above 1,700 m', 'below 1,700 m', 'at 900 to 1,700 m'."""
    lo, hi = zone.get("min_m"), zone.get("max_m")
    if lo is not None and hi is not None:
        return f"at {lo:,.0f} to {hi:,.0f} m"
    if lo is not None:
        return f"above {lo:,.0f} m"
    if hi is not None:
        return f"below {hi:,.0f} m"
    return ""


def for_field(crop: str, elevation_m: Optional[float] = None, data: Optional[dict] = None) -> dict:
    """The profile of one crop for a field: its varieties get `fits` (True, False or None), `zone_label` and
    `fit_note`, and those that fit come first (then unknown, then the others), each group in the file's order.
    `sources` lists every source the profile cites, in order of first use, with the Kinyarwanda names' source."""
    data = data or load()
    if crop not in data["crops"]:
        raise KeyError(f"No profile for crop {crop!r}")
    prof = copy.deepcopy(data["crops"][crop])
    zones = data.get("zones", {})
    for v in prof.get("varieties", []):
        zone = zones.get(v.get("zone")) if v.get("zone") else None
        v["fits"] = fits(zone, elevation_m)
        v["zone_label"] = zone["label"] if zone else None
        if v["fits"] is True:
            v["fit_note"] = "Fits this field"
        elif v["fits"] is False:
            v["fit_note"] = f"Better {zone_words(zone)}"
        else:
            v["fit_note"] = ""
    order = {True: 0, None: 1, False: 2}
    prof["varieties"] = sorted(prof.get("varieties", []), key=lambda v: order[v["fits"]])   # stable sort
    prof["fitting"] = sum(1 for v in prof["varieties"] if v["fits"] is True)
    prof["elevation_m"] = elevation_m
    prof["crop"] = crop
    prof["inputs"] = copy.deepcopy(data.get("inputs"))
    prof["diseases"] = copy.deepcopy(data.get("diseases", {}))
    keys = []
    for item in _cited(prof) + [{"source": data.get("names_source")}]:
        for k in source_keys(item.get("source")):
            if k not in keys:
                keys.append(k)
    prof["sources"] = [dict(key=k, **data["sources"][k]) for k in keys]
    return prof


def _cited(prof: dict) -> list:
    """Every part of a profile that carries a `source` field."""
    parts = list(prof.get("facts", [])) + list(prof.get("varieties", []))
    for key in ("watch_out", "inputs"):
        if prof.get(key):
            parts.append(prof[key])
    return parts


def _texts(node) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _texts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _texts(v)


def check(data: Optional[dict] = None) -> list:
    """Problems in the profiles file (empty when it is fine): a crop missing, a claim without a known source, a
    source without a link, an unknown zone, colour or disease code, or a long dash or arrow anywhere."""
    data = data or load()
    problems = []
    sources = data.get("sources", {})
    for k, s in sources.items():
        if not s.get("url", "").startswith("https://"):
            problems.append(f"source {k}: no https link")
        if not s.get("text") or not s.get("short"):
            problems.append(f"source {k}: needs text and short")
    if data.get("names_source") not in sources:
        problems.append("names_source is not a known source")
    for crop in CROPS:
        prof = data.get("crops", {}).get(crop)
        if not prof:
            problems.append(f"{crop}: no profile")
            continue
        for field in ("name", "kinyarwanda", "scientific", "intro", "facts", "watch_out", "varieties"):
            if not prof.get(field):
                problems.append(f"{crop}: missing {field}")
        parts = _cited(prof) + ([data["inputs"]] if data.get("inputs") else [])
        for part in parts:
            keys = source_keys(part.get("source"))
            if not keys:
                problems.append(f"{crop}: no source for {part.get('name') or part.get('big') or part.get('text', '')[:40]}")
            for k in keys:
                if k not in sources:
                    problems.append(f"{crop}: unknown source {k}")
        for v in prof.get("varieties", []):
            if v.get("zone") and v["zone"] not in data.get("zones", {}):
                problems.append(f"{crop} {v['name']}: unknown zone {v['zone']}")
            for label_key, field in (("seed_labels", "seed"), ("skin_labels", "skin"), ("flesh_labels", "flesh")):
                if v.get(field) and v[field] not in prof.get(label_key, {}):
                    problems.append(f"{crop} {v['name']}: no label for {field} {v[field]}")
            for code in list(v.get("resistant", [])) + list(v.get("tolerant", [])):
                if code not in data.get("diseases", {}):
                    problems.append(f"{crop} {v['name']}: unknown disease code {code}")
    for text in _texts(data):
        for ch in FORBIDDEN:
            if ch in text:
                problems.append(f"long dash or arrow in: {text[:60]}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crop", choices=CROPS, default="beans")
    ap.add_argument("--elevation", type=float, default=None, help="field elevation in metres")
    a = ap.parse_args(argv)
    problems = check()
    prof = for_field(a.crop, a.elevation)
    print(f"{prof['name']} ({prof['kinyarwanda']}, {prof['scientific']}), field at {a.elevation} m: "
          f"{prof['fitting']} of {len(prof['varieties'])} varieties fit")
    for v in prof["varieties"]:
        print(f"  {v['name']:<22} {v['zone_label'] or '':<22} {v['fit_note']}")
    print("problems:", problems or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
