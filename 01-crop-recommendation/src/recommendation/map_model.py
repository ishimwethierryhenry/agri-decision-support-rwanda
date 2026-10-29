"""Crop recommendation for one point on the map (dashboard map page and the Colab notebook).

What happens when someone clicks a point:
  1. District: the district polygon the point falls in (data/reference/rwanda_districts.geojson, geoBoundaries,
     source Open Data Rwanda, CC BY 4.0). Only the four pilot districts have CIAT data.
  2. Field inputs: the feature table has no coordinates any more (privacy), so soil and elevation come from the
     CIAT fields in the same 2 km block (Part A's block_id, cells of 0.018 degrees), or from the nearest block
     in the same district. The answer says how far that block is.
  3. District inputs: season climate (NASA POWER) and NISR crop shares of season B 2025, the season the model
     learned from. The model only knows season B, so another season is shown as a note, not as an answer.
  4. Two methods side by side, each with its tested score in that district (reports/eval_v1):
     XGBoost on all inputs, trained on all 1,904 fields with the protocol settings, and the B1 expert rules.
  5. Fertiliser: the rule engine answers for the model's first crop. Its rule file stays empty until the RAB
     guideline is in, so it says "no matching rule" instead of inventing a rate.

Quick check from the repo root:
    python 01-crop-recommendation/src/recommendation/map_model.py --lat -1.458 --lon 29.556
    python 01-crop-recommendation/src/recommendation/map_model.py --point -1.458 29.556 --point -1.35 30.30
    python 01-crop-recommendation/src/recommendation/map_model.py --export <folder>   (model, importance, meta)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1]                      # 01-crop-recommendation
for _p in (HERE, HERE.parent / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluate as ev  # noqa: E402
import expert_baseline as b1  # noqa: E402
import fertiliser_engine as fert  # noqa: E402

TABLE_PATH = MODULE / "data" / "processed" / "feature_table_v2.csv"
BOUNDARIES_PATH = MODULE / "data" / "reference" / "rwanda_districts.geojson"
EVAL_DIR = MODULE / "reports" / "eval_v1"
FERTILISER_RULES_PATH = HERE / "rules" / "fertiliser_rules.yaml"

BLOCK_DEG = 0.018          # Part A's block_id is round(lat / 0.018) and round(lon / 0.018)
MODEL_NAME = "XGB[all]"    # the protocol's XGBoost on all inputs (view "all")
MODEL_SEASON = "B"         # the only season in the training data (CIAT, season B 2025)
NEAR_KM = 5.0              # nearest CIAT block further than this: the answer warns that the data is far away
BORDER_KM = 1.5            # a point this close to a CIAT block, just across a district line, uses that block
CROP_NAMES = {"beans": "Beans", "irish_potato": "Irish potato", "maize": "Maize", "rice": "Rice"}
SEASON_NAMES = {"A": "Season A (Sep to Jan)", "B": "Season B (Feb to Jun)", "C": "Season C (Jul to Aug)"}


# ------------------------------------------------------------------ small helpers
def current_season(day: Optional[date] = None) -> str:
    """Rwanda's agricultural season for a date: A from September to January, B from February to June
    (the window used for the training data), C in July and August."""
    m = (day or date.today()).month
    if m >= 9 or m == 1:
        return "A"
    if 2 <= m <= 6:
        return "B"
    return "C"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def block_id(lat: float, lon: float) -> str:
    return f"{int(np.round(lat / BLOCK_DEG))}_{int(np.round(lon / BLOCK_DEG))}"


def texture_class(sand: float, silt: float, clay: float) -> str:
    """USDA soil texture class. iSDAsoil predicts the three fractions separately, so they are scaled to 100."""
    total = sand + silt + clay
    if not total or any(map(math.isnan, (sand, silt, clay))):
        return "unknown"
    sa, si, cl = (100 * v / total for v in (sand, silt, clay))
    if si + 1.5 * cl < 15:
        return "sand"
    if si + 2 * cl < 30:
        return "loamy sand"
    if (7 <= cl < 20 and sa > 52) or (cl < 7 and si < 50):
        return "sandy loam"
    if 7 <= cl < 27 and 28 <= si < 50 and sa <= 52:
        return "loam"
    if (si >= 50 and 12 <= cl < 27) or (50 <= si < 80 and cl < 12):
        return "silt loam"
    if si >= 80 and cl < 12:
        return "silt"
    if 20 <= cl < 35 and si < 28 and sa > 45:
        return "sandy clay loam"
    if 27 <= cl < 40 and 20 < sa <= 45:
        return "clay loam"
    if 27 <= cl < 40 and sa <= 20:
        return "silty clay loam"
    if cl >= 35 and sa > 45:
        return "sandy clay"
    if cl >= 40 and si >= 40:
        return "silty clay"
    if cl >= 40:
        return "clay"
    return "unclassified"


def _in_ring(x: float, y: float, ring: list) -> bool:
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class Districts:
    """District polygons, for 'which district is this point in?'."""

    def __init__(self, geojson: dict):
        self.geojson = geojson
        self.shapes = []
        for f in geojson["features"]:
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            xs = [p[0] for poly in polys for p in poly[0]]
            ys = [p[1] for poly in polys for p in poly[0]]
            self.shapes.append((f["properties"]["district"], polys, (min(xs), min(ys), max(xs), max(ys))))

    @classmethod
    def load(cls, path: Path = BOUNDARIES_PATH) -> "Districts":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def at(self, lat: float, lon: float) -> Optional[str]:
        for name, polys, (x0, y0, x1, y1) in self.shapes:
            if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                continue
            for poly in polys:
                if _in_ring(lon, lat, poly[0]) and not any(_in_ring(lon, lat, hole) for hole in poly[1:]):
                    return name
        return None


# ------------------------------------------------------------------ the recommender
class MapRecommender:
    """Trains the protocol's XGBoost on all fields once, then answers one point at a time."""

    def __init__(self, raw: pd.DataFrame, districts: Districts, evaluation_dir: Optional[Path] = EVAL_DIR,
                 n_estimators: int = 300):
        self.districts = districts
        self.main = ev.prepare(raw)                                 # the 1,904 training fields
        self.model = next(m for m in ev.default_models(n_estimators=n_estimators) if m.name == MODEL_NAME)
        self.model.fit(self.main, self.main["label"])
        self.crop_prior = self.main["label"].value_counts(normalize=True).to_dict()
        self.rules = b1.load_rules()
        self.fert_rules = fert.load_rules(FERTILISER_RULES_PATH)
        self._build_lookup(raw)
        self.tested = self._load_tested(evaluation_dir)

    @classmethod
    def load(cls, table: Path = TABLE_PATH, boundaries: Path = BOUNDARIES_PATH,
             evaluation_dir: Optional[Path] = EVAL_DIR) -> "MapRecommender":
        return cls(pd.read_csv(table, low_memory=False), Districts.load(boundaries), evaluation_dir)

    # ----------------------------------------------------------- lookup tables (no farm coordinates)
    def _build_lookup(self, raw: pd.DataFrame) -> None:
        qc = raw[ev._qc_mask(raw, 900.0, 31.0)].copy()              # every field with good GPS, any crop
        ij = qc["block_id"].astype(str).str.split("_", expand=True).astype(int)
        qc["block_lat"], qc["block_lon"] = ij[0] * BLOCK_DEG, ij[1] * BLOCK_DEG
        soil = ev.SOIL_TOP + ev.SOIL_SUB
        agg = {"lat": ("block_lat", "first"), "lon": ("block_lon", "first"), "n_fields": ("GPS_Elev", "size"),
               "elevation_m": ("GPS_Elev", "median"), **{c: (c, "mean") for c in soil}}
        self.blocks = qc.groupby(["District", "block_id"]).agg(**agg).reset_index()
        self.district_values: dict[str, dict] = {}
        district_cols = ["precip_mm", "tmax_c", "tmin_c", "solar_mj_m2"] + ev.NISR_SHARES
        for name, g in qc.groupby("District"):
            values = {}
            for c in district_cols:
                vals = g[c].dropna().unique()
                assert len(vals) == 1, f"{c} is not one value in {name}"
                values[c] = float(vals[0])
            values["ref_elevation_m"] = float(g["GPS_Elev"].median())
            values["province"] = str(g["Province"].mode().iloc[0])
            values["n_fields"] = int(len(g))
            self.district_values[name] = values

    @staticmethod
    def _load_tested(evaluation_dir: Optional[Path]) -> dict:
        """Tested macro-F1 per district for the model and the rules, from the official evaluation."""
        out: dict = {}
        if not evaluation_dir:
            return out
        for test, fname in (("new_fields", "main_blockcv_results.csv"), ("new_district", "main_lodo_results.csv")):
            path = Path(evaluation_dir) / fname
            if not path.exists():
                continue
            res = pd.read_csv(path)
            for method, model in (("model", MODEL_NAME), ("rules", ev.BASELINE)):
                for _, r in res[res["model"] == model].iterrows():
                    out.setdefault(r["district"], {}).setdefault(method, {})[test] = round(float(r["macro_f1"]), 3)
        return out

    # ----------------------------------------------------------- one point
    def feature_row(self, lat: float, lon: float, district: str,
                    elevation_m: Optional[float] = None) -> tuple[dict, dict]:
        blocks = self.blocks[self.blocks["District"] == district]
        bid = block_id(lat, lon)
        here = blocks[blocks["block_id"] == bid]
        if here.empty:
            dist = blocks.apply(lambda b: haversine_km(lat, lon, b["lat"], b["lon"]), axis=1)
            b = blocks.loc[dist.idxmin()]
            km = float(dist.min())
        else:
            b, km = here.iloc[0], 0.0
        d = self.district_values[district]
        elev = float(elevation_m) if elevation_m is not None else float(b["elevation_m"])
        tmean = (d["tmax_c"] + d["tmin_c"]) / 2
        row = {"elevation_m": elev, "precip_mm": d["precip_mm"], "tmax_c": d["tmax_c"], "tmin_c": d["tmin_c"],
               "tmean_c": tmean, "solar_mj_m2": d["solar_mj_m2"],
               "tmean_point_c": float(b1.adjust_to_height(tmean, elev, d["ref_elevation_m"])),
               "tmin_point_c": float(b1.adjust_to_height(d["tmin_c"], elev, d["ref_elevation_m"])),
               **{c: d[c] for c in ev.NISR_SHARES},
               **{c: float(b[c]) for c in ev.SOIL_TOP + ev.SOIL_SUB}}
        info = {"block_id": str(b["block_id"]), "block_lat": float(b["lat"]), "block_lon": float(b["lon"]),
                "distance_km": round(km, 2), "n_fields": int(b["n_fields"]),
                "elevation_source": "user" if elevation_m is not None else "median GPS elevation of the CIAT fields in the block"}
        return row, info

    def recommend(self, lat: float, lon: float, season: Optional[str] = None, today: Optional[date] = None,
                  elevation_m: Optional[float] = None) -> dict[str, Any]:
        now = current_season(today)
        asked = season or now
        out: dict[str, Any] = {"point": {"lat": round(lat, 5), "lon": round(lon, 5)},
                               "season": {"today": now, "asked": asked, "used": MODEL_SEASON},
                               "warnings": []}
        district = self.districts.at(lat, lon)
        if district not in self.district_values:
            # Block centres near a border can fall just outside the simplified district outline. A point right
            # next to CIAT fields is treated as part of the district the fields were recorded in.
            dist = self.blocks.apply(lambda b: haversine_km(lat, lon, b["lat"], b["lon"]), axis=1)
            if len(dist) and dist.min() <= BORDER_KM:
                near = str(self.blocks.loc[dist.idxmin(), "District"])
                out["warnings"].append(f"This point is just outside the {near} outline on the map, next to CIAT "
                                       f"fields recorded in {near}, so it is treated as {near}.")
                district = near
        out["district"] = district
        if district is None:
            out["status"] = "outside_rwanda"
            out["message"] = "This point is outside Rwanda's district boundaries."
            return out
        if district not in self.district_values:
            out["status"] = "outside_pilot"
            out["message"] = (f"{district} is not one of the four pilot districts (Musanze, Nyabihu, Nyagatare, "
                              "Ruhango). There is no CIAT data here, so no recommendation is given.")
            return out
        out["status"] = "ok"
        if asked != MODEL_SEASON:
            out["warnings"].insert(0, (
                f"{SEASON_NAMES[asked]} is not in the training data yet: the model and the rules were only tested on "
                f"season B (CIAT, 2025). The answer below is for season B, which starts on 1 February."))
        row, info = self.feature_row(lat, lon, district, elevation_m)
        d = self.district_values[district]
        if info["distance_km"] > NEAR_KM:
            out["warnings"].append(f"No CIAT field within {NEAR_KM:.0f} km of this point. Soil and elevation come "
                                   f"from the nearest sampled block, {info['distance_km']:.1f} km away.")
        if district in ("Musanze", "Nyabihu"):
            out["warnings"].append("Musanze and Nyabihu share one NASA POWER weather cell, so their rainfall and "
                                   "temperature are the same. Per-field rainfall (CHIRPS) comes later.")
        out["inputs"] = {
            "block": info,
            "elevation_m": round(row["elevation_m"], 1),
            "soil_texture": texture_class(row["sand_0_20cm"], row["silt_0_20cm"], row["clay_0_20cm"]),
            "soil_0_20cm": {c.replace("_0_20cm", ""): round(row[c], 2) for c in ev.SOIL_TOP},
            "climate_season_b_2025": {"rainfall_mm": round(d["precip_mm"], 1), "tmax_c": round(d["tmax_c"], 2),
                                      "tmin_c": round(d["tmin_c"], 2),
                                      "mean_temp_at_field_c": round(row["tmean_point_c"], 2),
                                      "solar_mj_m2_day": round(d["solar_mj_m2"], 2)},
            "nisr_crop_shares": {CROP_NAMES[k.replace("nisr_share_", "").replace("paddy_", "")]: round(d[k], 3)
                                 for k in ev.NISR_SHARES},
            "province": d["province"],
        }
        proba = self.model.predict_proba(pd.DataFrame([row]))[0]
        order = np.argsort(-proba)
        out["model"] = {"name": "XGBoost, all inputs",
                        "ranking": [{"crop": str(self.model.classes_[i]), "probability": round(float(proba[i]), 3)}
                                    for i in order],
                        "tested": self.tested.get(district, {}).get("model", {})}
        ranked = [r for r in b1.rank_crops(row, self.rules, self.crop_prior) if r["score"] is not None]
        out["rules"] = {"name": "Expert rules (B1)",
                        "ranking": [{"crop": r["crop"], "score": round(float(r["score"]), 3),
                                     "weak": r.get("reliability_cap") == "low", "reasons": r["reasons"]}
                                    for r in ranked],
                        "tested": self.tested.get(district, {}).get("rules", {})}
        if len(ranked) > 1:
            tied = [CROP_NAMES[r["crop"]] for r in ranked if abs(r["score"] - ranked[0]["score"]) < 1e-9]
            if len(tied) > 1:
                out["warnings"].append(f"The rules rate {', '.join(tied)} equally ({ranked[0]['score']:.2f}); their "
                                       "order is only a tie-break.")
        m_t = out["model"]["tested"].get("new_fields")
        r_t = out["rules"]["tested"].get("new_fields")
        out["better_in_testing"] = (None if m_t is None or r_t is None
                                    else "model" if m_t > r_t + 0.01 else "rules" if r_t > m_t + 0.01 else "same")
        top = out["model"]["ranking"][0]["crop"]
        out["fertiliser"] = fert.recommend(self.fert_rules, top, {})
        if out["fertiliser"]["status"] == "no_matching_rule":
            out["warnings"].append("No fertiliser advice yet: the rule file is waiting for the RAB fertiliser "
                                   "guideline, so no rate is given.")
        return out

    # ----------------------------------------------------------- for the notebook
    def feature_importance(self) -> pd.DataFrame:
        booster = self.model.estimator.get_booster()
        gain = booster.get_score(importance_type="gain")
        names = self.model.used_
        rows = [{"feature": names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, "gain": v}
                for k, v in gain.items()]
        df = pd.DataFrame(rows).sort_values("gain", ascending=False)
        df["share"] = df["gain"] / df["gain"].sum()
        return df.reset_index(drop=True)

    def export(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.estimator.save_model(str(folder / "xgboost_all_inputs.json"))
        self.feature_importance().to_csv(folder / "feature_importance.csv", index=False)
        import sklearn
        import xgboost
        meta = {"model": MODEL_NAME, "trained_on_fields": int(len(self.main)), "features": self.model.used_,
                "classes": [str(c) for c in self.model.classes_], "season": MODEL_SEASON,
                "settings": repr(self.model.estimator),
                "versions": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__}}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return folder


@lru_cache(maxsize=1)
def default_recommender() -> MapRecommender:
    return MapRecommender.load()


def _print(result: dict) -> None:
    print(f"Point {result['point']['lat']}, {result['point']['lon']}: {result.get('district') or 'no district'}")
    if result.get("status") != "ok":
        print(result.get("message", ""))
    else:
        inp = result["inputs"]
        print(f"  soil {inp['soil_texture']}, pH {inp['soil_0_20cm']['ph']}, elevation {inp['elevation_m']} m, "
              f"block {inp['block']['block_id']} ({inp['block']['n_fields']} CIAT fields, "
              f"{inp['block']['distance_km']} km away)")
        print("  XGBoost: " + ", ".join(f"{CROP_NAMES[r['crop']]} {r['probability']:.2f}"
                                       for r in result["model"]["ranking"]))
        print("  Rules:   " + ", ".join(f"{CROP_NAMES[r['crop']]} {r['score']:.2f}" for r in result["rules"]["ranking"]))
        print(f"  tested on new fields here: model {result['model']['tested'].get('new_fields')}, "
              f"rules {result['rules']['tested'].get('new_fields')}")
    for w in result["warnings"]:
        print("  note: " + w)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Crop recommendation for one point or several points")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--point", nargs=2, type=float, action="append", metavar=("LAT", "LON"), default=[],
                    help="a point to answer; repeat for several points")
    ap.add_argument("--season", choices=["A", "B", "C"], help="default: today's season")
    ap.add_argument("--export", help="Folder to save the trained model, feature importance and meta data")
    args = ap.parse_args(argv)
    rec = default_recommender()
    if args.export:
        print(f"Saved to {rec.export(Path(args.export))}")
    points = list(args.point)
    if args.lat is not None and args.lon is not None:
        points.insert(0, [args.lat, args.lon])
    for i, (lat, lon) in enumerate(points):
        if i:
            print()
        _print(rec.recommend(lat, lon, args.season))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
