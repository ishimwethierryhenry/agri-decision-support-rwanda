"""Build district_profiles.json, the small summary file the recommendation service reads at run time.

The service must not depend on raw data files, so this script turns Part A's processed files into
district-level summaries that contain no farm-level data:
  - median, 10th and 90th percentile elevation of the CIAT fields that pass qc_ok
  - season climate normals from NASA POWER (mean over all start years in the file)
  - the overall share of each crop among the CIAT single-crop fields (used only to break ties)
  - how the expert rules (B1) scored in each district in the official evaluation (reports/eval_v1),
    which the service uses to set reliability

Since 28 Oct the feature table has no coordinates, so the district centres are kept from the previous
profiles file (they were computed earlier as the median location of the CIAT fields in each district).

Run from the repo root after Part A updates the data or a new evaluation is committed:
    python 01-crop-recommendation/src/recommendation/build_district_profiles.py
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1]                 # 01-crop-recommendation
REPO = HERE.parents[2]
PROCESSED = MODULE / "data" / "processed"
TABLE_CANDIDATES = [PROCESSED / "feature_table_v2.csv", PROCESSED / "feature_table.csv"]
CLIMATE_CANDIDATES = [REPO / "data" / "raw" / "nasa_power" / "nasa_power_seasonal.csv",
                      PROCESSED / "nasa_power_seasonal.csv"]
EVAL_DEFAULT = MODULE / "reports" / "eval_v1" / "main_lodo_results.csv"
OUT_DEFAULT = HERE / "district_profiles.json"

LABEL_MAP = {"Maize": "maize", "Beans": "beans", "Irish potatoes": "irish_potato", "Rice": "rice"}
CLIMATE_VARS = {"PRECTOTCORR": "precip_mm", "T2M_MAX": "tmax_c", "T2M_MIN": "tmin_c"}
MIN_ELEV, MAX_LON = 900.0, 31.0  # Part A's GPS checks, used only when the table has no qc_ok column
# Season windows of Part A's NASA POWER fetcher (start-year convention). Season B starts on 1 Feb since 28 Oct.
SEASON_WINDOWS = {"A": "1 Sep to 31 Jan", "B": "1 Feb to 30 Jun", "C": "1 Jun to 31 Aug"}
B1_NAME = "B1_expert_rules"


def _first_existing(paths: list[Path], what: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"No {what} found. Looked for: " + ", ".join(map(str, paths)))


def _qc(df: pd.DataFrame) -> pd.DataFrame:
    if "qc_ok" in df.columns:
        return df[df["qc_ok"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})]
    lon_col = next(c for c in ("GPS_Long", "longitude", "longitude_x") if c in df.columns)
    return df[(df["GPS_Elev"] > MIN_ELEV) & (df[lon_col] < MAX_LON)]


def _centre(g: pd.DataFrame, name: str, previous: dict) -> dict:
    lat_col = next((c for c in ("GPS_Lat", "latitude", "latitude_x") if c in g.columns), None)
    lon_col = next((c for c in ("GPS_Long", "longitude", "longitude_x") if c in g.columns), None)
    if lat_col and lon_col:
        return {"lat": round(float(g[lat_col].median()), 4), "lon": round(float(g[lon_col].median()), 4)}
    try:
        return previous["districts"][name]["centre"]
    except KeyError as e:
        raise KeyError(f"The table has no coordinates and the previous profiles have no centre for {name}") from e


def b1_tested(evaluation: Path) -> dict:
    """B1's scores per district in the main test of the official evaluation (B1 is not trained, so this is
    simply how the rules did on that district's CIAT fields)."""
    res = pd.read_csv(evaluation)
    b1 = res[res["model"] == B1_NAME]
    if b1.empty:
        raise ValueError(f"{evaluation} has no {B1_NAME} rows")
    return {r["district"]: {"macro_f1": round(float(r["macro_f1"]), 3), "top1": round(float(r["top1"]), 3),
                            "n_fields": int(r["n_test"]), "crops_scored": str(r["crops_scored"])}
            for _, r in b1.iterrows()}


def build(table: Path, climate: Path, evaluation: Path, previous: dict) -> dict:
    df = _qc(pd.read_csv(table, low_memory=False))

    clim = pd.read_csv(climate)
    clim = clim[clim["variable"].isin(CLIMATE_VARS)]
    wide = clim.pivot_table(index=["district", "season", "year"], columns="variable", values="value").reset_index()
    wide = wide.rename(columns=CLIMATE_VARS)

    pure = df[df["Crop_plant"].isin(LABEL_MAP)]
    prior = pure["Crop_plant"].map(LABEL_MAP).value_counts(normalize=True).round(4).to_dict()
    tested = b1_tested(evaluation)

    districts = {}
    for name, g in df.groupby("District"):
        cw = wide[wide["district"] == name]
        seasons = {}
        for season, s in cw.groupby("season"):
            seasons[season] = {
                "precip_mm": round(float(s["precip_mm"].mean()), 1),
                "tmax_c": round(float(s["tmax_c"].mean()), 2),
                "tmin_c": round(float(s["tmin_c"].mean()), 2),
                "years": f"{int(s['year'].min())} to {int(s['year'].max())} (start years)",
            }
        point = clim[clim["district"] == name][["lat", "lon"]].iloc[0]
        districts[name] = {
            "province": str(g["Province"].mode().iloc[0]),
            "in_training_data": True,
            "n_points": int(len(g)),
            "centre": _centre(g, name, previous),
            "elevation_m": {"median": round(float(g["GPS_Elev"].median()), 1),
                            "p10": round(float(g["GPS_Elev"].quantile(0.1)), 1),
                            "p90": round(float(g["GPS_Elev"].quantile(0.9)), 1)},
            "nasa_power_point": {"lat": round(float(point["lat"]), 4), "lon": round(float(point["lon"]), 4)},
            "seasons": seasons,
            "b1_tested": tested.get(name),
        }
    try:
        eval_rel = evaluation.resolve().relative_to(REPO).as_posix()
    except ValueError:
        eval_rel = str(evaluation)
    return {
        "built_on": date.today().isoformat(),
        "source_table": table.name,
        "climate_source": "NASA POWER seasonal sums (precipitation) and means (temperature), averaged over all "
                          "start years",
        "season_windows": SEASON_WINDOWS,
        "elevation_source": "CIAT GPS elevation of the fields that pass qc_ok",
        "b1_tested_source": f"{eval_rel}: B1 rows of the main test (docs/training_protocol.md), season B 2025",
        "crop_prior": prior,
        "districts": districts,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build district_profiles.json for the recommendation service")
    ap.add_argument("--table", default=None)
    ap.add_argument("--climate", default=None)
    ap.add_argument("--evaluation", default=str(EVAL_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)
    table = Path(args.table) if args.table else _first_existing(TABLE_CANDIDATES, "feature table")
    climate = Path(args.climate) if args.climate else _first_existing(CLIMATE_CANDIDATES, "NASA POWER seasonal file")
    out = Path(args.out)
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    profiles = build(table, climate, Path(args.evaluation), previous)
    out.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} for {len(profiles['districts'])} districts from {table.name} and {climate.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
