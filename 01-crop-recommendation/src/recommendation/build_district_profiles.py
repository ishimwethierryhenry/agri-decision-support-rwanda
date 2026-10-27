"""Build district_profiles.json, the small summary file the recommendation service reads at run time.

The service must not depend on raw data files (they are kept out of Git), so this script turns
Part A's processed files into district-level summaries that contain no farm-level data:
  - median, 10th and 90th percentile elevation of the CIAT points, and the median point location
  - season climate normals from NASA POWER (mean over all start years in the file)
  - the overall share of each crop among the CIAT pure-crop points (used only to break ties)

Run from the repo root after Part A updates the data:
    python 01-crop-recommendation/src/recommendation/build_district_profiles.py
    python 01-crop-recommendation/src/recommendation/build_district_profiles.py --table <feature table csv> --climate <nasa_power_seasonal.csv>
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROCESSED = HERE.parents[1] / "data" / "processed"
TABLE_CANDIDATES = [PROCESSED / "feature_table_v2.csv", PROCESSED / "feature_table.csv"]
CLIMATE_DEFAULT = PROCESSED / "nasa_power_seasonal.csv"
OUT_DEFAULT = HERE / "district_profiles.json"

LABEL_MAP = {"Maize": "maize", "Beans": "beans", "Irish potatoes": "irish_potato", "Rice": "rice"}
CLIMATE_VARS = {"PRECTOTCORR": "precip_mm", "T2M_MAX": "tmax_c", "T2M_MIN": "tmin_c"}
MIN_ELEV, MAX_LON = 900.0, 31.0  # Part A's quality filters


def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("No feature table found. Pass --table. Looked for: " + ", ".join(map(str, paths)))


def build(table: Path, climate: Path) -> dict:
    df = pd.read_csv(table)
    lon_col = "GPS_Long" if "GPS_Long" in df.columns else "longitude"
    lat_col = "GPS_Lat" if "GPS_Lat" in df.columns else "latitude"
    df = df[(df["GPS_Elev"] > MIN_ELEV) & (df[lon_col] < MAX_LON)]

    clim = pd.read_csv(climate)
    clim = clim[clim["variable"].isin(CLIMATE_VARS)]
    wide = clim.pivot_table(index=["district", "season", "year"], columns="variable", values="value").reset_index()
    wide = wide.rename(columns=CLIMATE_VARS)

    pure = df[df["Crop_plant"].isin(LABEL_MAP)]
    prior = pure["Crop_plant"].map(LABEL_MAP).value_counts(normalize=True).round(4).to_dict()

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
            "centre": {"lat": round(float(g[lat_col].median()), 4), "lon": round(float(g[lon_col].median()), 4)},
            "elevation_m": {"median": round(float(g["GPS_Elev"].median()), 1),
                            "p10": round(float(g["GPS_Elev"].quantile(0.1)), 1),
                            "p90": round(float(g["GPS_Elev"].quantile(0.9)), 1)},
            "nasa_power_point": {"lat": round(float(point["lat"]), 4), "lon": round(float(point["lon"]), 4)},
            "seasons": seasons,
        }
    return {
        "built_on": date.today().isoformat(),
        "source_table": table.name,
        "climate_source": "NASA POWER seasonal sums (precipitation) and means (temperature), averaged over all start years",
        "season_windows": {"A": "1 Sep to 31 Jan", "B": "1 Mar to 30 Jun", "C": "1 Jun to 31 Aug"},
        "elevation_source": "CIAT GPS elevation of the field points (filtered above 900 m)",
        "crop_prior": prior,
        "districts": districts,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build district_profiles.json for the recommendation service")
    ap.add_argument("--table", default=None)
    ap.add_argument("--climate", default=str(CLIMATE_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)
    table = Path(args.table) if args.table else _first_existing(TABLE_CANDIDATES)
    profiles = build(table, Path(args.climate))
    Path(args.out).write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} for {len(profiles['districts'])} districts from {table.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
