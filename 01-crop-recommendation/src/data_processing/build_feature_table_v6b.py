"""
build_feature_table_v6b.py

Training table for protocol 1.1: the columns of feature_table_v5.csv, with the satellite values of
feature_table_v6.csv.

Why: v5 joined its satellite values on F_index, which is not unique (274 IDs are used by two fields each),
so 548 rows got another field's values. Part A fixed the join in v6 (matched on GPS), but v6 was rebuilt from
an older layout: it lost qc_ok, block_id and the NISR crop shares, brought back the NISR yield columns, and
contains the field coordinates. v6b keeps v5's layout (no coordinates) and takes only the five satellite
columns from v6, matched on F_id, which is unique.

The SRTM elevation is saved as srtm_elevation_m, because the evaluation already uses the name elevation_m for
the GPS elevation of each field.

Run from the repo root:
    python 01-crop-recommendation/src/data_processing/build_feature_table_v6b.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODULE = Path(__file__).resolve().parents[2]
PROCESSED = MODULE / "data" / "processed"
SATELLITE = {"elevation_m": "srtm_elevation_m", "chirps_precip_mm": "chirps_precip_mm", "ndvi_mean": "ndvi_mean",
             "ndvi_annual": "ndvi_annual", "swir_ratio": "swir_ratio"}
COORDINATES = {"longitude", "latitude", "lat", "lon", "GPS_Long", "GPS_Lat", "longitude_x", "latitude_x"}


def field_key(s: pd.Series) -> pd.Series:
    """F_id as a whole number. Some tables store it as 474888137.99999994 instead of 474888138."""
    return pd.to_numeric(s, errors="raise").round().astype("int64")


def build(v5: pd.DataFrame, v6: pd.DataFrame) -> pd.DataFrame:
    base = v5.drop(columns=[c for c in SATELLITE if c in v5.columns])
    sat = v6[["F_id", *SATELLITE]].rename(columns=SATELLITE)
    base = base.assign(_key=field_key(base["F_id"]))
    sat = sat.assign(_key=field_key(sat["F_id"])).drop(columns="F_id")
    if base["_key"].duplicated().any() or sat["_key"].duplicated().any():
        raise ValueError("F_id must be unique in both tables")
    if set(base["_key"]) != set(sat["_key"]):
        raise ValueError("v5 and v6 do not contain the same fields")
    out = base.merge(sat, on="_key", how="left", validate="one_to_one").drop(columns="_key")
    out["F_id"] = field_key(out["F_id"])
    leaked = COORDINATES & set(out.columns)
    if leaked:
        raise ValueError(f"coordinates must never be saved: {sorted(leaked)}")
    missing = out[list(SATELLITE.values())].isna().sum()
    if missing.any():
        raise ValueError(f"satellite values missing: {missing[missing > 0].to_dict()}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--v5", type=Path, default=PROCESSED / "feature_table_v5.csv")
    ap.add_argument("--v6", type=Path, default=PROCESSED / "feature_table_v6.csv")
    ap.add_argument("--out", type=Path, default=PROCESSED / "feature_table_v6b.csv")
    args = ap.parse_args(argv)
    out = build(pd.read_csv(args.v5, low_memory=False), pd.read_csv(args.v6, low_memory=False))
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(out)} rows and {out.shape[1]} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
