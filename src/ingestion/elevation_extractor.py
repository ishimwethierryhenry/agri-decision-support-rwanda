"""
elevation_extractor.py
----------------------
Samples SRTM 30m elevation at each CIAT fieldwork point and adds
an elevation_m column to the clean CIAT CSV.

Two modes:
  1. API mode (default): queries the open-elevation.com API, no files needed.
  2. File mode (--dem-file): samples a local SRTM .tif with rasterio.

API mode is simpler to run. File mode is faster for large point sets
and works offline.

Output: data/raw/ciat/ciat_points_with_elevation.csv

Usage:
    # API mode (internet required):
    python src/ingestion/elevation_extractor.py

    # File mode (download SRTM tiles first):
    python src/ingestion/elevation_extractor.py --dem-file data/raw/dem/srtm_rwanda.tif
"""

import csv, json, time, sys, argparse
import urllib.request, urllib.error
from pathlib import Path

CIAT_CLEAN   = Path(__file__).resolve().parents[2] / "data/raw/ciat/ciat_points_clean.csv"
OUTPUT_PATH  = Path(__file__).resolve().parents[2] / "data/raw/ciat/ciat_points_with_elevation.csv"

OPEN_ELEV_URL = "https://api.open-elevation.com/api/v1/lookup"
BATCH_SIZE    = 100   # points per API request


# ---------------------------------------------------------------------------
# API mode: open-elevation.com (free, no auth)
# ---------------------------------------------------------------------------

def fetch_elevation_api(coords):
    """
    coords: list of {"latitude": y, "longitude": x}
    Returns list of elevation values (metres), same order, None on failure.
    """
    payload = json.dumps({"locations": coords}).encode()
    req = urllib.request.Request(
        OPEN_ELEV_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            return [r.get("elevation") for r in data["results"]]
        except Exception as e:
            if attempt == 3:
                print(f"    elevation API failed: {e}")
                return [None] * len(coords)
            time.sleep(3)
    return [None] * len(coords)


# ---------------------------------------------------------------------------
# File mode: rasterio point sampling
# ---------------------------------------------------------------------------

def fetch_elevation_raster(lats, lons, dem_path):
    """
    Sample a local GeoTIFF at (lat, lon) pairs using rasterio.
    Returns list of elevation values (metres).
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        print("rasterio not installed. Run: pip install rasterio")
        sys.exit(1)

    with rasterio.open(dem_path) as src:
        elevations = []
        for lat, lon in zip(lats, lons):
            try:
                row, col = rowcol(src.transform, lon, lat)
                val = list(src.sample([(lon, lat)]))[0][0]
                # SRTM uses -32768 as nodata
                elevations.append(None if val == -32768 else float(val))
            except Exception:
                elevations.append(None)
    return elevations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols: return c
        if c.lower() in lower_map: return lower_map[c.lower()]
    return None


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or list(rows[0].keys()) if rows else []
    return rows, cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", default=str(CIAT_CLEAN),
                        help="Path to ciat_points_clean.csv")
    parser.add_argument("--dem-file", default=None,
                        help="Path to a local SRTM GeoTIFF (file mode)")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    points_path = Path(args.points)
    if not points_path.exists():
        print(f"CIAT points not found at {points_path}")
        print("Run ciat_download_audit.py first.")
        sys.exit(1)

    rows, cols = load_csv(points_path)
    lat_col = _find_col(cols, ["latitude", "lat", "y", "Y"])
    lon_col = _find_col(cols, ["longitude", "lon", "x", "X"])

    if not lat_col or not lon_col:
        print(f"Cannot find lat/lon columns. Available: {cols}")
        sys.exit(1)

    lats = [float(r[lat_col]) for r in rows]
    lons = [float(r[lon_col]) for r in rows]
    elevations = [None] * len(rows)

    if args.dem_file:
        print(f"Sampling elevation from {args.dem_file} ...")
        elevations = fetch_elevation_raster(lats, lons, args.dem_file)
        print(f"Done. {sum(e is not None for e in elevations)}/{len(elevations)} values extracted.")
    else:
        print(f"Fetching elevation via open-elevation API for {len(rows)} points ...")
        print(f"Batching in groups of {BATCH_SIZE} ...")
        for i in range(0, len(rows), BATCH_SIZE):
            batch_lats = lats[i:i+BATCH_SIZE]
            batch_lons = lons[i:i+BATCH_SIZE]
            coords = [{"latitude": la, "longitude": lo}
                      for la, lo in zip(batch_lats, batch_lons)]
            print(f"  Batch {i//BATCH_SIZE + 1} ({i+1}-{min(i+BATCH_SIZE, len(rows))}) ...", end=" ", flush=True)
            batch_elev = fetch_elevation_api(coords)
            elevations[i:i+BATCH_SIZE] = batch_elev
            ok = sum(e is not None for e in batch_elev)
            print(f"{ok}/{len(batch_elev)} OK")
            time.sleep(1)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = cols + (["elevation_m"] if "elevation_m" not in cols else [])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for row, elev in zip(rows, elevations):
            row["elevation_m"] = round(elev, 1) if elev is not None else ""
            writer.writerow(row)

    non_null = sum(e is not None for e in elevations)
    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(f"Elevation coverage: {non_null}/{len(rows)} points ({non_null/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
