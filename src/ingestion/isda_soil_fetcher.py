"""
isda_soil_fetcher.py
--------------------
Fetches iSDAsoil properties for the four pilot districts using point queries
at each CIAT fieldwork point coordinate, then summarises to district level.

If CIAT points are not yet available, falls back to district centroid coordinates.

Output: data/raw/isda/isda_soil_points.csv   (one row per CIAT point)
        data/raw/isda/isda_soil_district.csv  (district-level means)

iSDAsoil REST API: https://www.isda-africa.com/isdasoil/
  Endpoint: GET /query?lon=<lon>&lat=<lat>&property=<prop>&depth=<depth>
  No authentication required.
  Coverage: Africa only, 30m resolution.
  Depths available: 0-20, 20-50 (cm)

Properties fetched (agronomically relevant for fertilizer recommendation):
  nitrogen          - total nitrogen (g/kg)
  phosphorous_extractable - available P (mg/kg)
  potassium_extractable   - exchangeable K (cmol/kg)
  ph                - soil pH (in water)
  sand              - sand fraction (%)
  clay              - clay fraction (%)
  silt              - silt fraction (%)
  carbon_organic    - organic carbon (g/kg)
  cation_exchange_capacity - CEC (cmol/kg)

Usage:
    # With CIAT points (recommended):
    python src/ingestion/isda_soil_fetcher.py --points data/raw/ciat/ciat_points_clean.csv

    # Fallback to district centroids only:
    python src/ingestion/isda_soil_fetcher.py --centroids-only
"""

import csv, json, time, sys, argparse
import urllib.request, urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ISDA_BASE = "https://api.isda-africa.com/v1/soilproperty"

PROPERTIES = [
    "nitrogen",
    "phosphorous_extractable",
    "potassium_extractable",
    "ph",
    "sand",
    "clay",
    "silt",
    "carbon_organic",
    "cation_exchange_capacity",
]

DEPTHS = ["0-20", "20-50"]   # cm

DISTRICT_CENTROIDS = {
    "Nyagatare": {"lat": -1.338251, "lon": 30.379895},
    "Musanze":   {"lat": -1.498533, "lon": 29.606584},
    "Nyabihu":   {"lat": -1.647387, "lon": 29.510505},
    "Ruhango":   {"lat": -2.193599, "lon": 29.771734},
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "isda"
RETRY_WAIT  = 3
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def fetch_isda(lat, lon, prop, depth):
    """
    Query iSDAsoil for a single property/depth at a point.
    Returns float value, or None on missing data or error.
    """
    url = f"{ISDA_BASE}?lat={lat}&lon={lon}&property={prop}&depth={depth}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            # Response schema: {"property": {"value": ..., "unit": ...}}
            val = data.get(prop, {}).get("value")
            return float(val) if val is not None else None
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_WAIT)
    return None


def fetch_all_properties(lat, lon):
    """Return dict of {prop_depth: value} for one point."""
    result = {}
    for prop in PROPERTIES:
        for depth in DEPTHS:
            key = f"{prop}_{depth.replace('-','_')}cm"
            result[key] = fetch_isda(lat, lon, prop, depth)
    return result


# ---------------------------------------------------------------------------
# Load CIAT points
# ---------------------------------------------------------------------------

def load_ciat_points(csv_path):
    """Load CIAT clean CSV and return list of {district, lat, lon, point_id} dicts."""
    import csv as csv_mod
    points = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        cols = reader.fieldnames or []
        # Find district, lat, lon columns
        district_col = _find_col(cols, ["district", "adm2", "District", "ADM2_EN"])
        lat_col      = _find_col(cols, ["latitude", "lat", "y", "Y"])
        lon_col      = _find_col(cols, ["longitude", "lon", "x", "X"])
        id_col       = _find_col(cols, ["id", "ID", "point_id", "FID", "fid"])

        if not lat_col or not lon_col:
            raise RuntimeError(f"Cannot find lat/lon columns in {csv_path}. Columns: {cols}")

        for i, row in enumerate(reader):
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (ValueError, TypeError):
                continue
            points.append({
                "point_id": row.get(id_col, str(i)) if id_col else str(i),
                "district": row.get(district_col, "Unknown") if district_col else "Unknown",
                "lat": lat,
                "lon": lon,
            })
    return points


def _find_col(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", default=None,
                        help="Path to ciat_points_clean.csv")
    parser.add_argument("--centroids-only", action="store_true",
                        help="Use district centroids instead of CIAT points")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Limit number of CIAT points queried (for testing)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of points to query
    if args.centroids_only:
        points = [
            {"point_id": name, "district": name, "lat": c["lat"], "lon": c["lon"]}
            for name, c in DISTRICT_CENTROIDS.items()
        ]
        print(f"Using {len(points)} district centroids.")
    else:
        ciat_path = Path(args.points) if args.points else \
                    Path(__file__).resolve().parents[2] / "data/raw/ciat/ciat_points_clean.csv"
        if not ciat_path.exists():
            print(f"CIAT points not found at {ciat_path}")
            print("Run ciat_download_audit.py first, or use --centroids-only")
            sys.exit(1)
        points = load_ciat_points(ciat_path)
        print(f"Loaded {len(points)} CIAT points from {ciat_path}")
        if args.max_points:
            points = points[:args.max_points]
            print(f"Limited to {len(points)} points (--max-points)")

    # Column names for output
    prop_cols = [f"{p}_{d.replace('-','_')}cm" for p in PROPERTIES for d in DEPTHS]
    fieldnames = ["point_id", "district", "lat", "lon"] + prop_cols

    points_out = OUTPUT_DIR / "isda_soil_points.csv"
    rows = []

    print(f"\nQuerying iSDAsoil for {len(points)} points x {len(PROPERTIES)} properties x {len(DEPTHS)} depths ...")
    print(f"Estimated time: {len(points) * len(PROPERTIES) * len(DEPTHS) * 0.5 / 60:.1f} min\n")

    for i, pt in enumerate(points):
        print(f"  [{i+1}/{len(points)}] {pt['district']} point {pt['point_id']} ...", end=" ", flush=True)
        soil = fetch_all_properties(pt["lat"], pt["lon"])
        row = {
            "point_id": pt["point_id"],
            "district": pt["district"],
            "lat": pt["lat"],
            "lon": pt["lon"],
            **soil
        }
        rows.append(row)
        print("OK")
        time.sleep(0.2)   # be polite to the API

    with open(points_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {points_out}")

    # District-level summary (means across points)
    if not args.centroids_only and len(rows) > len(DISTRICT_CENTROIDS):
        import statistics
        district_out = OUTPUT_DIR / "isda_soil_district.csv"
        district_rows = []
        districts = sorted(set(r["district"] for r in rows))
        for d in districts:
            d_rows = [r for r in rows if r["district"] == d]
            summary = {"district": d, "n_points": len(d_rows)}
            for col in prop_cols:
                vals = [r[col] for r in d_rows if r[col] is not None]
                summary[col] = round(statistics.mean(vals), 4) if vals else ""
            district_rows.append(summary)

        dist_fieldnames = ["district", "n_points"] + prop_cols
        with open(district_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dist_fieldnames)
            writer.writeheader()
            writer.writerows(district_rows)
        print(f"Wrote district summary to {district_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
