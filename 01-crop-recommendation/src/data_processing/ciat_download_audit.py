"""
ciat_download_audit.py
----------------------
Downloads the CIAT ground reference fieldwork points from Harvard Dataverse,
unpacks them, and produces:
  1. data/raw/ciat/fieldwork_points.*  (original files, preserved)
  2. data/raw/ciat/ciat_points_clean.csv  (flattened, one row per point)
  3. data/raw/ciat/ciat_audit.txt  (label counts per district, missing values per column)

Harvard Dataverse DOI: 10.7910/DVN/O7IDTD
File: 02a. Fieldwork_Points.zip  (~665 KB)

Usage:
    python src/ingestion/ciat_download_audit.py

Requirements:
    pip install requests geopandas pandas
    (geopandas already installed if you ran the centroid work)
"""

import io, json, zipfile, sys
from pathlib import Path

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATAVERSE_BASE  = "https://dataverse.harvard.edu"
DATASET_DOI     = "doi:10.7910/DVN/O7IDTD"
# The persistent file ID for 02a. Fieldwork_Points.zip (V2 of the dataset)
# Obtained from the Dataverse file listing API.
POINTS_FILE_ID  = None   # resolved at runtime via API

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ciat"
AUDIT_PATH = OUTPUT_DIR / "ciat_audit.txt"
CLEAN_CSV  = OUTPUT_DIR / "ciat_points_clean.csv"

PILOT_DISTRICTS = ["Nyagatare", "Musanze", "Nyabihu", "Ruhango"]


# ---------------------------------------------------------------------------
# Step 1: resolve file ID from Dataverse API
# ---------------------------------------------------------------------------

def get_file_id(doi):
    """Return the file ID for Fieldwork_Points.zip from the dataset metadata."""
    url = f"{DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId={doi}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    files = resp.json()["data"]["latestVersion"]["files"]
    for f in files:
        label = f.get("label", "")
        if "Fieldwork_Points.zip" in label or "02a" in label:
            return f["dataFile"]["id"]
    # fallback: print all labels so user can identify manually
    print("Available files in dataset:")
    for f in files:
        print(f"  id={f['dataFile']['id']}  label={f.get('label','?')}")
    raise RuntimeError("Could not find Fieldwork_Points.zip in dataset file list.")


# ---------------------------------------------------------------------------
# Step 2: download and unpack
# ---------------------------------------------------------------------------

def download_and_unpack(file_id, output_dir):
    """Download the zip and extract to output_dir/fieldwork_points/."""
    url = f"{DATAVERSE_BASE}/api/access/datafile/{file_id}"
    print(f"Downloading from {url} ...")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunks = []
    for chunk in resp.iter_content(chunk_size=65536):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r  {downloaded/1024:.0f} KB / {total/1024:.0f} KB  ({pct:.0f}%)", end="", flush=True)
    print()

    raw = b"".join(chunks)
    dest = output_dir / "fieldwork_points"
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(dest)
    print(f"Extracted to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Step 3: read the shapefile or CSV inside the zip
# ---------------------------------------------------------------------------

def load_points(extract_dir):
    """
    Read the fieldwork points into a GeoDataFrame or plain DataFrame.
    The zip may contain a .shp, a .gpkg, or a .csv.
    """
    # Try shapefile first
    shps = list(extract_dir.rglob("*.shp"))
    gpkgs = list(extract_dir.rglob("*.gpkg"))
    csvs = list(extract_dir.rglob("*.csv"))

    if shps:
        import geopandas as gpd
        gdf = gpd.read_file(shps[0])
        print(f"Loaded shapefile: {shps[0].name}  ({len(gdf)} features)")
        # Add lat/lon columns from geometry
        gdf = gdf.to_crs(epsg=4326)
        gdf["longitude"] = gdf.geometry.x
        gdf["latitude"]  = gdf.geometry.y
        return gdf.drop(columns="geometry")

    if gpkgs:
        import geopandas as gpd
        gdf = gpd.read_file(gpkgs[0])
        print(f"Loaded GeoPackage: {gpkgs[0].name}  ({len(gdf)} features)")
        gdf = gdf.to_crs(epsg=4326)
        gdf["longitude"] = gdf.geometry.x
        gdf["latitude"]  = gdf.geometry.y
        return gdf.drop(columns="geometry")

    if csvs:
        df = pd.read_csv(csvs[0])
        print(f"Loaded CSV: {csvs[0].name}  ({len(df)} rows)")
        return df

    raise RuntimeError(f"No .shp, .gpkg or .csv found under {extract_dir}")


# ---------------------------------------------------------------------------
# Step 4: audit
# ---------------------------------------------------------------------------

def run_audit(df, audit_path):
    lines = []

    lines.append("=" * 60)
    lines.append("CIAT FIELDWORK POINTS AUDIT")
    lines.append("=" * 60)
    lines.append(f"Total points: {len(df)}")
    lines.append(f"Columns ({len(df.columns)}): {list(df.columns)}")
    lines.append("")

    # Label counts per district
    lines.append("--- Label counts per district ---")
    district_col = _find_col(df, ["district", "adm2", "District", "ADM2_EN", "district_name"])
    crop_col     = _find_col(df, ["crop_type", "crop", "Crop", "label", "class"])

    if district_col and crop_col:
        tbl = df.groupby([district_col, crop_col]).size().reset_index(name="count")
        lines.append(tbl.to_string(index=False))
    elif district_col:
        tbl = df[district_col].value_counts()
        lines.append(tbl.to_string())
    else:
        lines.append("Could not identify district/crop columns automatically.")
        lines.append(f"Available columns: {list(df.columns)}")

    lines.append("")

    # Missing values per column
    lines.append("--- Missing values per column ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    miss_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    miss_df = miss_df[miss_df["missing"] > 0].sort_values("missing", ascending=False)
    if miss_df.empty:
        lines.append("No missing values.")
    else:
        lines.append(miss_df.to_string())

    lines.append("")

    # Geometry check
    lat_col = _find_col(df, ["latitude", "lat", "y", "Y"])
    lon_col = _find_col(df, ["longitude", "lon", "x", "X"])
    if lat_col and lon_col:
        lines.append("--- Coordinate bounds ---")
        lines.append(f"  lat: {df[lat_col].min():.4f} to {df[lat_col].max():.4f}")
        lines.append(f"  lon: {df[lon_col].min():.4f} to {df[lon_col].max():.4f}")

    report = "\n".join(lines)
    audit_path.write_text(report)
    print(f"\n{report}")
    print(f"\nAudit written to {audit_path}")


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extract_dir = OUTPUT_DIR / "fieldwork_points"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        print("Resolving file ID from Harvard Dataverse ...")
        try:
            file_id = get_file_id(DATASET_DOI)
            print(f"File ID: {file_id}")
        except Exception as e:
            print(f"ERROR resolving file ID: {e}")
            print("\nManual fallback: download 02a. Fieldwork_Points.zip from")
            print("  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O7IDTD")
            print(f"and extract it to: {extract_dir}")
            sys.exit(1)

        download_and_unpack(file_id, OUTPUT_DIR)
    else:
        print(f"fieldwork_points/ already exists, skipping download.")

    print("\nLoading points ...")
    df = load_points(extract_dir)

    # Save clean CSV
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Clean CSV written to {CLEAN_CSV}  ({len(df)} rows)")

    # Run audit
    run_audit(df, AUDIT_PATH)


if __name__ == "__main__":
    main()
