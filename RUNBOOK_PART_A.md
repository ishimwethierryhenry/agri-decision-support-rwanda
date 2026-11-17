# Part A Runbook — Arsene
Data ingestion pipeline for the Agricultural DSS practicum.
Run every script from the project root: `cd ~/projects/agri-dss`

---

## Status summary

| Script | Status | Output |
|---|---|---|
| nisr_excel_loader.py | DONE | data/raw/nisr/*.csv (4 files) |
| nasa_power_fetcher.py | NOT RUN | data/raw/nasa_power/nasa_power_seasonal.csv |
| ciat_download_audit.py | NOT RUN | data/raw/ciat/ciat_points_clean.csv + ciat_audit.txt |
| elevation_extractor.py | NOT RUN | data/raw/ciat/ciat_points_with_elevation.csv |
| isda_soil_fetcher.py | NOT RUN | data/raw/isda/isda_soil_points.csv + isda_soil_district.csv |

---

## Step 1: NASA POWER (run first, takes ~3 min)

```bash
python src/ingestion/nasa_power_fetcher.py
```

Fetches daily climate data 2017-2025 for Nyagatare, Musanze, Nyabihu, Ruhango.
Aggregates by season A/B/C. Writes 384 rows.

Expected output:
```
Nyagatare  (-1.338251, 30.379895)
  2017-09-01 to 2018-08-31 ... OK
  ...
Wrote 384 rows to data/raw/nasa_power/nasa_power_seasonal.csv  (expected 384)
```

---

## Step 2: CIAT download and audit (run second, takes ~2 min)

```bash
pip install requests geopandas
python src/ingestion/ciat_download_audit.py
```

Downloads Fieldwork_Points.zip from Harvard Dataverse (no login needed).
Extracts and reads the shapefile/CSV inside.
Writes the clean CSV and prints the audit for Thierry.

If the automatic download fails (Harvard Dataverse sometimes requires accepting
custom terms), download manually:
1. Go to: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O7IDTD
2. Download: 02a. Fieldwork_Points.zip
3. Extract to: data/raw/ciat/fieldwork_points/
4. Re-run the script (it will skip the download step):
   python src/ingestion/ciat_download_audit.py

---

## Step 3: Elevation extraction (run after Step 2, takes ~5 min)

```bash
python src/ingestion/elevation_extractor.py
```

Queries open-elevation.com API at each CIAT point coordinate.
Adds elevation_m column. Writes ciat_points_with_elevation.csv.

No additional installs needed (uses stdlib urllib).

---

## Step 4: iSDAsoil (run after Step 2, takes ~20-30 min for all points)

```bash
# Full run against all CIAT points:
python src/ingestion/isda_soil_fetcher.py --points data/raw/ciat/ciat_points_clean.csv

# Quick test (first 10 points only):
python src/ingestion/isda_soil_fetcher.py --points data/raw/ciat/ciat_points_clean.csv --max-points 10

# Fallback if CIAT data not yet available:
python src/ingestion/isda_soil_fetcher.py --centroids-only
```

Queries iSDAsoil API for 9 soil properties at 0-20cm and 20-50cm depth.
Writes per-point CSV and a district-level summary.

---

## Answering Thierry's questions (after running Steps 2-3)

After Step 2 completes, the audit file at data/raw/ciat/ciat_audit.txt will contain:
- Label counts per district (answers question 7)
- Missing value counts per column (answers question 7)
- Geometry bounds confirming these are points not polygons (answers question 3)

Manual answers to his other questions:

**Q2. Season windows (Feb 01 to Aug = Season B)**
Standard Season B is March 1 to June 30. His Feb 1 start is slightly early.
More importantly, CIAT collected May 6 to June 7 2025, which is mid-Season B.
Labels are a single-season snapshot. Flag this in the baseline notes.

**Q3. CIAT labels: points or polygons?**
Both exist in the dataset. Fieldwork_Points.zip contains GPS survey points.
Fieldwork_Polygons.zip contains digitized polygons derived from those points.
For the ML feature table, join on points (or polygon centroids).

**Q4. Elevation added?**
Yes, after running elevation_extractor.py.

**Q5. Swamp/irrigation flag**
Not feasible in the 7-day window without a dedicated WaPOR pipeline.
State as a limitation: "Marshland/irrigation status not available at point level;
rice predictions may be weaker without this signal."

**Q6. Cassava in CIAT data?**
No. The dataset covers only maize, beans, rice, and Irish potato
(Rwanda Crop Intensification Programme priority crops).

**Q7. Label counts per district + missing values**
See data/raw/ciat/ciat_audit.txt after running Step 2.
