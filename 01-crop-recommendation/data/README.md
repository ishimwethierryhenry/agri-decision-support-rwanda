# Crop Recommendation Data

## Overview

This directory holds all datasets used to build the feature table for the crop recommendation model. All joins and decisions are documented in `notebooks/01_feature_engineering.ipynb`. The final feature table is at `processed/feature_table.csv`.

---

## Datasets

### 1. CIAT Fieldwork Points (base table)

- **File:** `processed/ciat_points_clean.csv`
- **Rows:** 2,739 georeferenced crop observations
- **Collected:** May 6 to June 7, 2025, across four pilot districts: Nyagatare, Musanze, Nyabihu, Ruhango
- **Key columns:** `F_index` (point ID), `latitude`, `longitude`, `GPS_Elev`, `District`, `landcover`
- **Labelled rows:** 1,567 (have a crop label in `landcover`)
- **Unlabelled rows:** 1,172 (NaN in `landcover`, no crop recorded at that point)
- **Label type:** Each row is a single GPS point, not a polygon or centroid.
- **Data quality filters (apply before training, not removed from raw file):**
  - `GPS_Elev > 900`: removes GPS acquisition failures (Rwanda's lowest point is ~950m)
  - `longitude < 31.0`: removes coordinate outliers outside Rwanda's eastern boundary

### 2. NASA POWER Seasonal Climate

- **File:** `processed/nasa_power_seasonal.csv`
- **Source:** NASA POWER API, daily data aggregated by season
- **Coverage:** Four pilot districts, seasons A/B/C, years 2017-2024
- **Variables:** precipitation (mm), max temperature (C), min temperature (C), surface solar radiation (MJ/m2)
- **Join key:** district + season + year
- **Season/year alignment decision:** CIAT collected in May-June 2025 which is Season B. NASA POWER uses a start-year convention where Season B labelled year=2024 covers March to June 2025. All joins use `season=B, year=2024`. Joining on year=2025 returns all nulls.
- **Note for Thierry:** Season B runs March 1 to June 30, not February 1 to August. February falls in Season A. Using Feb 1 as the season start will pull the wrong climate window.

### 3. NISR Seasonal Agricultural Survey (SAS)

- **Files:** `processed/nisr_yield_kg_ha.csv`, `nisr_production_mt.csv`, `nisr_harvested_area_ha.csv`, `nisr_cultivated_area_ha.csv`
- **Source:** Rwanda National Institute of Statistics, SAS district-level reports
- **Coverage:** Four pilot districts, seasons A/B/C, multiple years
- **Join key:** district + season + year + crop
- **Crop mapping decision (Option 1):** CIAT uses plot-level intercrop labels (e.g. "Bean and Maize"). NISR reports single crops only. Intercrop plots are mapped to the dominant/primary crop so all 1,567 labelled rows receive NISR statistics. The original CIAT label is preserved in the `landcover` column. This decision was made to retain all labelled rows for training rather than leaving intercrop points without NISR features. Documented for Prof. Okeyo review.

  | CIAT label | NISR crop |
  |---|---|
  | Maize | Maize |
  | Bean | Beans |
  | Rice | Paddy rice |
  | Irish potato / Irish Potato | Irish potato |
  | Bean and Maize | Maize |
  | Maize and Bean | Maize |
  | Irish Potato and Maize | Irish potato |
  | Irish potato and Maize | Irish potato |
  | Bean and Irish potato | Beans |
  | Cassava and Maize | Maize |

### 4. iSDAsoil Soil Properties

- **File:** `processed/isda_soil_points_v2.csv` (push pending, fetch in progress)
- **Source:** iSDAsoil REST API v2 (https://api.isda-africa.com)
- **Coverage:** All 2,739 CIAT points, queried by lat/lon
- **Properties:** nitrogen, phosphorus, potassium, pH, sand, clay, silt, organic carbon, cation exchange capacity
- **Depths:** 0-20cm and 20-50cm
- **Column naming:** `nitrogen_0_20cm`, `ph_20_50cm`, etc.
- **Status:** Fetch running, expected complete by 2026-10-28. Re-run `src/ingestion/build_feature_table.py` after file lands to include soil columns in the feature table.

---

## Feature Table

- **File:** `processed/feature_table.csv`
- **Rows:** 2,739
- **Columns:** 73 without soil, 94 with soil (once iSDAsoil fetch completes)
- **Target variable:** `nisr_crop` (cleaned, NISR-aligned crop label for modelling)
- **Raw label:** `landcover` (original CIAT label, kept for reference)
- **Elevation:** available as `GPS_Elev` directly from CIAT shapefile, 100% coverage. No external DEM needed.
- **Built by:** `src/ingestion/build_feature_table.py`
- **Full column definitions:** `docs/feature_schema.md`

### Pre-processing checklist before training

```python
df = df[df['GPS_Elev'] > 900]      # remove GPS elevation errors
df = df[df['longitude'] < 31.0]    # remove coordinate outliers outside Rwanda
```

The 1,172 unlabelled rows (NaN in `landcover`) have no NISR statistics. Drop them for supervised learning or use for semi-supervised approaches.

---

## District breakdown

| District | Total points | Labelled | Unlabelled | Notes |
|---|---|---|---|---|
| Nyagatare | 1,627 | 1,164 | 463 | Dominated by Bean and Maize intercrop (642 points). Dry lowland. |
| Ruhango | 755 | 153 | 602 | 80% unlabelled, very limited labelled data for modelling. Mid-altitude. |
| Nyabihu | 174 | 116 | 58 | Mostly Irish potato. Volcanic highland. |
| Musanze | 183 | 134 | 49 | Mixed crops. Volcanic highland. |

**Note for Thierry:** Ruhango has only 153 labelled points out of 755 total. Stratify your train/test split by district to avoid Ruhango being underrepresented in training. District is also a strong feature given the agronomic differences between highland (Musanze, Nyabihu) and lowland (Nyagatare) zones.

### Crop labels per district

**Nyagatare:** Bean and Maize 642, Maize 415, Bean 55, Rice 35, Irish Potato 6, Maize and Bean 4, Irish potato and Maize 4, Irish potato 3

**Ruhango:** Maize 58, Bean and Maize 48, Bean 30, Rice 8, Irish Potato 4, Bean and Irish potato 1, Maize and Bean 1, Irish potato 1, Irish potato and Maize 1, Cassava and Maize 1

**Musanze:** Bean 46, Irish Potato 31, Irish potato 23, Maize 17, Bean and Maize 8, Irish Potato and Maize 5, Irish potato and Maize 2, Rice 1, Bean and Irish potato 1

**Nyabihu:** Irish potato 52, Maize 26, Irish Potato 16, Bean 15, Bean and Maize 4, Irish potato and Maize 2, Bean and Irish potato 1

---

## Answers to Part B questions

**Are CIAT labels points or polygons?**
Points. Each row is a single GPS coordinate with one crop label. No polygon centroids.

**Is elevation available?**
Yes. `GPS_Elev` is in the feature table with 100% coverage, sourced directly from the CIAT shapefile. No external DEM is needed.

**Is there a swamp or irrigation flag?**
No. None of the available data sources include a swamp or irrigation indicator at point level. This is a known limitation: rice is largely grown in marshland in Rwanda, so the rice prediction rule will be weaker without it. Stated as a limitation in the report.

**Is cassava in the CIAT labels?**
One point only: "Cassava and Maize" in Ruhango. Mapped to Maize under Option 1. Cassava cannot be predicted as a standalone class.

**Season window for Season B:**
Season B runs March 1 to June 30. CIAT collected May 6 to June 7, 2025, which is fully within Season B. The season start is March 1, not February 1. February 1 falls in Season A (September to January).

---

## Excluded Datasets

| Dataset | Reason excluded |
|---|---|
| Rwanda Govt Soil Fertility (148 samples) | All samples in Gicumbi district, not a pilot district |
| CIMMYT 2022 soil (104 samples) | Coordinates truncated for privacy, cannot join spatially |
