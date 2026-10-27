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
- **Authentication:** JWT token via POST /login, expires 1 hour, auto-refreshed every 55 minutes
- **Status:** Fetch running overnight, expected to complete by morning of 2026-10-28. Re-run `src/ingestion/build_feature_table.py` after file lands to include soil columns in the feature table.

---

## Feature Table

- **File:** `processed/feature_table.csv`
- **Rows:** 2,739
- **Columns:** 73 without soil, 94 with soil (once iSDAsoil fetch completes)
- **Target variable:** `nisr_crop` (cleaned, NISR-aligned crop label for modelling)
- **Raw label:** `landcover` (original CIAT label, kept for reference)
- **Built by:** `src/ingestion/build_feature_table.py`
- **Full column definitions:** `docs/feature_schema.md`

### Pre-processing checklist before training

```python
df = df[df['GPS_Elev'] > 900]      # remove GPS elevation errors
df = df[df['longitude'] < 31.0]    # remove coordinate outliers outside Rwanda
```

The 1,172 unlabelled rows (NaN in `landcover`) have no NISR statistics. Drop them for supervised learning or use for semi-supervised approaches.

---

## Excluded Datasets

| Dataset | Reason excluded |
|---|---|
| Rwanda Govt Soil Fertility (148 samples) | All samples in Gicumbi district, not a pilot district |
| CIMMYT 2022 soil (104 samples) | Coordinates truncated for privacy, cannot join spatially |
