# Crop Recommendation Data

## Overview

This directory holds all datasets used to build the feature table for the crop recommendation model. All joins and decisions are documented in `notebooks/01_feature_engineering.ipynb`. The final feature tables are at `processed/feature_table.csv` (v1, reference) and `processed/feature_table_v2.csv` (v2, recommended for training).

---

## Datasets

### 1. CIAT Fieldwork Points (base table)

- **File:** `raw/ciat/ciat_points_clean.csv`
- **Rows:** 2,739 georeferenced crop observations
- **Collected:** May 6 to June 7, 2025, across four pilot districts: Nyagatare, Musanze, Nyabihu, Ruhango
- **Key columns:** `F_index` (point ID), `latitude`, `longitude`, `GPS_Elev`, `District`, `landcover`, `Crop_plant`
- **Labelled rows (landcover):** 1,567 rows have a crop label in `landcover`
- **Labelled rows (Crop_plant, pure crops only):** 1,904 rows have a single-crop label in `Crop_plant`
- **Unlabelled rows:** 1,172 rows have NaN in `landcover`
- **Label type:** Each row is a single GPS point, not a polygon or centroid.
- **Two label columns explained:**
  - `landcover`: original CIAT field label, includes intercrop combinations like "Bean and Maize". 1,567 non-null values.
  - `Crop_plant`: a parallel column in the same shapefile that records only the primary planted crop per plot, leaving intercrop plots as "Mixed crops" rather than combining them. 1,904 pure-crop labels. Preferred for training (see Clarifications, Q3).
- **Data quality filters (apply before training, not removed from raw file):**
  - `GPS_Elev > 900`: removes GPS acquisition failures. Rwanda's lowest point is around 950 m, so any elevation below 900 m is a failed GPS fix, not a real location.
  - `longitude < 31.0`: removes coordinate outliers that fall outside Rwanda's eastern boundary. A small number of points have garbled longitude values that place them east of Rwanda.

### 2. NASA POWER Seasonal Climate

- **File:** `raw/nasa_power/nasa_power_seasonal.csv`
- **Source:** NASA POWER API, daily data aggregated to seasonal means and totals
- **Coverage:** Four pilot districts, seasons A/B/C, years 2017-2024
- **Variables:** precipitation (mm), max temperature (C), min temperature (C), surface solar radiation (MJ/m2)
- **Join key:** district + season + year
- **Season/year alignment:** CIAT collected in May to June 2025, which is Season B. NASA POWER uses a start-year convention: Season B labelled `year=2024` covers March to June 2025. All joins use `season=B, year=2024`. Joining on `year=2025` returns all nulls because NASA POWER has not yet published 2025 data under that label.
- **Season B window:** March 1 to June 30. Season A runs September to January. February 1 falls in Season A, not Season B.
- **Known limitation:** NASA POWER provides one value per ~0.5 degree grid cell. Musanze and Nyabihu fall in the same grid cell, so their climate features are identical. The model cannot distinguish the two districts on climate alone. See Clarifications Q2 for the planned fix.

### 3. NISR Seasonal Agricultural Survey (SAS)

- **Files:** `raw/nisr/nisr_cultivated_area_ha.csv`, `nisr_harvested_area_ha.csv`, `nisr_production_mt.csv`, `nisr_yield_kg_ha.csv`
- **Source:** Rwanda National Institute of Statistics, Seasonal Agricultural Survey, district-level reports
- **Coverage:** Four pilot districts, seasons A/B/C, multiple years
- **Join key:** district + season + year (v2) or district + season + year + crop (v1)
- **Why v1 used crop-specific NISR stats and why that was wrong:** In feature table v1, each point received the NISR statistics for its own crop. A Maize point got Maize yield, a Rice point got Rice yield. This leaks the label into the features: the model learns that high rice yield predicts Rice, which is circular. A real agricultural officer does not know the crop before making the recommendation.
- **How v2 fixes this:** Instead of crop-specific statistics, v2 computes the district-level crop share, which is the fraction of the four model crops' total cultivated area each crop occupies, per district and season. Every point in a district gets the same four share values regardless of what crop is growing there. The features describe the agricultural landscape of the district, not the crop at the specific point.
- **v2 NISR columns:** `nisr_share_maize`, `nisr_share_beans`, `nisr_share_irish_potato`, `nisr_share_paddy_rice`, `nisr_total_model_crop_area_ha`
- **Crop mapping decision (Option 1):** CIAT intercrop labels (e.g. "Bean and Maize") are mapped to the dominant or primary crop so all labelled rows receive NISR features. The original CIAT label is preserved in `landcover`.

  | CIAT label | Mapped NISR crop |
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

- **Note on paddy rice share in highland districts:** Musanze and Nyabihu do not grow rice. NISR has no rice row for those districts, so `nisr_share_paddy_rice` is filled with 0 for all points in Musanze and Nyabihu. This is correct, not a data error.

### 4. iSDAsoil Soil Properties

- **File:** `raw/isda/isda_soil_points_v2.csv`
- **Source:** iSDAsoil REST API v2 (https://api.isda-africa.com)
- **Coverage:** All 2,739 CIAT points, queried individually by latitude and longitude
- **Properties:** nitrogen, phosphorus, potassium, pH, sand, clay, silt, organic carbon, cation exchange capacity
- **Depths:** 0-20 cm and 20-50 cm
- **Column naming:** `nitrogen_0_20cm`, `ph_20_50cm`, etc.
- **Authentication:** JWT token via POST /login, token expires after 1 hour, auto-refreshed every 55 minutes by the fetcher script
- **Fetch script:** `src/data_processing/isda_soil_fetcher_v2.py`, supports checkpoint/resume so it can be stopped and restarted without losing progress
- **Status as of 2026-10-27:** Fetch running, 1,161 of 2,739 points complete (42%). Expected to finish overnight 2026-10-27 to 2026-10-28. Re-run `src/data_processing/build_feature_table_v2.py` after file lands to include soil columns.
- **Fallback:** If the API becomes unavailable, the same soil layers are accessible on Google Earth Engine under the iSDAsoil dataset. The fetcher script can be adapted to use the GEE Python API instead.

---

## Feature Tables

### feature_table_v2.csv (recommended for training)

- **File:** `processed/feature_table_v2.csv`
- **Rows:** 2,739
- **Columns:** 26 without soil, 47 with soil (once iSDAsoil fetch completes and table is rebuilt)
- **Target variable (recommended):** `Crop_plant` filtered to pure-crop labels only (1,904 rows). See Clarifications Q3.
- **Target variable (alternative):** `nisr_crop` derived from `landcover` via Option 1 mapping (1,567 rows)
- **NISR features:** district crop share, no label leakage
- **Built by:** `src/data_processing/build_feature_table_v2.py`

### feature_table.csv (v1, kept as reference)

- **File:** `processed/feature_table.csv`
- **Rows:** 2,739
- **Columns:** 73 without soil, 94 with soil
- **Do not use for training:** NISR columns are crop-specific and leak the label. Kept for comparison and audit.
- **Built by:** `src/data_processing/build_feature_table.py`

### Pre-processing checklist before training (applies to both tables)

```python
df = df[df['GPS_Elev'] > 900]      # remove GPS elevation errors
df = df[df['longitude'] < 31.0]    # remove coordinate outliers outside Rwanda
```

For supervised learning, drop the 1,172 rows where both `landcover` and `Crop_plant` are null or "Mixed crops". Stratify train/test split by district to avoid Ruhango being underrepresented (see District breakdown below).

---

## District Breakdown

| District | Total points | Labelled (landcover) | Unlabelled | Notes |
|---|---|---|---|---|
| Nyagatare | 1,627 | 1,164 | 463 | Dominated by Bean and Maize intercrop (642 points). Dry lowland (~1,100 m). |
| Ruhango | 755 | 153 | 602 | 80% unlabelled. Mid-altitude (~1,600 m). See cassava limitation below. |
| Nyabihu | 174 | 116 | 58 | Mostly Irish potato. Volcanic highland (~2,200 m). |
| Musanze | 183 | 134 | 49 | Mixed crops. Volcanic highland (~1,800 m). |

### Crop labels per district

**Nyagatare:** Bean and Maize 642, Maize 415, Bean 55, Rice 35, Irish Potato 6, Maize and Bean 4, Irish potato and Maize 4, Irish potato 3

**Ruhango:** Maize 58, Bean and Maize 48, Bean 30, Rice 8, Irish Potato 4, Bean and Irish potato 1, Maize and Bean 1, Irish potato 1, Irish potato and Maize 1, Cassava and Maize 1

**Musanze:** Bean 46, Irish Potato 31, Irish potato 23, Maize 17, Bean and Maize 8, Irish Potato and Maize 5, Irish potato and Maize 2, Rice 1, Bean and Irish potato 1

**Nyabihu:** Irish potato 52, Maize 26, Irish Potato 16, Bean 15, Bean and Maize 4, Irish potato and Maize 2, Bean and Irish potato 1

### Ruhango cassava gap (known limitation)

Ruhango has 602 unlabelled points out of 755 total (80%). This is not a data collection failure. CIAT excluded cassava from the label set for this survey. According to NISR Season B data, cassava is the single largest crop in Ruhango at 18,706 ha cultivated, far ahead of the next crop. The CIAT enumerators visited fields but recorded no label when the crop was cassava, because it was outside the survey scope. Those 602 unlabelled points in Ruhango are almost certainly cassava plots.

Consequence for modelling: the model cannot predict cassava in Ruhango. Recommendations for Ruhango will be drawn from the 153 labelled points only, which are biased toward Maize and Beans. This is stated as a limitation in the report. A follow-up survey that includes cassava would be needed to make the model useful for Ruhango.

---

## Clarifications

### Q1: NISR label leakage and the fix

**Thierry's concern:** In feature table v1, each point received NISR statistics for its own crop (a Rice point got Rice yield). This leaks the label into the features. A real officer does not know the crop before using the model, so the model is learning a tautology.

**Resolution:** Feature table v2 replaces crop-specific NISR statistics with district-level crop share. For each district and season, the share columns record what fraction of the four model crops' total cultivated area each crop takes up. A Rice point and a Maize point in the same district get the same four share values. The features describe the agricultural landscape, not the crop at the point. Use `feature_table_v2.csv` for all modelling. Feature table v1 is kept only as a reference.

### Q2: Climate resolution and CHIRPS

**Thierry's concern:** NASA POWER gives one value per ~0.5 degree grid cell. Musanze and Nyabihu share a cell and are therefore identical on all climate features. The model may be learning district identity rather than climate.

**Current state:** NASA POWER climate is in both feature tables. Musanze and Nyabihu are indeed identical on all four climate columns. This is a known limitation.

**Planned improvement:** CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) provides rainfall at ~5 km resolution and can be queried per point using each CIAT coordinate. This would give each of the 2,739 points its own rainfall value instead of a district average. For temperature, the lapse rate adjustment by elevation (approximately -6.5 C per 1,000 m) is a reasonable approximation and can be applied to the NASA POWER district temperature using `GPS_Elev`. CHIRPS data is publicly available and accessible via the Google Earth Engine Python API or direct download. Not yet implemented for the midterm submission; flagged as a post-midterm improvement.

### Q3: Crop_plant versus landcover as the training target

**Thierry's concern:** `Crop_plant` and `landcover` disagree on 137 plots where `landcover` is "Bean and Maize" but `Crop_plant` says Beans. Using `nisr_crop` (derived from `landcover` via Option 1 mapping) assigns those 137 plots to Maize, which conflicts with what `Crop_plant` records as the primary planted crop.

**Recommendation: use Crop_plant with pure crops only.**

Reasons:
- `Crop_plant` gives 1,904 pure-crop labels versus 1,567 from `landcover`. More training data.
- `Crop_plant` avoids the intercrop ambiguity entirely. There is no need to decide which crop an intercrop plot belongs to.
- The Option 1 mapping of intercrop to dominant crop introduces noise. "Bean and Maize" mapped to Maize may be wrong for fields where beans occupy more area.
- `Crop_plant` is already in both feature tables. Filter with `df['Crop_plant'].isin(['Maize', 'Beans', 'Irish potatoes', 'Rice'])` before training.

`landcover` and `nisr_crop` are retained in the feature table for reference and for any ablation study comparing the two label strategies.

### Q4: iSDAsoil data availability

**Thierry's question:** When will the iSDAsoil data be ready? If the login fails, the same layers are on Google Earth Engine.

**Current status:** The JWT login is working. The fetcher script (`src/data_processing/isda_soil_fetcher_v2.py`) has been running since 2026-10-27 morning and has completed 1,161 of 2,739 points as of 17:22 on 2026-10-27. Expected to finish overnight. Once done, `build_feature_table_v2.py` will be re-run to produce the feature table with full soil columns.

**Fallback confirmed:** The same iSDAsoil layers (nitrogen, phosphorus, potassium, pH, sand, clay, silt, organic carbon, cation exchange capacity at 0-20 cm and 20-50 cm) are available on Google Earth Engine. If the API fetch fails or stalls, the GEE approach is the backup.

### Q5: GPS coordinates in the repo

**Thierry's concern:** The `.gitignore` had a typo (`cdcrop-recommendation` instead of `01-crop-recommendation`), which meant the data directory was not ignored. CSVs including farm GPS coordinates and village names were committed to the public repo.

**Resolution:** The typo is fixed. All CSVs were removed from git tracking with `git rm --cached`. The raw data files remain on local disk and on the shared drive. Only the processed feature tables (`feature_table.csv` and `feature_table_v2.csv`) are committed to the repo, since those contain only district-level joins and no individual GPS coordinates. The raw CIAT file with coordinates stays off the repo permanently.

---

## Excluded Datasets

| Dataset | Reason excluded |
|---|---|
| Rwanda Govt Soil Fertility (148 samples) | All samples in Gicumbi district, not a pilot district |
| CIMMYT 2022 soil (104 samples) | Coordinates truncated for privacy, cannot join spatially |
