# Feature Schema: Crop Recommendation Model
**Agri-DSS Practicum — CMU Africa Fall 2026**
Authors: Arsene (Part A, data ingestion) | Thierry (Part B, modelling)
Last updated: 2026-10-26

---

## Overview

The feature table joins four data sources into one row per labelled crop observation.
The CIAT fieldwork points are the base table (2,739 rows). Climate, soil, and historical
statistics are joined to each point by district and season.

Target variable: `crop_type` (maize, beans, rice, Irish potato)

---

## Join keys

| Key | Description |
|---|---|
| `district` | One of: Nyagatare, Musanze, Nyabihu, Ruhango |
| `season` | Rwanda season: A (Sep-Jan), B (Mar-Jun), C (Jun-Aug) |
| `year` | Season start year, 2017-2025 |
| `point_id` | CIAT point identifier (from F_index in original shapefile) |

CIAT collection was May-June 2025, which falls in Season B, start year 2024.
Climate and soil features are joined at district level.
iSDAsoil features are joined at point level (nearest API query).

---

## Target variable

| Column | Source | Type | Notes |
|---|---|---|---|
| `crop_type` | CIAT | string | maize, beans, rice, Irish potato |
| `district` | CIAT | string | pilot district name |
| `season` | derived | string | A, B, or C |
| `latitude` | CIAT GPS_Lat | float | WGS84, degrees |
| `longitude` | CIAT GPS_Long | float | WGS84, degrees |
| `elevation_m` | CIAT GPS_Elev | float | metres, 100% coverage |

---

## Climate features (NASA POWER)

Joined by district + season + year. Source: nasa_power_seasonal.csv

| Column | Unit | Aggregation | Notes |
|---|---|---|---|
| `precip_mm` | mm | seasonal sum | PRECTOTCORR |
| `tmax_c` | C | seasonal mean | T2M_MAX |
| `tmin_c` | C | seasonal mean | T2M_MIN |
| `solar_mj_m2` | MJ/m2 | seasonal mean | ALLSKY_SFC_SW_DWN |

Coverage: 4 districts x 8 start years (2017-2024) x 3 seasons = 96 district-season combinations.

---

## Soil features (iSDAsoil)

Joined at point level. Source: isda_soil_points_v2.csv
Two depths: 0-20cm and 20-50cm. Column naming: {property}_{depth}cm

| Column | Unit | Notes |
|---|---|---|
| `nitrogen_0_20cm` | g/kg | total nitrogen, topsoil |
| `nitrogen_20_50cm` | g/kg | total nitrogen, subsoil |
| `phosphorous_extractable_0_20cm` | mg/kg | available phosphorus |
| `phosphorous_extractable_20_50cm` | mg/kg | |
| `potassium_extractable_0_20cm` | cmol/kg | exchangeable K |
| `potassium_extractable_20_50cm` | cmol/kg | |
| `ph_0_20cm` | unitless | soil pH in water |
| `ph_20_50cm` | unitless | |
| `sand_0_20cm` | % | sand fraction |
| `sand_20_50cm` | % | |
| `clay_0_20cm` | % | clay fraction |
| `clay_20_50cm` | % | |
| `silt_0_20cm` | % | silt fraction |
| `silt_20_50cm` | % | |
| `carbon_organic_0_20cm` | g/kg | soil organic carbon |
| `carbon_organic_20_50cm` | g/kg | |
| `cation_exchange_capacity_0_20cm` | cmol/kg | CEC |
| `cation_exchange_capacity_20_50cm` | cmol/kg | |

Source: iSDAsoil REST API, 30m resolution, Africa-wide model.

---

## Historical crop statistics (NISR)

Joined by district + season + year. Source: nisr_*.csv

| Column | Unit | Notes |
|---|---|---|
| `nisr_cultivated_area_ha` | hectares | cultivated area for that crop in that district-season |
| `nisr_harvested_area_ha` | hectares | harvested area |
| `nisr_production_mt` | metric tonnes | total production |
| `nisr_yield_kg_ha` | kg/ha | yield per hectare |

Coverage: 2017-2025, 4 districts, seasons A and B.

---

## Known limitations

| Limitation | Impact |
|---|---|
| Rwanda Govt soil fertility dataset covers Gicumbi only, not pilot districts | iSDAsoil modelled estimates used instead; API v2 requires JWT auth; credentials obtained Oct 2026, fetch running, soil join complete by 2026-10-28 |
| CIMMYT 2022 has 104 samples with truncated coordinates | Excluded, cannot join spatially |
| CIAT GPS_Elev includes some 0.0m readings | Filter elevation_m > 900 before training |
| CIAT GPS_Long includes outliers above 31.0 | Filter longitude < 31.0 before training |
| No swamp/irrigation flag at point level | Rice predictions may underperform |
| No cassava in CIAT labels | Cassava cannot be predicted |
| CIAT collection was single-season (Season B, start year 2024 only) | Labels are a snapshot |

---

## Recommended pre-processing before training

1. Filter: elevation_m > 900
2. Filter: longitude < 31.0
3. Join climate features on district + season + year
4. Join NISR statistics on district + season + year
5. Soil features already joined at point level in isda_soil_points_v2.csv
6. Drop columns with more than 30% missing values
7. One-hot encode district and season for tree models
