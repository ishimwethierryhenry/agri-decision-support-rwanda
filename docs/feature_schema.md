# Feature schema (Part A to Part B handoff)

Status: DRAFT v0.1, proposed by Thierry (Part B) on 20 Oct 2026. Not agreed until Arsene (Part A) confirms it below.

Owner of the table: Arsene (Part A). Consumer: Thierry (Part B), who builds the expert-rules baseline, Random Forest and XGBoost against this schema.

Rule from the task sheet: after both of us agree, any column change needs a conversation first. Every change also raises `schema_version`.

## 1. What one row is

One row is one labelled location in one season. In the CIAT data this is one field point with a crop label. All raster layers (rainfall, temperature, NDVI, elevation, soil) are sampled at that point and summarised over the season window.

If the CIAT labels turn out to be polygons or village centroids instead of points, the row stays the same (one labelled location per season) and `lat` and `lon` become the centroid. Arsene to confirm when the file is opened.

## 2. Columns

Required columns (needed for the baseline and the first model run):

| # | Column | Type | Unit | Source | Notes and known gaps |
|---|--------|------|------|--------|----------------------|
| 1 | `point_id` | string | none | CIAT | Unique id per row. Keep the CIAT id if one exists. |
| 2 | `district` | string | none | CIAT | One of Nyagatare, Musanze, Nyabihu, Ruhango. This is the split key: one district is held out per run. |
| 3 | `province` | string | none | CIAT | East, North, West, South. For reports only, not a feature. |
| 4 | `season` | string | none | CIAT | Code such as `2025B`. Only one season exists so far. |
| 5 | `lat` | float | decimal degrees, WGS84 | CIAT | Used to sample rasters. |
| 6 | `lon` | float | decimal degrees, WGS84 | CIAT | Used to sample rasters. |
| 7 | `elevation_m` | float | metres | DEM (to be chosen) | NEW request. The expert-rules baseline needs altitude, and it is not on the sheet yet. |
| 8 | `rain_total_mm` | float | mm over the season window | CHIRPS | Sum of daily rainfall inside the window in section 3. |
| 9 | `tmean_c` | float | degrees C | NASA POWER | Mean of daily mean air temperature in the window. NASA POWER is coarse (about 50 km cells), so nearby points can share a value. |
| 10 | `ndvi_mean` | float | index, -1 to 1 | Sentinel-2 GeoMAD | Mean NDVI at the point. GeoMAD is a composite, so check which period it covers. |
| 11 | `soil_ph` | float | pH in water | iSDAsoil | Back-transform the raw scaled values and note the factor used. |
| 12 | `soil_soc_gkg` | float | g per kg | iSDAsoil | Soil organic carbon. Say which depth layer is used. |
| 13 | `soil_n_gkg` | float | g per kg | iSDAsoil | Total nitrogen. Same depth layer as SOC. |
| 14 | `soil_source` | string | none | pipeline | `isdasoil` for every row for now (see section 4). |
| 15 | `crop_label` | string | none | CIAT | Allowed values in section 5. This is the target. |
| 16 | `schema_version` | string | none | pipeline | `0.1` for this draft. |

Optional columns (add only if cheap; Part B will not wait for them):

| Column | Type | Unit | Source | Why it helps |
|--------|------|------|--------|--------------|
| `tmax_c`, `tmin_c` | float | degrees C | NASA POWER | Frost and heat limits in the baseline rules. |
| `rain_days_gt1mm` | int | days | CHIRPS | Rain distribution, not only the total. |
| `ndvi_max` | float | index | Sentinel-2 GeoMAD | Peak greenness. |
| `soil_p_mgkg` | float | mg per kg | iSDAsoil | Fertiliser rules use phosphorus. |
| `soil_k_mgkg` | float | mg per kg | iSDAsoil | Fertiliser rules use potassium. |
| `soil_clay_pct`, `soil_sand_pct` | float | percent | iSDAsoil | Texture. |

Missing values: leave the cell empty (read as NaN). Do not use -999 or 0 for missing. If a column is more than 30 percent empty in a district, say so in section 6.

## 3. Season window (to agree)

Proposal: season B means 1 February to 30 June. Arsene to confirm against the CIAT field calendar, because the rainfall and temperature columns depend on it. The same window is used for every district.

## 4. Two soil roles, kept apart

- The CIAT districts (Nyagatare, Musanze, Nyabihu, Ruhango) do not overlap with the measured soil samples Prof sent (GAIA: Burera, Ngororero, Nyaruguru; government spreadsheet: ten sectors in northern Rwanda). So the soil columns in the feature table come from iSDAsoil.
- The measured samples go in a separate validation table, `soil_validation`, used to check iSDAsoil against measured values and to build the fertiliser rules. Suggested columns: `sample_id`, `dataset`, `lat`, `lon`, `ph_water`, `soc_pct`, `tn_pct`, `p`, `k`, `cec`, `texture`, `isdasoil_ph`, `isdasoil_soc`. Units must be converted first (the datasets use mg per kg, meq per 100 g and percent).

## 5. Crop label vocabulary

Lower case, underscore, exactly as follows:

`beans`, `irish_potato`, `maize`, `rice`

Cassava is not in the CIAT labels as far as we know. The SoW lists maize, beans and cassava, so the team still has to agree one list. Until then Part B builds on the four labels above.

## 6. Known gaps (Arsene to fill in after the quality check)

- Missing values per column: to do.
- Label counts per district and crop: to do.
- Districts too thin for ML (fewer than a threshold set with Thierry): to do.
- Only season B 2025 exists, so we cannot test across seasons.

## 7. Handoff

- Schema document: this file, kept in `docs/feature_schema.md` in the repo.
- The data itself is not committed to git (`.gitignore` blocks csv and parquet). Share it on the team drive and name it `features_v0.1.csv` so the version matches the schema.
- Dates from the task sheet: schema agreed by 20 Oct, clean feature table to Part B by 27 Oct.

## 8. Agreement

- [ ] Arsene confirms columns and units
- [ ] Arsene confirms season window
- [ ] Arsene confirms point or centroid
- [ ] Thierry confirms the schema is enough for the baseline and the models
