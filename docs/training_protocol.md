# Training protocol: crop recommendation (version 1.0)

Written on 28 Oct 2026 by Thierry (Part B), with the data decisions made the same day by Arsène (Part A), before any official result. Every number in `01-crop-recommendation/reports/eval_v1/` follows this file. If anything here changes after we see results, the report says so and shows both versions.

Run everything with one command from the repo root:

```
python 01-crop-recommendation/src/models/evaluate.py
```

It writes `01-crop-recommendation/reports/eval_v1/` (report, CSV files and `run_info.json`). To get exactly the official numbers, install the pinned versions first: `python -m pip install -r 01-crop-recommendation/requirements-eval.txt`.

## 1. Data

- Table: `01-crop-recommendation/data/processed/feature_table_v2.csv` from Part A (commit `41e350d`). Every run saves the table's SHA-256 fingerprint and the code commit in `run_info.json`.
- One season: CIAT fieldwork of May to June 2025, season B (stored as `season=B`, `year=2024`, start-year convention).
- Season B window for climate: 1 February to 30 June. Part A moved the start from 1 March on 28 Oct because 1,072 of the 1,904 training fields were planted in January or February (`Plant_date`).
- Soil: iSDAsoil, joined on `F_id` (the 274 fields that had another field's soil were refetched on 28 Oct).

## 2. Rows

- Keep fields with `qc_ok == True` (GPS elevation above 900 m and longitude below 31.0): 2,728 of 2,739.
- Label: `Crop_plant`, single crops only: maize, beans, Irish potato, rice. Mixed crops are dropped.
- **Main set: all 1,904 single-crop fields.**
- **Second check: the 650 fields CIAT marked `Status == "Used"`.** Part A suggested training on these only. We keep them as the second check because CIAT's README says Status marks the plots CIAT used in its own satellite crop model, not plots with wrong labels: Used plots are mostly large, pure, consolidated fields (69 percent pure cropping against 44 percent), while Not Used plots carry the same crop labels (CIAT's own `landcover` names the same crop in 100 percent of both groups) and have better GPS precision. Decided by Thierry on 28 Oct before any run on the fixed table.

## 3. Inputs

Allowed, and nothing else (the code refuses any other column):

- `elevation_m`: GPS elevation of the field.
- Season climate, one value per district (NASA POWER): `precip_mm`, `tmax_c`, `tmin_c`, `tmean_c`, `solar_mj_m2`.
- Point temperature: `tmean_point_c`, `tmin_point_c`. The district temperature moved to the field's height with the standard lapse rate (6.5 C per 1,000 m). Reference height: median elevation of all CIAT fields in the district that pass `qc_ok`.
- NISR district crop shares: `nisr_share_*`, the same for every field in a district.
- Soil 0 to 20 cm, 9 iSDAsoil properties. Soil 20 to 50 cm only in the extra view.

Never used as inputs: survey answers about the crop (area, previous crop, seed, fertiliser, harvest, yields), the district name or any ID (`F_id`, `F_index`, `block_id`), CIAT `Status`, and satellite data from the same season, because it shows the crop that is already planted.

Feature views:

| View | Inputs |
| --- | --- |
| district | elevation, season climate, point temperature, NISR shares |
| field | elevation, soil 0 to 20 cm |
| all | both of the above |
| all_subsoil (extra, Random Forest only) | all, plus soil 20 to 50 cm |

## 4. Tests

- **Main test: leave one district out.** Train on three districts and test on the fourth, which the model has never seen. Every model sees the same four folds. This test decides adoption.
- **Second test: new fields inside the four known districts.** 5 folds (`StratifiedGroupKFold`, seed 0). Each 2 km block (`block_id`) stays on one side, so neighbouring fields are never split between training and test. Scores are computed per district so the table has the same shape as the main test. This test was added on 28 Oct after the dry run (see section 9) and is reported as secondary.

## 5. Scores

- **Main score: macro-F1 per test district, over the crops with at least 5 fields in that district, then the mean over the four districts.** A crop with one or two fields in a district (for example, 1 rice field in Musanze) would otherwise swing the score by a quarter.
- Also reported: top-1, top-3, recall per crop, macro-F1 over all four crops (`macro_f1_all4`, the old definition) and full confusion counts.
- Top-3 does not decide anything. With four crops, a fixed ranking that never looks at the field already scored 98.5 percent in the dry run, more than the expert rules (95 percent).

## 6. Models

- B0: majority crop (crop shares in the training districts).
- B1: expert rules (`src/models/rules/crop_suitability.yaml`), from published crop ranges. Not tuned on our data.
- Logistic Regression: standardised inputs, class-balanced, max 5,000 iterations.
- Random Forest: 300 trees, at least 5 fields per leaf, class-balanced.
- XGBoost: 300 trees, depth 4, learning rate 0.05, class-balanced sample weights.
- Seed 0 everywhere. Missing values are filled with the training median. Nothing is tuned on a test district. Tuning (tracker T051) comes after the midterm and will only use the training districts (nested).

## 7. Adoption rule

On the main test, an ML model replaces B1 only if both are true:

1. its mean main score is higher than B1's, and
2. it beats B1 by more than 0.01 in at least 3 of the 4 districts.

The same check is printed for the second test and for the second check, but only the main test decides. The margin stops a tiny win from counting: in the dry run, Random Forest on elevation alone beat B1 in Musanze by 0.002 while its average was lower.

## 8. Changes after results

Any change after seeing results (rules, inputs, filters, settings, models) is reported with both versions and the reason.

## 9. History

- 28 Oct, morning: dry run on the table as it was then (before the soil fix). It was used only to plan this protocol and its numbers are not reported. The main score and the 0.01 margin were set after it. Under both the old and the new score, it gave the same picture: no ML model beat B1 when leaving one district out.
- 28 Oct, afternoon: Part A fixed the soil join, removed the coordinates, added `qc_ok` and `block_id`, set rice share to 0 in Musanze and Nyabihu, and moved the season B start to 1 February. The Status decision (section 2) was made before the first run on this table.
- 28 Oct, evening: official run on Thierry's laptop (Windows, Python 3.13), committed as `eval_v1`. Rerun on Colab Pro (Linux, Python 3.12) with the same library versions: every result file was identical (`01-crop-recommendation/notebooks/02_reproduce_eval_v1_colab.ipynb`). Library versions are now pinned in `01-crop-recommendation/requirements-eval.txt`, because other versions move the numbers.
- 28 Oct, evening: the table fingerprint now ignores Windows line endings. `eval_v1/run_info.json` shows `e84d21c8...`, the Windows checkout of the table; the same table gives `5df52fa6...` with line endings normalised, which is what later runs record. The results do not change.
- 29 Oct: XGBoost training and scores, as agreed at the stand-up of 28 Oct (`src/models/train_xgboost.py`, run on Colab with `notebooks/03_xgboost_training_and_map_colab.ipynb`; results go to `reports/xgboost_v1/`). Same folds and settings as eval_v1, and every score per district is the same as in eval_v1. It adds overall scores, scores per crop and confusion matrices. Nothing in this protocol changes and neither does the decision: B1 stays the default in the API. The final XGBoost on all inputs, trained on all 1,904 fields, is shown next to B1 on the dashboard map page, each with its tested score in the clicked district.
- 18 Nov: version 1.1 written down (section 10) before any run on the satellite inputs.

## 10. Version 1.1 (18 Nov 2026): inputs measured at each field from satellites

Written on 18 Nov 2026 by Thierry, before any run on these inputs. Everything in sections 1 to 8 stays the same unless this section says otherwise. Results go to `01-crop-recommendation/reports/eval_v2/`; `eval_v1` is not touched.

Why: at the midterm, no model beat the expert rules in a district it had never seen (best 0.365 against 0.378). More than half of XGBoost's gain came from inputs that are the same for a whole district (rainfall above all), so the models partly learned which district a field is in. Part A has now added inputs that change from field to field and exist for any point on a map.

**Table.** `01-crop-recommendation/data/processed/feature_table_v6b.csv`, built by `src/data_processing/build_feature_table_v6b.py`: the columns of `feature_table_v5.csv` (same rows, same order and same model inputs as v2) with the satellite values of Part A's `feature_table_v6.csv`, matched on `F_id`. v5 had joined its satellite values on `F_index`, which 274 pairs of fields share, so 548 rows had another field's values; v6 fixed that. v6 itself is not used for training because it lost `qc_ok`, `block_id` and the NISR crop shares and contains the field coordinates. Check: SRTM elevation against the CIAT GPS elevation, r = 0.995, median difference 11 m.

**New inputs** (all measured at the field and known before planting):

- `srtm_elevation_m`: SRTM elevation, 30 m. Saved under this name because `elevation_m` is already the GPS elevation.
- `chirps_precip_mm`: CHIRPS rainfall at the field's pixel (about 5 km), 1 February to 30 June 2025, the same window as the climate inputs.
- `ndvi_annual`: Sentinel-2 vegetation index, median over 2024, the year before the season.
- `swir_ratio`: Sentinel-2 shortwave infrared ratio (B11/B8), median over 2024. It is higher on flooded rice paddies.

**Not used:** `ndvi_mean` (Sentinel-2 during season B 2025). Section 3 already rules out satellite data from the same season, because it can show the crop that is planted.

**New views** (the four views of version 1.0 do not change):

| View | Inputs |
| --- | --- |
| field_sat | field, plus the four new inputs (14) |
| all_sat | all, plus the four new inputs (25) |

Logistic Regression, Random Forest and XGBoost run on both, with the settings of section 6: 6 new combinations, 18 in total.

**Unchanged:** rows (1,904 fields, and the 650 "Used" fields as the second check), both tests, the main score, the adoption rule, the model settings and seed 0.

**Expected before the run:** (a) on new fields in known districts, all_sat should beat all; (b) field_sat has no input that is the same for a whole district, so it is the best chance to transfer to a district never seen. If any model passes the adoption rule on the main test, it replaces B1 as the default answer of the API.

**Built-in check:** the 12 combinations of version 1.0 use exactly the same inputs and rows in v6b as in v2, so their results in `eval_v2` must equal `eval_v1`.
