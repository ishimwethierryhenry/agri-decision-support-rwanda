# Training protocol: crop recommendation (version 1.0)

Written on 28 Oct 2026 by Thierry (Part B), with the data decisions made the same day by Arsène (Part A), before any official result. Every number in `01-crop-recommendation/reports/eval_v1/` follows this file. If anything here changes after we see results, the report says so and shows both versions.

Run everything with one command from the repo root:

```
python 01-crop-recommendation/src/models/evaluate.py
```

It writes `01-crop-recommendation/reports/eval_v1/` (report, CSV files and `run_info.json`).

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
