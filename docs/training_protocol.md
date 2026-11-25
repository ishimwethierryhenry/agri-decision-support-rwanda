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
- 18 Nov, evening: official run of version 1.1, committed as `eval_v2`. Built-in check passed: the 12 combinations of version 1.0 give exactly the `eval_v1` numbers. No model passes the adoption rule, so B1 stays the default of the API. Main test: the best trained model is now Random Forest on field + satellite inputs, 0.368 against 0.378 for B1, ahead in 2 of 4 districts (at the midterm the best was 0.365). New fields: XGBoost on all + satellite inputs is best, 0.456 against 0.451 for XGBoost on all inputs. Expectation (a) held for all three methods; expectation (b) held for Random Forest only. The map keeps XGBoost on all inputs, because the gain is too small to change it.
- 22 Nov: the dashboard map shows one answer per point instead of both methods side by side, as the client asked on 19 Nov. This changes how results are presented, not the evaluation: in each district the answer comes from the method with the higher main score on new fields in `eval_v1` (ahead by more than 0.01, otherwise the rules), which today means the rules in Musanze and Nyabihu and XGBoost in Nyagatare and Ruhango. The confidence level shown with it comes from that method's tested top-1 accuracy on new fields in the district, with fixed thresholds (high from 0.75, medium from 0.50; one step lower for a close call within 0.10 or data more than 5 km away; low for an untested season). The adoption rule and the API default (B1) do not change. Logic: `01-crop-recommendation/src/recommendation/map_model.py`.
- 23 Nov: version 1.2 written down (section 11) before any tuning run.
- 23 Nov, afternoon: official run of version 1.2, committed as `tuning_v1` (1.3 hours of computing, two jobs at a time). Built-in check passed: the section 6 settings give exactly the `eval_v2` numbers (88 rows). Main test: no tuned model passes the adoption rule, so B1 stays the default answer of the API. Settings chosen inside three districts did not carry over to the fourth: tuning lowered five of the nine models there, all three Random Forests among them (on field + satellite inputs, from 0.368 to 0.276). New fields: every tuned model did better than with the section 6 settings. The best is Random Forest on all + satellite inputs, 0.514 against 0.441 untuned and 0.451 for the model on the map, with the right first crop for 72 percent of the fields, so under decision 2 the map moves to it (the change to the map is a separate step). Expectation (a) did not hold on new fields: the gains were larger than a few hundredths, up to 0.126. Expectation (b) held in both tests.
- 23 Nov, after the official run: decision 2 applied. The settings of the model on the map were chosen on all 1,904 fields with the same comparison as inside the nested test (3 folds over 2 km blocks, seed 0, `tune.py --final`): candidate 30 of the Random Forest (entropy, trees up to 16 levels, a third of the inputs tried at each split, 70 percent of the fields for each tree, 1 field per leaf, no class weights), which the nested test also chose in two of its five folds (`reports/tuning_v1/final_model.json`). The dashboard map and `POST /v1/recommend/point` (contract 1.1) now serve it, with its satellite inputs taken from the CIAT fields of the 2 km block, like soil. Its answer comes from the model in Musanze, Nyagatare and Ruhango and from the rules in Nyabihu, with the confidence from its tested hit rates in `tuning_v1`. The adoption rule and the API default (B1) do not change.
- 24 Nov: version 1.3 written down (section 12) before any run of this version.
- 24 and 25 Nov: official run of version 1.3, committed as `tuning_v2` (4.5 hours of computing, two jobs at a time). The machine's two-hour limit stopped the run once, after 9 of the 12 jobs had been saved; the three unfinished jobs then ran again from the start with the same seed, so the results are those of one uninterrupted run. Built-in check passed: Random Forest and XGBoost with the section 6 settings and the baselines give exactly the `eval_v2` numbers (48 rows). Main test: no tuned model or blend passes the adoption rule, so B1 stays the default answer of the API (best: Random Forest on all + satellite inputs, 0.308 against 0.378 for B1, ahead in 1 of 4 districts). New fields: the best is Extra Trees on field + satellite inputs, 0.521, against 0.514 for the model on the map, so under decision 2 the map does not change (`tuning_v2/final_model.json` says so). Expectation (a) did not hold: the nearby-crop inputs changed the main score by -0.027 to +0.021 depending on the model, and the best model with them (Extra Trees on all + satellite inputs, 0.511) stayed below the best without them. Expectation (b) held: the best of this round is 0.007 above the best of version 1.2. Expectation (c) held. The tuned Random Forest on all + satellite inputs scored 0.496 in this round against 0.514 in version 1.2: with more candidates, the comparison inside the training fields chose other settings, which did worse on the test folds. Differences of about 0.02 between tuned models are therefore within the noise of choosing settings on this table. The nearest-fields reference (crop with the highest nearby share, h = 4 km) gets the first crop right for 72 percent of the fields with a main score of 0.413: the models match it on the first crop and do better on the less common crops.
- 25 Nov: the report text of `tuning_v2` was rebuilt from the saved jobs after a fix in how two tables are printed (districts ahead of B1 as whole numbers, tuned labels in the table of chosen settings). The numbers did not change. The scores page of the dashboard now says that this round found nothing better than the model on the map.
- 25 Nov, change after results (section 8): on the dashboard map and in `POST /v1/recommend/point`, the method that answers in each district is now chosen by first crop right on new fields in testing (ahead by more than 0.01, otherwise the rules), instead of by the main score. Reason: the map gives one crop, and its confidence level already starts from first crop right, so the choice now uses the same measure. With the main score the rules answered in Nyabihu (0.645 against 0.627), although the Random Forest named the right crop first more often there (74 percent of the fields against 66 percent). The rules also rate maize, Irish potato and rice the same at many Nyabihu points, so 55 percent of the Nyabihu fields got a close call and a low confidence. Both versions, replayed on the out-of-fold predictions of the new-fields test of `tuning_v1` (`src/recommendation/check_map_confidence.py`): the map's first crop is right for 71.6 percent of the fields before and 72.4 percent after; low confidence goes from 55 to 12 percent of the Nyabihu fields and from 8.0 to 4.3 percent of all fields; Musanze, Nyagatare and Ruhango do not change. In both versions, the first crop is right for 77 percent of the fields at high confidence, 66 percent at medium and 47 to 48 percent at low. Nothing was retrained and no tested number changed. The cost, in Nyabihu: the Random Forest finds maize less often than the rules (recall 0.27 against 0.70) and Irish potato more often (0.92 against 0.66). The adoption rule and the default of `POST /v1/recommend` (B1) do not change. API contract 1.2.

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

## 11. Version 1.2 (23 Nov 2026): tuning the model settings

Written on 23 Nov 2026 by Thierry, before any tuning run. Everything in sections 1 to 10 stays the same unless this section says otherwise. Results go to `01-crop-recommendation/reports/tuning_v1/`; `eval_v1` and `eval_v2` are not touched.

Why: at the meeting of 19 Nov, the client asked why anyone should use the model. Every model so far used the fixed settings of section 6, which planned tuning for after the midterm, on training data only (tracker T051). This version does that tuning for Logistic Regression, Random Forest and XGBoost, in a way that never lets the test fields choose the settings. It also adds the analyses that show when the model can be trusted.

**Table and rows.** Same as version 1.1: `feature_table_v6b.csv` and the 1,904 single-crop fields. The CIAT "Used" second check is not tuned: it would double the run time and it does not decide anything.

**Models and views.** Logistic Regression, Random Forest and XGBoost on three views of version 1.1: `all`, `field_sat` and `all_sat`, so 9 tuned models. `district` is left out because it has no input measured at the field, and `field` is part of `field_sat`.

**How the settings are chosen (nested tests).** The settings are chosen again inside the training part of every fold of the two tests, with the training fields of that fold only:

- Main test (leave one district out): for each held-out district, the candidates are compared by leaving out each of the three training districts in turn (train on two, score on the third). The best candidate is then trained on the three training districts and scored once on the held-out district.
- Second test (new fields): for each of the 5 block folds of section 4, the candidates are compared on 3 folds over 2 km blocks inside the training part (`StratifiedGroupKFold`, seed 0). The best candidate is then trained on the whole training part and scored once on the test fold.

Candidates are compared on the main score of section 5, computed the same way as in the outer test (per district, then the mean). A tie goes to the candidate drawn first. Median filling and class weights are fitted on the training fields of each step only.

**Candidates.** For each method, 61 candidates. The settings of section 6 always come first, so tuning can only keep them or pick something that did better in the comparison. Then come 60 random draws from the ranges below (scikit-learn `ParameterSampler`, seed 0). The same candidates are used in every fold and every view. With 60 random draws, there is a 95 percent chance that at least one lands in the best 5 percent of the ranges (0.95 to the power 60 is about 0.05) [1].

| Method | Settings searched |
| --- | --- |
| Logistic Regression | C from 0.001 to 100 (log scale). Input shape: straight, as in section 6, or curved: cubic splines with 5 equally spaced knots for each input measured at the field (inputs that are the same for a whole district stay straight), flat outside the training range. Class weight strength 0, 0.5 or 1. |
| Random Forest | 300 trees, as in section 6. Maximum depth none, 6, 10 or 16. At least 1, 2, 3, 5, 8, 12 or 20 fields per leaf. Inputs tried at each split: square root of the number of inputs, 20, 33 or 50 percent. Fields drawn for each tree: 100, 70 or 50 percent. Split rule Gini or entropy. Class weight strength 0, 0.5 or 1. |
| XGBoost | 100, 200, 300, 500 or 800 trees. Learning rate from 0.01 to 0.3 (log scale). Depth 2, 3, 4, 5, 6 or 8. Minimum child weight 1, 2, 5 or 10. Share of fields for each tree from 0.6 to 1.0, share of inputs for each tree from 0.5 to 1.0. L2 penalty from 0.1 to 10 (log scale). Minimum gain to split 0, 0.1, 0.5 or 1. Class weight strength 0, 0.5 or 1. |

Class weight strength p gives each crop the weight (n / (k x n_crop)) to the power p, where n is the number of training fields, k the number of crops among them and n_crop the fields of that crop. Strength 1 is the class balance of section 6 and 0 is no weighting.

**Decisions fixed now.**

1. The adoption rule of section 7 does not change. It is checked for the 9 tuned models on the main test. If one passes, it replaces B1 as the default answer of the API, as section 10 already says.
2. The map serves a tuned model only if its main score on new fields (nested) beats the model on the map today, XGBoost on `all` with the section 6 settings (0.451), by more than 0.01. If several do, the one with the highest score. The map can get every input of the three views for any point from the CIAT fields of the block, as it does today for soil and elevation. The choice between the model and the rules in each district and the confidence levels then come from that model's nested results. If no tuned model beats it by more than 0.01, the map does not change.
3. Untuned (`eval_v2`) and tuned results are always reported side by side, per district, with the settings chosen in every fold.

**Built-in check.** The section 6 settings, scored on the same outer folds, must give the same scores as `eval_v2` in every district for the 9 models.

**Also reported (none of these decide anything).**

- How often the first crop is right at each level of the model's own probability for it, in bands of 0.1, on the new-fields test, for the model the map serves after this version: the share of fields in each band and how often the first crop was right there.
- Simple alternatives on the same folds of the new-fields test: the most common crop among all training fields (B0), the most common crop of the field's own district among the training fields, and B1.
- Input importance for the tuned Logistic Regression, Random Forest and XGBoost, on both tests: how much the main score drops when the values of one input are shuffled among the test fields (5 shuffles, seed 0), and the same for groups of inputs that carry the same information (elevation with the field temperatures, district climate, NISR shares, soil, satellite rainfall and vegetation). In the main test, an input that is the same for the whole district cannot change inside the held-out district, so its importance there is zero by construction.
- How mixed the land is: the share of fields whose crop is the most common crop of their own 2 km block. No model can know this in advance. It shows how often neighbouring fields grow different crops.

**Expected before the run:** (a) tuning gives small gains on a table of this size, a few hundredths of main score at most; (b) Logistic Regression gains more than the tree models, because the curved shape lets it follow the elevation bands of the crops, which a straight line cannot.

**Run.** `python 01-crop-recommendation/src/models/tune.py` with the pinned versions of `requirements-eval.txt`. It takes a few hours on two processor cores, so the official run is made once and saved with its `run_info.json`. `--quick` runs a tiny version for checks only, and its numbers are never reported.

## 12. Version 1.3 (24 Nov 2026): second round of tuning

Written on 24 Nov 2026 by Thierry, before any run of this version. Everything in sections 1 to 11 stays the same unless this section says otherwise. Results go to `01-crop-recommendation/reports/tuning_v2/`; `eval_v1`, `eval_v2` and `tuning_v1` are not touched.

Why: version 1.2 moved the map to a tuned Random Forest (0.514 on new fields). This round asks how far the data lets us go, with three additions: a wider search, a third tree method with a blend of the three, and inputs that say which crops are grown on the surveyed fields of the nearby 2 km blocks. This round was decided after seeing the results of version 1.2, so it is reported as a second round next to version 1.2, never instead of it (section 8).

**Table and rows.** Same as versions 1.1 and 1.2: `feature_table_v6b.csv` and the 1,904 single-crop fields.

**Methods.**

- Random Forest and XGBoost, as in version 1.2, with the wider ranges below.
- Extra Trees (scikit-learn `ExtraTreesClassifier`): like Random Forest, but each split point is drawn at random instead of searched, which often overfits less on a small table.
- Blend: the average of the probabilities of the tuned Random Forest, Extra Trees and XGBoost on the same view and the same fold, with equal weights fixed now. Each member is chosen inside the nested test as usual; the blend itself has nothing to tune, so its outer score is as fair as theirs.
- Logistic Regression is not run again. In version 1.2 its best tuned score was 0.044 below the best tree model on new fields (0.470 against 0.514) and 0.033 below on the main test (0.293 against 0.326).

**Views.**

| View | Inputs | Tests |
| --- | --- | --- |
| field_sat | as in version 1.1 (14) | both |
| all_sat | as in version 1.1 (25) | both |
| field_sat_near | field_sat plus the 5 nearby-crop inputs (19) | new fields only |
| all_sat_near | all_sat plus the 5 nearby-crop inputs (30) | new fields only |

`all` is dropped: `all_sat` has every input of `all`, and in version 1.2 the satellite views beat `all` on new fields for every method. The nearby-crop views are not run in the main test: a district never seen has no surveyed field nearby by definition, so they would answer with the crops of another district.

**Nearby-crop inputs (new).** For a field in 2 km block b:

- Every training field in another block counts with the weight exp(-0.5 (d / h)^2), where d is the distance in km between the centres of the two blocks and h is a setting (2, 4 or 8 km). Blocks farther than 3h do not count.
- `near_share_beans`, `near_share_irish_potato`, `near_share_maize` and `near_share_rice` are the weighted shares of each crop, and `near_weight` is the sum of the weights, which says how much nearby data there is. With no field counted, the shares are those of all training fields and `near_weight` is 0.
- Fields of the field's own block are never counted, for training fields too. So no field ever sees its own crop or the crops of its own block, which is exactly the situation of the new-fields test, where the whole block is unseen.
- They are computed inside every step from the training fields of that step only, so the test fields of the outer fold and the validation fields of the inner folds never inform them. The map follows the same rule: the fields of the clicked point's own block are left out.

**Candidates.** For each method, the settings of section 6 come first (Extra Trees takes the Random Forest settings of section 6), then 100 random draws from the ranges below (scikit-learn `ParameterSampler`, seed 0). Random Forest also gets the settings of the map model of version 1.2 (candidate 30 of `tuning_v1`) as its second candidate. The distance h is part of every candidate (4 km for the section 6 settings) and is used only by the nearby-crop views. The same candidates are used in every fold and every view. How they are compared, the nested tests and the tie rule are those of section 11.

| Method | Settings searched |
| --- | --- |
| Random Forest and Extra Trees | 300 trees. Maximum depth none, 6, 10, 16 or 24. At least 1, 2, 3, 5, 8 or 12 fields per leaf. Inputs tried at each split: square root of the number of inputs, 20, 33, 50 or 70 percent. Fields drawn for each tree: all, 85, 70 or 50 percent (Extra Trees draws fields only below all). Split rule Gini or entropy. Class weight strength 0, 0.5, 1 or 1.5. Distance h 2, 4 or 8 km. |
| XGBoost | 100, 200, 300, 500, 800, 1,200 or 1,500 trees. Learning rate from 0.005 to 0.3 (log scale). Depth 2, 3, 4, 5, 6, 8 or 10. Minimum child weight 1, 2, 5, 10 or 20. Share of fields for each tree from 0.5 to 1.0, share of inputs for each tree from 0.3 to 1.0. L2 penalty from 0.01 to 100 (log scale). L1 penalty 0, 0.01, 0.1 or 1. Minimum gain to split 0, 0.1, 0.5, 1 or 2. Class weight strength 0, 0.5, 1 or 1.5. Distance h 2, 4 or 8 km. |

Class weight strength 1.5 weights the rare crops more than the class balance of section 6.

**Decisions fixed now.**

1. The adoption rule of section 7 does not change. It is checked on the main test for the tuned models and blends of `field_sat` and `all_sat`. If one passes, it replaces B1 as the default answer of the API.
2. The map moves only if the best tuned model or blend on new fields (nested) beats the model on the map today, the tuned Random Forest on `all_sat` of version 1.2 (0.514 in `tuning_v1`), by more than 0.01. Its final settings are then chosen on all 1,904 fields with the same comparison as inside the nested test, as in version 1.2. The choice between the model and the rules in each district and the confidence levels come from its nested results, with the rules of 22 Nov unchanged. If nothing beats 0.514 by more than 0.01, the map does not change.
3. Results with the section 6 settings, tuned results and the version 1.2 results are reported side by side.

**Built-in check.** Random Forest and XGBoost with the section 6 settings on `field_sat` and `all_sat` must give the `eval_v2` numbers in every district of both tests. Extra Trees and the nearby-crop views have no earlier numbers to compare with.

**Also reported (none of these decide anything).**

- Nearest surveyed fields: on new fields, the crop with the highest nearby share (h = 4 km), next to B0, the most common crop of the district and B1. It shows how much the models add to simply repeating what is grown nearby.
- For the best model on new fields: how often the first crop is right at each level of its own probability, the share of fields it would answer if it answered only when sure, its confusion counts, and input importance (one input at a time, and groups of inputs, with the five nearby-crop inputs as one group), as in version 1.2.

**Expected before the run:** (a) the nearby-crop inputs give the largest gain on new fields, because neighbouring blocks tend to grow the same crops (version 1.2: 49 percent of the blocks grow more than one crop, and the most common crop of a field's own block is right for 80 percent of the fields); (b) the wider search and Extra Trees add little, up to about 0.01; (c) on the main test, no model passes the adoption rule.

**Run.** `python 01-crop-recommendation/src/models/tune_v2.py` with the pinned versions of `requirements-eval.txt`, about three hours on two processor cores. Finished jobs are saved, so a stopped run starts again where it stopped. `--quick` runs a tiny version for checks only, and its numbers are never reported.

[1] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *J. Mach. Learn. Res.*, vol. 13, pp. 281-305, 2012.
