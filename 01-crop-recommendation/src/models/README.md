# Crop Recommendation Models

| File | What it is |
| --- | --- |
| `expert_baseline.py` | B0 (majority crop) and B1 (published crop ranges) baselines. B1 is not trained; it scores each crop against `rules/crop_suitability.yaml`. |
| `rules/crop_suitability.yaml` | B1 rules as data. Every range cites `docs/baseline_research_notes.md`. Draft until each page is checked. |
| `evaluate.py` | Evaluation harness v1. Follows `docs/training_protocol.md`: leave one district out (main test), new fields inside the known districts (second test), and both again on CIAT "Used" fields (second check). `--protocol 1.1` adds the satellite inputs (section 10) and writes `eval_v2`. |
| `tune.py` | Tuning with nested tests (protocol 1.2, section 11). 61 candidate settings per method for Logistic Regression, Random Forest and XGBoost on the views `all`, `field_sat` and `all_sat`. The settings are chosen inside the training part of every fold of both tests, so the test fields never choose them. Also gives input importance, how often the model is right at each level of its own probability, simple alternatives and how mixed the 2 km blocks are. |
| `tune_v2.py` | Second round of tuning (protocol 1.3, section 12). Random Forest, Extra Trees and XGBoost with wider ranges (102 candidates for Random Forest, 101 for the others), a blend of the three, and the views `field_sat` and `all_sat` plus two views with inputs that say which crops are grown on the surveyed fields of the nearby 2 km blocks (new-fields test only). Same nested tests as `tune.py`, which it reuses. |
| `tune_v3.py` | Third round of tuning (protocol 1.4, section 13). Same candidates as `tune_v2.py`, but the 8 best of each fold are compared again on two more inner splits (a steadier choice), then one weight per crop is chosen at decision time, and a blend picks its group of models inside the training part. Built-in check: the version 1.3 choices fitted again give exactly the `tuning_v2` scores. |
| `train_xgboost.py` | XGBoost training and scores (agreed at the stand-up of 28 Oct). Same folds and settings as `evaluate.py`, plus overall scores, scores per crop, confusion matrices, a check against `eval_v1`, and the final XGBoost on all inputs and all 1,904 fields (the model behind the dashboard map). |
| `model_scores.py` | Reads the official evaluation (`reports/eval_v1`) and gives every score the dashboard and the notebooks show: main score per model and district, overall scores over all test fields, and precision, recall and F1 of each crop in each district. Needs only pandas and numpy. |
| `compare_results.py` | Compares a rerun with the committed result files, file by file and model by model (used by the Colab notebooks). |

Run the evaluation from the repository root. To get exactly the official numbers, install the pinned versions first (`python -m pip install -r 01-crop-recommendation/requirements-eval.txt`, Python 3.12 or newer):

```
python 01-crop-recommendation/src/models/evaluate.py
```

It reads `01-crop-recommendation/data/processed/feature_table_v2.csv` and writes `01-crop-recommendation/reports/eval_v1/`: `report.md`, one CSV per test, `confusion_counts.csv` and `run_info.json` (table fingerprint, code commit, settings, library versions). It takes one to three minutes on a laptop. Commit the code first, then run, so `run_info.json` points to the code that produced the numbers. To check the results on Colab, use `../../notebooks/02_reproduce_eval_v1_colab.ipynb`: it reinstalls the pinned versions and compares every result file with the committed ones.

XGBoost training and scores (under a minute, same pinned versions):

```
python 01-crop-recommendation/src/models/train_xgboost.py
```

It writes `01-crop-recommendation/reports/xgboost_v1/`: `report.md`, the score tables as CSV, `check.csv` (every score per district must be the same as in `eval_v1`), `feature_importance.csv`, `run_info.json`, and the final model in `model/` (not committed, it takes seconds to retrain). On Colab, use `../../notebooks/03_xgboost_training_and_map_colab.ipynb`.

Tuning (protocol 1.2, about an hour on two cores, same pinned versions):

```
python 01-crop-recommendation/src/models/tune.py
```

It writes `01-crop-recommendation/reports/tuning_v1/`: `report.md`, the results per test and district with the protocol settings and the tuned settings side by side, `chosen_settings.csv`, `inner_scores.csv` (every candidate in every fold), `candidates.json`, `importance.csv`, `confidence_bands.csv`, `answer_when_sure.csv`, `mixed_land.csv`, `confusion_counts.csv` and `run_info.json`. Finished jobs are saved in `tuning_v1/_work/` (not committed), so a stopped run starts again where it stopped. `--quick` runs a tiny version into a temporary folder to check the code; its numbers are never reported. After the run, `--final` applies decision 2: it chooses the settings of the model on the map on all 1,904 fields, with the same comparison as inside the nested test, and writes `tuning_v1/final_model.json` (about two minutes).

Second round of tuning (protocol 1.3, four and a half hours of computing, so about two and a half hours on two cores):

```
python 01-crop-recommendation/src/models/tune_v2.py
```

It writes `01-crop-recommendation/reports/tuning_v2/` with the same files as `tuning_v1`, plus the nearest-fields reference and the blends in the result tables. Finished jobs are saved in `tuning_v2/_work/` (not committed), so a stopped run starts again where it stopped. `--quick` and `--final` work as in `tune.py`; `--final` only chooses new settings when decision 2 moves the map, otherwise it writes a note that the map keeps the model of `tuning_v1`. `--assemble-only` rebuilds the report from the saved jobs in a few seconds. The official run (25 Nov 2026) did not move the map: the best of the round, the tuned Extra Trees on `field_sat`, scored 0.521 on new fields against 0.514 for the model on the map.

Third round of tuning (protocol 1.4, about half an hour on two cores):

```
python 01-crop-recommendation/src/models/tune_v3.py
```

It writes `01-crop-recommendation/reports/tuning_v3/`: `report.md`, the results and summaries of both tests, `choices.csv` (the candidate, crop weights and blend group chosen in every fold) and `run_info.json`. Finished tasks are kept in `tuning_v3/_work/` (not committed). `--quick` checks the code on one fold; its numbers are never reported. `--final` chooses the settings and crop weights of the model the map serves on all 1,904 fields (about five minutes). The official run (25 Nov 2026) moved the map to the Random Forest on `field_sat` with crop weights (0.531 against 0.514 on new fields).

Rules for using results:

- The main score is macro-F1 per held-out district over the crops with at least 5 fields there, then the mean over the four districts. With four crops, top-3 accuracy cannot tell models apart.
- ML replaces B1 only if, on the main test, its mean main score is higher AND it beats B1 by more than 0.01 in at least 3 of the 4 districts.
- The harness refuses any column outside the protocol's feature list (crop details, IDs, district, CIAT Status, NISR statistics looked up from a field's own crop). District crop shares are allowed.
- Do not change the B1 rules, inputs, filters or settings after looking at test results without saying so in the report and showing both versions.
