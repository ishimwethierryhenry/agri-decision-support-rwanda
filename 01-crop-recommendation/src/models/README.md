# Crop Recommendation Models

| File | What it is |
| --- | --- |
| `expert_baseline.py` | B0 (majority crop) and B1 (published crop ranges) baselines. B1 is not trained; it scores each crop against `rules/crop_suitability.yaml`. |
| `rules/crop_suitability.yaml` | B1 rules as data. Every range cites `docs/baseline_research_notes.md`. Draft until each page is checked. |
| `evaluate.py` | Evaluation harness v1. Follows `docs/training_protocol.md`: leave one district out (main test), new fields inside the known districts (second test), and both again on CIAT "Used" fields (second check). |
| `train_xgboost.py` | XGBoost training and scores (agreed at the stand-up of 28 Oct). Same folds and settings as `evaluate.py`, plus overall scores, scores per crop, confusion matrices, a check against `eval_v1`, and the final XGBoost on all inputs and all 1,904 fields (the model behind the dashboard map). |

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

Rules for using results:

- The main score is macro-F1 per held-out district over the crops with at least 5 fields there, then the mean over the four districts. With four crops, top-3 accuracy cannot tell models apart.
- ML replaces B1 only if, on the main test, its mean main score is higher AND it beats B1 by more than 0.01 in at least 3 of the 4 districts.
- The harness refuses any column outside the protocol's feature list (crop details, IDs, district, CIAT Status, NISR statistics looked up from a field's own crop). District crop shares are allowed.
- Do not change the B1 rules, inputs, filters or settings after looking at test results without saying so in the report and showing both versions.
