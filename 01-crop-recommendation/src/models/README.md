# Crop Recommendation Models

| File | What it is |
| --- | --- |
| `expert_baseline.py` | B0 (majority crop) and B1 (published crop ranges) baselines. B1 is not trained; it scores each crop against `rules/crop_suitability.yaml`. |
| `rules/crop_suitability.yaml` | B1 rules as data. Every range cites `docs/baseline_research_notes.md`. Draft until each page is checked. |
| `evaluate.py` | Evaluation harness v1. Follows `docs/training_protocol.md`: leave one district out (main test), new fields inside the known districts (second test), and both again on CIAT "Used" fields (second check). |

Run the evaluation from the repository root (XGBoost is required: `python -m pip install xgboost`):

```
python 01-crop-recommendation/src/models/evaluate.py
```

It reads `01-crop-recommendation/data/processed/feature_table_v2.csv` and writes `01-crop-recommendation/reports/eval_v1/`: `report.md`, one CSV per test, `confusion_counts.csv` and `run_info.json` (table fingerprint, code commit, settings, library versions). It takes one to three minutes on a laptop. Commit the code first, then run, so `run_info.json` points to the code that produced the numbers.

Rules for using results:

- The main score is macro-F1 per held-out district over the crops with at least 5 fields there, then the mean over the four districts. With four crops, top-3 accuracy cannot tell models apart.
- ML replaces B1 only if, on the main test, its mean main score is higher AND it beats B1 by more than 0.01 in at least 3 of the 4 districts.
- The harness refuses any column outside the protocol's feature list (crop details, IDs, district, CIAT Status, NISR statistics looked up from a field's own crop). District crop shares are allowed.
- Do not change the B1 rules, inputs, filters or settings after looking at test results without saying so in the report and showing both versions.
