# Crop Recommendation Models

| File | What it is |
| --- | --- |
| `expert_baseline.py` | B0 (majority crop) and B1 (published crop ranges) baselines. B1 is not trained; it scores each crop against `rules/crop_suitability.yaml`. |
| `rules/crop_suitability.yaml` | B1 rules as data. Every range cites `docs/baseline_research_notes.md`. Draft until each page is checked. |
| `evaluate.py` | Leave-one-district-out harness: trains on three districts, tests on the fourth, for every model. |

Run the evaluation from the repository root:

```
python 01-crop-recommendation/src/models/evaluate.py
```

It reads the newest feature table in `01-crop-recommendation/data/processed/` (`feature_table_v2.csv`, else `feature_table.csv`) and writes `01-crop-recommendation/reports/eval_v0/`.

Rules for using results:

- Read top-1 accuracy and macro-F1 first. With four crops, top-3 accuracy is easy (random guessing gets 75 percent).
- ML is adopted only if it beats B1 on macro-F1 in most held-out districts.
- The harness refuses columns that leak the label (NISR statistics looked up from a point's own crop, CIAT crop and farm details). District crop shares are allowed.
- Do not change the B1 rules after looking at test results without saying so in the report.
