# Crop Recommendation Reports

Store concise model evaluation summaries, experimental results, limitations, and workstream findings. Distinguish preliminary findings from reviewed results and reference the experiment configuration.

| Folder | What it is |
| --- | --- |
| `eval_v1/` | Official crop recommendation evaluation under `docs/training_protocol.md` (version 1.0). Made by `src/models/evaluate.py`. |
| `xgboost_v1/` | XGBoost training and scores under the same protocol, and the final model behind the dashboard map. Made by `src/models/train_xgboost.py` on a clean checkout (the code commit is in `run_info.json`); notebook 03 on Colab gives the same numbers. `check.csv` confirms every score per district equals `eval_v1`. The `model/` folder is not committed. |
