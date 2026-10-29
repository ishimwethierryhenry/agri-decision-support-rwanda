# Crop Recommendation Reports

Store concise model evaluation summaries, experimental results, limitations, and workstream findings. Distinguish preliminary findings from reviewed results and reference the experiment configuration.

| Folder | What it is |
| --- | --- |
| `eval_v1/` | Official crop recommendation evaluation under `docs/training_protocol.md` (version 1.0). Made by `src/models/evaluate.py`. |
| `xgboost_v1/` | XGBoost training and scores under the same protocol, and the final model behind the dashboard map. Made by `src/models/train_xgboost.py`, run on Colab with `notebooks/03_xgboost_training_and_map_colab.ipynb`. The `model/` folder is not committed. |
