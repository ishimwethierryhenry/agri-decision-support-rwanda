# Crop Recommendation Reports

Store concise model evaluation summaries, experimental results, limitations, and workstream findings. Distinguish preliminary findings from reviewed results and reference the experiment configuration.

| Folder | What it is |
| --- | --- |
| `eval_v1/` | Official crop recommendation evaluation under `docs/training_protocol.md` (version 1.0). Made by `src/models/evaluate.py`. |
| `xgboost_v1/` | XGBoost training and scores under the same protocol, and the model behind the dashboard map until 23 Nov 2026 (since then, the tuned Random Forest of `tuning_v1`). Made by `src/models/train_xgboost.py` on a clean checkout (the code commit is in `run_info.json`); notebook 03 on Colab gives the same numbers. `check.csv` confirms every score per district equals `eval_v1`. The `model/` folder is not committed. |
| `eval_v2/` | Evaluation under protocol version 1.1 (satellite inputs, section 10), on `feature_table_v6b.csv`. Made by `src/models/evaluate.py --protocol 1.1`. The 12 combinations of version 1.0 give exactly the `eval_v1` numbers. |
| `tuning_v1/` | Tuned settings under protocol version 1.2 (section 11), with nested tests so the test fields never choose the settings. Made by `src/models/tune.py`. The protocol settings give exactly the `eval_v2` numbers (built-in check at the top of `report.md`). `final_model.json` holds the settings of the model the map serves since 23 Nov (`tune.py --final`). |
