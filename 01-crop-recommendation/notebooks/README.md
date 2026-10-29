# Crop Recommendation Notebooks

Place exploratory data analysis and reproducible experiments here. Record each notebook's purpose, inputs, execution requirements, and findings. Remove secrets and large embedded outputs before committing.

| Notebook | Purpose | How to run |
| --- | --- | --- |
| `01_feature_engineering.ipynb` | Part A: how the feature table was built (joins and decisions). | Locally, see `../data/README.md`. |
| `02_reproduce_eval_v1_colab.ipynb` | Reruns the official evaluation on Colab and checks that every result file matches `reports/eval_v1`. | Colab. Upload a zip of the repo made with `git archive`. |
| `03_xgboost_training_and_map_colab.ipynb` | Trains XGBoost with the protocol settings, shows its scores next to the other models, trains the final model and opens the map dashboard with a link. Made for the whole team (Gabin asked for access). | Colab, CPU is enough. Runtime, then Run all. It gets the repo by itself (public clone, or a token in Colab secrets, or a zip). |
