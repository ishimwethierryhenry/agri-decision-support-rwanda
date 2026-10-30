# XGBoost training and scores (protocol 1.0)

Made by `src/models/train_xgboost.py` on 2026-10-30 23:26 (UTC). Code commit `237c0a4537276da3e47297af5e792cb0e8875030`. Table `01-crop-recommendation\data\processed\feature_table_v2.csv`, fingerprint `5df52fa6350fa15e...`, 1904 training fields (single crops, GPS checks passed). Settings: the protocol's, nothing tuned.

## In short

- A district it never saw (main test): XGBoost on all inputs scores 0.294 against 0.378 for the expert rules (B1). It does not pass the adoption rule, so the rules stay the default.
- New fields in the four known districts (second test): XGBoost on all inputs scores 0.451 against 0.378 for B1, better in 2 of the 4 districts.
- District-level inputs (season weather and NISR shares, one value per district) carry 53% of the model's gain: about half of what the model uses tells it which district a field is in. That helps inside a known district and does not carry over to a new one.
- Every score per district is the same as in `eval_v1`.

Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over the four districts (protocol, rule 5). It goes from 0 to 1.

## 1. Main score per model

Test: a district the model never saw (main test, decides adoption)

| model           |   main score |   worst district |   accuracy (mean of districts) | districts better than B1   | passes adoption rule   |
|:----------------|-------------:|-----------------:|-------------------------------:|:---------------------------|:-----------------------|
| B1_expert_rules |        0.378 |            0.090 |                          0.523 |                            |                        |
| XGB[district]   |        0.365 |            0.239 |                          0.616 | 2 of 4                     | no                     |
| XGB[field]      |        0.308 |            0.210 |                          0.522 | 2 of 4                     | no                     |
| XGB[all]        |        0.294 |            0.215 |                          0.554 | 2 of 4                     | no                     |
| B0_majority     |        0.104 |            0.085 |                          0.227 | 0 of 4                     |                        |

Test: new fields in the four known districts (second test)

| model           |   main score |   worst district |   accuracy (mean of districts) | districts better than B1   | passes adoption rule   |
|:----------------|-------------:|-----------------:|-------------------------------:|:---------------------------|:-----------------------|
| XGB[all]        |        0.451 |            0.330 |                          0.659 | 2 of 4                     | no                     |
| XGB[field]      |        0.426 |            0.327 |                          0.652 | 2 of 4                     | no                     |
| XGB[district]   |        0.405 |            0.248 |                          0.652 | 2 of 4                     | no                     |
| B1_expert_rules |        0.378 |            0.090 |                          0.523 |                            |                        |
| B0_majority     |        0.135 |            0.085 |                          0.355 | 1 of 4                     |                        |

## 2. XGBoost and the rules, district by district

| test                                                         | district   |   XGBoost (all inputs) |   Expert rules (B1) |   difference |
|:-------------------------------------------------------------|:-----------|-----------------------:|--------------------:|-------------:|
| a district the model never saw (main test, decides adoption) | Musanze    |                  0.342 |               0.592 |       -0.250 |
| a district the model never saw (main test, decides adoption) | Nyabihu    |                  0.376 |               0.645 |       -0.269 |
| a district the model never saw (main test, decides adoption) | Nyagatare  |                  0.242 |               0.090 |        0.152 |
| a district the model never saw (main test, decides adoption) | Ruhango    |                  0.215 |               0.186 |        0.029 |
| new fields in the four known districts (second test)         | Musanze    |                  0.528 |               0.592 |       -0.065 |
| new fields in the four known districts (second test)         | Nyabihu    |                  0.577 |               0.645 |       -0.068 |
| new fields in the four known districts (second test)         | Nyagatare  |                  0.372 |               0.090 |        0.282 |
| new fields in the four known districts (second test)         | Ruhango    |                  0.330 |               0.186 |        0.144 |

## 3. Overall scores (all test predictions together, for information)

Accuracy counts every field the same, so it mostly reflects Nyagatare (1,084 of the 1,904 fields). Balanced accuracy and macro-F1 give each crop the same weight. Cohen's kappa is 0 for chance agreement and 1 for perfect. Top-3 is near 1 for every method because there are only four crops.

Test: a district the model never saw (main test, decides adoption)

| model           |   fields |   main score |   accuracy (all fields) |   balanced accuracy |   macro-F1 (all fields) |   weighted F1 |   Cohen's kappa |   top-3 (all fields) |
|:----------------|---------:|-------------:|------------------------:|--------------------:|------------------------:|--------------:|----------------:|---------------------:|
| B0_majority     |     1904 |        0.104 |                   0.239 |               0.153 |                   0.130 |         0.215 |          -0.286 |                0.975 |
| B1_expert_rules |     1904 |        0.378 |                   0.388 |               0.373 |                   0.304 |         0.281 |           0.105 |                0.961 |
| XGB[district]   |     1904 |        0.365 |                   0.596 |               0.472 |                   0.476 |         0.607 |           0.343 |                0.981 |
| XGB[field]      |     1904 |        0.308 |                   0.446 |               0.411 |                   0.391 |         0.445 |           0.137 |                0.980 |
| XGB[all]        |     1904 |        0.294 |                   0.519 |               0.443 |                   0.421 |         0.521 |           0.238 |                0.984 |

Test: new fields in the four known districts (second test)

| model           |   fields |   main score |   accuracy (all fields) |   balanced accuracy |   macro-F1 (all fields) |   weighted F1 |   Cohen's kappa |   top-3 (all fields) |
|:----------------|---------:|-------------:|------------------------:|--------------------:|------------------------:|--------------:|----------------:|---------------------:|
| B0_majority     |     1904 |        0.135 |                   0.530 |               0.250 |                   0.173 |         0.368 |           0.000 |                0.975 |
| B1_expert_rules |     1904 |        0.378 |                   0.388 |               0.373 |                   0.304 |         0.281 |           0.105 |                0.962 |
| XGB[district]   |     1904 |        0.405 |                   0.623 |               0.547 |                   0.518 |         0.640 |           0.403 |                0.982 |
| XGB[field]      |     1904 |        0.426 |                   0.651 |               0.616 |                   0.584 |         0.654 |           0.430 |                0.989 |
| XGB[all]        |     1904 |        0.451 |                   0.656 |               0.632 |                   0.599 |         0.658 |           0.437 |                0.988 |

## 4. Scores per crop, XGBoost on all inputs

Test: a district the model never saw (main test, decides adoption)

| crop         |   precision |   recall |    f1 |   fields |
|:-------------|------------:|---------:|------:|---------:|
| beans        |       0.381 |    0.494 | 0.430 |      601 |
| irish_potato |       0.629 |    0.785 | 0.698 |      246 |
| maize        |       0.635 |    0.494 | 0.556 |     1010 |
| rice         |       0.000 |    0.000 | 0.000 |       47 |

Test: new fields in the four known districts (second test)

| crop         |   precision |   recall |    f1 |   fields |
|:-------------|------------:|---------:|------:|---------:|
| beans        |       0.554 |    0.554 | 0.554 |      601 |
| irish_potato |       0.686 |    0.711 | 0.699 |      246 |
| maize        |       0.736 |    0.708 | 0.721 |     1010 |
| rice         |       0.342 |    0.553 | 0.423 |       47 |

## 5. Confusion matrix, XGBoost on all inputs (rows: true crop, columns: predicted crop)

Test: a district the model never saw (main test, decides adoption)

| true         |   beans |   irish_potato |   maize |   rice |
|:-------------|--------:|---------------:|--------:|-------:|
| beans        |     297 |             50 |     239 |     15 |
| irish_potato |      29 |            193 |      24 |      0 |
| maize        |     431 |             64 |     499 |     16 |
| rice         |      23 |              0 |      24 |      0 |

Test: new fields in the four known districts (second test)

| true         |   beans |   irish_potato |   maize |   rice |
|:-------------|--------:|---------------:|--------:|-------:|
| beans        |     333 |             36 |     219 |     13 |
| irish_potato |      41 |            175 |      26 |      4 |
| maize        |     221 |             41 |     715 |     33 |
| rice         |       6 |              3 |      12 |     26 |

## 6. Final model

XGBoost on all inputs (21 inputs), trained on all 1904 fields with the protocol's settings. It is the model behind the dashboard map page (`src/recommendation/map_model.py` trains the same model with the same data and settings). Saved in `model/` (not committed: it takes seconds to retrain).

Inputs by gain importance (top 10):

| feature                         | level    |   share |
|:--------------------------------|:---------|--------:|
| precip_mm                       | district |   0.456 |
| tmean_point_c                   | field    |   0.121 |
| silt_0_20cm                     | field    |   0.070 |
| nisr_share_beans                | district |   0.055 |
| elevation_m                     | field    |   0.054 |
| tmin_point_c                    | field    |   0.051 |
| cation_exchange_capacity_0_20cm | field    |   0.038 |
| sand_0_20cm                     | field    |   0.025 |
| nitrogen_0_20cm                 | field    |   0.022 |
| ph_0_20cm                       | field    |   0.021 |

## 7. Same as eval_v1?

| test         | model           |   largest_difference | same_as_eval_v1   |
|:-------------|:----------------|---------------------:|:------------------|
| new_district | B0_majority     |                0.000 | yes               |
| new_district | B1_expert_rules |                0.000 | yes               |
| new_district | XGB[district]   |                0.000 | yes               |
| new_district | XGB[field]      |                0.000 | yes               |
| new_district | XGB[all]        |                0.000 | yes               |
| new_fields   | B0_majority     |                0.000 | yes               |
| new_fields   | B1_expert_rules |                0.000 | yes               |
| new_fields   | XGB[district]   |                0.000 | yes               |
| new_fields   | XGB[field]      |                0.000 | yes               |
| new_fields   | XGB[all]        |                0.000 | yes               |

## 8. Limits

- One season: CIAT season B 2025. Seasons A and C are not in the data yet.
- Weather is one value per district (NASA POWER), and Musanze and Nyabihu share one weather cell. Per-field rainfall (CHIRPS) is the next input to add.
- Settings are not tuned yet (tracker T051, after the midterm, nested inside the training districts).
