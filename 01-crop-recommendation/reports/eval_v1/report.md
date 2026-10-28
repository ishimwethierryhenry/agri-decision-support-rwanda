# Crop recommendation evaluation (eval_v1)

Protocol: `docs/training_protocol.md` (version 1.0). Run at 2026-10-28 17:10 UTC.
Table: `01-crop-recommendation/data/processed/feature_table_v2.csv` (SHA-256 `e84d21c8a859ef88...`). Code commit: `cf38b291ad`.

Rows: 2739 in the table, 2728 pass qc_ok, 1904 single-crop fields in the main set, 650 in the CIAT 'Used' second check.

## Fields per district and crop (main set)

| district   |   beans |   irish_potato |   maize |   rice |   All |
|:-----------|--------:|---------------:|--------:|-------:|------:|
| Musanze    |      47 |             92 |      24 |      1 |   164 |
| Nyabihu    |      28 |            103 |      37 |      0 |   168 |
| Nyagatare  |     237 |             19 |     792 |     36 |  1084 |
| Ruhango    |     289 |             32 |     157 |     10 |   488 |
| All        |     601 |            246 |    1010 |     47 |  1904 |

## Main test: leave one district out

Train on three districts, test on the fourth. This decides adoption.

### Main set: leave one district out

**No ML model passes the adoption rule, so B1 stays. Best ML model: XGB[district], mean macro-F1 0.365 vs B1 0.378, better than B1 in 2 of 4 districts.**

Mean over the four districts:

| model           |   macro_f1_mean |   macro_f1_all4_mean |   top1_mean |   top3_mean |   worst_district_macro_f1 |   districts_beating_B1 | passes_rule   |
|:----------------|----------------:|---------------------:|------------:|------------:|--------------------------:|-----------------------:|:--------------|
| B1_expert_rules |           0.378 |                0.301 |       0.523 |       0.948 |                     0.09  |                    nan |               |
| XGB[district]   |           0.365 |                0.304 |       0.616 |       0.973 |                     0.239 |                      2 | no            |
| RF[district]    |           0.365 |                0.305 |       0.598 |       0.983 |                     0.235 |                      2 | no            |
| RF[all_subsoil] |           0.354 |                0.292 |       0.617 |       0.987 |                     0.16  |                      1 | no            |
| RF[field]       |           0.338 |                0.281 |       0.608 |       0.991 |                     0.215 |                      2 | no            |
| RF[all]         |           0.334 |                0.275 |       0.601 |       0.987 |                     0.153 |                      1 | no            |
| XGB[field]      |           0.308 |                0.257 |       0.522 |       0.984 |                     0.21  |                      2 | no            |
| XGB[all]        |           0.294 |                0.249 |       0.554 |       0.987 |                     0.215 |                      2 | no            |
| LR[all]         |           0.254 |                0.208 |       0.509 |       0.958 |                     0.055 |                      1 | no            |
| LR[field]       |           0.241 |                0.208 |       0.472 |       0.929 |                     0.103 |                      2 | no            |
| LR[district]    |           0.229 |                0.186 |       0.501 |       0.963 |                     0.01  |                      1 | no            |
| B0_majority     |           0.104 |                0.091 |       0.227 |       0.985 |                     0.085 |                      0 |               |

Main score (macro-F1) per district:

| model           |   Musanze |   Nyabihu |   Nyagatare |   Ruhango |
|:----------------|----------:|----------:|------------:|----------:|
| B1_expert_rules |     0.592 |     0.645 |       0.09  |     0.186 |
| XGB[district]   |     0.563 |     0.419 |       0.239 |     0.24  |
| RF[district]    |     0.447 |     0.518 |       0.261 |     0.235 |
| RF[all_subsoil] |     0.512 |     0.483 |       0.262 |     0.16  |
| RF[field]       |     0.405 |     0.509 |       0.223 |     0.215 |
| RF[all]         |     0.475 |     0.462 |       0.245 |     0.153 |
| XGB[field]      |     0.366 |     0.438 |       0.216 |     0.21  |
| XGB[all]        |     0.342 |     0.376 |       0.242 |     0.215 |
| LR[all]         |     0.268 |     0.474 |       0.221 |     0.055 |
| LR[field]       |     0.277 |     0.31  |       0.103 |     0.272 |
| LR[district]    |     0.401 |     0.295 |       0.211 |     0.01  |
| B0_majority     |     0.085 |     0.12  |       0.09  |     0.122 |

Recall per crop (mean over the districts where the crop is grown):

| model           |   beans |   irish_potato |   maize |   rice |
|:----------------|--------:|---------------:|--------:|-------:|
| B1_expert_rules |    0.84 |           0.31 |    0.32 |   0    |
| XGB[district]   |    0.44 |           0.45 |    0.33 |   0.02 |
| RF[district]    |    0.44 |           0.47 |    0.36 |   0    |
| RF[all_subsoil] |    0.42 |           0.46 |    0.45 |   0.02 |
| RF[field]       |    0.37 |           0.48 |    0.39 |   0    |
| RF[all]         |    0.33 |           0.47 |    0.45 |   0    |
| XGB[field]      |    0.42 |           0.49 |    0.24 |   0.01 |
| XGB[all]        |    0.36 |           0.49 |    0.22 |   0    |
| LR[all]         |    0.14 |           0.51 |    0.25 |   0.33 |
| LR[field]       |    0.35 |           0.49 |    0.18 |   0.36 |
| LR[district]    |    0.09 |           0.49 |    0.25 |   0.33 |
| B0_majority     |    0.25 |           0    |    0.75 |   0    |

## Second test: new fields inside the four known districts

5 folds over 2 km blocks (block_id). Reported, but it does not decide adoption.

### Main set: new fields inside the known districts

**For information only (this test does not decide): no ML model would pass the same rule here. Best ML model: XGB[all], mean macro-F1 0.451 vs B1 0.378, better than B1 in 2 of 4 districts.**

Mean over the four districts:

| model           |   macro_f1_mean |   macro_f1_all4_mean |   top1_mean |   top3_mean |   worst_district_macro_f1 |   districts_beating_B1 | passes_rule   |
|:----------------|----------------:|---------------------:|------------:|------------:|--------------------------:|-----------------------:|:--------------|
| XGB[all]        |           0.451 |                0.382 |       0.659 |       0.991 |                     0.33  |                      2 | no            |
| XGB[field]      |           0.426 |                0.363 |       0.652 |       0.991 |                     0.327 |                      2 | no            |
| RF[all_subsoil] |           0.426 |                0.36  |       0.691 |       0.991 |                     0.293 |                      2 | no            |
| RF[all]         |           0.424 |                0.359 |       0.687 |       0.99  |                     0.276 |                      2 | no            |
| RF[field]       |           0.418 |                0.356 |       0.678 |       0.99  |                     0.295 |                      2 | no            |
| XGB[district]   |           0.405 |                0.335 |       0.652 |       0.985 |                     0.248 |                      2 | no            |
| B1_expert_rules |           0.378 |                0.301 |       0.523 |       0.948 |                     0.09  |                    nan |               |
| LR[all]         |           0.365 |                0.311 |       0.628 |       0.982 |                     0.265 |                      2 | no            |
| RF[district]    |           0.362 |                0.304 |       0.626 |       0.98  |                     0.254 |                      2 | no            |
| LR[district]    |           0.353 |                0.292 |       0.612 |       0.987 |                     0.221 |                      2 | no            |
| LR[field]       |           0.329 |                0.291 |       0.595 |       0.979 |                     0.292 |                      2 | no            |
| B0_majority     |           0.135 |                0.122 |       0.355 |       0.985 |                     0.085 |                      1 |               |

Main score (macro-F1) per district:

| model           |   Musanze |   Nyabihu |   Nyagatare |   Ruhango |
|:----------------|----------:|----------:|------------:|----------:|
| XGB[all]        |     0.528 |     0.577 |       0.372 |     0.33  |
| XGB[field]      |     0.503 |     0.518 |       0.358 |     0.327 |
| RF[all_subsoil] |     0.479 |     0.564 |       0.366 |     0.293 |
| RF[all]         |     0.472 |     0.564 |       0.383 |     0.276 |
| RF[field]       |     0.472 |     0.515 |       0.39  |     0.295 |
| XGB[district]   |     0.575 |     0.549 |       0.25  |     0.248 |
| B1_expert_rules |     0.592 |     0.645 |       0.09  |     0.186 |
| LR[all]         |     0.393 |     0.485 |       0.319 |     0.265 |
| RF[district]    |     0.448 |     0.486 |       0.261 |     0.254 |
| LR[district]    |     0.461 |     0.507 |       0.222 |     0.221 |
| LR[field]       |     0.305 |     0.411 |       0.292 |     0.306 |
| B0_majority     |     0.085 |     0.12  |       0.211 |     0.122 |

Recall per crop (mean over the districts where the crop is grown):

| model           |   beans |   irish_potato |   maize |   rice |
|:----------------|--------:|---------------:|--------:|-------:|
| XGB[all]        |    0.54 |           0.45 |    0.36 |   0.34 |
| XGB[field]      |    0.51 |           0.46 |    0.34 |   0.29 |
| RF[all_subsoil] |    0.59 |           0.47 |    0.31 |   0.24 |
| RF[all]         |    0.59 |           0.47 |    0.31 |   0.29 |
| RF[field]       |    0.58 |           0.46 |    0.3  |   0.27 |
| XGB[district]   |    0.53 |           0.47 |    0.32 |   0.11 |
| B1_expert_rules |    0.84 |           0.31 |    0.32 |   0    |
| LR[all]         |    0.46 |           0.5  |    0.21 |   0.55 |
| RF[district]    |    0.53 |           0.48 |    0.24 |   0.09 |
| LR[district]    |    0.52 |           0.49 |    0.17 |   0.41 |
| LR[field]       |    0.35 |           0.51 |    0.21 |   0.72 |
| B0_majority     |    0    |           0    |    1    |   0    |

## Second check: CIAT 'Used' fields only

Same tests on the fields CIAT used in its own satellite crop model (Status == 'Used').

| district   |   beans |   irish_potato |   maize |   rice |   All |
|:-----------|--------:|---------------:|--------:|-------:|------:|
| Musanze    |      30 |             55 |      13 |      1 |    99 |
| Nyabihu    |       9 |             51 |      14 |      0 |    74 |
| Nyagatare  |      91 |              6 |     274 |     23 |   394 |
| Ruhango    |      29 |              2 |      45 |      7 |    83 |
| All        |     159 |            114 |     346 |     31 |   650 |

### Used only: leave one district out

**For information only (this test does not decide): no ML model would pass the same rule here. Best ML model: RF[all_subsoil], mean macro-F1 0.429 vs B1 0.384, better than B1 in 2 of 4 districts.**

Mean over the four districts:

| model           |   macro_f1_mean |   macro_f1_all4_mean |   top1_mean |   top3_mean |   worst_district_macro_f1 |   districts_beating_B1 | passes_rule   |
|:----------------|----------------:|---------------------:|------------:|------------:|--------------------------:|-----------------------:|:--------------|
| RF[all_subsoil] |           0.429 |                0.341 |       0.7   |       0.978 |                     0.306 |                      2 | no            |
| RF[district]    |           0.428 |                0.336 |       0.697 |       0.984 |                     0.233 |                      2 | no            |
| XGB[field]      |           0.409 |                0.32  |       0.566 |       0.972 |                     0.209 |                      2 | no            |
| RF[field]       |           0.392 |                0.311 |       0.646 |       0.978 |                     0.271 |                      2 | no            |
| RF[all]         |           0.384 |                0.305 |       0.662 |       0.981 |                     0.272 |                      2 | no            |
| B1_expert_rules |           0.384 |                0.294 |       0.472 |       0.967 |                     0.094 |                    nan |               |
| XGB[all]        |           0.376 |                0.295 |       0.56  |       0.969 |                     0.207 |                      2 | no            |
| XGB[district]   |           0.348 |                0.287 |       0.54  |       0.972 |                     0.246 |                      2 | no            |
| LR[district]    |           0.302 |                0.239 |       0.555 |       0.98  |                     0.053 |                      1 | no            |
| LR[all]         |           0.3   |                0.238 |       0.551 |       0.977 |                     0.174 |                      1 | no            |
| LR[field]       |           0.231 |                0.185 |       0.476 |       0.934 |                     0.107 |                      2 | no            |
| B0_majority     |           0.106 |                0.08  |       0.219 |       0.962 |                     0.008 |                      1 |               |

Main score (macro-F1) per district:

| model           |   Musanze |   Nyabihu |   Nyagatare |   Ruhango |
|:----------------|----------:|----------:|------------:|----------:|
| RF[all_subsoil] |     0.544 |     0.534 |       0.306 |     0.331 |
| RF[district]    |     0.554 |     0.555 |       0.233 |     0.371 |
| XGB[field]      |     0.543 |     0.549 |       0.209 |     0.335 |
| RF[field]       |     0.432 |     0.534 |       0.271 |     0.332 |
| RF[all]         |     0.511 |     0.478 |       0.272 |     0.275 |
| B1_expert_rules |     0.655 |     0.615 |       0.094 |     0.173 |
| XGB[all]        |     0.515 |     0.426 |       0.207 |     0.356 |
| XGB[district]   |     0.36  |     0.515 |       0.246 |     0.27  |
| LR[district]    |     0.478 |     0.471 |       0.205 |     0.053 |
| LR[all]         |     0.34  |     0.481 |       0.205 |     0.174 |
| LR[field]       |     0.258 |     0.281 |       0.107 |     0.279 |
| B0_majority     |     0.077 |     0.106 |       0.008 |     0.234 |

Recall per crop (mean over the districts where the crop is grown):

| model           |   beans |   irish_potato |   maize |   rice |
|:----------------|--------:|---------------:|--------:|-------:|
| RF[all_subsoil] |    0.5  |           0.48 |    0.46 |   0.14 |
| RF[district]    |    0.62 |           0.48 |    0.38 |   0    |
| XGB[field]      |    0.32 |           0.41 |    0.55 |   0.12 |
| RF[field]       |    0.38 |           0.49 |    0.4  |   0.12 |
| RF[all]         |    0.36 |           0.48 |    0.45 |   0.09 |
| B1_expert_rules |    0.85 |           0.3  |    0.39 |   0    |
| XGB[all]        |    0.26 |           0.42 |    0.52 |   0.17 |
| XGB[district]   |    0.43 |           0.49 |    0.43 |   0.05 |
| LR[district]    |    0.25 |           0.49 |    0.25 |   0.33 |
| LR[all]         |    0.19 |           0.5  |    0.27 |   0.33 |
| LR[field]       |    0.25 |           0.48 |    0.2  |   0.36 |
| B0_majority     |    0    |           0.25 |    0.75 |   0    |

### Used only: new fields inside the known districts

**For information only (this test does not decide): XGB[all], XGB[district] would pass the same rule here. Best ML model: XGB[all], mean macro-F1 0.483 vs B1 0.384, better than B1 in 3 of 4 districts.**

Mean over the four districts:

| model           |   macro_f1_mean |   macro_f1_all4_mean |   top1_mean |   top3_mean |   worst_district_macro_f1 |   districts_beating_B1 | passes_rule   |
|:----------------|----------------:|---------------------:|------------:|------------:|--------------------------:|-----------------------:|:--------------|
| XGB[all]        |           0.483 |                0.385 |       0.661 |       0.983 |                     0.355 |                      3 | yes           |
| XGB[field]      |           0.459 |                0.368 |       0.643 |       0.983 |                     0.381 |                      2 | no            |
| RF[all]         |           0.454 |                0.362 |       0.669 |       0.982 |                     0.354 |                      2 | no            |
| RF[all_subsoil] |           0.45  |                0.36  |       0.667 |       0.982 |                     0.363 |                      2 | no            |
| RF[field]       |           0.447 |                0.358 |       0.656 |       0.982 |                     0.366 |                      2 | no            |
| XGB[district]   |           0.446 |                0.349 |       0.625 |       0.98  |                     0.234 |                      3 | yes           |
| RF[district]    |           0.42  |                0.332 |       0.615 |       0.983 |                     0.272 |                      2 | no            |
| LR[all]         |           0.419 |                0.335 |       0.59  |       0.976 |                     0.293 |                      2 | no            |
| B1_expert_rules |           0.384 |                0.294 |       0.472 |       0.967 |                     0.094 |                    nan |               |
| LR[district]    |           0.355 |                0.28  |       0.544 |       0.989 |                     0.139 |                      1 | no            |
| LR[field]       |           0.345 |                0.278 |       0.577 |       0.965 |                     0.283 |                      2 | no            |
| B0_majority     |           0.156 |                0.13  |       0.39  |       0.962 |                     0.077 |                      2 |               |

Main score (macro-F1) per district:

| model           |   Musanze |   Nyabihu |   Nyagatare |   Ruhango |
|:----------------|----------:|----------:|------------:|----------:|
| XGB[all]        |     0.492 |     0.664 |       0.355 |     0.423 |
| XGB[field]      |     0.487 |     0.571 |       0.381 |     0.4   |
| RF[all]         |     0.548 |     0.494 |       0.354 |     0.419 |
| RF[all_subsoil] |     0.554 |     0.498 |       0.363 |     0.385 |
| RF[field]       |     0.542 |     0.512 |       0.366 |     0.368 |
| XGB[district]   |     0.532 |     0.637 |       0.234 |     0.381 |
| RF[district]    |     0.554 |     0.536 |       0.272 |     0.319 |
| LR[all]         |     0.55  |     0.503 |       0.329 |     0.293 |
| B1_expert_rules |     0.655 |     0.615 |       0.094 |     0.173 |
| LR[district]    |     0.548 |     0.518 |       0.218 |     0.139 |
| LR[field]       |     0.466 |     0.283 |       0.305 |     0.327 |
| B0_majority     |     0.077 |     0.106 |       0.205 |     0.234 |

Recall per crop (mean over the districts where the crop is grown):

| model           |   beans |   irish_potato |   maize |   rice |
|:----------------|--------:|---------------:|--------:|-------:|
| XGB[all]        |    0.55 |           0.47 |    0.41 |   0.18 |
| XGB[field]      |    0.51 |           0.45 |    0.39 |   0.21 |
| RF[all]         |    0.59 |           0.48 |    0.34 |   0.24 |
| RF[all_subsoil] |    0.59 |           0.48 |    0.33 |   0.21 |
| RF[field]       |    0.55 |           0.47 |    0.34 |   0.24 |
| XGB[district]   |    0.59 |           0.47 |    0.31 |   0.12 |
| RF[district]    |    0.59 |           0.48 |    0.27 |   0.17 |
| LR[all]         |    0.55 |           0.48 |    0.19 |   0.47 |
| B1_expert_rules |    0.85 |           0.3  |    0.39 |   0    |
| LR[district]    |    0.46 |           0.48 |    0.16 |   0.31 |
| LR[field]       |    0.29 |           0.48 |    0.29 |   0.19 |
| B0_majority     |    0    |           0    |    1    |   0    |

## How to read this

- Main score: macro-F1 per test district, over the crops with at least 5 fields there, then the mean over the four districts. `macro_f1_all4` is the old definition (all four crops, so a crop a district does not grow counts as zero).
- Adoption: ML replaces B1 only if its mean main score is higher AND it beats B1 by more than 0.01 in at least 3 of the 4 districts, on the main test.
- Top-3 cannot decide anything with four crops: a fixed ranking that never looks at the field already scores close to 100 percent.
- B0 gives the training share of each crop. B1 uses published crop ranges only and is not trained.
- Views: district = elevation, season climate and NISR district shares; field = elevation and soil 0 to 20 cm; all = both; all_subsoil = all plus soil 20 to 50 cm.
- Climate and NISR shares are one value per district, so in the main test they mostly tell the model which district it is in, and the held-out district has values the model has never seen.
- Full numbers: the CSV files in this folder. Settings and versions: run_info.json.
