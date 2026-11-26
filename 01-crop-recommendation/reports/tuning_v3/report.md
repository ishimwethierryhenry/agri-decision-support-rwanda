# Crop recommendation: third round of tuning (tuning_v3)

Protocol: `docs/training_protocol.md`, version 1.4 (section 13). Run at 2026-11-25 09:38 UTC: 0.4 hours of computing, 2 tasks at a time. Code commit: `103d25f183`. Report text rebuilt from the saved tasks at commit `fa46868477`, with the same tasks.

**Built-in check: passed.** Fitting the version 1.3 choice of every fold again gives the `tuning_v2` main score in 48 of 48 rows (model and district).

## Decisions fixed before the run

1. **Adoption (main test).** No model of this round passes the adoption rule, so B1 stays the default answer of the API. Best: Extra Trees, field_sat, steadier choice + crop weights, 0.313 against 0.378 for B1.
2. **Map.** The map moves to Random Forest, field_sat, steadier choice + crop weights.

Expected before the run: (a) the steadier choice recovers part of the loss of version 1.3, to about the level of version 1.2; (b) crop weights raise the main score by about 0.01 and lower the share of first crops right; (c) nothing passes decision 2, and no model passes the adoption rule.
(a) Steadier choice minus version 1.3, same model and view: -0.003, -0.003, +0.011, +0.012, +0.004, +0.008 (mean +0.005).
(b) Crop weights minus the steadier choice: main score +0.016, +0.006, +0.005, +0.009, +0.004, +0.010 (mean +0.008); first crop right +0.004, -0.001, -0.004, +0.001, -0.007, +0.003 (mean -0.001). The weights stayed at no change in 9% of the folds.
(c) Decision 2 moves the map; no model passes the adoption rule.

## New fields in the known districts (decides the map)

Everything chosen inside the training part of each block fold, then scored once on the test fold.

| model                                                             | main score   | worst district   | first crop right, all fields   |
|:------------------------------------------------------------------|:-------------|:-----------------|:-------------------------------|
| Random Forest, field_sat, steadier choice + crop weights          | 0.531        | 0.337            | 0.722                          |
| Extra Trees, field_sat, steadier choice + crop weights            | 0.525        | 0.320            | 0.722                          |
| Extra Trees, field_sat, version 1.3 tuned                         | 0.521        | 0.344            | 0.722                          |
| Blend of the three, all_sat, group chosen inside + crop weights   | 0.519        | 0.313            | 0.718                          |
| Extra Trees, field_sat, steadier choice                           | 0.519        | 0.342            | 0.722                          |
| Random Forest, field_sat, version 1.3 tuned                       | 0.517        | 0.288            | 0.717                          |
| Random Forest, all_sat, steadier choice + crop weights            | 0.517        | 0.322            | 0.722                          |
| Random Forest, field_sat, steadier choice                         | 0.514        | 0.298            | 0.718                          |
| Blend of the three, field_sat, group chosen inside + crop weights | 0.514        | 0.346            | 0.709                          |
| Blend of the three, field_sat, group chosen inside                | 0.513        | 0.361            | 0.709                          |
| XGBoost, field_sat, steadier choice + crop weights                | 0.511        | 0.332            | 0.693                          |
| Random Forest, all_sat, steadier choice                           | 0.508        | 0.330            | 0.721                          |
| Extra Trees, all_sat, steadier choice + crop weights              | 0.508        | 0.330            | 0.715                          |
| XGBoost, field_sat, steadier choice                               | 0.507        | 0.344            | 0.697                          |
| Extra Trees, all_sat, steadier choice                             | 0.504        | 0.307            | 0.722                          |
| XGBoost, all_sat, steadier choice + crop weights                  | 0.503        | 0.297            | 0.705                          |
| Blend of the three, all_sat, group chosen inside                  | 0.502        | 0.313            | 0.716                          |
| Extra Trees, all_sat, version 1.3 tuned                           | 0.500        | 0.305            | 0.723                          |
| Random Forest, all_sat, version 1.3 tuned                         | 0.496        | 0.293            | 0.723                          |
| XGBoost, field_sat, version 1.3 tuned                             | 0.496        | 0.338            | 0.708                          |
| XGBoost, all_sat, steadier choice                                 | 0.493        | 0.306            | 0.703                          |
| XGBoost, all_sat, version 1.3 tuned                               | 0.485        | 0.330            | 0.695                          |

References on the same folds (from `tuning_v2`): B0, most common crop 0.135 (first crop right 0.530); most common crop of the district 0.223 (first crop right 0.670); B1, expert rules 0.378 (first crop right 0.388).

Groups the blend chose inside the training part (new fields, 2 views x 5 folds): XGB 3, RF+XGB 2, ET 2, RF+ET 2, RF+ET+XGB 1.

## When the model the map serves is sure (Random Forest, field_sat, steadier choice + crop weights, new fields)

If it answered only when its probability for the first crop is at least this high (out-of-fold, crop weights applied):

| probability at least   | fields answered   | share answered   | first crop right   |
|:-----------------------|:------------------|:-----------------|:-------------------|
| 0.300                  | 1903              | 0.999            | 0.723              |
| 0.400                  | 1880              | 0.987            | 0.727              |
| 0.500                  | 1769              | 0.929            | 0.736              |
| 0.600                  | 1483              | 0.779            | 0.765              |
| 0.700                  | 1098              | 0.577            | 0.797              |
| 0.800                  | 642               | 0.337            | 0.819              |
| 0.900                  | 202               | 0.106            | 0.847              |

## Main test: leave one district out (decides adoption)

| model                                                             | main score   | worst district   | first crop right, all fields   | better than B1 in   | passes the adoption rule   |
|:------------------------------------------------------------------|:-------------|:-----------------|:-------------------------------|:--------------------|:---------------------------|
| Extra Trees, field_sat, steadier choice + crop weights            | 0.313        | 0.204            | 0.599                          | 2 of 4              | no                         |
| Random Forest, all_sat, steadier choice                           | 0.308        | 0.122            | 0.522                          | 1 of 4              | no                         |
| Random Forest, all_sat, version 1.3 tuned                         | 0.308        | 0.122            | 0.522                          | 1 of 4              | no                         |
| Blend of the three, field_sat, group chosen inside + crop weights | 0.302        | 0.156            | 0.583                          | 1 of 4              | no                         |
| Blend of the three, field_sat, group chosen inside                | 0.302        | 0.156            | 0.538                          | 1 of 4              | no                         |
| Extra Trees, field_sat, steadier choice                           | 0.299        | 0.141            | 0.426                          | 2 of 4              | no                         |
| Extra Trees, field_sat, version 1.3 tuned                         | 0.299        | 0.141            | 0.426                          | 2 of 4              | no                         |
| Blend of the three, all_sat, group chosen inside                  | 0.297        | 0.148            | 0.522                          | 1 of 4              | no                         |
| Random Forest, all_sat, steadier choice + crop weights            | 0.295        | 0.122            | 0.519                          | 1 of 4              | no                         |
| XGBoost, field_sat, steadier choice                               | 0.293        | 0.156            | 0.535                          | 1 of 4              | no                         |
| XGBoost, field_sat, version 1.3 tuned                             | 0.293        | 0.156            | 0.535                          | 1 of 4              | no                         |
| Blend of the three, all_sat, group chosen inside + crop weights   | 0.292        | 0.129            | 0.519                          | 1 of 4              | no                         |
| XGBoost, field_sat, steadier choice + crop weights                | 0.291        | 0.156            | 0.580                          | 1 of 4              | no                         |
| XGBoost, all_sat, steadier choice                                 | 0.282        | 0.186            | 0.619                          | 1 of 4              | no                         |
| XGBoost, all_sat, version 1.3 tuned                               | 0.282        | 0.186            | 0.619                          | 1 of 4              | no                         |
| XGBoost, all_sat, steadier choice + crop weights                  | 0.263        | 0.152            | 0.612                          | 1 of 4              | no                         |
| Extra Trees, all_sat, steadier choice + crop weights              | 0.262        | 0.125            | 0.606                          | 1 of 4              | no                         |
| Random Forest, field_sat, steadier choice + crop weights          | 0.256        | 0.111            | 0.335                          | 1 of 4              | no                         |
| Random Forest, field_sat, version 1.3 tuned                       | 0.256        | 0.111            | 0.335                          | 1 of 4              | no                         |
| Random Forest, field_sat, steadier choice                         | 0.256        | 0.111            | 0.335                          | 1 of 4              | no                         |
| Extra Trees, all_sat, steadier choice                             | 0.235        | 0.111            | 0.329                          | 1 of 4              | no                         |
| Extra Trees, all_sat, version 1.3 tuned                           | 0.235        | 0.111            | 0.329                          | 1 of 4              | no                         |

## How to read this

- Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over the four districts (section 5). 'First crop right' is the share of all test fields whose first crop was the crop planted.
- 'Version 1.3 tuned' is the model of `tuning_v2`, fitted again (built-in check). 'Steadier choice' compared the 8 best candidates of version 1.3 on two more inner splits. 'Crop weights' multiply the probabilities by one weight per crop chosen inside the training part. The blend averages the group of tuned models chosen inside.
- Every choice of every fold is in `choices.csv`. Settings, versions and timing: `run_info.json`.
