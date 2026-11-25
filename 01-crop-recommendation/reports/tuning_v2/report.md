# Crop recommendation: second round of tuning (tuning_v2)

Protocol: `docs/training_protocol.md`, version 1.3 (section 12). Run at 2026-11-25 07:18 UTC: 4.5 hours of computing, 2 jobs at a time.
Table: `01-crop-recommendation/data/processed/feature_table_v6b.csv` (SHA-256 `0bb9096237a38bc0...`). Code commit: `a6dc702920`. Report text rebuilt from the saved jobs at commit `f5efb4b6ff`, with the same numbers.
Candidates per method: RF 102, ET 101, XGB 101 (the section 6 settings, then 100 random draws; Random Forest also has the map model of version 1.2 as candidate 1). 3 inner block folds, 5 shuffles per input.

**Built-in check: passed.** Random Forest and XGBoost with the section 6 settings and the baselines give the `eval_v2` numbers in 48 of 48 rows (district and model).

## Decisions fixed before the run

1. **Adoption (main test).** No tuned model or blend passes the adoption rule, so B1 stays the default answer of the API. Best on the main test: Random Forest, all_sat, tuned, 0.308 against 0.378 for B1, better than B1 in 1 of 4 districts.
2. **Map.** The map keeps Random Forest, all_sat, tuned (0.514 on new fields): the best of this round, Extra Trees, field_sat, tuned, scores 0.521, not more than 0.01 ahead.

Expected before the run: (a) the nearby-crop inputs give the largest gain on new fields; (b) the wider search and Extra Trees add little, up to about 0.01; (c) no model passes the adoption rule.
(a) Best with nearby crops: Extra Trees, all_sat_near, tuned, 0.511; best without: Extra Trees, field_sat, tuned, 0.521.
(b) Best without nearby crops in this round 0.521, against 0.514 for the best of version 1.2 (Random Forest, all_sat, tuned). Extra Trees minus Random Forest on the same view: field_sat +0.004, all_sat +0.004, field_sat_near -0.009, all_sat_near +0.042.
(c) No model passes the adoption rule on the main test.

## New fields in the known districts (decides the map)

Settings chosen inside the training part of each block fold only, then scored once on the test fold. The nearby-crop inputs of a test field come from the training fields of other blocks only.

| model                                     | main score   | worst district   | first crop right, all fields   | section 6 settings   | version 1.2, tuned   |
|:------------------------------------------|:-------------|:-----------------|:-------------------------------|:---------------------|:---------------------|
| Extra Trees, field_sat, tuned             | 0.521        | 0.344            | 0.722                          | 0.409                |                      |
| Random Forest, field_sat, tuned           | 0.517        | 0.288            | 0.717                          | 0.442                | 0.484                |
| Extra Trees, all_sat_near, tuned          | 0.511        | 0.342            | 0.733                          | 0.427                |                      |
| Random Forest, field_sat_near, tuned      | 0.506        | 0.296            | 0.728                          | 0.461                |                      |
| XGBoost, all_sat_near, tuned              | 0.506        | 0.398            | 0.714                          | 0.465                |                      |
| Blend of the three, all_sat, tuned        | 0.503        | 0.313            | 0.721                          |                      |                      |
| Extra Trees, all_sat, tuned               | 0.500        | 0.305            | 0.723                          | 0.418                |                      |
| Blend of the three, field_sat_near, tuned | 0.499        | 0.306            | 0.728                          |                      |                      |
| Extra Trees, field_sat_near, tuned        | 0.497        | 0.293            | 0.725                          | 0.424                |                      |
| Random Forest, all_sat, tuned             | 0.496        | 0.293            | 0.723                          | 0.441                | 0.514                |
| XGBoost, field_sat, tuned                 | 0.496        | 0.338            | 0.708                          | 0.453                | 0.509                |
| XGBoost, field_sat_near, tuned            | 0.496        | 0.384            | 0.713                          | 0.456                |                      |
| Blend of the three, field_sat, tuned      | 0.495        | 0.297            | 0.719                          |                      |                      |
| Blend of the three, all_sat_near, tuned   | 0.491        | 0.313            | 0.730                          |                      |                      |
| XGBoost, all_sat, tuned                   | 0.485        | 0.330            | 0.695                          | 0.456                | 0.485                |
| Random Forest, all_sat_near, tuned        | 0.469        | 0.304            | 0.725                          | 0.447                |                      |

Simple choices and the map model of version 1.2 on the same folds:

|                                                               | main score   | first crop right, all fields   |
|:--------------------------------------------------------------|:-------------|:-------------------------------|
| B0: the most common crop of all training fields               | 0.135        | 0.530                          |
| the most common crop of the field's district                  | 0.223        | 0.670                          |
| B1: expert rules                                              | 0.378        | 0.388                          |
| nearest surveyed fields (h = 4 km)                            | 0.413        | 0.716                          |
| Random Forest, all_sat, tuned (map since 23 Nov, version 1.2) | 0.514        | 0.724                          |

Main score per district (the six best and two references):

| model                                | Musanze   | Nyabihu   | Nyagatare   | Ruhango   |
|:-------------------------------------|:----------|:----------|:------------|:----------|
| Extra Trees, field_sat, tuned        | 0.674     | 0.647     | 0.421       | 0.344     |
| Random Forest, field_sat, tuned      | 0.662     | 0.670     | 0.448       | 0.288     |
| Extra Trees, all_sat_near, tuned     | 0.548     | 0.698     | 0.456       | 0.342     |
| Random Forest, field_sat_near, tuned | 0.605     | 0.681     | 0.444       | 0.296     |
| XGBoost, all_sat_near, tuned         | 0.501     | 0.664     | 0.461       | 0.398     |
| Blend of the three, all_sat, tuned   | 0.623     | 0.612     | 0.463       | 0.313     |
| B1_expert_rules                      | 0.592     | 0.645     | 0.090       | 0.186     |
| near_fields_4km                      | 0.452     | 0.588     | 0.316       | 0.295     |

## What the nearby-crop inputs add (new fields)

| model                         | without nearby crops   | with nearby crops   | change   | first crop right, with   |
|:------------------------------|:-----------------------|:--------------------|:---------|:-------------------------|
| Random Forest, field_sat      | 0.517                  | 0.506               | -0.011   | 0.728                    |
| Random Forest, all_sat        | 0.496                  | 0.469               | -0.027   | 0.725                    |
| Extra Trees, field_sat        | 0.521                  | 0.497               | -0.024   | 0.725                    |
| Extra Trees, all_sat          | 0.500                  | 0.511               | 0.011    | 0.733                    |
| XGBoost, field_sat            | 0.496                  | 0.496               | 0.000    | 0.713                    |
| XGBoost, all_sat              | 0.485                  | 0.506               | 0.021    | 0.714                    |
| Blend of the three, field_sat | 0.495                  | 0.499               | 0.004    | 0.728                    |
| Blend of the three, all_sat   | 0.503                  | 0.491               | -0.012   | 0.730                    |

## Main test: leave one district out (decides adoption)

Settings chosen inside the three training districts only, then scored once on the district left out. The nearby-crop views do not run here: a district never seen has no surveyed field nearby.

| model                                | main score   | worst district   | better than B1 in   | passes the adoption rule   | section 6 settings   | version 1.2, tuned   |
|:-------------------------------------|:-------------|:-----------------|:--------------------|:---------------------------|:---------------------|:---------------------|
| Random Forest, all_sat, tuned        | 0.308        | 0.122            | 1 of 4              | no                         | 0.355                | 0.321                |
| Extra Trees, field_sat, tuned        | 0.299        | 0.141            | 2 of 4              | no                         | 0.355                |                      |
| XGBoost, field_sat, tuned            | 0.293        | 0.156            | 1 of 4              | no                         | 0.308                | 0.326                |
| Blend of the three, field_sat, tuned | 0.293        | 0.182            | 1 of 4              | no                         |                      |                      |
| XGBoost, all_sat, tuned              | 0.282        | 0.186            | 1 of 4              | no                         | 0.296                | 0.266                |
| Blend of the three, all_sat, tuned   | 0.276        | 0.139            | 1 of 4              | no                         |                      |                      |
| Random Forest, field_sat, tuned      | 0.256        | 0.111            | 1 of 4              | no                         | 0.368                | 0.276                |
| Extra Trees, all_sat, tuned          | 0.235        | 0.111            | 1 of 4              | no                         | 0.344                |                      |

B1 (expert rules): 0.378. B0 (most common crop): 0.104.

## Settings chosen in each fold

Candidate 0 is the section 6 settings. Every candidate is in `candidates.json`; the inner score of every candidate in every fold is in `inner_scores.csv`.

| test       | model                                | candidate chosen in each fold                              | folds that kept the section 6 settings   |
|:-----------|:-------------------------------------|:-----------------------------------------------------------|:-----------------------------------------|
| main       | Random Forest, field_sat, tuned      | Musanze: 3, Nyabihu: 92, Nyagatare: 97, Ruhango: 35        | 0 of 4                                   |
| main       | Random Forest, all_sat, tuned        | Musanze: 33, Nyabihu: 49, Nyagatare: 45, Ruhango: 15       | 0 of 4                                   |
| main       | Extra Trees, field_sat, tuned        | Musanze: 9, Nyabihu: 51, Nyagatare: 76, Ruhango: 16        | 0 of 4                                   |
| main       | Extra Trees, all_sat, tuned          | Musanze: 18, Nyabihu: 54, Nyagatare: 35, Ruhango: 25       | 0 of 4                                   |
| main       | XGBoost, field_sat, tuned            | Musanze: 10, Nyabihu: 38, Nyagatare: 92, Ruhango: 94       | 0 of 4                                   |
| main       | XGBoost, all_sat, tuned              | Musanze: 10, Nyabihu: 32, Nyagatare: 34, Ruhango: 54       | 0 of 4                                   |
| new fields | Random Forest, field_sat, tuned      | fold 1: 97, fold 2: 26, fold 3: 80, fold 4: 81, fold 5: 66 | 0 of 5                                   |
| new fields | Random Forest, all_sat, tuned        | fold 1: 23, fold 2: 13, fold 3: 13, fold 4: 45, fold 5: 31 | 0 of 5                                   |
| new fields | Random Forest, field_sat_near, tuned | fold 1: 65, fold 2: 26, fold 3: 80, fold 4: 81, fold 5: 45 | 0 of 5                                   |
| new fields | Random Forest, all_sat_near, tuned   | fold 1: 25, fold 2: 60, fold 3: 40, fold 4: 22, fold 5: 45 | 0 of 5                                   |
| new fields | Extra Trees, field_sat, tuned        | fold 1: 12, fold 2: 65, fold 3: 25, fold 4: 67, fold 5: 25 | 0 of 5                                   |
| new fields | Extra Trees, all_sat, tuned          | fold 1: 79, fold 2: 65, fold 3: 25, fold 4: 76, fold 5: 67 | 0 of 5                                   |
| new fields | Extra Trees, field_sat_near, tuned   | fold 1: 74, fold 2: 12, fold 3: 79, fold 4: 76, fold 5: 79 | 0 of 5                                   |
| new fields | Extra Trees, all_sat_near, tuned     | fold 1: 13, fold 2: 65, fold 3: 79, fold 4: 79, fold 5: 79 | 0 of 5                                   |
| new fields | XGBoost, field_sat, tuned            | fold 1: 98, fold 2: 32, fold 3: 32, fold 4: 92, fold 5: 98 | 0 of 5                                   |
| new fields | XGBoost, all_sat, tuned              | fold 1: 3, fold 2: 57, fold 3: 41, fold 4: 43, fold 5: 71  | 0 of 5                                   |
| new fields | XGBoost, field_sat_near, tuned       | fold 1: 12, fold 2: 3, fold 3: 54, fold 4: 28, fold 5: 32  | 0 of 5                                   |
| new fields | XGBoost, all_sat_near, tuned         | fold 1: 35, fold 2: 98, fold 3: 74, fold 4: 28, fold 5: 28 | 0 of 5                                   |

## When the best model of this round is sure (Extra Trees, field_sat, tuned, new fields)

Probability the model gives its first crop, and how often that first crop was the crop actually planted. Out-of-fold predictions: no field was predicted by a model that saw its 2 km block.

| probability of the first crop   | fields   | share of fields   | first crop right   |
|:--------------------------------|:---------|:------------------|:-------------------|
| 0.2 to 0.3                      | 0        | 0.000             |                    |
| 0.3 to 0.4                      | 16       | 0.008             | 0.562              |
| 0.4 to 0.5                      | 130      | 0.068             | 0.485              |
| 0.5 to 0.6                      | 316      | 0.166             | 0.541              |
| 0.6 to 0.7                      | 443      | 0.233             | 0.711              |
| 0.7 to 0.8                      | 480      | 0.252             | 0.785              |
| 0.8 to 0.9                      | 346      | 0.182             | 0.827              |
| 0.9 to 1.0                      | 173      | 0.091             | 0.890              |

If it answered only when its probability for the first crop is at least this high:

| probability at least   | fields answered   | share answered   | first crop right   |
|:-----------------------|:------------------|:-----------------|:-------------------|
| 0.300                  | 1904              | 1.000            | 0.722              |
| 0.400                  | 1888              | 0.992            | 0.724              |
| 0.500                  | 1758              | 0.923            | 0.741              |
| 0.600                  | 1442              | 0.757            | 0.785              |
| 0.700                  | 999               | 0.525            | 0.818              |
| 0.800                  | 519               | 0.273            | 0.848              |
| 0.900                  | 173               | 0.091            | 0.890              |

Per district, answering only from 0.7:

| district   | fields answered   | share answered   | first crop right   |
|:-----------|:------------------|:-----------------|:-------------------|
| Musanze    | 87                | 0.530            | 0.828              |
| Nyabihu    | 96                | 0.571            | 0.885              |
| Nyagatare  | 657               | 0.606            | 0.831              |
| Ruhango    | 159               | 0.326            | 0.717              |

## Input importance (tuned models, new fields)

How much the main score drops when a group of inputs is shuffled among the test fields (mean of the shuffles). The nearby-crop inputs are shuffled together by shuffling the blocks they come from. Every input and its spread: `importance.csv`.

| model                                | elevation and field temperature   | district climate   | NISR crop shares   | soil 0 to 20 cm   | satellite rainfall and vegetation   | crops grown nearby   |
|:-------------------------------------|:----------------------------------|:-------------------|:-------------------|:------------------|:------------------------------------|:---------------------|
| Extra Trees, all_sat, tuned          | 0.239                             | 0.001              | -0.007             | 0.108             | 0.058                               |                      |
| Extra Trees, all_sat_near, tuned     | 0.165                             | 0.005              | 0.001              | 0.070             | 0.019                               | 0.105                |
| Extra Trees, field_sat, tuned        | 0.195                             |                    |                    | 0.173             | 0.073                               |                      |
| Extra Trees, field_sat_near, tuned   | 0.102                             |                    |                    | 0.049             | 0.021                               | 0.115                |
| Random Forest, all_sat, tuned        | 0.234                             | -0.001             | 0.000              | 0.056             | 0.050                               |                      |
| Random Forest, all_sat_near, tuned   | 0.131                             | -0.004             | 0.000              | 0.038             | 0.013                               | 0.096                |
| Random Forest, field_sat, tuned      | 0.228                             |                    |                    | 0.139             | 0.100                               |                      |
| Random Forest, field_sat_near, tuned | 0.133                             |                    |                    | 0.077             | 0.020                               | 0.199                |
| XGBoost, all_sat, tuned              | 0.162                             | -0.017             | -0.007             | 0.031             | 0.052                               |                      |
| XGBoost, all_sat_near, tuned         | 0.084                             | 0.011              | 0.000              | 0.029             | 0.037                               | 0.095                |
| XGBoost, field_sat, tuned            | 0.126                             |                    |                    | 0.054             | 0.065                               |                      |
| XGBoost, field_sat_near, tuned       | 0.065                             |                    |                    | 0.054             | 0.018                               | 0.079                |

## How mixed the land is

If every field got the most common crop of its own 2 km block (which no model can know in advance):

| district   | fields   | blocks   | blocks with more than one crop   | fields in those blocks   | block's most common crop is right   | main score of the block's most common crop   |
|:-----------|:---------|:---------|:---------------------------------|:-------------------------|:------------------------------------|:---------------------------------------------|
| all        | 1904     | 368      | 0.486                            | 0.698                    | 0.795                               | 0.681                                        |
| Musanze    | 164      | 48       | 0.375                            | 0.530                    | 0.823                               | 0.753                                        |
| Nyabihu    | 168      | 45       | 0.311                            | 0.476                    | 0.887                               | 0.848                                        |
| Nyagatare  | 1084     | 195      | 0.503                            | 0.701                    | 0.808                               | 0.573                                        |
| Ruhango    | 488      | 80       | 0.613                            | 0.824                    | 0.725                               | 0.550                                        |

## How to read this

- Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over the four districts (section 5). 'First crop right' is the share of fields whose first crop was the crop actually planted.
- 'Section 6 settings' is the model with the fixed settings of the protocol. 'Tuned' chose its settings again in every fold, inside the training fields of that fold. 'Version 1.2, tuned' is the same model in `tuning_v1`.
- The blend averages the probabilities of the tuned Random Forest, Extra Trees and XGBoost of the same view.
- The adoption rule (section 7) and the map rule (section 12) were written before this run.
- Full numbers: the CSV files in this folder. Settings, versions and timing: `run_info.json`.
