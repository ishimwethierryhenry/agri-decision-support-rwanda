# Crop recommendation: tuned settings (tuning_v1)

Protocol: `docs/training_protocol.md`, version 1.2 (section 11). Run at 2026-11-23 11:57 UTC: 1.3 hours of computing, 2 jobs at a time.
Table: `01-crop-recommendation/data/processed/feature_table_v6b.csv` (SHA-256 `0bb9096237a38bc0...`). Code commit: `06026db101`. Report text rebuilt from the saved jobs at commit `cc891c8a46`, with the same numbers.
61 candidates per method (the protocol settings, then 60 random draws), 3 inner block folds, 5 shuffles per input.

**Built-in check: passed.** The protocol settings and the baselines give the `eval_v2` numbers in 88 of 88 rows (district and model).

## Decisions fixed before the run

1. **Adoption (main test).** No tuned model passes the adoption rule, so B1 stays the default answer of the API. Best tuned model on the main test: XGB[field_sat] tuned, 0.326 against 0.378 for B1, better than B1 in 2 of 4 districts.
2. **Map.** The map moves to RF[all_sat] tuned: on new fields it scores 0.514 against 0.451 for XGB[all] with the protocol settings, more than 0.01 ahead. Tuned models ahead by more than 0.01: RF[all_sat] tuned (0.514), XGB[field_sat] tuned (0.509), XGB[all_sat] tuned (0.485), RF[field_sat] tuned (0.484), LR[field_sat] tuned (0.470), LR[all_sat] tuned (0.468), XGB[all] tuned (0.463), RF[all] tuned (0.462).

Expected before the run: (a) small gains, a few hundredths of main score at most; (b) Logistic Regression gains more than the tree models.
(a) Largest gain: +0.126 (LR[field_sat], new fields).
(b) Mean gain per method: main test: LR +0.039, RF -0.047, XGB -0.012; new fields: LR +0.091, RF +0.051, XGB +0.032. Logistic Regression gained the most in the main test and the new-fields test.

## Main test: leave one district out (decides adoption)

Settings chosen inside the three training districts only, then scored once on the district left out.

| model          | main score, protocol settings   | main score, tuned   | change   | worst district, tuned   | better than B1 in, tuned   | passes the adoption rule, tuned   | first crop right, all fields, tuned   |
|:---------------|:--------------------------------|:--------------------|:---------|:------------------------|:---------------------------|:----------------------------------|:--------------------------------------|
| LR[all]        | 0.254                           | 0.288               | 0.033    | 0.113                   | 1                          | no                                | 0.266                                 |
| LR[field_sat]  | 0.248                           | 0.267               | 0.019    | 0.141                   | 1                          | no                                | 0.387                                 |
| LR[all_sat]    | 0.228                           | 0.293               | 0.065    | 0.134                   | 1                          | no                                | 0.487                                 |
| RF[all]        | 0.334                           | 0.318               | -0.016   | 0.122                   | 1                          | no                                | 0.454                                 |
| RF[field_sat]  | 0.368                           | 0.276               | -0.091   | 0.107                   | 1                          | no                                | 0.341                                 |
| RF[all_sat]    | 0.355                           | 0.321               | -0.034   | 0.122                   | 1                          | no                                | 0.463                                 |
| XGB[all]       | 0.294                           | 0.269               | -0.025   | 0.125                   | 1                          | no                                | 0.510                                 |
| XGB[field_sat] | 0.308                           | 0.326               | 0.018    | 0.203                   | 2                          | no                                | 0.474                                 |
| XGB[all_sat]   | 0.296                           | 0.266               | -0.030   | 0.158                   | 1                          | no                                | 0.409                                 |

B1 (expert rules): 0.378. B0 (most common crop): 0.104.

Main score per district:

|                 | Musanze, protocol   | Nyabihu, protocol   | Nyagatare, protocol   | Ruhango, protocol   | Musanze, tuned   | Nyabihu, tuned   | Nyagatare, tuned   | Ruhango, tuned   |
|:----------------|:--------------------|:--------------------|:----------------------|:--------------------|:-----------------|:-----------------|:-------------------|:-----------------|
| B1_expert_rules | 0.592               | 0.645               | 0.090                 | 0.186               |                  |                  |                    |                  |
| LR[all]         | 0.268               | 0.474               | 0.221                 | 0.055               | 0.373            | 0.512            | 0.113              | 0.153            |
| LR[field_sat]   | 0.278               | 0.369               | 0.108                 | 0.237               | 0.278            | 0.420            | 0.230              | 0.141            |
| LR[all_sat]     | 0.348               | 0.264               | 0.232                 | 0.067               | 0.315            | 0.507            | 0.216              | 0.134            |
| RF[all]         | 0.475               | 0.462               | 0.245                 | 0.153               | 0.470            | 0.463            | 0.215              | 0.122            |
| RF[field_sat]   | 0.495               | 0.508               | 0.238                 | 0.229               | 0.348            | 0.483            | 0.107              | 0.166            |
| RF[all_sat]     | 0.512               | 0.491               | 0.264                 | 0.153               | 0.470            | 0.464            | 0.226              | 0.122            |
| XGB[all]        | 0.342               | 0.376               | 0.242                 | 0.215               | 0.293            | 0.399            | 0.258              | 0.125            |
| XGB[field_sat]  | 0.431               | 0.471               | 0.149                 | 0.183               | 0.337            | 0.494            | 0.272              | 0.203            |
| XGB[all_sat]    | 0.437               | 0.383               | 0.202                 | 0.161               | 0.304            | 0.397            | 0.203              | 0.158            |

## Second test: new fields in the known districts

Settings chosen inside the training part of each block fold only, then scored once on the test fold.

| model          | main score, protocol settings   | main score, tuned   | change   | worst district, tuned   | better than B1 in, tuned   | would pass the same rule, tuned (information only)   | first crop right, all fields, tuned   |
|:---------------|:--------------------------------|:--------------------|:---------|:------------------------|:---------------------------|:-----------------------------------------------------|:--------------------------------------|
| LR[all]        | 0.365                           | 0.432               | 0.066    | 0.294                   | 2                          | no                                                   | 0.670                                 |
| LR[field_sat]  | 0.344                           | 0.470               | 0.126    | 0.326                   | 2                          | no                                                   | 0.705                                 |
| LR[all_sat]    | 0.389                           | 0.468               | 0.079    | 0.276                   | 2                          | no                                                   | 0.705                                 |
| RF[all]        | 0.424                           | 0.462               | 0.038    | 0.311                   | 2                          | no                                                   | 0.700                                 |
| RF[field_sat]  | 0.442                           | 0.484               | 0.041    | 0.301                   | 2                          | no                                                   | 0.719                                 |
| RF[all_sat]    | 0.441                           | 0.514               | 0.073    | 0.337                   | 3                          | yes                                                  | 0.724                                 |
| XGB[all]       | 0.451                           | 0.463               | 0.011    | 0.299                   | 2                          | no                                                   | 0.668                                 |
| XGB[field_sat] | 0.453                           | 0.509               | 0.056    | 0.343                   | 3                          | yes                                                  | 0.703                                 |
| XGB[all_sat]   | 0.456                           | 0.485               | 0.029    | 0.352                   | 2                          | no                                                   | 0.699                                 |

B1 (expert rules): 0.378. This test does not decide adoption (section 7): the rule column only says whether a model would pass the same rule here, as in `eval_v2`.

Main score per district:

|                 | Musanze, protocol   | Nyabihu, protocol   | Nyagatare, protocol   | Ruhango, protocol   | Musanze, tuned   | Nyabihu, tuned   | Nyagatare, tuned   | Ruhango, tuned   |
|:----------------|:--------------------|:--------------------|:----------------------|:--------------------|:-----------------|:-----------------|:-------------------|:-----------------|
| B1_expert_rules | 0.592               | 0.645               | 0.090                 | 0.186               |                  |                  |                    |                  |
| LR[all]         | 0.393               | 0.485               | 0.319                 | 0.265               | 0.472            | 0.643            | 0.317              | 0.294            |
| LR[field_sat]   | 0.359               | 0.429               | 0.297                 | 0.290               | 0.484            | 0.631            | 0.438              | 0.326            |
| LR[all_sat]     | 0.430               | 0.470               | 0.399                 | 0.256               | 0.523            | 0.625            | 0.449              | 0.276            |
| RF[all]         | 0.472               | 0.564               | 0.383                 | 0.276               | 0.570            | 0.610            | 0.355              | 0.311            |
| RF[field_sat]   | 0.473               | 0.535               | 0.444                 | 0.318               | 0.554            | 0.642            | 0.438              | 0.301            |
| RF[all_sat]     | 0.477               | 0.532               | 0.446                 | 0.308               | 0.661            | 0.627            | 0.432              | 0.337            |
| XGB[all]        | 0.528               | 0.577               | 0.372                 | 0.330               | 0.588            | 0.579            | 0.386              | 0.299            |
| XGB[field_sat]  | 0.478               | 0.564               | 0.447                 | 0.321               | 0.641            | 0.598            | 0.453              | 0.343            |
| XGB[all_sat]    | 0.478               | 0.569               | 0.458                 | 0.318               | 0.533            | 0.591            | 0.463              | 0.352            |

## Settings chosen in each fold

Candidate 0 is the protocol settings. The settings of every candidate are in `candidates.json`; the inner score of every candidate in every fold is in `inner_scores.csv`.

| test       | model          | candidate chosen in each fold                              | folds that kept the protocol settings   |
|:-----------|:---------------|:-----------------------------------------------------------|:----------------------------------------|
| main       | LR[all]        | Musanze: 14, Nyabihu: 53, Nyagatare: 20, Ruhango: 51       | 0 of 4                                  |
| main       | LR[field_sat]  | Musanze: 0, Nyabihu: 34, Nyagatare: 42, Ruhango: 49        | 1 of 4                                  |
| main       | LR[all_sat]    | Musanze: 53, Nyabihu: 53, Nyagatare: 24, Ruhango: 5        | 0 of 4                                  |
| main       | RF[all]        | Musanze: 13, Nyabihu: 3, Nyagatare: 22, Ruhango: 43        | 0 of 4                                  |
| main       | RF[field_sat]  | Musanze: 13, Nyabihu: 13, Nyagatare: 9, Ruhango: 9         | 0 of 4                                  |
| main       | RF[all_sat]    | Musanze: 2, Nyabihu: 6, Nyagatare: 9, Ruhango: 29          | 0 of 4                                  |
| main       | XGB[all]       | Musanze: 57, Nyabihu: 31, Nyagatare: 20, Ruhango: 5        | 0 of 4                                  |
| main       | XGB[field_sat] | Musanze: 6, Nyabihu: 31, Nyagatare: 31, Ruhango: 48        | 0 of 4                                  |
| main       | XGB[all_sat]   | Musanze: 6, Nyabihu: 26, Nyagatare: 22, Ruhango: 45        | 0 of 4                                  |
| new fields | LR[all]        | fold 1: 48, fold 2: 60, fold 3: 52, fold 4: 49, fold 5: 48 | 0 of 5                                  |
| new fields | LR[field_sat]  | fold 1: 60, fold 2: 7, fold 3: 7, fold 4: 54, fold 5: 1    | 0 of 5                                  |
| new fields | LR[all_sat]    | fold 1: 60, fold 2: 55, fold 3: 28, fold 4: 28, fold 5: 28 | 0 of 5                                  |
| new fields | RF[all]        | fold 1: 5, fold 2: 38, fold 3: 5, fold 4: 39, fold 5: 4    | 0 of 5                                  |
| new fields | RF[field_sat]  | fold 1: 39, fold 2: 22, fold 3: 22, fold 4: 45, fold 5: 30 | 0 of 5                                  |
| new fields | RF[all_sat]    | fold 1: 39, fold 2: 30, fold 3: 30, fold 4: 39, fold 5: 4  | 0 of 5                                  |
| new fields | XGB[all]       | fold 1: 34, fold 2: 51, fold 3: 54, fold 4: 60, fold 5: 51 | 0 of 5                                  |
| new fields | XGB[field_sat] | fold 1: 54, fold 2: 25, fold 3: 17, fold 4: 48, fold 5: 31 | 0 of 5                                  |
| new fields | XGB[all_sat]   | fold 1: 39, fold 2: 29, fold 3: 17, fold 4: 35, fold 5: 31 | 0 of 5                                  |

## When the model is sure (RF[all_sat] tuned, new fields)

Probability the model gives its first crop, and how often that first crop was the crop actually planted. Out-of-fold predictions of the new-fields test: no field was predicted by a model that saw its 2 km block.

| probability of the first crop   | fields   | share of fields   | first crop right   |
|:--------------------------------|:---------|:------------------|:-------------------|
| 0.2 to 0.3                      | 0        | 0.000             |                    |
| 0.3 to 0.4                      | 20       | 0.011             | 0.350              |
| 0.4 to 0.5                      | 127      | 0.067             | 0.559              |
| 0.5 to 0.6                      | 294      | 0.154             | 0.605              |
| 0.6 to 0.7                      | 397      | 0.209             | 0.673              |
| 0.7 to 0.8                      | 612      | 0.321             | 0.763              |
| 0.8 to 0.9                      | 277      | 0.145             | 0.827              |
| 0.9 to 1.0                      | 177      | 0.093             | 0.898              |

If the model answered only when its probability for the first crop is at least this high:

| probability at least   | fields answered   | share answered   | first crop right   |
|:-----------------------|:------------------|:-----------------|:-------------------|
| 0.300                  | 1904              | 1.000            | 0.724              |
| 0.400                  | 1884              | 0.989            | 0.728              |
| 0.500                  | 1757              | 0.923            | 0.740              |
| 0.600                  | 1463              | 0.768            | 0.767              |
| 0.700                  | 1066              | 0.560            | 0.802              |
| 0.800                  | 454               | 0.238            | 0.855              |
| 0.900                  | 177               | 0.093            | 0.898              |

Per district, answering only from 0.7:

| district   | fields answered   | share answered   | first crop right   |
|:-----------|:------------------|:-----------------|:-------------------|
| Musanze    | 97                | 0.591            | 0.845              |
| Nyabihu    | 102               | 0.607            | 0.814              |
| Nyagatare  | 711               | 0.656            | 0.814              |
| Ruhango    | 156               | 0.320            | 0.712              |

## Simple alternatives on the same folds (new fields)

B0 gives every field the most common crop of all training fields. The district's most common crop reads the district name, which no model may use as an input. B1 uses published crop ranges only.

|                                                   | main score   | first crop right, all fields   | first crop right, mean of the districts   |
|:--------------------------------------------------|:-------------|:-------------------------------|:------------------------------------------|
| B0: the most common crop of all training fields   | 0.135        | 0.530                          | 0.355                                     |
| the most common crop of the field's district      | 0.223        | 0.670                          | 0.624                                     |
| B1: expert rules                                  | 0.378        | 0.388                          | 0.523                                     |
| XGB[all], protocol settings (map before this run) | 0.451        | 0.656                          | 0.659                                     |
| RF[all_sat] tuned (map after this run)            | 0.514        | 0.724                          | 0.721                                     |

## How mixed the land is

If every field got the most common crop of its own 2 km block (which no model can know in advance):

| district   | fields   | blocks   | blocks with more than one crop   | fields in those blocks   | block's most common crop is right   | main score of the block's most common crop   |
|:-----------|:---------|:---------|:---------------------------------|:-------------------------|:------------------------------------|:---------------------------------------------|
| all        | 1904     | 368      | 0.486                            | 0.698                    | 0.795                               | 0.681                                        |
| Musanze    | 164      | 48       | 0.375                            | 0.530                    | 0.823                               | 0.753                                        |
| Nyabihu    | 168      | 45       | 0.311                            | 0.476                    | 0.887                               | 0.848                                        |
| Nyagatare  | 1084     | 195      | 0.503                            | 0.701                    | 0.808                               | 0.573                                        |
| Ruhango    | 488      | 80       | 0.613                            | 0.824                    | 0.725                               | 0.550                                        |

## Input importance (tuned models)

How much the main score drops when the values of one input, or of a group of inputs that carry the same information, are shuffled among the test fields (mean of the shuffles; 0 means the model does not use it). In the main test, an input that is the same for the whole district cannot change inside the district left out, so its drop there is zero by construction. Every input and its spread: `importance.csv`.

Groups of inputs:

| test       | model                | elevation and field temperature   | district climate   | NISR crop shares   | soil 0 to 20 cm   | satellite rainfall and vegetation   |
|:-----------|:---------------------|:----------------------------------|:-------------------|:-------------------|:------------------|:------------------------------------|
| main       | LR[all] tuned        | 0.071                             | 0.000              | 0.000              | -0.008            |                                     |
| main       | LR[field_sat] tuned  | 0.065                             |                    |                    | -0.023            | 0.001                               |
| main       | LR[all_sat] tuned    | 0.074                             | 0.000              | 0.000              | 0.018             | 0.003                               |
| main       | RF[all] tuned        | 0.063                             | 0.000              | 0.000              | 0.040             |                                     |
| main       | RF[field_sat] tuned  | 0.056                             |                    |                    | 0.047             | 0.007                               |
| main       | RF[all_sat] tuned    | 0.084                             | 0.000              | 0.000              | 0.019             | -0.003                              |
| main       | XGB[all] tuned       | 0.040                             | 0.000              | 0.000              | 0.016             |                                     |
| main       | XGB[field_sat] tuned | 0.068                             |                    |                    | 0.040             | -0.003                              |
| main       | XGB[all_sat] tuned   | 0.025                             | 0.000              | 0.000              | 0.005             | -0.001                              |
| new fields | LR[all] tuned        | 0.114                             | 0.012              | 0.033              | 0.038             |                                     |
| new fields | LR[field_sat] tuned  | 0.132                             |                    |                    | 0.051             | 0.076                               |
| new fields | LR[all_sat] tuned    | 0.096                             | 0.007              | 0.006              | 0.032             | 0.053                               |
| new fields | RF[all] tuned        | 0.227                             | 0.008              | -0.002             | 0.062             |                                     |
| new fields | RF[field_sat] tuned  | 0.198                             |                    |                    | 0.093             | 0.068                               |
| new fields | RF[all_sat] tuned    | 0.255                             | 0.000              | 0.000              | 0.069             | 0.062                               |
| new fields | XGB[all] tuned       | 0.163                             | -0.005             | -0.001             | 0.048             |                                     |
| new fields | XGB[field_sat] tuned | 0.166                             |                    |                    | 0.063             | 0.081                               |
| new fields | XGB[all_sat] tuned   | 0.142                             | -0.005             | -0.008             | 0.029             | 0.050                               |

Single inputs:

| test       | model                | five inputs that matter most                                                                                                                                   |
|:-----------|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| main       | LR[all] tuned        | tmin_point_c (0.043), nitrogen_0_20cm (0.016), potassium_extractable_0_20cm (0.010), ph_0_20cm (0.009), cation_exchange_capacity_0_20cm (0.001)                |
| main       | LR[field_sat] tuned  | srtm_elevation_m (0.045), silt_0_20cm (0.023), cation_exchange_capacity_0_20cm (0.010), ndvi_annual (0.007), elevation_m (0.004)                               |
| main       | LR[all_sat] tuned    | tmin_point_c (0.037), cation_exchange_capacity_0_20cm (0.014), silt_0_20cm (0.013), tmean_point_c (0.012), chirps_precip_mm (0.009)                            |
| main       | RF[all] tuned        | tmean_point_c (0.035), elevation_m (0.010), tmin_point_c (0.010), carbon_organic_0_20cm (0.008), tmax_c (0.000)                                                |
| main       | RF[field_sat] tuned  | srtm_elevation_m (0.058), elevation_m (0.055), carbon_organic_0_20cm (0.031), nitrogen_0_20cm (0.015), swir_ratio (0.009)                                      |
| main       | RF[all_sat] tuned    | tmean_point_c (0.051), elevation_m (0.030), srtm_elevation_m (0.013), tmin_point_c (0.009), carbon_organic_0_20cm (0.004)                                      |
| main       | XGB[all] tuned       | cation_exchange_capacity_0_20cm (0.009), carbon_organic_0_20cm (0.008), elevation_m (0.004), phosphorous_extractable_0_20cm (0.004), ph_0_20cm (0.003)         |
| main       | XGB[field_sat] tuned | srtm_elevation_m (0.037), silt_0_20cm (0.013), cation_exchange_capacity_0_20cm (0.013), swir_ratio (0.008), phosphorous_extractable_0_20cm (0.007)             |
| main       | XGB[all_sat] tuned   | srtm_elevation_m (0.009), cation_exchange_capacity_0_20cm (0.007), tmin_point_c (0.005), phosphorous_extractable_0_20cm (0.005), carbon_organic_0_20cm (0.003) |
| new fields | LR[all] tuned        | tmin_point_c (0.044), carbon_organic_0_20cm (0.040), elevation_m (0.039), nisr_share_beans (0.022), silt_0_20cm (0.021)                                        |
| new fields | LR[field_sat] tuned  | chirps_precip_mm (0.060), carbon_organic_0_20cm (0.053), elevation_m (0.048), swir_ratio (0.024), srtm_elevation_m (0.012)                                     |
| new fields | LR[all_sat] tuned    | carbon_organic_0_20cm (0.047), chirps_precip_mm (0.042), swir_ratio (0.025), cation_exchange_capacity_0_20cm (0.021), tmin_point_c (0.017)                     |
| new fields | RF[all] tuned        | elevation_m (0.041), silt_0_20cm (0.029), tmin_point_c (0.020), tmean_point_c (0.016), nitrogen_0_20cm (0.015)                                                 |
| new fields | RF[field_sat] tuned  | chirps_precip_mm (0.048), srtm_elevation_m (0.028), elevation_m (0.019), nitrogen_0_20cm (0.014), swir_ratio (0.012)                                           |
| new fields | RF[all_sat] tuned    | srtm_elevation_m (0.058), elevation_m (0.042), chirps_precip_mm (0.041), silt_0_20cm (0.019), tmean_point_c (0.016)                                            |
| new fields | XGB[all] tuned       | silt_0_20cm (0.033), sand_0_20cm (0.028), carbon_organic_0_20cm (0.024), tmin_point_c (0.022), elevation_m (0.022)                                             |
| new fields | XGB[field_sat] tuned | chirps_precip_mm (0.065), srtm_elevation_m (0.049), elevation_m (0.046), swir_ratio (0.042), silt_0_20cm (0.039)                                               |
| new fields | XGB[all_sat] tuned   | tmin_point_c (0.032), chirps_precip_mm (0.031), swir_ratio (0.024), nitrogen_0_20cm (0.017), carbon_organic_0_20cm (0.011)                                     |

## How to read this

- Main score: macro-F1 per test district over the crops with at least 5 fields there, then the mean over the four districts (section 5). 'First crop right' is the share of fields whose first crop was the crop actually planted.
- 'Protocol settings' is the model of `eval_v2` (section 6 settings). 'Tuned' chose its settings again in every fold, inside the training fields of that fold.
- The adoption rule (section 7) and the map rule (section 11) were written before this run.
- Full numbers: the CSV files in this folder. Settings, versions and timing: `run_info.json`.
