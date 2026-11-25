# API contracts

**Status:** 1.2, 25 November 2026, by Thierry (Part B, ML and Geospatial). First proposed on 20 October 2026, updated on 28 October and 18 November 2026, final as 1.0 on 23 November 2026. Version 1.1 only adds optional fields, for the tuned model now on the map. Version 1.2 changes no field: the point route now lets the method that names the right crop first more often answer in each district (section 9). Both recommendation interfaces are implemented in `01-crop-recommendation/src/recommendation/service.py` and checked by the tests in section 2.6. The crop health diagnosis interface belongs to the CV sub-team and joins this document when the two dashboards are merged.

**Reviewers:** Arsene (Part A) for the inputs and the season decision. Anny and Gabin (CV sub-team) for section 1 when their service joins.

Each interface follows the template fields. The same contract exists as code in `03-shared/apis/recommend_schemas.py`, and the tests check that the examples and rules in this file hold. If you change one, change the other and raise the contract version.

The examples of `POST /v1/recommend` are placeholders that show the shape. The examples of `POST /v1/recommend/point` are real answers of the service. Neither is agronomic advice.

## Contents

| # | Interface | Method and route | Status |
| --- | --- | --- | --- |
| 1 | Crop and fertiliser recommendation (expert rules, district or point) | `POST /v1/recommend` | Implemented |
| 2 | One recommendation for a point, with a confidence level | `POST /v1/recommend/point` | Implemented, new in 1.0, model updated in 1.1, choice of method updated in 1.2 |
| 3 | Service metadata | `GET /v1/metadata` | Implemented |
| 4 | Service health | `GET /v1/health` | Implemented |
| 5 | Crop health diagnosis | `POST /v1/diagnose` | Owned by the CV sub-team, added when the two dashboards are merged |

## 1. Conventions shared by both services (to agree with the CV sub-team)

These are proposals. The goal is that the dashboard can call both services the same way.

| Topic | Proposal |
| --- | --- |
| Format | JSON, UTF-8, field names in `snake_case`. |
| Versioning | Major version in the path (`/v1/...`). Adding an optional field is a minor change. Renaming, removing or changing the meaning of a field is a major change and needs agreement from both sub-teams first. |
| Local ports | Recommendation service on 8001, diagnosis service on 8002. The Streamlit dashboard runs on its default port, 8501. |
| Cross-origin | Allow the dashboard origin in development. |
| Time | ISO 8601, UTC, for example `2026-10-21T08:00:00Z`. |
| Units | The unit is written in the field name or documented beside it. SI units. |
| Request id | Every response carries `request_id` in the body and in an `X-Request-ID` header. |
| Errors | One envelope for both services (section 2.4). |
| Uncertainty | Being unsure is not an error. The service answers `200` with an uncertain result, and the dashboard shows "uncertain, consult an extension officer". For recommendation this is `abstain: true`. For diagnosis the CV team defines the equivalent. |
| Levels, not probabilities | Recommendation answers carry a level (high, medium, low): `reliability` for each crop of `POST /v1/recommend`, and `confidence` for the one answer of `POST /v1/recommend/point` (section 3). Neither is a probability. Diagnosis uses a **confidence score** from the CV model. They are not the same thing and must not be shown as if they were. |
| Authentication | None in the prototype. |
| Speed target | 95 percent of requests under 3 seconds on a laptop. Measured on 23 Nov 2026 (test client, Linux laptop class machine): `POST /v1/recommend` about 6 ms; `POST /v1/recommend/point` about 1 s for the first call, which trains the model, then about 15 ms. |

## 2. Interface 1: Crop and fertiliser recommendation

| Field | Value |
| --- | --- |
| Interface name | Crop and fertiliser recommendation |
| Status | Implemented on the B1 expert rules, because no trained model passed the adoption rule for a district never seen (`eval_v1` and `eval_v2`, `docs/training_protocol.md`). |
| Responsible workstream | Crop recommendation (ML and Geospatial, Part B), with the shared dashboard |
| Purpose | An extension officer gives a location, a season and optional soil test values. The service returns ranked crops with reasons and a reliability level, and fertiliser guidance for one crop with the source of each rule. |
| Method and route | `POST /v1/recommend` |
| Contract version | 1.2. This route has not changed since 1.0, final for the practicum. |

### 2.1 Input fields

| Field | Type | Required | Unit and valid range | Notes |
| --- | --- | --- | --- | --- |
| `district` | string | one of `district` or `lat` and `lon` | name from `GET /v1/metadata` | Case-insensitive. Unknown names are rejected. |
| `lat` | number | with `lon` | decimal degrees, -2.9 to -1.0 | Approximate bounding box of Rwanda. |
| `lon` | number | with `lat` | decimal degrees, 28.8 to 31.0 | If `lat` and `lon` are given they win over `district`. |
| `elevation_m` | number | no | metres, 900 to 4,600 | Added in 0.2.0. Field elevation, for example from a phone GPS. If left out, the median elevation of the CIAT points in the district is used, with a warning. |
| `season` | string | yes | `A`, `B` or `C` | Rwandan agricultural seasons. Only season B was tested, because the only labelled data (CIAT, 2025) is season B. Seasons A and C get the warning `SEASON_NOT_EVALUATED` and reliability `low`. No other season is planned in this project (decided with Part A on 23 Nov 2026). |
| `year` | integer | no | 2000 to 2100 | Reserved. Not used by the first version. |
| `crop` | string | no | lower case code from `GET /v1/metadata` | The crop to give fertiliser advice for. If left out, advice is for the rank 1 crop. |
| `soil.ph_water` | number | no | pH, 3.0 to 10.0 | Range follows the Prof's soil variable dictionary. |
| `soil.organic_carbon_pct` | number | no | percent, 0 to 20 | Same dictionary. |
| `soil.total_nitrogen_pct` | number | no | percent, 0 to 5 | Proposed range. |
| `soil.phosphorus_mgkg` | number | no | mg per kg, 0 to 200 | Available phosphorus. Same dictionary. |
| `soil.potassium_cmolkg` | number | no | cmol(+) per kg, 0 to 10 | Exchangeable potassium. Proposed range. Numerically equal to meq per 100 g. |
| `soil.texture` | string | no | USDA class: sand, loamy_sand, sandy_loam, loam, silt_loam, silt, sandy_clay_loam, clay_loam, silty_clay_loam, sandy_clay, silty_clay, clay | |
| `top_k` | integer | no, default 3 | 1 to 5 | Number of crops returned. |
| `include` | list of strings | no, default both | `crops`, `fertiliser` | Ask for only one part if that is all the dashboard needs. |

Unknown fields are rejected, so a misspelt field is an error and is never ignored silently.

This route uses only the soil values given in the request: it does not look soil up. The crop rules (B1) do not use soil yet, so soil values only reach the fertiliser rules, and the response says so (`SOIL_NOT_USED_BY_METHOD`). `inputs_used.soil` lists the values used (source `user`) and the key values that are missing. For soil looked up at the point, use `POST /v1/recommend/point` (section 3).

Example request (district, with partial soil values):

```json
{
  "district": "Musanze",
  "season": "B",
  "soil": {
    "ph_water": 5.1,
    "organic_carbon_pct": 2.0
  },
  "top_k": 3
}
```

Example request (point, fertiliser advice only):

```json
{
  "lat": -1.5,
  "lon": 29.6,
  "season": "A",
  "crop": "maize",
  "include": [
    "fertiliser"
  ]
}
```

### 2.2 Output fields

| Field | Type | Meaning |
| --- | --- | --- |
| `request_id` | string | Id to quote when reporting a problem. |
| `api_version` | string | Contract version the service follows. |
| `model_version` | string or null | Trained model in use. `null` while only the expert rules run. |
| `rules_version` | string | Version of the rule set (crop rules and fertiliser rules). |
| `generated_at` | datetime | UTC time of the answer. |
| `location` | object | District, province, coordinates and `resolution` (`point` or `district`) as understood by the service. |
| `season` | string | Echo of the request. |
| `abstain` | boolean | `true` when the service cannot give a trustworthy answer. |
| `abstain_message` | string or null | Required when `abstain` is true. Text such as "Uncertain. Please consult an extension officer." |
| `crop_recommendations` | list | Ranked crops, best first. Empty if `abstain` is true or crops were not requested. |
| `crop_recommendations[].rank` | integer | 1, 2, 3 and so on, in order. |
| `crop_recommendations[].crop` | string | Crop code. |
| `crop_recommendations[].score` | number 0 to 1 | Suitability score used for ranking. It is **not** a probability unless the models are calibrated and we say so. |
| `crop_recommendations[].method` | string | `expert_rules`, `random_forest` or `xgboost`. |
| `crop_recommendations[].reliability` | object | `level` (`high`, `medium`, `low`) and `reason_codes` (section 6). |
| `crop_recommendations[].reasons` | list, at least one | Each has `factor`, `direction` (`supports`, `against`, `neutral`), `value`, `expected_range` and a `text` sentence. |
| `fertiliser` | object or null | Fertiliser guidance for one crop. |
| `fertiliser.crop` | string | The crop the advice is for. |
| `fertiliser.status` | string | `ok`, or `no_matching_rule` when no guideline rule fits. |
| `fertiliser.engine_version` | string | Version of the fertiliser rule set. |
| `fertiliser.items[]` | list | One entry per recommended input. Each has `input_type` (lime, nitrogen, phosphorus, potassium, manure, compound_fertiliser), optional `product`, `rate` and `rate_unit` (`kg/ha` or `t/ha`), optional `timing`, `condition_matched`, and a required `source`. |
| `fertiliser.items[].source` | object | `document`, and where possible `section`, `page` and `url`. **Every item must have one.** An item without a source is a contract violation. |
| `fertiliser.items[].explanation` | string or null | Plain language explanation. |
| `fertiliser.items[].explanation_method` | string | `rule_text` (written with the rule) or `rag` (retrieved guideline passages, see below). |
| `inputs_used.soil.values` | object | For each soil variable used: `value` and `source`. In 1.0 the source is always `user`. |
| `inputs_used.soil.missing` | list | Key soil inputs that were not available (pH, organic carbon, total nitrogen). |
| `inputs_used.site` | object | Added in 0.2.0. The elevation and climate values the method used, each with `value` and `source` (for example `user`, `district_median_ciat`, `nasa_power_normal_2017_2024`). |
| `warnings` | list | `code` and `message` pairs (section 6). |
| `disclaimer` | string | Fixed sentence: decision support, not a replacement for local advice. |

**How RAG fits.** The fertiliser rule engine decides the recommendation: input type, rate and timing. The RAG component only writes or improves the `explanation` text, using passages retrieved from the guideline documents. It cannot change an item, and it cannot add one. The `source` of each item always comes from the rule, even when the explanation is RAG text. If RAG finds nothing to cite, `explanation` is `null` and the item still ships with its rule source. **Status in 1.0:** RAG is not built, because its source, the RAB guideline, is not in yet. `explanation_method` is always `rule_text`, and the field stays so that RAG can be added without changing the contract.

Example response (placeholders only):

```json
{
  "request_id": "example-0001",
  "api_version": "1.0",
  "model_version": null,
  "rules_version": "rules-0.1-example",
  "generated_at": "2026-10-21T08:00:00Z",
  "location": {
    "district": "Musanze",
    "province": "North",
    "lat": null,
    "lon": null,
    "resolution": "district"
  },
  "season": "B",
  "abstain": false,
  "abstain_message": null,
  "crop_recommendations": [
    {
      "rank": 1,
      "crop": "maize",
      "score": 0.7,
      "method": "expert_rules",
      "reliability": {
        "level": "medium",
        "reason_codes": [
          "ML_NOT_ADOPTED",
          "INPUT_MISSING_TOTAL_NITROGEN"
        ]
      },
      "reasons": [
        {
          "factor": "soil_ph",
          "direction": "supports",
          "value": 5.1,
          "expected_range": "<range from the cited agronomy source>",
          "text": "ILLUSTRATIVE TEXT. Soil pH is inside the range used for this crop."
        },
        {
          "factor": "elevation_m",
          "direction": "neutral",
          "value": null,
          "expected_range": null,
          "text": "ILLUSTRATIVE TEXT. Elevation was not used in this example."
        }
      ]
    },
    {
      "rank": 2,
      "crop": "beans",
      "score": 0.6,
      "method": "expert_rules",
      "reliability": {
        "level": "medium",
        "reason_codes": [
          "ML_NOT_ADOPTED",
          "INPUT_MISSING_TOTAL_NITROGEN"
        ]
      },
      "reasons": [
        {
          "factor": "soil_ph",
          "direction": "supports",
          "value": 5.1,
          "expected_range": null,
          "text": "ILLUSTRATIVE TEXT."
        }
      ]
    }
  ],
  "fertiliser": {
    "crop": "maize",
    "status": "ok",
    "engine_version": "rules-0.1-example",
    "items": [
      {
        "input_type": "lime",
        "product": null,
        "rate": null,
        "rate_unit": null,
        "timing": null,
        "condition_matched": "ILLUSTRATIVE. Soil pH is below the threshold given in the cited guideline.",
        "source": {
          "document": "<guideline title, filled from the real source>",
          "section": "<section>",
          "page": null,
          "url": null
        },
        "explanation": "ILLUSTRATIVE TEXT ONLY. Rate and timing come from the cited guideline, not from this example.",
        "explanation_method": "rule_text"
      }
    ]
  },
  "inputs_used": {
    "soil": {
      "values": {
        "ph_water": {
          "value": 5.1,
          "source": "user"
        },
        "organic_carbon_pct": {
          "value": 2.0,
          "source": "user"
        }
      },
      "missing": [
        "total_nitrogen_pct"
      ]
    }
  },
  "warnings": [
    {
      "code": "SOIL_VALUE_MISSING",
      "message": "Total nitrogen was not given, so reliability is capped at medium."
    }
  ],
  "disclaimer": "Decision support for extension officers and agronomists. It does not replace local field advice."
}
```

Example response when the service abstains:

```json
{
  "request_id": "example-0002",
  "api_version": "1.0",
  "model_version": null,
  "rules_version": "rules-0.1-example",
  "generated_at": "2026-10-21T08:05:00Z",
  "location": {
    "district": null,
    "province": null,
    "lat": -1.5,
    "lon": 29.6,
    "resolution": "point"
  },
  "season": "A",
  "abstain": true,
  "abstain_message": "Uncertain. Please consult an extension officer.",
  "crop_recommendations": [],
  "fertiliser": null,
  "inputs_used": {
    "soil": {
      "values": {},
      "missing": [
        "ph_water",
        "organic_carbon_pct",
        "total_nitrogen_pct"
      ]
    }
  },
  "warnings": [
    {
      "code": "NO_CROP_ABOVE_MIN_SCORE",
      "message": "No crop reached the minimum score."
    }
  ],
  "disclaimer": "Decision support for extension officers and agronomists. It does not replace local field advice."
}
```

### 2.3 Validation

| Case | Behaviour |
| --- | --- |
| Body is not valid JSON | `400`, code `INVALID_JSON`. |
| Missing required field, wrong type, unknown field | `422`, code `VALIDATION_ERROR`, with the field named in `details`. |
| Neither `district` nor `lat` and `lon`, or only one of `lat` and `lon` | `422`, `VALIDATION_ERROR`. |
| `lat` or `lon` outside the Rwanda bounding box | `422`, `OUT_OF_BOUNDS`. |
| Soil value outside its range | `422`, `VALIDATION_ERROR`, naming the field and the allowed range. |
| District not in the metadata list | `422`, `UNKNOWN_DISTRICT`. |
| Crop not in the metadata list | `422`, `UNSUPPORTED_CROP`. |
| Valid request but district is outside the training data | `200`, expert rules only, reliability `low`, warning `DISTRICT_NOT_IN_TRAINING_DATA`. |
| Valid request but no crop reaches the minimum score | `200`, `abstain: true`. Draft minimum score: 0.5 (set in the service, to be reviewed after ML). |
| Point more than 25 km from every pilot district centre | `200`, reliability `low`, warning `DISTRICT_NOT_IN_TRAINING_DATA`. No climate is invented for it. |
| No fertiliser rule fits | `200`, `fertiliser.status: "no_matching_rule"`, no items. The service never invents a rule. |
| Only interim fertiliser rates exist for the crop (until the RAB guideline is in) | `200`, `fertiliser.status: "ok"`. Each item has `source.page` and `source.url`, its `explanation` starts with "Interim:", and the response has the warning `FERTILISER_INTERIM`. |
| Model files not loaded | `503`, `MODEL_UNAVAILABLE`. Expert rules should still answer, so this is only for a full failure. |

### 2.4 Error responses

All errors use the same envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request has invalid fields.",
    "details": [
      {
        "field": "soil.ph_water",
        "issue": "must be between 3.0 and 10.0"
      }
    ],
    "request_id": "example-0003"
  }
}
```

| HTTP status | `error.code` | When |
| --- | --- | --- |
| 400 | `INVALID_JSON` | Body cannot be parsed. |
| 422 | `VALIDATION_ERROR` | Field rules above. |
| 422 | `OUT_OF_BOUNDS` | Coordinates outside Rwanda. |
| 422 | `UNKNOWN_DISTRICT` | District not supported. |
| 422 | `UNSUPPORTED_CROP` | Crop not supported. |
| 503 | `MODEL_UNAVAILABLE` | Nothing can answer. |
| 500 | `INTERNAL_ERROR` | Unexpected failure. The message must not contain stack traces. |

FastAPI answers validation errors in its own format by default. The service replaces it with this envelope. `03-shared/apis/stub_app.py` shows how.

### 2.5 Data and privacy

- Farm-level coordinates can identify a household. Request logs keep coordinates rounded to two decimals (about 1 km) and keep no names or contact details. This is proposed, and the team should confirm it.
- Soil values are not personal data.
- The prototype stores no requests after answering.
- Do not commit real farmer data or raw datasets to the repository. See `CONTRIBUTING.md`.

### 2.6 Test coverage

- Contract tests: `03-shared/tests/test_recommend_contract.py` (38 checks: examples validate, range and location rules, abstain rules, rank order, every fertiliser item needs a source, error envelope, stub server behaviour, and the point interface of section 3).
- Service tests: `01-crop-recommendation/tests/test_recommend_service.py` (36 checks on the real service, including that `POST /v1/recommend/point` gives the same answer as the dashboard map).
- Run from the repository root: `python -m pytest 01-crop-recommendation/tests 03-shared/tests -q` (249 tests pass on 23 Nov 2026 with the pinned library versions).
- Needs: `pip install -r 03-shared/dashboard/requirements.txt pytest`. The point route needs scikit-learn only; XGBoost is needed by the tests of the training code.
- Integration tests with the diagnosis service: written when the two dashboards are merged.

## 3. Interface 2: One recommendation for a point

| Field | Value |
| --- | --- |
| Interface name | One recommendation for a point |
| Status | Implemented, new in 1.0. In 1.1 the trained model is the tuned Random Forest. In 1.2 the method that names the right crop first more often answers, which is the tuned Random Forest in all four districts. |
| Responsible workstream | Crop recommendation (ML and Geospatial, Part B), with the shared dashboard |
| Purpose | A user clicks or types a point in one of the four pilot districts. The service returns one crop, a confidence level with plain reasons, the fertiliser for that crop, and the inputs it used. It is the same answer as the dashboard map, as the client asked on 19 Nov 2026: one recommendation and how sure we are, not a comparison of methods. |
| Method and route | `POST /v1/recommend/point` |
| Contract version | 1.2 |

### 3.1 How the answer is made

1. **District.** From the district outline the point falls in. A point within 1.5 km of a CIAT block just across a simplified outline counts as that block's district. Outside the four pilot districts the service abstains.
2. **Inputs.** Soil (iSDAsoil, 0 to 20 and 20 to 50 cm), elevation and four satellite values (SRTM elevation, CHIRPS rainfall from 1 February to 30 June 2025, and the Sentinel-2 vegetation index and shortwave ratio of 2024) come from the CIAT fields of the same 2 km block, or of the nearest block in the district. An `elevation_m` given in the request replaces both elevations. Season B 2025 weather (NASA POWER) and NISR crop shares come from the district, the values the model learned from.
3. **Two methods rank the crops.** The trained model on the map and the expert rules (B1). Since 1.1 the trained model is the tuned Random Forest on all inputs plus satellite (`RF[all_sat] tuned`), which the rule written before the tuning run chose for the map (`docs/training_protocol.md`, section 11, decision 2). It is trained once on the 1,904 CIAT fields with the settings in `01-crop-recommendation/reports/tuning_v1/final_model.json`. In 1.0 it was XGBoost on all inputs with the protocol settings.
4. **One method answers.** In each district, the method whose first crop was right for more of the new fields in testing (`tuning_v1`), if it is ahead by more than 0.01; otherwise the rules, the default. The answer is one crop, so the choice uses the same measure as the confidence level. Today that means Random Forest in all four districts: its first crop was right for 75% of the new fields in Musanze, 74% in Nyabihu, 76% in Nyagatare and 63% in Ruhango, against 62%, 66%, 22% and 59% for the rules. Until 1.1 the choice used the main score, so the rules answered in Nyabihu (0.645 against 0.627), where they often rate maize, Irish potato and rice the same.
5. **Confidence level.** It starts from the method's tested hit rate in the district: the share of new fields where its first crop was the right one.

| Level | When |
| --- | --- |
| `high` | Tested hit rate of at least 0.75, season B, and no reason to lower it. |
| `medium` | Tested hit rate from 0.50 to 0.75, or at least 0.75 with one reason to lower it. |
| `low` | Tested hit rate below 0.50; or from 0.50 to 0.75 with one reason to lower it; or two reasons to lower it; or a season other than B, which was never tested. |

Each reason lowers the level by one step: a close call (the first two crops are less than 0.10 apart in the method's own score, a tie included), and data far from the point (the nearest CIAT data is more than 5 km away). The thresholds are fixed, not tuned on results. Today's tested hit rates (Random Forest in all four districts) are 0.75 in Musanze, 0.74 in Nyabihu, 0.76 in Nyagatare and 0.63 in Ruhango. So a season B answer is `high` in Musanze and Nyagatare and `medium` in Nyabihu and Ruhango, one step lower after a close call or with far data. `lowered_by` says in plain words why a level is lower than the hit rate alone would give.

6. **Fertiliser** for the recommended crop, from the same rule engine as section 2 (interim national rates until the RAB guideline is in).

### 3.2 Input fields

| Field | Type | Required | Unit and valid range | Notes |
| --- | --- | --- | --- | --- |
| `lat` | number | yes | decimal degrees, -2.9 to -1.0 | Approximate bounding box of Rwanda. |
| `lon` | number | yes | decimal degrees, 28.8 to 31.0 | |
| `season` | string | yes | `A`, `B` or `C` | The answer is always for season B, the only season tested. A and C get `low` confidence and the warning `SEASON_NOT_EVALUATED`. |
| `elevation_m` | number | no | metres, 900 to 4,600 | Field elevation, for example from a phone GPS. If left out, the median GPS elevation of the CIAT fields in the block. |
| `include_evaluation` | boolean | no, default false | | Also return both methods' rankings and tested scores, for evaluators. Users only need the recommendation. |

Unknown fields are rejected (a `district` field, for example).

Example request:

```json
{
  "lat": -1.35,
  "lon": 30.3,
  "season": "B"
}
```

### 3.3 Output fields

| Field | Type | Meaning |
| --- | --- | --- |
| `request_id`, `api_version`, `rules_version`, `generated_at`, `disclaimer` | | As in section 2.2. |
| `model_version` | string | The trained model on the map, `RF[all_sat] tuned` since 1.1. |
| `location` | object | District, province, the requested point, and `resolution: point`. |
| `season.asked`, `season.used` | strings | The season asked, and the season the answer is for (always `B`). |
| `abstain` | boolean | `true` outside the four pilot districts. Then `recommendation`, `fertiliser` and `inputs_used` are `null` and `abstain_message` says why. |
| `recommendation.crop` | string | The one recommended crop. |
| `recommendation.method` | string | `random_forest` or `expert_rules`: the method that answers in this district (section 3.1). `xgboost` stays a valid value; it was the model on the map in 1.0. |
| `recommendation.confidence.level` | string | `high`, `medium` or `low` (section 3.1). Not a probability. |
| `recommendation.confidence.tested_hit_rate` | number 0 to 1 | Share of new fields in the district where the method's first crop was right in testing. |
| `recommendation.confidence.lowered_by` | list of strings | Why the level is lower than the hit rate alone would give. Empty when nothing lowers it. |
| `recommendation.close_second` | string or null | A crop that fits almost as well, if any. |
| `recommendation.alternatives` | list | The method's next crops, in order. |
| `recommendation.why` | list, at least one | Plain sentences: why this method answers here, its hit rate, and for the rules, the checks behind the crop with their sources. |
| `fertiliser` | object or null | Same object as section 2.2, always for `recommendation.crop`. |
| `inputs_used` | object | `elevation_m` and `elevation_source`, `soil_texture`, `soil_0_20cm`, `field_data_from` (`n_ciat_fields` in the block and `distance_km` to it), `climate_season_b_2025`, `nisr_crop_shares`, and since 1.1 `satellite_at_field` (`srtm_elevation_m` in metres, `chirps_precip_mm` in mm from 1 February to 30 June 2025, `ndvi_annual` and `swir_ratio`, unitless indices of 2024). Units of `soil_0_20cm`: `ph` in water; `nitrogen` and `carbon_organic` in g/kg; `phosphorous_extractable` and `potassium_extractable` in mg/kg (ppm, as iSDAsoil gives them); `cation_exchange_capacity` in cmol(+)/kg; `sand`, `silt` and `clay` in percent. iSDAsoil predicts the three fractions separately, so they need not add up to 100 (the texture class scales them first). It never contains farm or block coordinates. |
| `evaluation` | object or null | Only with `include_evaluation: true`: `model` (the trained model, `random_forest`), `model_ranking` and `rules_ranking` (crop and score), and `tested_main_score_new_fields` for both methods in the district, keyed by method. `xgboost_ranking` is kept for 1.0 clients and is `null` unless the model on the map is XGBoost. |
| `warnings` | list | `code` and `message` pairs (section 6). |

Rules checked by the schema: `abstain: true` needs `abstain_message` and no recommendation; `abstain: false` needs a recommendation; the fertiliser advice is for the recommended crop.

Example response (a real answer of the service for a point in Nyagatare, with `request_id` and `generated_at` fixed):

```json
{
  "request_id": "example-0004",
  "api_version": "1.2",
  "model_version": "RF[all_sat] tuned, trained on the 1,904 CIAT fields",
  "rules_version": "crops-0.1-draft+fertiliser-0.1-interim-2026-11-18",
  "generated_at": "2026-11-25T09:30:00Z",
  "location": {
    "district": "Nyagatare",
    "province": "East",
    "lat": -1.35,
    "lon": 30.3,
    "resolution": "point"
  },
  "season": {
    "asked": "B",
    "used": "B"
  },
  "abstain": false,
  "abstain_message": null,
  "recommendation": {
    "crop": "maize",
    "method": "random_forest",
    "confidence": {
      "level": "high",
      "tested_hit_rate": 0.759,
      "lowered_by": []
    },
    "close_second": null,
    "alternatives": [
      "rice",
      "irish_potato",
      "beans"
    ],
    "why": [
      "In Nyagatare, Random Forest named the right crop first for 76% of the new fields in testing, against 22% for the expert rules, so the answer comes from Random Forest."
    ]
  },
  "fertiliser": {
    "crop": "maize",
    "status": "ok",
    "engine_version": "0.2.0+rules-0.1-interim-2026-11-18",
    "items": [
      {
        "input_type": "compound_fertiliser",
        "product": "DAP (18-46-0)",
        "rate": 100.0,
        "rate_unit": "kg/ha",
        "timing": "At planting",
        "condition_matched": "maize-national-interim-01: applies to this crop in general",
        "source": {
          "document": "B. T. Mudereri et al., Smart Fertiliser Recommendations for Maize in Rwanda, CGIAR Sustainable Farming Science Program Report, Nov. 2025",
          "section": "Table 2, treatment BR (blanket recommendation)",
          "page": "7",
          "url": "https://hdl.handle.net/10568/178714"
        },
        "explanation": "Interim: published national rate, not yet confirmed by RAB. National blanket recommendation as used in the RAB and CGIAR maize trials: 100 kg/ha of DAP at planting.",
        "explanation_method": "rule_text"
      },
      {
        "input_type": "nitrogen",
        "product": "Urea (46% N)",
        "rate": 100.0,
        "rate_unit": "kg/ha",
        "timing": "Top dressing at 6 to 8 leaves, about 4 to 6 weeks after emergence",
        "condition_matched": "maize-national-interim-01: applies to this crop in general",
        "source": {
          "document": "B. T. Mudereri et al., Smart Fertiliser Recommendations for Maize in Rwanda, CGIAR Sustainable Farming Science Program Report, Nov. 2025",
          "section": "Table 2, treatment BR (blanket recommendation)",
          "page": "7",
          "url": "https://hdl.handle.net/10568/178714"
        },
        "explanation": "Interim: published national rate, not yet confirmed by RAB. National blanket recommendation as used in the RAB and CGIAR maize trials: 100 kg/ha of urea as top dressing. An older report lists 50 kg/ha of urea for the government recommendation (IFDC, 2018, Table 2), so this rate needs RAB's confirmation.",
        "explanation_method": "rule_text"
      }
    ]
  },
  "inputs_used": {
    "elevation_m": 1350.2,
    "elevation_source": "median GPS elevation of the CIAT fields in the block",
    "soil_texture": "sandy clay loam",
    "soil_0_20cm": {
      "nitrogen": 1.37,
      "phosphorous_extractable": 18.09,
      "potassium_extractable": 171.07,
      "ph": 5.84,
      "sand": 49.78,
      "clay": 32.78,
      "silt": 20.44,
      "carbon_organic": 14.57,
      "cation_exchange_capacity": 10.94
    },
    "field_data_from": {
      "n_ciat_fields": 9,
      "distance_km": 0.0
    },
    "climate_season_b_2025": {
      "rainfall_mm": 441.6,
      "tmax_c": 27.67,
      "tmin_c": 16.42,
      "mean_temp_at_field_c": 22.42,
      "solar_mj_m2_day": 17.65
    },
    "nisr_crop_shares": {
      "Beans": 0.531,
      "Irish potato": 0.01,
      "Maize": 0.421,
      "Rice": 0.038
    },
    "satellite_at_field": {
      "srtm_elevation_m": 1362.0,
      "chirps_precip_mm": 399.8,
      "ndvi_annual": 0.465,
      "swir_ratio": 1.116
    }
  },
  "evaluation": null,
  "warnings": [
    {
      "code": "FERTILISER_INTERIM",
      "message": "Fertiliser rates are interim: national blanket rates from published sources, not yet confirmed by RAB, and the same for every field."
    }
  ],
  "disclaimer": "Decision support for extension officers and agronomists. It does not replace local field advice."
}
```

Example response outside the pilot districts (a point in Kigali):

```json
{
  "request_id": "example-0005",
  "api_version": "1.0",
  "model_version": "XGB[all], trained on the 1,904 CIAT fields",
  "rules_version": "crops-0.1-draft+fertiliser-0.1-interim-2026-11-18",
  "generated_at": "2026-11-23T08:01:00Z",
  "location": {
    "district": "Nyarugenge",
    "province": null,
    "lat": -1.95,
    "lon": 30.06,
    "resolution": "point"
  },
  "season": {
    "asked": "B",
    "used": "B"
  },
  "abstain": true,
  "abstain_message": "Nyarugenge is not one of the four pilot districts (Musanze, Nyabihu, Nyagatare, Ruhango). There is no CIAT data here, so no recommendation is given. Please consult an extension officer.",
  "recommendation": null,
  "fertiliser": null,
  "inputs_used": null,
  "evaluation": null,
  "warnings": [
    {
      "code": "OUTSIDE_PILOT_DISTRICTS",
      "message": "Nyarugenge is not one of the four pilot districts (Musanze, Nyabihu, Nyagatare, Ruhango). There is no CIAT data here, so no recommendation is given."
    }
  ],
  "disclaimer": "Decision support for extension officers and agronomists. It does not replace local field advice."
}
```

### 3.4 Validation and errors

| Case | Behaviour |
| --- | --- |
| Missing `lat`, `lon` or `season`, wrong type, unknown field | `422`, `VALIDATION_ERROR`. |
| `lat` or `lon` outside the Rwanda bounding box | `422`, `OUT_OF_BOUNDS`. |
| A point in Rwanda but outside the four pilot districts | `200`, `abstain: true`, warning `OUTSIDE_PILOT_DISTRICTS`. |
| A point inside the bounding box but outside Rwanda | `200`, `abstain: true`, warning `OUTSIDE_RWANDA`. |
| The settings or data files of the model on the map are missing | `503`, `MODEL_UNAVAILABLE`. `POST /v1/recommend` still answers with the expert rules. |

Errors use the envelope of section 2.4.

## 4. Interface 3: Service metadata

| Field | Value |
| --- | --- |
| Method and route | `GET /v1/metadata` |
| Status | Implemented |
| Purpose | Lets the dashboard fill its drop-down lists and lets people check which version is running. |
| Input | None. |
| Output | `api_version`, `model_version`, `rules_version`, `feature_schema_version`, `districts` (each with `name`, `province`, `in_training_data`), `crops` (codes), `seasons`. |
| Errors | `500 INTERNAL_ERROR` only. |

Crops: beans, irish_potato, maize and rice, the four crops of the CIAT data. This list is final for 1.0 (section 8).

## 5. Interface 4: Service health

| Field | Value |
| --- | --- |
| Method and route | `GET /v1/health` |
| Status | Implemented |
| Purpose | Quick check that the service is up and which parts are loaded. |
| Output | `status` (`ok`, `degraded`, `down`), `rules_loaded`, `model_loaded`, `api_version`. |
| Rule | `ok`: the rules answer, and the settings and data files of the model on the map are in place, so point requests can answer (the model trains on the first point request, and `model_loaded` becomes `true` then). `degraded`: only the rules answer, so `POST /v1/recommend` works and `POST /v1/recommend/point` returns `503`. `down`: nothing can answer. |

## 6. Reliability levels of POST /v1/recommend, and warning codes

The level is about how much the answer should be trusted, and it is shown to the extension officer. The confidence level of `POST /v1/recommend/point` has its own rules (section 3.1).

**Rules of `POST /v1/recommend`, on the B1 expert rules (since 0.2.1, unchanged in 1.0).** B1 is never `high`, because no trained model has been adopted for this route: in the official evaluations of 28 October (`eval_v1`) and 18 November (`eval_v2`, with the satellite inputs), no ML model passed the adoption rule for a district never seen (protocol in `docs/training_protocol.md`). The same evaluation shows where the rules themselves work. B1 scored 0.59 (Musanze) and 0.65 (Nyabihu) macro-F1, but 0.09 (Nyagatare) and 0.19 (Ruhango), where several crops fit the rules equally and the rules cannot tell which one farmers prefer. These scores are stored per district in `district_profiles.json` (`b1_tested`) and set the level below.

| Level | When |
| --- | --- |
| `medium` | The location is in a pilot district where B1 scored at least 0.5 macro-F1 when tested (now Musanze and Nyabihu), the season is B (the only season B1 was tested on), every rule input is available, and the crop's rules are not marked weak. |
| `low` | Any of: outside the pilot districts, a pilot district where B1 scored below 0.5 when tested (now Nyagatare and Ruhango), season A or C, a rule input is missing, or the crop's rules are marked weak (rice, because there is no water-access layer). |

The draft rules for an adopted ML model (0.2.0) are dropped in 1.0, because no model was adopted for this route. Where a trained model did better in testing, inside the four pilot districts, it answers through `POST /v1/recommend/point` (section 3).

Key soil inputs are pH, organic carbon and total nitrogen, the same three as in the feature schema.

Reason codes in `reliability.reason_codes`: `ML_NOT_ADOPTED`, `MODEL_BEAT_BASELINE`, `DISTRICT_NOT_IN_TRAINING`, `SEASON_NOT_EVALUATED`, `RULES_WEAK_IN_DISTRICT`, `CROP_RULES_WEAK`, `INPUT_MISSING_<VARIABLE>` (for example `INPUT_MISSING_TOTAL_NITROGEN` or `INPUT_MISSING_WATER_ACCESS`), `SOIL_FROM_ISDASOIL`.

Warning codes in `warnings[].code`: `DISTRICT_NOT_IN_TRAINING_DATA`, `LOCATION_MATCHED_TO_NEAREST_DISTRICT`, `DISTRICT_MISMATCH`, `ELEVATION_FROM_DISTRICT_MEDIAN`, `SEASON_NOT_EVALUATED`, `RULES_WEAK_IN_DISTRICT`, `CROPS_TIED`, `SOIL_VALUE_MISSING`, `SOIL_NOT_USED_BY_METHOD`, `SOIL_FROM_ISDASOIL`, `NO_CROP_ABOVE_MIN_SCORE`, `NO_GUIDELINE_RULE_FOUND`, `FERTILISER_INTERIM`, `FERTILISER_SKIPPED`, `INPUT_OUT_OF_TYPICAL_RANGE`, `STUB_RESPONSE` (only the stub server).

Added in 1.0 for `POST /v1/recommend/point`: `FIELD_DATA_FAR` (no CIAT field within 5 km; the inputs come from the nearest block), `SHARED_WEATHER_CELL` (Musanze and Nyabihu share one NASA POWER weather cell), `OUTSIDE_PILOT_DISTRICTS`, `OUTSIDE_RWANDA`. On that route, `CROPS_TIED` is only sent when the rules give the answer.

## 7. The service and the stub server

**The service (real answers).** `01-crop-recommendation/src/recommendation/service.py` implements the four interfaces of this file. `POST /v1/recommend`, metadata and health run on the B1 expert rules and the fertiliser engine. `POST /v1/recommend/point` runs `01-crop-recommendation/src/recommendation/map_model.py`, the same code as the dashboard map; it needs the files listed in that folder's README, and it trains the model on its first request (scikit-learn, a few seconds). It reads `district_profiles.json` (district elevation, NASA POWER season normals with season B from 1 February to 30 June, and B1's tested score per district, no farm-level data), so it needs no raw data at run time. Run it from the repository root with:

```
python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001
```

**Stub (placeholder answers).** `03-shared/apis/stub_app.py` is a small FastAPI app that validates requests with the real models and returns placeholder data marked `STUB_RESPONSE`, for both recommendation routes. It exists so the dashboard and the CV team can test without the models. Run it with:

```
cd 03-shared/apis
uvicorn stub_app:app --port 8001 --reload
```

Then open http://127.0.0.1:8001/docs for the interactive page.

## 8. Decisions and open questions

| # | Question | Status in 1.0 |
| --- | --- | --- |
| 1 | One crop list for the whole project. | **Decided:** beans, irish_potato, maize and rice, the four crops of the CIAT data. |
| 2 | Same error envelope, request id, ports and uncertainty rules for the diagnosis service? | **Open:** agreed with Anny and Gabin when the two dashboards are merged, after Part B is complete (team decision of 23 Nov 2026). |
| 3 | Should the dashboard send a district name only, or also let the user click a point on a map? | **Decided:** both. `POST /v1/recommend` takes a district or a point (expert rules); `POST /v1/recommend/point` gives the map's one answer. |
| 4 | Units the extension officers actually have on soil test reports (percent, mg per kg, cmol per kg). | **Open:** Prof. Okeyo, through the primary contact. The units of section 2.1 stay until then. |
| 5 | Minimum score for abstaining and the reliability thresholds. | **Decided:** abstain below 0.5; `medium` on `POST /v1/recommend` needs B1's tested macro-F1 of at least 0.5 in the district; the confidence thresholds of the point route are in section 3.1. |
| 6 | Does `district_default` soil exist? | **Decided:** no. `POST /v1/recommend` only uses soil values given by the user; the point route takes soil from the nearest CIAT block. |
| 7 | Is rounding coordinates in logs enough for privacy? | **Decided:** the prototype keeps no request logs (section 2.5), and the point route never returns farm or block coordinates. |
| 8 | Other seasons than B? | **Decided with Part A, 23 Nov 2026:** no. The only labelled data, CIAT 2025, is season B; collecting labels for A or C is out of scope. Seasons A and C are answered for season B with a warning and a low level. |

## 9. Change log

| Version | Date | Change |
| --- | --- | --- |
| 0.1.0 draft | 2026-10-20 | First draft of recommendation, metadata and health interfaces. |
| 0.2.0 draft | 2026-10-27 | Optional `elevation_m` in the request; `inputs_used.site` in the response; B1 reliability rules; new warning and reason codes; draft abstain threshold. All changes are additive, so 0.1.0 requests still work. |
| 0.2.1 draft | 2026-10-28 | Reliability now uses B1's tested score per district from `eval_v1` (new reason and warning code `RULES_WEAK_IN_DISTRICT`); new warning `CROPS_TIED` when crops fit the rules equally; season B climate normals now run from 1 February to 30 June, as in the training data. No field changes, so 0.2.0 clients still work. |
| 0.2.2 draft | 2026-11-18 | Interim fertiliser rates: until the RAB guideline is in, each crop has the national blanket rate copied from a published source (page and link in `source`), served with the new warning `FERTILISER_INTERIM` and an explanation that starts with "Interim:". No field changes, so 0.2.1 clients still work. |
| 1.0 final | 2026-11-23 | New `POST /v1/recommend/point`: one crop with a confidence level, the same answer as the dashboard map (client feedback of 19 Nov). New warning codes `FIELD_DATA_FAR`, `SHARED_WEATHER_CELL`, `OUTSIDE_PILOT_DISTRICTS` and `OUTSIDE_RWANDA`. Health is `ok` when the trained model can answer. Final crop list, season decision and closed questions (section 8). The text now says what `POST /v1/recommend` really does with soil (user values only). `POST /v1/recommend` itself is unchanged, so 0.2.x clients still work. |
| 1.1 | 2026-11-23 | The point route serves the tuned Random Forest on all inputs plus satellite, as decision 2 of the tuning protocol requires (`docs/training_protocol.md`, section 11). Only optional fields and values are added, so 1.0 clients still work: `random_forest` as a `recommendation.method`, `inputs_used.satellite_at_field`, and `evaluation.model` and `evaluation.model_ranking`; `evaluation.xgboost_ranking` is now `null`. Health no longer depends on XGBoost. Units of `inputs_used.soil_0_20cm` written down. `POST /v1/recommend` is unchanged. |
| 1.2 | 2026-11-25 | The point route chooses the method that answers in each district by first crop right on new fields in testing (ahead by more than 0.01, otherwise the rules), the measure the confidence level already uses, instead of the main score. In Nyabihu the tuned Random Forest now answers instead of the rules (first crop right 74% against 66%), so `recommendation.method` there is `random_forest` and the reasons in `why` say it in those words. No field changes, so 1.1 clients still work. The `low` row of the confidence table now also names one reason to lower a hit rate below 0.75. `POST /v1/recommend` is unchanged. |

## Review checklist

Confirm that field naming, units, error behavior, and confidence semantics match the workstream implementation before marking a contract as implemented. Do not invent endpoints or sample responses that appear production-ready.

- [ ] Anny and Gabin reviewed section 1 and the diagnosis side (when the two dashboards are merged)
- [ ] Arsene reviewed the inputs, soil units and the season decision
- [x] Contract and service tests pass (`python -m pytest 01-crop-recommendation/tests 03-shared/tests -q`, 233 tests on 23 Nov 2026)
- [x] Status changed to Final 1.0 for the implemented interfaces
