# API contracts

**Status:** DRAFT 0.2.0, proposed by Thierry (Part B, ML and Geospatial) on 20 October 2026 and updated on 27 October 2026. `POST /v1/recommend` now runs on the B1 expert-rules baseline (section 6). The examples below are still placeholders, not model output and not agronomic advice.

**Reviewers needed:** Anny and Gabin (CV sub-team) for section 1 and the dashboard side. Arsene for the feature schema link and reliability levels. Calvin or Prof. Okeyo only if they want to see the format.

This file replaces the empty template. Each interface follows the template fields. The same contract exists as code in `03-shared/apis/recommend_schemas.py`, and `03-shared/tests/test_recommend_contract.py` checks that the examples and rules in this file hold. If you change one, change the other and raise the contract version.

## Contents

| # | Interface | Method and route | Status |
| --- | --- | --- | --- |
| 1 | Crop and fertiliser recommendation | `POST /v1/recommend` | Proposed |
| 2 | Service metadata | `GET /v1/metadata` | Proposed |
| 3 | Service health | `GET /v1/health` | Proposed |
| 4 | Crop health diagnosis | `POST /v1/diagnose` | Owned by the CV sub-team, not defined here |

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
| Errors | One envelope for both services (section 6). |
| Uncertainty | Being unsure is not an error. The service answers `200` with an uncertain result, and the dashboard shows "uncertain, consult an extension officer". For recommendation this is `abstain: true`. For diagnosis the CV team defines the equivalent. |
| Two different words | Recommendation uses a **reliability level** (high, medium, low). Diagnosis uses a **confidence score**. They are not the same thing and must not be shown as if they were. |
| Authentication | None in the prototype. |
| Speed target | 95 percent of requests under 3 seconds on a laptop. Proposed, to be measured. |

## 2. Interface 1: Crop and fertiliser recommendation

| Field | Value |
| --- | --- |
| Interface name | Crop and fertiliser recommendation |
| Status | Proposed |
| Responsible workstream | Crop recommendation (ML and Geospatial, Part B), with the shared dashboard |
| Purpose | An extension officer gives a location, a season and optional soil test values. The service returns ranked crops with reasons and a reliability level, and fertiliser guidance for one crop with the source of each rule. |
| Method and route | `POST /v1/recommend` (proposed, not agreed) |
| Contract version | 0.2.0 draft. Reviewer and review date: to fill in. |

### 2.1 Input fields

| Field | Type | Required | Unit and valid range | Notes |
| --- | --- | --- | --- | --- |
| `district` | string | one of `district` or `lat` and `lon` | name from `GET /v1/metadata` | Case-insensitive. Unknown names are rejected. |
| `lat` | number | with `lon` | decimal degrees, -2.9 to -1.0 | Approximate bounding box of Rwanda. |
| `lon` | number | with `lat` | decimal degrees, 28.8 to 31.0 | If `lat` and `lon` are given they win over `district`. |
| `elevation_m` | number | no | metres, 900 to 4,600 | Added in 0.2.0. Field elevation, for example from a phone GPS. If left out, the median elevation of the CIAT points in the district is used, with a warning. |
| `season` | string | yes | `A`, `B` or `C` | Rwandan agricultural seasons. |
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

If soil values are not given and the request has `lat` and `lon`, the service looks them up from iSDAsoil. If only a district is given, it uses a district default when one exists. Either way the response says where each soil value came from in `inputs_used`.

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
| `crop_recommendations[].reliability` | object | `level` (`high`, `medium`, `low`) and `reason_codes` (section 4). |
| `crop_recommendations[].reasons` | list, at least one | Each has `factor`, `direction` (`supports`, `against`, `neutral`), `value`, `expected_range` and a `text` sentence. |
| `fertiliser` | object or null | Fertiliser guidance for one crop. |
| `fertiliser.crop` | string | The crop the advice is for. |
| `fertiliser.status` | string | `ok`, or `no_matching_rule` when no guideline rule fits. |
| `fertiliser.engine_version` | string | Version of the fertiliser rule set. |
| `fertiliser.items[]` | list | One entry per recommended input. Each has `input_type` (lime, nitrogen, phosphorus, potassium, manure, compound_fertiliser), optional `product`, `rate` and `rate_unit` (`kg/ha` or `t/ha`), optional `timing`, `condition_matched`, and a required `source`. |
| `fertiliser.items[].source` | object | `document`, and where possible `section`, `page` and `url`. **Every item must have one.** An item without a source is a contract violation. |
| `fertiliser.items[].explanation` | string or null | Plain language explanation. |
| `fertiliser.items[].explanation_method` | string | `rule_text` (written with the rule) or `rag` (retrieved guideline passages, see below). |
| `inputs_used.soil.values` | object | For each soil variable used: `value` and `source` (`user`, `isdasoil`, `district_default`). |
| `inputs_used.soil.missing` | list | Key soil inputs that were not available (pH, organic carbon, total nitrogen). |
| `inputs_used.site` | object | Added in 0.2.0. The elevation and climate values the method used, each with `value` and `source` (for example `user`, `district_median_ciat`, `nasa_power_normal_2017_2024`). |
| `warnings` | list | `code` and `message` pairs (section 5). |
| `disclaimer` | string | Fixed sentence: decision support, not a replacement for local advice. |

**How RAG fits.** The fertiliser rule engine decides the recommendation: input type, rate and timing. The RAG component only writes or improves the `explanation` text, using passages retrieved from the guideline documents. It cannot change an item, and it cannot add one. The `source` of each item always comes from the rule, even when the explanation is RAG text. If RAG finds nothing to cite, `explanation` is `null` and the item still ships with its rule source.

Example response (placeholders only):

```json
{
  "request_id": "example-0001",
  "api_version": "0.2.0",
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
  "api_version": "0.2.0",
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

- Contract tests: `03-shared/tests/test_recommend_contract.py` (22 checks: examples validate, range and location rules, abstain rules, rank order, every fertiliser item needs a source, error envelope, stub server behaviour).
- Run from the repository root: `python -m pytest 03-shared/tests -q`
- Needs: `pip install pydantic fastapi httpx pytest`
- Integration tests with the diagnosis service and dashboard: not written yet.

## 3. Interface 2: Service metadata

| Field | Value |
| --- | --- |
| Method and route | `GET /v1/metadata` |
| Status | Proposed |
| Purpose | Lets the dashboard fill its drop-down lists and lets people check which version is running. |
| Input | None. |
| Output | `api_version`, `model_version`, `rules_version`, `feature_schema_version`, `districts` (each with `name`, `province`, `in_training_data`), `crops` (codes), `seasons`. |
| Errors | `500 INTERNAL_ERROR` only. |

The crop list is not final. The candidates are beans, irish_potato, maize, rice and cassava. The team has to agree one list (see section 7).

## 4. Interface 3: Service health

| Field | Value |
| --- | --- |
| Method and route | `GET /v1/health` |
| Status | Proposed |
| Purpose | Quick check that the service is up and which parts are loaded. |
| Output | `status` (`ok`, `degraded`, `down`), `rules_loaded`, `model_loaded`, `api_version`. |
| Rule | `ok` needs rules and models loaded. `degraded` means the rules answer but models are not loaded. `down` means nothing can answer. |

## 5. Reliability levels (draft meaning)

The level is about how much the answer should be trusted, and it is shown to the extension officer. The rules below are drafts. They are reviewed after the ML models are evaluated and then recorded in `04-data-reporting`.

**Rules used now, with the B1 expert-rules baseline (0.2.0).** B1 is never `high`, because no trained model has been adopted (on 27 October, ML did not beat B1 on held-out districts).

| Level | When |
| --- | --- |
| `medium` | The location is in a pilot district, the season is B (the only season B1 was tested on), every rule input is available, and the crop's rules are not marked weak. |
| `low` | Any of: outside the pilot districts, season A or C, a rule input is missing, or the crop's rules are marked weak (rice, because there is no water-access layer). |

**Rules planned for the ML models.**

| Level | Draft meaning |
| --- | --- |
| `high` | The district is in the training data, the ML model beat the expert-rules baseline in evaluation and was adopted, and all key soil inputs are available. |
| `medium` | The district is in the training data and at most one key soil input is missing. |
| `low` | The district is outside the training data, or two or more key soil inputs are missing. |

Key soil inputs are pH, organic carbon and total nitrogen, the same three as in the feature schema.

Reason codes in `reliability.reason_codes`: `ML_NOT_ADOPTED`, `MODEL_BEAT_BASELINE`, `DISTRICT_NOT_IN_TRAINING`, `SEASON_NOT_EVALUATED`, `CROP_RULES_WEAK`, `INPUT_MISSING_<VARIABLE>` (for example `INPUT_MISSING_TOTAL_NITROGEN` or `INPUT_MISSING_WATER_ACCESS`), `SOIL_FROM_ISDASOIL`.

Warning codes in `warnings[].code`: `DISTRICT_NOT_IN_TRAINING_DATA`, `LOCATION_MATCHED_TO_NEAREST_DISTRICT`, `DISTRICT_MISMATCH`, `ELEVATION_FROM_DISTRICT_MEDIAN`, `SEASON_NOT_EVALUATED`, `SOIL_VALUE_MISSING`, `SOIL_NOT_USED_BY_METHOD`, `SOIL_FROM_ISDASOIL`, `NO_CROP_ABOVE_MIN_SCORE`, `NO_GUIDELINE_RULE_FOUND`, `FERTILISER_SKIPPED`, `INPUT_OUT_OF_TYPICAL_RANGE`, `STUB_RESPONSE` (only the stub server).

## 6. The B1 service and the stub server

**B1 service (real answers).** `01-crop-recommendation/src/recommendation/service.py` implements all three interfaces on the B1 expert-rules baseline and the fertiliser engine. It reads `district_profiles.json` (district elevation and NASA POWER season normals, no farm-level data), so it needs no raw data at run time. Run it from the repository root with:

```
python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001
```

**Stub (placeholder answers).** `03-shared/apis/stub_app.py` is a small FastAPI app that validates requests with the real models and returns placeholder data marked `STUB_RESPONSE`. It exists so the dashboard and the CV team can test without the rules. Run it with:

```
cd 03-shared/apis
uvicorn stub_app:app --port 8001 --reload
```

Then open http://127.0.0.1:8001/docs for the interactive page. The B1 service will be replaced by the trained models once they beat the baseline, without changing the contract.

## 7. Open questions

| # | Question | Who decides |
| --- | --- | --- |
| 1 | One crop list for the whole project (beans, irish_potato, maize, rice, cassava?). | Team, with Calvin and Prof. Okeyo |
| 2 | Same error envelope, request id, ports and uncertainty rules for the diagnosis service? | Anny and Gabin with Thierry |
| 3 | Should the dashboard send a district name only, or also let the user click a point on a map? | Whoever owns the dashboard (not yet named) |
| 4 | Units the extension officers actually have on soil test reports (percent, mg per kg, cmol per kg). | Prof. Okeyo, through the primary contact |
| 5 | Minimum score for abstaining (draft 0.5) and the final reliability thresholds. | Thierry and Arsene, after the ML evaluation |
| 6 | Does `district_default` soil exist? It needs a district level soil table from the data pipeline. | Arsene |
| 7 | Is rounding coordinates in logs enough for privacy? | Team |

## 8. Change log

| Version | Date | Change |
| --- | --- | --- |
| 0.1.0 draft | 2026-10-20 | First draft of recommendation, metadata and health interfaces. |
| 0.2.0 draft | 2026-10-27 | Optional `elevation_m` in the request; `inputs_used.site` in the response; B1 reliability rules; new warning and reason codes; draft abstain threshold. All changes are additive, so 0.1.0 requests still work. |

## Review checklist

Confirm that field naming, units, error behavior, and confidence semantics match the workstream implementation before marking a contract as implemented. Do not invent endpoints or sample responses that appear production-ready.

- [ ] Anny and Gabin reviewed section 1 and the diagnosis side
- [ ] Arsene reviewed reliability levels and soil units
- [ ] Contract tests pass (`python -m pytest 03-shared/tests -q`)
- [ ] Status changed from Proposed to Agreed
