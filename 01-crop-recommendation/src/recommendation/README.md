# Crop Recommendation Logic

This folder holds the recommendation service and the fertiliser rule engine.

| File | What it is |
| --- | --- |
| `service.py` | FastAPI service for `GET /v1/health`, `GET /v1/metadata` and `POST /v1/recommend` (contract in `docs/api_contracts.md`). |
| `district_profiles.json` | District summaries the service reads: CIAT elevation (median, 10th and 90th percentile), NASA POWER season normals, crop shares. No farm-level data. |
| `build_district_profiles.py` | Rebuilds `district_profiles.json` from Part A's processed files. Run it when Part A updates the data. |
| `fertiliser_engine.py`, `rules/fertiliser_rules.yaml` | Fertiliser rule engine. The rules file stays empty until the RAB guideline is confirmed. |

Run the service from the repository root:

```
python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001
```

## Inputs

- A pilot district (Musanze, Nyabihu, Nyagatare, Ruhango), or `lat` and `lon`. A point is matched to the nearest district centre within 25 km.
- The season (A, B or C).
- Optional field elevation (`elevation_m`). Without it, the district median of the CIAT points is used.
- Optional soil test values. They are recorded and passed to the fertiliser rules. The crop baseline does not use soil yet.

## Decision assumptions

- Crops are ranked by the B1 expert rules (`../models/rules/crop_suitability.yaml`). Every range cites a source, and the file is a draft until a teammate checks each page.
- Temperature is the NASA POWER season normal for the district (mean over start years 2017 to 2024), moved to the field's height with the standard lapse rate of 6.5 C per 1000 m. The reference height is the district median of the CIAT points.
- Musanze and Nyabihu share one NASA POWER cell, so their climate values are identical.

## Failure behaviour

- If no crop scores at least 0.5, the service abstains and asks the user to consult an extension officer.
- If no fertiliser rule fits, it returns `no_matching_rule` and never invents a rate.
- Unknown districts and crops are refused with the contract error envelope.

## Uncertainty

- B1 is never "high" reliability, because no trained model has been adopted.
- Reliability drops to "low" for season A or C (B1 was tested on season B only), for points outside the pilot districts, for missing inputs, and for rice (no water-access layer).
- On held-out districts, B1 reached a macro-F1 of about 0.30 (27 October 2026). Treat its answers as a starting point for discussion, not as advice.

Do not add agronomic rules without validated evidence and stakeholder review.
