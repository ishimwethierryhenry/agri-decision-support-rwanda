# Crop Recommendation Logic

This folder holds the recommendation service and the fertiliser rule engine.

| File | What it is |
| --- | --- |
| `service.py` | FastAPI service for `GET /v1/health`, `GET /v1/metadata` and `POST /v1/recommend` (contract in `docs/api_contracts.md`). |
| `district_profiles.json` | District summaries the service reads: CIAT elevation (median, 10th and 90th percentile), NASA POWER season normals (season B from 1 Feb to 30 Jun), crop shares, and how B1 scored in each district in `reports/eval_v1`. No farm-level data. |
| `build_district_profiles.py` | Rebuilds `district_profiles.json` from Part A's files and the official evaluation. Run it when Part A updates the data or a new evaluation is committed. |
| `fertiliser_engine.py`, `rules/fertiliser_rules.yaml` | Fertiliser rule engine. The rules file stays empty until the RAB guideline is confirmed. |
| `map_model.py` | Recommendation for one point on the map (dashboard map page and Colab notebook 03): finds the district, builds the input row for the point, ranks the crops with XGBoost and with the B1 rules, and adds the tested score of each in that district. |

Run the service from the repository root:

```
python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001
```

## Map page (`map_model.py`)

What happens when someone clicks a point:

1. District: the outline the point falls in (`../../data/reference/rwanda_districts.geojson`, geoBoundaries, Open Data Rwanda, CC BY 4.0). Only the four pilot districts get a crop. A point within 1.5 km of a CIAT block just across a simplified outline counts as that block's district.
2. Field inputs: the feature table no longer has coordinates, so soil (iSDAsoil) and elevation come from the CIAT fields in the same 2 km block (Part A's `block_id`), or from the nearest block in the district. The answer says how far that block is and warns beyond 5 km.
3. District inputs: season B 2025 weather (NASA POWER) and NISR crop shares, the values the model learned from, so the map and the training data agree.
4. Two methods side by side: XGBoost on all inputs, trained on all 1,904 fields with the protocol settings, and the B1 rules. Each comes with its tested macro-F1 in that district from `reports/eval_v1` (new fields, and district left out), and every crop in each ranking carries its tested precision, recall and F1 in that district (`tested_here`, from `../models/model_scores.py`). Crops with fewer than 5 fields in the district are marked as not scored.
5. Season: today's season is preselected (A from September to January, B from February to June, C in July and August). Only season B is in the training data, so another season gets the season B answer with a note.
6. Fertiliser: the rule engine answers for XGBoost's first crop, which is `no_matching_rule` until the RAB guideline is in.

Quick check from the repository root: `python 01-crop-recommendation/src/recommendation/map_model.py --lat -1.458 --lon 29.556`

## Service inputs

- A pilot district (Musanze, Nyabihu, Nyagatare, Ruhango), or `lat` and `lon`. A point is matched to the nearest district centre within 25 km.
- The season (A, B or C).
- Optional field elevation (`elevation_m`). Without it, the district median of the CIAT points is used.
- Optional soil test values. They are recorded and passed to the fertiliser rules. The crop baseline does not use soil yet.

## Decision assumptions

- Crops are ranked by the B1 expert rules (`../models/rules/crop_suitability.yaml`). Every range cites a source, and the file is a draft until a teammate checks each page.
- Temperature is the NASA POWER season normal for the district (mean over start years 2017 to 2024), moved to the field's height with the standard lapse rate of 6.5 C per 1000 m. The reference height is the district median of the CIAT points.
- Season B runs from 1 February to 30 June, the same window as the training data (most CIAT fields were planted in February).
- Musanze and Nyabihu share one NASA POWER cell, so their climate values are identical.

## Failure behaviour

- If no crop scores at least 0.5, the service abstains and asks the user to consult an extension officer.
- If no fertiliser rule fits, it returns `no_matching_rule` and never invents a rate.
- Unknown districts and crops are refused with the contract error envelope.

## Uncertainty

- B1 is never "high" reliability, because no trained model has been adopted.
- Reliability drops to "low" for season A or C (B1 was tested on season B only), for points outside the pilot districts, for missing inputs, for rice (no water-access layer), and in districts where B1 scored below 0.5 macro-F1 in the official evaluation (now Nyagatare 0.09 and Ruhango 0.19; Musanze 0.59 and Nyabihu 0.65 can be "medium").
- When several crops fit the rules equally, the response says so (`CROPS_TIED`): their order is a tie-break, not a real difference. This happens in the lowlands, where beans and maize both fit.
- In the official evaluation (`reports/eval_v1`, 28 October 2026), B1 scored 0.378 mean macro-F1 and no ML model passed the adoption rule. Treat its answers as a starting point for discussion, not as advice.

Do not add agronomic rules without validated evidence and stakeholder review.
