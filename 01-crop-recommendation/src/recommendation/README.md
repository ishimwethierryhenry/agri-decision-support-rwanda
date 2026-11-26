# Crop Recommendation Logic

This folder holds the recommendation service and the fertiliser rule engine.

| File | What it is |
| --- | --- |
| `service.py` | FastAPI service for `GET /v1/health`, `GET /v1/metadata` and `POST /v1/recommend` (contract in `docs/api_contracts.md`). |
| `district_profiles.json` | District summaries the service reads: CIAT elevation (median, 10th and 90th percentile), NASA POWER season normals (season B from 1 Feb to 30 Jun), crop shares, and how B1 scored in each district in `reports/eval_v1`. No farm-level data. |
| `build_district_profiles.py` | Rebuilds `district_profiles.json` from Part A's files and the official evaluation. Run it when Part A updates the data or a new evaluation is committed. |
| `fertiliser_engine.py`, `rules/fertiliser_rules.yaml` | Fertiliser rule engine. Until the RAB guideline is confirmed, the rules file holds interim national rates copied from published sources (page and link for each), served with the label "Interim". |
| `map_model.py` | Recommendation for one point on the map (dashboard map page and Colab notebook 03): finds the district, builds the input row for the point, ranks the crops with the trained model on the map (since 25 Nov, the Random Forest of the third round of tuning, on field + satellite inputs with one weight per crop, that decision 2 of the protocol chose: settings and weights in `../../reports/tuning_v3/final_model.json`; from 23 to 25 Nov the tuned Random Forest of `tuning_v1`) and with the B1 rules, and adds the tested score of each in that district. |
| `build_map_confidence.py` | Builds `../../reports/map_confidence/`: how often answers like each one were right in testing, per district, kind of answer (one crop, or two crops when the first two are close) and distance of the field data (within 5 km, 5 to 10 km, 10 to 20 km). It refits the five fold models of the new-fields test of the model on the map (`tuning_v3` since 25 Nov) with the settings and crop weights each fold chose (built-in check: exactly its hit rates) and replays the old and new rules on the tested fields and on a 1 km grid over the four districts. About a minute. |
| `crop_profiles.py`, `rules/crop_profiles.yaml` | Crop profiles for the map page (since 26 Nov 2026): names in English, Kinyarwanda (as NISR writes them) and Latin, four short facts per crop, and the varieties for Rwanda: the ten iron beans of 2010 and 2012 (Mulambu et al., 2017, Table 10.3), 15 potato varieties (Rwanda Potato Variety Description Guide, CIP and RAB, 2025, CC BY 4.0), six RAB maize hybrids (RAB Annual Report 2019-2020, Tables 2 and 7) and six rice varieties (RAB, and the 2025 farmer trials). Every claim cites its source by key; a value no source gives is left out. `for_field` marks the varieties whose altitude zone fits the clicked field, with RAB's split for maize (mid altitude 900 to 1,700 m, high altitude above 1,700 m), and puts them first. `check` lists any claim without a source, unknown colour or zone, or long dash. |

Run the service from the repository root:

```
python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001
```

## Map page (`map_model.py`)

What happens when someone clicks a point:

1. District: the outline the point falls in (`../../data/reference/rwanda_districts.geojson`, geoBoundaries, Open Data Rwanda, CC BY 4.0). Only the four pilot districts get a crop. A point within 1.5 km of a CIAT block just across a simplified outline counts as that block's district.
2. Field inputs: the feature table no longer has coordinates, so soil (iSDAsoil), elevation and the four satellite inputs (SRTM elevation, CHIRPS rainfall of the season, Sentinel-2 vegetation index and shortwave ratio of 2024) come from the CIAT fields in the same 2 km block (Part A's `block_id`, table `feature_table_v6b.csv`), or from the nearest block in the district. An elevation given by the user replaces both elevations. The answer says how far that block is and warns beyond 5 km.
3. District inputs: season B 2025 weather (NASA POWER) and NISR crop shares, the values the model learned from, so the map and the training data agree.
4. Two methods: the Random Forest of the third round of tuning, on field + satellite inputs, trained on all 1,904 fields with the settings chosen by `tune_v3.py --final`, its probabilities multiplied by one weight per crop and normalised again, and the B1 rules. Each comes with its tested macro-F1 in that district from `reports/tuning_v3` (new fields, and district left out), and every crop in each ranking carries its tested precision, recall and F1 in that district (`tested_here`, from `../models/model_scores.py`). Crops with fewer than 5 fields in the district are marked as not scored.
5. Season: today's season is preselected (A from September to January, B from February to June, C in July and August). Only season B is in the training data, so another season gets the season B answer with a note.
6. One answer for the user (`answer`), as the client asked on 19 Nov: one crop and a confidence level, not the comparison. In each district the answer comes from the method whose first crop was right more often on new fields there in testing, by more than 0.01 (otherwise the rules, the API default): today Random Forest in all four districts (76, 72, 76 and 64 percent in Musanze, Nyabihu, Nyagatare and Ruhango, against 62, 66, 22 and 59 percent for the rules, in `tuning_v3`). Until 25 Nov the choice used the main score, which made the rules answer in Nyabihu, where they often rate maize, Irish potato and rice the same. Since 25 Nov the confidence is how often answers like this one were right in testing, from `../../reports/map_confidence` (built by `build_map_confidence.py`): same district, same kind of answer, and field data from about as far away (within 5 km, 5 to 10 km, 10 to 20 km). The answer (`answer["kind"]`) is one crop when the model's first two crops are 0.10 apart or more, and both crops when they are closer (`answer["crops"]`, with the fertiliser for each). High from 75%, medium from 50%. When even the answer's tested rate is below 50%, when the nearest surveyed field is more than 20 km away, or for a season that was never tested, there is no answer (`kind` "none", with `answer["message"]`) instead of a low one, so a farmer never gets one crop with a low confidence. These thresholds are fixed, not tuned. `answer["why"]` gives the reasons in plain words. The dashboard shows both methods only in its evaluation view.
7. Fertiliser: the rule engine answers for the recommended crop with the interim national rate for that crop, labelled as interim, until the RAB guideline is in.
8. Crop profile (dashboard only, since 26 Nov): `crop_profiles.for_field` gives the profile of each crop in the answer, with the varieties that fit the field's elevation first. The dashboard draws it (`03-shared/dashboard/crop_ui.py`). The API answer does not change.

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
