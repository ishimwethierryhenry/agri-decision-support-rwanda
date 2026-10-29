# Dashboard

Streamlit app (D6, task T037) with three pages:

| Page | What it does |
| --- | --- |
| Map: crop for a field (default) | Click a point in Musanze, Nyabihu, Nyagatare or Ruhango. The page finds the district, shows the soil, elevation, weather and NISR crop shares used for that point, and ranks the crops with XGBoost and with the expert rules (B1), each with its tested score in that district. The season follows today's date. |
| Crop and fertiliser | Calls `POST /v1/recommend` (Part B service, contract 0.2.1). |
| Crop health diagnosis | Placeholder, built by the CV sub-team against `/detect`. |

Run from this folder (Python 3.12 or newer, because of the pinned XGBoost):

    python -m pip install -r requirements.txt
    streamlit run app.py

`requirements.txt` includes `01-crop-recommendation/requirements-eval.txt`, so the map page trains XGBoost with the same library versions as the official evaluation. Training takes a few seconds, once, when the map page first opens.

## Map page

- The map needs `folium` and `streamlit-folium`. Without them, the page still works with the latitude and longitude boxes under the map.
- Logic: `01-crop-recommendation/src/recommendation/map_model.py` (explained in that folder's README). The page only displays what that module returns.
- What the map shows: district outlines (geoBoundaries, Open Data Rwanda, CC BY 4.0) and one dot per 2 km block with CIAT fields. It never shows a farm's GPS point.
- On Colab, `01-crop-recommendation/notebooks/03_xgboost_training_and_map_colab.ipynb` starts this app and prints a link.

## Crop and fertiliser page

Baseline mode (default) runs the real recommendation service on the B1 expert rules inside the app, so no server is needed. It needs the files in `01-crop-recommendation/src/recommendation` and `01-crop-recommendation/src/models`.

Mock mode uses the in-process stub from `03-shared/apis`. The data is placeholder data and the page says so.

Live mode calls a running service over HTTP. From the repository root, start the real service with:

    python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001

or the stub with `cd 03-shared/apis` and `uvicorn stub_app:app --port 8001`.

Then choose "Live service" in the sidebar.

This page only shows what the API returns. Nothing is computed or invented in the dashboard. When the real service replaces the stub, no dashboard code changes as long as the contract version stays the same.

## Tests

From the repository root: `python -m pytest 03-shared/tests -q` (the map page tests need XGBoost).
