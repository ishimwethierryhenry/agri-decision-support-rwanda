# Dashboard

Streamlit app (D6, task T037) with four pages:

| Page | What it does |
| --- | --- |
| Map: crop for a field (default) | Click a point in Musanze, Nyabihu, Nyagatare or Ruhango. The page shows one recommended crop, or two crops when they fit about equally, with its confidence level (high or medium: how often answers like it were right in testing) and the reasons, then the fertiliser for each crop. Where nothing tested is reliable enough (a season other than B, or more than 20 km from any surveyed field) it says so and gives no crop, never one crop with a low confidence. Then it shows the soil, elevation, satellite values, weather and NISR crop shares used for that point. In each district the answer comes from the method that named the right crop first more often in testing on new fields there (the tuned Random Forest or the expert rules, B1); today that is the tuned Random Forest in all four districts. The **Evaluation view** switch in the sidebar shows both methods side by side, with their tested scores, crop by crop. The page opens on season B, the only season with labelled data; today's season is marked and can be picked, and gets no answer because it was never tested. Under the answer, a crop profile (since 26 Nov 2026) shows each recommended crop: an animated drawing, its names in English, Kinyarwanda and Latin, four short facts, the varieties for Rwanda with those that fit the field's elevation first (each drawn in its own colours: bean seed, potato skin and flesh, maize kernels, rice grain), what to watch out for, how farmers get seed and fertiliser through the Smart Nkunganire System, and the numbered sources. |
| Model scores | How well every model did in the official evaluation (reports/eval_v2): main score of the 18 combinations on both tests, the score in each district, overall scores (accuracy, balanced accuracy, macro-F1, weighted F1, Cohen's kappa), and precision, recall and F1 of each crop in each district. Then the three tuning rounds (reports/tuning_v1, tuning_v2 and tuning_v3): every method with the protocol settings and tuned, and why the Random Forest of the third round is the model on the map, with how often it is right when it is sure. |
| Crop and fertiliser | Calls `POST /v1/recommend` (Part B service, contract 1.4). The map's one answer is also served as `POST /v1/recommend/point` (see `docs/api_contracts.md`, section 3). |
| Crop health diagnosis | Pick the crop, add one leaf photo, and the page calls `POST /v1/detect` on the crop health detection service (CV workstream, port 8002). It shows the diagnosis with its confidence, "uncertain", or what to fix in the photo. |

Run from this folder (Python 3.12 or newer, because of the pinned library versions):

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

`python -m streamlit` also works when the `streamlit` command is not on the PATH, which happens on Windows when pip installs into the user folder.

`requirements.txt` includes `01-crop-recommendation/requirements-eval.txt`, so the map page trains its model (the tuned Random Forest, scikit-learn) with the same library versions as the official evaluation. Training takes a few seconds, once, when the map page first opens.

## Look and navigation

`ui_theme.py` holds the shared look: the crop-row background, the frosted panel, the top bar with one pill
button per page (the current page is filled), cards and form widgets. `app.py` applies it once before running
the page, so the pages themselves stay plain Streamlit. `rec_ui.py` adds the parts that the Map, Model scores
and Crop and fertiliser pages share with the Diagnosis page: numbered steps, card headings, and the answer shown
as a confidence badge, the crop name and a bar. The sidebar keeps the settings (data source, Evaluation
view, detection service URL). It starts closed so the pages get the full width; the >> button at the top left
opens it. `.streamlit/config.toml` sets the green theme; Streamlit reads it when the app is
started from this folder.

## Crop profile (map page)

- Data and fit logic: `01-crop-recommendation/src/recommendation/crop_profiles.py` and `rules/crop_profiles.yaml`. Every claim cites a source; the page numbers the sources in IEEE style.
- Drawings: `crop_art.py`. Every picture is an original SVG drawing (no photos), so it works offline and needs no licence. The scenes are animated with CSS only (plants grow, leaves sway, the season B rain falls) and stand still for anyone whose system asks for reduced motion.
- Layout: `crop_ui.py`. One crop gets one profile, two crops get one tab each. Potato shows its first 8 varieties and folds the rest.

## Diagnosis page

Start the detection service from the [Agriculture-CV](https://github.com/annychristelle/Agriculture-CV)
repository (it needs the trained models in its `models/` folder):

    uvicorn api.detect.main:app --port 8002

The page calls `http://127.0.0.1:8002` by default. Change it in the sidebar, or set `DETECT_API_URL`.
The contract is `docs/api_detect.md` in that repository.

## Map page

- The map needs `folium` and `streamlit-folium`. Without them, the page still works with the latitude and longitude boxes under the map.
- Logic: `01-crop-recommendation/src/recommendation/map_model.py` (explained in that folder's README). The page only displays what that module returns.
- What the map shows: district outlines (geoBoundaries, Open Data Rwanda, CC BY 4.0) and one dot per 2 km block with CIAT fields. It never shows a farm's GPS point.
- On Colab, `01-crop-recommendation/notebooks/03_xgboost_training_and_map_colab.ipynb` starts this app and prints a link.

## Model scores page

- Logic: `01-crop-recommendation/src/models/model_scores.py`. It recomputes every number from the committed result files (summaries, results per district and confusion counts), and a test checks that the main score it recomputes equals the official one for every model, district and test.
- Nothing on this page is trained or tuned: it only reads `01-crop-recommendation/reports/eval_v2` and the three tuning rounds, `tuning_v1`, `tuning_v2` and `tuning_v3`.

## Crop and fertiliser page

Baseline mode (default) runs the real recommendation service on the B1 expert rules inside the app, so no server is needed. It needs the files in `01-crop-recommendation/src/recommendation` and `01-crop-recommendation/src/models`.

Mock mode uses the in-process stub from `03-shared/apis`. The data is placeholder data and the page says so.

Live mode calls a running service over HTTP. From the repository root, start the real service with:

    python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001

or the stub with `cd 03-shared/apis` and `uvicorn stub_app:app --port 8001`.

Then choose "Live service" in the sidebar.

This page only shows what the API returns. Nothing is computed or invented in the dashboard. When the real service replaces the stub, no dashboard code changes as long as the contract version stays the same.

## Tests

From the repository root: `python -m pytest 03-shared/tests -q`.
