# Dashboard skeleton

Streamlit app for the Recommendation page and a placeholder Diagnosis page (D6, task T037).

Run from this folder:

    pip install -r requirements.txt
    streamlit run app.py

Baseline mode (default) runs the real recommendation service on the B1 expert rules inside the app, so no server is needed. It needs the files in `01-crop-recommendation/src/recommendation` and `01-crop-recommendation/src/models`.

Mock mode uses the in-process stub from `03-shared/apis`. The data is placeholder data and the page says so.

Live mode calls a running service over HTTP. From the repository root, start the real service with:

    python -m uvicorn service:app --app-dir 01-crop-recommendation/src/recommendation --port 8001

or the stub with `cd 03-shared/apis` and `uvicorn stub_app:app --port 8001`.

Then choose "Live service" in the sidebar.

The page only shows what the API returns. Nothing is computed or invented in the dashboard. When the real service replaces the stub, no dashboard code changes as long as the contract version stays the same.
