# Dashboard skeleton

Streamlit app for the Recommendation page and a placeholder Diagnosis page (D6, task T037).

Run from this folder:

    pip install -r requirements.txt
    streamlit run app.py

Mock mode (default) uses the in-process stub from `03-shared/apis`, so no server is needed. The data is placeholder data and the page says so.

Live mode calls a running service. Start the stub (or later the real service) with:

    cd ../apis
    uvicorn stub_app:app --port 8001

then choose "Live service" in the sidebar.

The page only shows what the API returns. Nothing is computed or invented in the dashboard. When the real service replaces the stub, no dashboard code changes as long as the contract version stays the same.
