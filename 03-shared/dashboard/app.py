"""Dashboard (D6). Run from this folder:  streamlit run app.py

Pages:
  Map            : click a point in a pilot district; XGBoost and the expert rules side by side (default page).
  Model scores   : how well every model did in the official evaluation, per district and per crop.
  Recommendation : calls POST /v1/recommend (Part B).
  Diagnosis      : placeholder, built by the CV sub-team against /detect.
The data source in the sidebar is for the Recommendation page. Baseline mode (default) runs the real service on
the B1 expert rules inside the app, so no server is needed. Mock mode shows clearly marked placeholder data from
the stub. Live mode calls a running service. The Map page always runs its models inside the app.
"""
import streamlit as st

st.set_page_config(page_title="Smallholder decision support", layout="wide")

with st.sidebar:
    st.title("Decision support")
    st.caption("Team 8 prototype. Not agronomic advice.")
    modes = {"baseline": "Baseline rules (B1, built in)", "mock": "Mock (placeholder stub)", "live": "Live service"}
    st.session_state.setdefault("mode", "baseline")
    st.session_state["mode"] = st.radio(
        "Data source", list(modes), index=list(modes).index(st.session_state["mode"]), format_func=modes.get,
    )
    st.session_state.setdefault("base_url", "http://127.0.0.1:8001")
    if st.session_state["mode"] == "live":
        st.session_state["base_url"] = st.text_input("Service URL", st.session_state["base_url"])
    st.caption("The data source is for the Crop and fertiliser page. The map page runs XGBoost and the rules "
               "inside the app.")

pages = [
    st.Page("pages/map.py", title="Map: crop for a field", default=True),
    st.Page("pages/scores.py", title="Model scores"),
    st.Page("pages/recommendation.py", title="Crop and fertiliser"),
    st.Page("pages/diagnosis.py", title="Crop health diagnosis"),
]
st.navigation(pages).run()
