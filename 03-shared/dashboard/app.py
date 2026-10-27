"""Dashboard skeleton (D6). Run from this folder:  streamlit run app.py

Pages:
  Recommendation : calls POST /v1/recommend (Part B).
  Diagnosis      : placeholder, built by the CV sub-team against /detect.
Mock mode needs no running service and shows clearly marked placeholder data.
"""
import streamlit as st

st.set_page_config(page_title="Smallholder decision support", layout="wide")

with st.sidebar:
    st.title("Decision support")
    st.caption("Team 8 prototype. Not agronomic advice.")
    st.session_state.setdefault("mode", "mock")
    st.session_state["mode"] = st.radio(
        "Data source", ["mock", "live"], index=0 if st.session_state["mode"] == "mock" else 1,
        format_func=lambda m: "Mock (built-in stub)" if m == "mock" else "Live service",
    )
    st.session_state.setdefault("base_url", "http://127.0.0.1:8001")
    if st.session_state["mode"] == "live":
        st.session_state["base_url"] = st.text_input("Service URL", st.session_state["base_url"])

pages = [
    st.Page("pages/recommendation.py", title="Crop and fertiliser", default=True),
    st.Page("pages/diagnosis.py", title="Crop health diagnosis"),
]
st.navigation(pages).run()
