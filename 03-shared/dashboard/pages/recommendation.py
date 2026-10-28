import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client  # noqa: E402

# Season windows as defined by Part A (01-crop-recommendation/data/README.md).
SEASONS = {"A": "Season A (Sep to Jan)", "B": "Season B (Feb to Jun)", "C": "Season C (Jun to Aug, valleys and marshland)"}
LEVEL_LABEL = {"high": "High reliability", "medium": "Medium reliability", "low": "Low reliability"}
DIRECTION_LABEL = {"supports": "Supports", "against": "Against", "neutral": "Note"}
REASON_CODE_TEXT = {
    "ML_NOT_ADOPTED": "rules only, no trained model adopted yet",
    "SEASON_NOT_EVALUATED": "the rules were tested on season B only",
    "CROP_RULES_WEAK": "the rules for this crop are weak",
    "DISTRICT_NOT_IN_TRAINING": "outside the four pilot districts",
    "RULES_WEAK_IN_DISTRICT": "the rules scored low in this district when tested",
}
SITE_LABEL = {"elevation_m": ("Elevation", "m"), "precip_mm": ("Season rainfall", "mm"),
              "tmean_point_c": ("Mean temperature", "C"), "tmin_point_c": ("Minimum temperature", "C")}


def code_text(code: str) -> str:
    if code.startswith("INPUT_MISSING_"):
        return "missing " + code.removeprefix("INPUT_MISSING_").lower().replace("_", " ")
    return REASON_CODE_TEXT.get(code, code)


mode = st.session_state.get("mode", "baseline")
base_url = st.session_state.get("base_url", client.DEFAULT_BASE_URL)

st.header("Crop and fertiliser recommendation")
if mode == "mock":
    st.warning("Mock mode: the answers below are placeholders from the stub, not real recommendations.")
elif mode == "baseline":
    st.info("Baseline mode: crops are ranked with published crop requirements (rules B1), not machine learning. "
            "The rules are a draft for team review. This is not agronomic advice.")

try:
    meta = client.get_metadata(mode, base_url)
    districts = [d["name"] for d in meta["districts"]]
    crops = meta["crops"]
except client.ApiError as exc:
    st.error(f"Could not load the district and crop lists. {exc}")
    st.stop()

with st.form("request"):
    c1, c2, c3 = st.columns(3)
    district = c1.selectbox("District", districts)
    season = c2.selectbox("Season", list(SEASONS), format_func=SEASONS.get, index=1)
    top_k = c3.slider("Number of crops", 1, 5, 3)
    e1, e2 = st.columns(2)
    elevation = e1.number_input("Field elevation in metres (optional)", 900.0, 4600.0, value=None, step=10.0,
                                help="From a phone GPS, for example. Leave empty to use the district median.")
    include = e2.multiselect("What do you need?", ["crops", "fertiliser"], default=["crops", "fertiliser"])
    crop = st.selectbox("Crop for fertiliser advice (optional)", [""] + crops,
                        format_func=lambda c: c.replace("_", " ") if c else "Use the top ranked crop")
    with st.expander("Soil test values (optional)"):
        s1, s2, s3 = st.columns(3)
        ph = s1.number_input("pH in water", 3.0, 10.0, value=None, step=0.1)
        oc = s2.number_input("Organic carbon (%)", 0.0, 20.0, value=None, step=0.1)
        tn = s3.number_input("Total nitrogen (%)", 0.0, 5.0, value=None, step=0.01)
        p = s1.number_input("Phosphorus (mg/kg)", 0.0, 200.0, value=None, step=1.0)
        k = s2.number_input("Potassium (cmol/kg)", 0.0, 10.0, value=None, step=0.1)
        tex = s3.selectbox("Texture", [""] + [
            "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt", "sandy_clay_loam",
            "clay_loam", "silty_clay_loam", "sandy_clay", "silty_clay", "clay"], format_func=lambda t: t or "Not known")
    submitted = st.form_submit_button("Get recommendation", type="primary")

if submitted:
    if not include:
        st.error("Choose at least one of crops or fertiliser.")
        st.stop()
    payload = client.build_payload(district, season, top_k, include, crop or None, {
        "ph_water": ph, "organic_carbon_pct": oc, "total_nitrogen_pct": tn,
        "phosphorus_mgkg": p, "potassium_cmolkg": k, "texture": tex or None}, elevation_m=elevation)
    try:
        st.session_state["result"] = client.recommend(payload, mode, base_url)
    except client.ApiError as exc:
        st.session_state.pop("result", None)
        st.error(f"{exc} (code {exc.code})")
        for d in exc.details:
            st.write(f"- {d.get('field') or 'request'}: {d.get('issue')}")

res = st.session_state.get("result")
if res:
    if res["abstain"]:
        st.error(res.get("abstain_message") or "Uncertain. Please consult an extension officer.")
    if res["crop_recommendations"]:
        loc = res["location"]
        where = loc.get("district") or "this location"
        st.subheader(f"Suggested crops for {where}, season {res['season']}")
        for rec in res["crop_recommendations"]:
            with st.container(border=True):
                a, b = st.columns([3, 2])
                a.markdown(f"### {rec['rank']}. {rec['crop'].replace('_', ' ').title()}")
                level = rec["reliability"]["level"]
                b.metric("Suitability score", f"{rec['score']:.2f}")
                b.caption(f"{LEVEL_LABEL[level]}  |  method: {rec['method'].replace('_', ' ')}")
                if rec["reliability"]["reason_codes"]:
                    st.caption("Why this reliability: " + "; ".join(code_text(c) for c in rec["reliability"]["reason_codes"]))
                for r in rec["reasons"]:
                    st.write(f"- **{DIRECTION_LABEL[r['direction']]}**: {r['text']}")
    fert = res.get("fertiliser")
    if fert:
        st.subheader(f"Fertiliser guidance for {fert['crop'].replace('_', ' ')}")
        if fert["status"] == "no_matching_rule":
            st.info("No guideline rule fits these conditions, so no fertiliser advice is given.")
        for it in fert["items"]:
            with st.container(border=True):
                rate = f"{it['rate']} {it['rate_unit']}" if it.get("rate") is not None else "rate not set"
                st.markdown(f"**{it['input_type'].replace('_', ' ').title()}**: {rate}")
                st.caption(f"Matched rule: {it['condition_matched']}")
                if it.get("explanation"):
                    st.write(it["explanation"])
                src = it["source"]
                st.caption("Source: " + ", ".join(str(v) for v in (src.get("document"), src.get("section"), src.get("page")) if v))
    used = res.get("inputs_used", {})
    with st.expander("Inputs used and notes"):
        for key, v in used.get("site", {}).items():
            name, unit = SITE_LABEL.get(key, (key, ""))
            st.write(f"- {name}: {v['value']:g} {unit} (source: {v['source'].replace('_', ' ')})")
        soil = used.get("soil", {})
        for key, v in soil.get("values", {}).items():
            st.write(f"- {key.replace('_', ' ')}: {v['value']} (source: {v['source']})")
        if soil.get("missing"):
            st.write("- Soil values not given: " + ", ".join(m.replace("_", " ") for m in soil["missing"]))
        for w in res.get("warnings", []):
            (st.warning if w["code"] == "STUB_RESPONSE" else st.caption)(w["message"])
    st.caption(res["disclaimer"])
    with st.expander("Technical details"):
        st.json({k: res[k] for k in ("request_id", "api_version", "model_version", "rules_version", "generated_at")})
