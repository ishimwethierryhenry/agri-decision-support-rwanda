"""Diagnosis page: one leaf photo -> POST /v1/detect -> result, uncertainty or retake advice.

Calls the crop health detection service of the CV workstream (contract 1.0.0, port 8002; code and spec in
github.com/annychristelle/Agriculture-CV, api/detect/main.py and docs/api_detect.md). Set the service URL in
the sidebar or with DETECT_API_URL. The page only shows what the service returns.
"""
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui_theme import hero, html  # noqa: E402

# Crops and class codes of the detection contract (docs/api_detect.md, section 2.2)
CROPS = {
    "maize": ["HLT", "FAW", "MSV", "MLB", "MLN"],
    "beans": ["HLT", "ALS", "BR"],
    "cassava": ["HLT", "CMD", "CBSD"],
}
NAMES = {
    "HLT": "Healthy", "FAW": "Fall armyworm", "MSV": "Maize streak virus", "MLB": "Maize leaf blight",
    "MLN": "Maize lethal necrosis", "ALS": "Angular leaf spot", "BR": "Bean rust",
    "CMD": "Cassava mosaic disease", "CBSD": "Cassava brown streak disease",
}
CROP_LABELS = {"maize": "🌽 Maize", "beans": "🫘 Beans", "cassava": "🌿 Cassava"}

st.session_state.setdefault("detect_url", os.environ.get("DETECT_API_URL", "http://127.0.0.1:8002"))
with st.sidebar:
    st.session_state["detect_url"] = st.text_input("Detection service URL", st.session_state["detect_url"])
    st.caption("Diagnosis page only. Start the service from the Agriculture-CV repository with "
               "`uvicorn api.detect.main:app --port 8002`.")

# Styles for this page only; the shared ones are in ui_theme.py
html("""
<style>
.result-badge { display: inline-block; font-size: .78rem; font-weight: 600; border-radius: 999px; padding: 4px 12px; }
.result-badge.ok { background: #e3f1cc; color: #3f6b0c; }
.result-badge.alert { background: #fbe7c6; color: #8a5300; }
.result-badge.unsure { background: #e1e8f5; color: var(--navy); }
.result-badge.retake { background: #f8d9d3; color: #9b2c1a; }
.confidence { height: 10px; border-radius: 999px; background: #fff; border: 1px solid var(--line); overflow: hidden; margin: 6px 0 4px; }
.confidence > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--lime-soft), var(--lime)); }
.muted { color: var(--muted); font-size: .85rem; }
.conditions { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }
.tips li { color: var(--muted); font-size: .88rem; margin-bottom: 4px; }
</style>
""")


def detect(crop: str, photo, data: dict) -> dict:
    url = st.session_state["detect_url"].rstrip("/")
    mime = photo.type or "image/jpeg"
    try:
        r = httpx.post(f"{url}/v1/detect", data={"crop": crop, **data},
                       files={"image": (photo.name or "photo.jpg", photo.getvalue(), mime)}, timeout=60)
    except httpx.HTTPError as e:
        st.error(f"Could not reach the detection service at {url}. Is it running? ({type(e).__name__})")
        st.stop()
    if r.status_code != 200:
        try:
            st.error(r.json()["error"]["message"])
        except (ValueError, KeyError, TypeError):
            st.error(f"Service returned HTTP {r.status_code}.")
        st.stop()
    return r.json()


def show_result(res: dict) -> None:
    name = res.get("label_name") or NAMES.get(res.get("label") or "", res.get("label"))
    if res["status"] == "ok":
        c = res["confidence"]
        shown = ">99%" if c > 0.99 else f"{c:.0%}"
        healthy = res["label"] == "HLT"
        html(f"""
<span class="result-badge {'ok' if healthy else 'alert'}">{'Looks healthy' if healthy else 'Problem detected'}</span>
<div class="card-title" style="margin-top:12px">{name}</div>
<div class="confidence"><div style="width:{c:.0%}"></div></div>
<div class="muted">{shown} confidence</div>
<p class="muted" style="margin-top:12px">Confirm with an extension officer before treating the field.</p>""")
    elif res["status"] == "uncertain":
        guess = (f'<p class="muted">Best guess (low confidence, do not act on it alone): {name}, '
                 f'{res["confidence"]:.0%}</p>') if name else ""
        html(f"""
<span class="result-badge unsure">Not sure</span>
<div class="card-title" style="margin-top:12px">{res['message']}</div>{guess}""")
    else:
        html(f"""
<span class="result-badge retake">Retake the photo</span>
<div class="card-title" style="margin-top:12px">{res['message']}</div>
<p class="muted">Tip: one leaf filling the frame, in daylight, with the phone held still.</p>""")


hero("Leaf diagnosis", "What is <em>harming</em> my crop?",
     "Pick the crop, then photograph one leaf in daylight, filling the frame. The crop's own model "
     "checks it for diseases and pests.")
st.write("")

input_col, result_col = st.columns([1.1, 1], gap="large")
with input_col:
    with st.container(key="card_input"):
        html('<div class="card-head"><span><span class="step-n">1</span>Pick the crop</span></div>')
        crop = st.radio("Crop", list(CROPS), format_func=CROP_LABELS.get, horizontal=True,
                        label_visibility="collapsed", key="pills_crop")
        st.write("")
        html('<div class="card-head"><span><span class="step-n">2</span>Add one leaf photo</span></div>')
        upload_tab, camera_tab = st.tabs(["⬆️ Upload a photo", "📷 Take a photo"])
        with upload_tab:
            upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        with camera_tab:
            camera = st.camera_input("Take a photo", label_visibility="collapsed")
        photo = camera or upload

        with st.expander("Location and season (optional)"):
            use_loc = st.checkbox("Include location")
            lat = st.number_input("Latitude", value=-1.95, format="%.5f", disabled=not use_loc)
            lon = st.number_input("Longitude", value=30.06, format="%.5f", disabled=not use_loc)
            season = st.selectbox("Season", ["", "A", "B", "C"])

        html('<div class="card-head"><span><span class="step-n">3</span>Get the result</span></div>')
        diagnose = st.button("Diagnose", type="primary", disabled=not photo, width="stretch")

with result_col:
    with st.container(key="card_result"):
        if photo and diagnose:
            data = {}
            if use_loc:
                data |= {"lat": str(lat), "lon": str(lon)}
            if season:
                data["season"] = season
            # location and season are accepted by the contract; the current models do not use them
            with st.spinner("Checking the photo…"):
                res = detect(crop, photo, data)
            html('<div class="card-head">Result <b>···</b></div>')
            st.image(photo, width="stretch")
            show_result(res)
        else:
            chips = "".join(f'<span class="chip{" on" if c == "HLT" else ""}">{NAMES[c]}</span>' for c in CROPS[crop])
            html(f"""
<div class="card-head">Result <b>···</b></div>
<div class="card-title" style="margin-top:14px">{'Ready when you are' if photo else 'Your result will appear here'}</div>
<p class="muted">{'Press <b>Diagnose</b> to check this photo.' if photo else 'Add a photo to begin.'}</p>
<p class="muted" style="margin-top:16px">The {crop} model recognises:</p>
<div class="conditions">{chips}</div>
<p class="muted" style="margin-top:16px">For a good photo:</p>
<ul class="tips"><li>One leaf filling the frame</li><li>Daylight, not too dark or too bright</li>
<li>Hold the phone still so the photo is sharp</li></ul>""")
            if photo:
                st.image(photo, width="stretch")
