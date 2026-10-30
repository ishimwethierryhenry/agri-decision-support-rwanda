"""Map page: click a field in one of the four pilot districts and get the crops for that point.

Flow agreed at the stand-up of 28 Oct 2026: click a point, the page finds the district, reads the soil, elevation,
weather and crop shares for that point, and shows the XGBoost ranking next to the expert rules, each with its
tested score in that district, plus the fertiliser status. The season follows today's date.
The model runs inside the dashboard (no server needed). Logic: 01-crop-recommendation/src/recommendation/map_model.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SERVICE_DIR = Path(__file__).resolve().parents[3] / "01-crop-recommendation" / "src" / "recommendation"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
import map_model as mm  # noqa: E402

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:  # the page still works with typed coordinates
    folium = None

PILOTS = ["Musanze", "Nyabihu", "Nyagatare", "Ruhango"]
PROVINCES = {"North": "Northern Province", "West": "Western Province", "East": "Eastern Province",
             "South": "Southern Province"}
START = {"lat": -1.458, "lon": 29.556}  # a 2 km block in Musanze with many CIAT fields


@st.cache_resource(show_spinner="Training XGBoost on the 1,904 CIAT fields (a few seconds, only once)...")
def recommender() -> mm.MapRecommender:
    return mm.MapRecommender.load()


def crop(c: str) -> str:
    return mm.CROP_NAMES.get(c, c)


def draw_map(rec: mm.MapRecommender, point: dict):
    """District outlines and CIAT blocks stay the same between clicks, so the map keeps its zoom.
    Only the marker of the selected point is redrawn."""
    m = folium.Map(location=[-1.94, 29.87], zoom_start=8, tiles="OpenStreetMap", control_scale=True)
    # Chrome draws a black box around the last clicked district outline; hide it
    m.get_root().header.add_child(folium.Element("<style>.leaflet-interactive:focus{outline:none;}</style>"))
    folium.GeoJson(
        rec.districts.geojson, name="Districts",
        style_function=lambda f: {
            "fillColor": "#2a9d8f" if f["properties"]["district"] in PILOTS else "#bbbbbb",
            "color": "#264653" if f["properties"]["district"] in PILOTS else "#999999",
            "weight": 2 if f["properties"]["district"] in PILOTS else 0.5,
            "fillOpacity": 0.15 if f["properties"]["district"] in PILOTS else 0.03},
        tooltip=folium.GeoJsonTooltip(fields=["district"], labels=False),
    ).add_to(m)
    for b in rec.blocks.itertuples():
        folium.CircleMarker([b.lat, b.lon], radius=1.5 + min(b.n_fields, 40) / 10, weight=0, fill=True,
                            fill_color="#e76f51", fill_opacity=0.6,
                            tooltip=f"{b.District}: {b.n_fields} CIAT field(s) in this 2 km block").add_to(m)
    selected = folium.FeatureGroup(name="Selected point")
    selected.add_child(folium.CircleMarker([point["lat"], point["lon"]], radius=8, color="#ffffff", weight=3,
                                           fill=True, fill_color="#1d3557", fill_opacity=1,
                                           tooltip="Selected point"))
    return st_folium(m, height=560, use_container_width=True, returned_objects=["last_clicked"],
                     feature_group_to_add=selected, key="crop_map")


def tested_note(item: dict) -> str:
    """How the method did for this crop in this district, on new fields (the situation of the map)."""
    x = (item.get("tested_here") or {}).get("new_fields")
    if not x:
        return ""
    if not x["scored"]:
        return f", not tested here (only {x['fields']} field{'' if x['fields'] == 1 else 's'})"
    return f", tested F1 here {x['f1']:.2f}"


def ranking(title: str, rows: list, value_key: str, tested: dict, district: str):
    st.markdown(f"**{title}**")
    for r in rows:
        label = crop(r["crop"]) + (" (weak rule: no water-access data)" if r.get("weak") else "")
        st.progress(min(max(float(r[value_key]), 0.0), 1.0), text=f"{label}: {r[value_key]:.2f}{tested_note(r)}")
    if tested.get("new_fields") is not None:
        st.caption(f"Tested score in {district}: {tested['new_fields']:.2f} on new fields, "
                   f"{tested.get('new_district', float('nan')):.2f} when {district} was left out of training.")


def crop_table(rec: mm.MapRecommender, district: str, test: str) -> pd.DataFrame:
    """Precision, recall and F1 of each crop in one district, for XGBoost and for the rules."""
    rows = []
    for c in mm.CROP_NAMES:
        found = {name: rec.crop_tested.get(district, {}).get(method, {}).get(c, {}).get(test)
                 for name, method in (("XGBoost", "model"), ("Rules", "rules"))}
        if not found["XGBoost"]:
            continue
        row = {"Crop": crop(c), "Fields": found["XGBoost"]["fields"]}
        # F1 first, so the main numbers are visible without scrolling in the narrow column
        for key, label in (("f1", "F1"), ("precision", "precision"), ("recall", "recall")):
            for name, x in found.items():
                row[f"{name} {label}"] = float(x[key]) if x and x["scored"] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("Crop")


def tested_table(rec: mm.MapRecommender) -> pd.DataFrame:
    rows = []
    for d in PILOTS:
        t = rec.tested.get(d, {})
        rows.append({"District": d,
                     "XGBoost, new fields": t.get("model", {}).get("new_fields"),
                     "Rules, new fields": t.get("rules", {}).get("new_fields"),
                     "XGBoost, district left out": t.get("model", {}).get("new_district"),
                     "Rules, district left out": t.get("rules", {}).get("new_district")})
    return pd.DataFrame(rows).set_index("District")


st.header("Click a field on the map")
st.caption("XGBoost and the expert rules side by side, for any point in Musanze, Nyabihu, Nyagatare or Ruhango. "
           "Team 8 prototype. Not agronomic advice.")

rec = recommender()
st.session_state.setdefault("point", dict(START))

with st.expander("How this page works"):
    st.markdown(
        "1. **District**: the district outline the point falls in (geoBoundaries, Open Data Rwanda, CC BY 4.0). "
        "Only the four pilot districts have CIAT data.\n"
        "2. **Field inputs**: soil (iSDAsoil, 0 to 20 cm) and elevation from the CIAT fields in the same 2 km "
        "block, or from the nearest block in the district. The page says how far that block is.\n"
        "3. **District inputs**: season B 2025 weather (NASA POWER) and NISR crop shares, the values the model "
        "learned from.\n"
        "4. **Two methods**: XGBoost on all inputs, trained on all 1,904 CIAT fields with the protocol settings, "
        "and the expert rules (B1). Each shows its tested score in this district (eval_v1).\n"
        "5. **Fertiliser**: no rate until the RAB guideline is in the rule file. Nothing is invented.")
    st.markdown("**Tested scores (macro-F1, 0 to 1, protocol 1.0)**")
    st.dataframe(tested_table(rec).style.format("{:.2f}", na_rep=""))
    st.caption("New fields: fields from 2 km blocks the model never saw, in a district it knows. District left "
               "out: the model never saw the district at all. The API default stays the rules, because no model "
               "passed the adoption rule when a district was left out.")

today = mm.current_season()
seasons = list(mm.SEASON_NAMES)
season = st.selectbox("Season", seasons, index=seasons.index(today),
                      format_func=lambda s: mm.SEASON_NAMES[s] + (" (today)" if s == today else ""))

left, right = st.columns([3, 2], gap="large")
with left:
    if folium is None or st.session_state.get("hide_map"):
        if folium is None:
            st.info("Install streamlit-folium to click on the map: python -m pip install streamlit-folium. "
                    "Until then, type the coordinates below.")
    else:
        clicked = draw_map(rec, st.session_state["point"])
        last = (clicked or {}).get("last_clicked")
        if last:
            seen = (round(float(last["lat"]), 6), round(float(last["lng"]), 6))
            if seen != st.session_state.get("seen_click"):     # a new click, not the one already handled
                st.session_state["seen_click"] = seen
                st.session_state["point"] = {"lat": seen[0], "lon": seen[1]}
                st.rerun()
        st.caption("Green: the four pilot districts. Orange dots: 2 km blocks with CIAT fields "
                   "(bigger dot, more fields). Dark blue dot: the selected point. "
                   "Outlines: geoBoundaries, Open Data Rwanda, CC BY 4.0.")
    with st.form("coordinates"):
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=float(st.session_state["point"]["lat"]), format="%.5f", step=0.01)
        lon = c2.number_input("Longitude", value=float(st.session_state["point"]["lon"]), format="%.5f", step=0.01)
        if st.form_submit_button("Use these coordinates"):
            st.session_state["point"] = {"lat": float(lat), "lon": float(lon)}
            st.rerun()

with right:
    p = st.session_state["point"]
    res = rec.recommend(p["lat"], p["lon"], season)
    if res["status"] != "ok":
        st.subheader(res.get("district") or "Outside Rwanda")
        st.warning(res["message"])
        for w in res["warnings"]:
            st.caption(w)
    else:
        inp = res["inputs"]
        st.subheader(f"{res['district']} ({PROVINCES.get(inp['province'], inp['province'])})")
        st.caption(f"Point {res['point']['lat']}, {res['point']['lon']}")
        if res["season"]["asked"] != res["season"]["used"]:
            st.info(f"{mm.SEASON_NAMES[res['season']['asked']]} is not in the training data yet. The answer below "
                    "is for season B (from 1 February), the only season the model and the rules were tested on.")
        top_model, top_rules = res["model"]["ranking"][0], res["rules"]["ranking"][0]
        b1, b2 = st.columns(2)
        b1.caption("XGBoost suggests")
        b1.markdown(f"### {crop(top_model['crop'])}")
        b1.caption(f"Model score {top_model['probability']:.2f} out of 1")
        b2.caption("Expert rules suggest")
        b2.markdown(f"### {crop(top_rules['crop'])}")
        b2.caption(f"Rule score {top_rules['score']:.2f} out of 1")
        better = res["better_in_testing"]
        mt, rt = res["model"]["tested"].get("new_fields"), res["rules"]["tested"].get("new_fields")
        if better == "model":
            st.success(f"On new fields in {res['district']}, XGBoost did better than the rules in testing "
                       f"({mt:.2f} against {rt:.2f}).")
        elif better == "rules":
            st.warning(f"On new fields in {res['district']}, the rules did better than XGBoost in testing "
                       f"({rt:.2f} against {mt:.2f}). Read both rankings.")
        elif better == "same":
            st.info(f"On new fields in {res['district']}, XGBoost and the rules scored about the same in testing.")
        c1, c2 = st.columns(2)
        with c1:
            ranking("XGBoost (all inputs)", res["model"]["ranking"], "probability", res["model"]["tested"],
                    res["district"])
        with c2:
            ranking("Expert rules (B1)", res["rules"]["ranking"], "score", res["rules"]["tested"], res["district"])
        with st.expander(f"How each method did in {res['district']}, crop by crop"):
            test = st.radio("Test", ["new_fields", "new_district"], horizontal=True, key="crop_test",
                            format_func={"new_fields": "New fields in this district",
                                         "new_district": f"{res['district']} left out of training"}.get)
            table = crop_table(rec, res["district"], test)
            if len(table):
                shown = table.copy()
                for c in shown.columns:
                    if c != "Fields":   # text, so a crop with too few fields reads "not scored" instead of None
                        shown[c] = [("not scored" if pd.isna(v) else f"{v:.2f}") for v in table[c]]
                st.dataframe(shown)
            st.caption("Precision: when the method says this crop, how often it is right. Recall: of the fields "
                       "with this crop, how many the method found. F1 combines the two, from 0 to 1. Crops with "
                       "fewer than 5 fields in the district are not scored. From the official evaluation (eval_v1). "
                       "The Model scores page has every model.")
        with st.expander("Why the rules say this"):
            for r in res["rules"]["ranking"]:
                st.markdown(f"**{crop(r['crop'])}** ({r['score']:.2f})")
                for reason in r["reasons"]:
                    st.write(f"- {reason['text']}")
        with st.expander("Inputs for this point", expanded=True):
            blk = inp["block"]
            where = ("the same 2 km block" if blk["distance_km"] == 0
                     else f"the nearest 2 km block, {blk['distance_km']:.1f} km away")
            st.write(f"Soil and elevation come from {blk['n_fields']} CIAT field(s) in {where}.")
            soil = inp["soil_0_20cm"]
            st.write(f"Soil, 0 to 20 cm (iSDAsoil): **{inp['soil_texture']}**, pH {soil['ph']:.1f}, organic carbon "
                     f"{soil['carbon_organic']:.1f} g/kg, nitrogen {soil['nitrogen']:.2f} g/kg, phosphorus "
                     f"{soil['phosphorous_extractable']:.1f} mg/kg, potassium {soil['potassium_extractable']:.0f} "
                     f"mg/kg, CEC {soil['cation_exchange_capacity']:.1f} cmol/kg "
                     f"(sand {soil['sand']:.0f}%, silt {soil['silt']:.0f}%, clay {soil['clay']:.0f}%).")
            clim = inp["climate_season_b_2025"]
            st.write(f"Elevation {inp['elevation_m']:.0f} m. Season B 2025 weather for the district (NASA POWER): "
                     f"rainfall {clim['rainfall_mm']:.0f} mm, mean temperature at this height "
                     f"{clim['mean_temp_at_field_c']:.1f} C.")
            shares = ", ".join(f"{k} {v:.0%}" for k, v in inp["nisr_crop_shares"].items())
            st.write(f"District crop shares (NISR, area of the four crops): {shares}.")
        st.markdown("**Fertiliser**")
        fert = res["fertiliser"]
        if fert["status"] == "no_matching_rule":
            st.write(f"No advice yet for {crop(top_model['crop'])}: the rule file is waiting for the RAB fertiliser "
                     "guideline, so no rate is given.")
        else:
            for item in fert.get("items", []):
                rate = f"{item.get('rate')} {item.get('rate_unit') or ''}".strip()
                st.write(f"- {item.get('input_type')}: {item.get('product') or ''} {rate}, {item.get('timing') or ''} "
                         f"(source: {item.get('source', {}).get('document', 'not given')})")
        if res["warnings"]:
            with st.expander("Notes"):
                for w in res["warnings"]:
                    st.write(f"- {w}")
