"""Map page: click a field in one of the four pilot districts and get the crops for that point.

Flow agreed at the stand-up of 28 Oct 2026: click a point, the page finds the district, reads the soil, elevation,
satellite values, weather and crop shares for that point, and shows one recommended crop with its confidence and its
fertiliser. The trained model is the one the protocol's rule chose for the map: since 25 Nov 2026 the Random Forest of
the third round of tuning, on field + satellite inputs with crop weights (reports/tuning_v3; from 23 to 25 Nov the tuned
Random Forest of reports/tuning_v1); the Evaluation view shows it next to the expert rules. The page opens on season B, the only
season in the training data; today's season is marked and can be picked.
The model runs inside the dashboard (no server needed). Logic: 01-crop-recommendation/src/recommendation/map_model.py
Since 24 Nov 2026 the page uses the shared Agri DSS look (ui_theme.py, rec_ui.py): a hero heading, the steps in one
card and the answer in another, like the Diagnosis page. Since 25 Nov 2026 the answer in each district comes from the
method that named the right crop first more often in testing (the measure the confidence uses), which is the tuned
Random Forest in all four districts. Also since 25 Nov 2026 the confidence is how often answers like this one were
right in testing (reports/map_confidence): one crop, two crops when they fit about equally, or no answer, never one
crop with a low confidence.
Since 26 Nov 2026 a crop profile follows the answer (crop_ui.py): an animated drawing of the crop, its names in
English, Kinyarwanda and Latin, short facts, and the varieties released in Rwanda, those that fit the field's
elevation first, each drawn in its own colours, with numbered sources.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SERVICE_DIR = Path(__file__).resolve().parents[3] / "01-crop-recommendation" / "src" / "recommendation"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crop_ui  # noqa: E402
import map_model as mm  # noqa: E402
import rec_ui as ui  # noqa: E402
from ui_theme import hero  # noqa: E402

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:  # the page still works with typed coordinates
    folium = None

PILOTS = ["Musanze", "Nyabihu", "Nyagatare", "Ruhango"]
PROVINCES = {"North": "Northern Province", "West": "Western Province", "East": "Eastern Province",
             "South": "Southern Province"}
START = {"lat": -1.458, "lon": 29.556}  # a 2 km block in Musanze with many CIAT fields


@st.cache_resource(show_spinner=f"Training the {mm.MODEL_LABEL} on the 1,904 CIAT fields (a few seconds, only once)...")
def recommender() -> mm.MapRecommender:
    return mm.MapRecommender.load()


def crop(c: str) -> str:
    return mm.CROP_NAMES.get(c, c)


def show_fertiliser(items: list) -> None:
    """Fertiliser items as short lines, then the label if the rates are interim, then each source once."""
    interim = False
    for item in items:
        text = item.get("explanation") or ""
        if text.startswith(mm.fert.INTERIM_NOTE):
            interim, text = True, text[len(mm.fert.INTERIM_NOTE):].strip()
        rate = f"{item['rate']:g} {item['rate_unit']}" if item.get("rate") is not None else "rate not given"
        when = f", {item['timing'][0].lower()}{item['timing'][1:]}" if item.get("timing") else ""
        st.write(f"- **{item.get('product') or item['input_type'].replace('_', ' ')}**: {rate}{when}")
        if text:
            st.caption(text)
    if interim:
        st.warning("Interim rates: national blanket rates from published sources, not yet confirmed by RAB, and the "
                   "same for every field. Check with the local agronomist before applying.")
    seen = []
    for item in items:
        src = item.get("source", {})
        if src.get("document") and src["document"] not in seen:
            seen.append(src["document"])
            where = f", p. {src['page']}" if src.get("page") else ""
            link = f" ([link]({src['url']}))" if src.get("url") else ""
            st.caption(f"Source: {src['document']}{where}{link}")


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
    """Precision, recall and F1 of each crop in one district, for the trained model and for the rules."""
    rows = []
    for c in mm.CROP_NAMES:
        found = {name: rec.crop_tested.get(district, {}).get(method, {}).get(c, {}).get(test)
                 for name, method in ((mm.MODEL_LABEL, "model"), ("Rules", "rules"))}
        if not found[mm.MODEL_LABEL]:
            continue
        row = {"Crop": crop(c), "Fields": found[mm.MODEL_LABEL]["fields"]}
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
                     f"{mm.MODEL_LABEL}, new fields": t.get("model", {}).get("new_fields"),
                     "Rules, new fields": t.get("rules", {}).get("new_fields"),
                     f"{mm.MODEL_LABEL}, district left out": t.get("model", {}).get("new_district"),
                     "Rules, district left out": t.get("rules", {}).get("new_district")})
    return pd.DataFrame(rows).set_index("District")


hero("Crop recommendation", "Which <em>crop</em> fits this field?",
     "Click a point in Musanze, Nyabihu, Nyagatare or Ruhango. You get one crop, how sure we are, and its "
     "fertiliser. Team 8 prototype, not agronomic advice.")
ui.styles()

rec = recommender()
st.session_state.setdefault("point", dict(START))
evaluation = st.sidebar.toggle("Evaluation view", key="eval_view",
                               help="For the team: both methods at this point, with their tested scores.")
answered_by = {d: rec.method_for(d) for d in PILOTS}

with st.expander("How this page works"):
    st.markdown(
        "1. **District**: the district outline the point falls in (geoBoundaries, Open Data Rwanda, CC BY 4.0). "
        "Only the four pilot districts have CIAT data.\n"
        "2. **Field inputs**: soil (iSDAsoil, 0 to 20 cm), elevation and four satellite values (SRTM elevation, "
        "CHIRPS rainfall of the season, and two Sentinel-2 indices of 2024) from the CIAT fields in the same 2 km "
        "block, or from the nearest block in the district. The page says how far that block is.\n"
        "3. **District inputs**: season B 2025 weather (NASA POWER) and NISR crop shares, the values the model "
        "learned from.\n"
        f"4. **One answer**: two methods were tested, the {mm.MODEL_TITLE} (trained on all 1,904 CIAT fields, "
        "with the settings and crop weights the third round of tuning, on 25 November, chose without ever looking at "
        "the test fields) and the "
        "expert rules (B1). In each district, the answer comes from the one that named the right crop first more "
        "often in testing on new fields there: "
        + (f"{mm.MODEL_LABEL} in all four districts" if all(answered_by[d] == "model" for d in PILOTS) else
           ", ".join(f"{mm.MODEL_LABEL if answered_by[d] == 'model' else 'the rules'} in {d}" for d in PILOTS))
        + ".\n"
        "5. **Confidence**: how often answers like this one were right in testing: same district, same kind of "
        "answer, and field data from about as far away (01-crop-recommendation/reports/map_confidence). High from "
        "75%, medium from 50%. When the first two crops are close, the page gives both, with how often the crop "
        "planted was one of the two. When even that was right for less than half of such fields, when the nearest "
        "surveyed field is more than 20 km away, or for a season that was never tested, the page gives no answer "
        "instead of a low one. Only season B has labelled data, so the page opens on season B.\n"
        "6. **Fertiliser**: for the recommended crop. Until the RAB guideline is in, the rates are interim: "
        "national blanket rates copied from published sources, each shown with its page and link. Nothing is "
        "invented.\n\n"
        "Turn on **Evaluation view** in the sidebar to see both methods and their tested scores.")
st.write("")

today = mm.current_season()
seasons = list(mm.SEASON_NAMES)
left, right = st.columns([1.3, 1], gap="large")
with left:
    with st.container(key="card_point"):
        ui.step(1, "Pick the season")
        # Opens on season B, the season the model and the rules were trained and tested on (CIAT, season B 2025)
        season = st.radio("Season", seasons, index=seasons.index(mm.MODEL_SEASON), horizontal=True,
                          key="pills_season", label_visibility="collapsed",
                          format_func=lambda s: mm.SEASON_NAMES[s] + (", tested" if s == mm.MODEL_SEASON else "")
                          + (", today" if s == today else ""))
        st.write("")
        ui.step(2, "Click a field on the map")
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
        ui.step(3, "Or type its coordinates")
        with st.form("coordinates", border=False):
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=float(st.session_state["point"]["lat"]), format="%.5f",
                                  step=0.01)
            lon = c2.number_input("Longitude", value=float(st.session_state["point"]["lon"]), format="%.5f",
                                  step=0.01)
            if st.form_submit_button("Use these coordinates", type="primary"):
                st.session_state["point"] = {"lat": float(lat), "lon": float(lon)}
                st.rerun()

with right:
    with st.container(key="card_result"):
        ui.head("Recommended crop")
        p = st.session_state["point"]
        res = rec.recommend(p["lat"], p["lon"], season)
        if res["status"] != "ok":
            ui.place(res.get("district") or "Outside Rwanda")
            st.warning(res["message"])
            for w in res["warnings"]:
                st.caption(w)
        else:
            inp = res["inputs"]
            ui.place(f"{res['district']} ({PROVINCES.get(inp['province'], inp['province'])}), point "
                     f"{res['point']['lat']}, {res['point']['lon']}")
            ans = res["answer"]
            blk_km = inp["block"]["distance_km"]
            far = (f" The nearest surveyed field is {blk_km:.1f} km away, so the soil, elevation and satellite "
                   "values come from there." if blk_km > mm.NEAR_KM else "")
            if ans["kind"] == "none":
                title = ("No answer for this season" if res["season"]["asked"] != res["season"]["used"]
                         else "No reliable answer here")
                ui.no_answer(title, ans["message"] + " Please ask the local agronomist.")
            elif ans["kind"] == "two_crops":
                first, second = (crop(c) for c in ans["crops"])
                ui.answer(f"{first} or {second}", ans["confidence"], ans["hit_rate"],
                          f"Both fit this field about equally. In testing, the crop planted was one of these two "
                          f"about {round(ans['hit_rate'] * 10)} times out of 10. Pick between them by the seed you "
                          "have and where you can sell." + far)
            else:
                ui.answer(crop(ans["crop"]), ans["confidence"], ans["hit_rate"],
                          f"In testing, answers like this one were right about {round(ans['hit_rate'] * 10)} times "
                          "out of 10." + far)
            title = {"none": "Why there is no answer", "two_crops": "Why these crops and this confidence"}.get(
                ans["kind"], "Why this crop and this confidence")
            with st.expander(title):
                for line in ans["why"]:
                    st.write(f"- {line}")
                if ans["others"]:
                    st.write("Next in order: " + ", ".join(crop(c) for c in ans["others"]) + ".")
            for c, advice in zip(ans["crops"], (res["fertiliser"], res["fertiliser_second"])):
                st.write("")
                ui.head(f"Fertiliser for {crop(c)}")
                if advice["status"] == "no_matching_rule":
                    st.write(f"No fertiliser rule fits {crop(c)}, so no rate is given.")
                else:
                    show_fertiliser(advice["items"])
            with st.expander("Inputs for this point"):
                blk = inp["block"]
                where = ("the same 2 km block" if blk["distance_km"] == 0
                         else f"the nearest 2 km block, {blk['distance_km']:.1f} km away")
                st.write(f"Soil, elevation and satellite values come from {blk['n_fields']} CIAT field(s) in "
                         f"{where}.")
                soil = inp["soil_0_20cm"]
                st.write(f"Soil, 0 to 20 cm (iSDAsoil): **{inp['soil_texture']}**, pH {soil['ph']:.1f}, organic "
                         f"carbon {soil['carbon_organic']:.1f} g/kg, nitrogen {soil['nitrogen']:.2f} g/kg, "
                         f"phosphorus {soil['phosphorous_extractable']:.1f} mg/kg, potassium "
                         f"{soil['potassium_extractable']:.0f} mg/kg, CEC {soil['cation_exchange_capacity']:.1f} "
                         f"cmol/kg (sand {soil['sand']:.0f}%, silt {soil['silt']:.0f}%, clay {soil['clay']:.0f}%).")
                clim = inp["climate_season_b_2025"]
                st.write(f"Elevation {inp['elevation_m']:.0f} m. Season B 2025 weather for the district (NASA "
                         f"POWER): rainfall {clim['rainfall_mm']:.0f} mm, mean temperature at this height "
                         f"{clim['mean_temp_at_field_c']:.1f} C.")
                sat = inp.get("satellite_at_field") or {}
                if sat:
                    st.write(f"Satellite at the field: SRTM elevation {sat['srtm_elevation_m']:.0f} m, CHIRPS "
                             f"rainfall from February to June 2025 {sat['chirps_precip_mm']:.0f} mm, Sentinel-2 "
                             f"vegetation index of 2024 {sat['ndvi_annual']:.2f}, shortwave ratio of 2024 "
                             f"{sat['swir_ratio']:.2f}.")
                shares = ", ".join(f"{k} {v:.0%}" for k, v in inp["nisr_crop_shares"].items())
                st.write(f"District crop shares (NISR, area of the four crops): {shares}.")
            if res["warnings"]:
                with st.expander("Notes"):
                    for w in res["warnings"]:
                        st.write(f"- {w}")

if res["status"] == "ok" and res["answer"]["kind"] != "none":
    st.write("")
    with st.container(key="card_crops"):
        crop_ui.show(res["answer"]["crops"], res["inputs"].get("elevation_m"))

if evaluation and res["status"] == "ok":
    st.write("")
    with st.container(key="card_eval"):
        ui.head("Evaluation view: both methods at this point")
        st.caption("For the team. Users only see the answer above.")
        st.markdown("**Tested scores (macro-F1, 0 to 1, protocol 1.4)**")
        st.dataframe(tested_table(rec).style.format("{:.2f}", na_rep=""))
        st.caption("New fields: fields from 2 km blocks the model never saw, in a district it knows. District "
                   "left out: the model never saw the district at all. The API default stays the rules, because "
                   "no model passed the adoption rule when a district was left out.")
        top_model, top_rules = res["model"]["ranking"][0], res["rules"]["ranking"][0]
        b1, b2 = st.columns(2)
        b1.caption(f"{mm.MODEL_LABEL} suggests")
        b1.markdown(f"### {crop(top_model['crop'])}")
        b1.caption(f"Model score {top_model['probability']:.2f} out of 1")
        b2.caption("Expert rules suggest")
        b2.markdown(f"### {crop(top_rules['crop'])}")
        b2.caption(f"Rule score {top_rules['score']:.2f} out of 1")
        better = res["better_in_testing"]
        mt, rt = res["model"]["tested"].get("new_fields"), res["rules"]["tested"].get("new_fields")
        hits = rec.hit_rates.get(res["district"], {})
        first = ("" if hits.get("model") is None or hits.get("rules") is None else
                 f" First crop right: {hits['model']:.0%} for the {mm.MODEL_LABEL} and {hits['rules']:.0%} for the "
                 "rules. The one answer for users goes to the method that names the right crop first more often.")
        if better == "model":
            st.success(f"On new fields in {res['district']}, the {mm.MODEL_LABEL} did better than the rules in "
                       f"testing (main score {mt:.2f} against {rt:.2f})." + first)
        elif better == "rules":
            st.warning(f"On new fields in {res['district']}, the rules did better than the {mm.MODEL_LABEL} in "
                       f"testing (main score {rt:.2f} against {mt:.2f})." + first + " Read both rankings.")
        elif better == "same":
            st.info(f"On new fields in {res['district']}, the {mm.MODEL_LABEL} and the rules had about the same "
                    "main score in testing." + first)
        c1, c2 = st.columns(2)
        with c1:
            ranking(mm.MODEL_TITLE, res["model"]["ranking"], "probability", res["model"]["tested"],
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
                       "fewer than 5 fields in the district are not scored. From the third round of tuning (tuning_v3). "
                       "The Model scores page has every model.")
        with st.expander("Why the rules say this"):
            for r in res["rules"]["ranking"]:
                st.markdown(f"**{crop(r['crop'])}** ({r['score']:.2f})")
                for reason in r["reasons"]:
                    st.write(f"- {reason['text']}")
