"""The crop profile under the map's answer ("Get to know the crop"): an animated drawing of the crop, its names in
English, Kinyarwanda and Latin, four short facts, the varieties released in Rwanda with the ones that fit the
clicked field first (each drawn in its own colours), what to watch out for, where farmers get seed and fertiliser
(the Smart Nkunganire System), and the numbered sources (IEEE style).

The data and the fit logic are in 01-crop-recommendation/src/recommendation/crop_profiles.py and its YAML file;
the drawings are in crop_art.py. Everything shown comes from those files, never from what a user typed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

SERVICE_DIR = Path(__file__).resolve().parents[2] / "01-crop-recommendation" / "src" / "recommendation"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
import crop_profiles as cp  # noqa: E402
import crop_art as art  # noqa: E402
from ui_theme import html  # noqa: E402

CSS = """
<style>
.kc-top { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 24px; align-items: center; }
.kc-cap { color: var(--muted, #5f6b5c); font-size: .78rem; margin-top: 6px; text-align: center; }
.kc-name { font-size: 2.1rem; letter-spacing: -.045em; color: var(--navy, #1b2a4a); font-weight: 500; line-height: 1.05; }
.kc-rw { display: flex; align-items: baseline; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
.kc-rw b { font-size: 1.35rem; color: var(--lime, #6a9e12); font-weight: 600; letter-spacing: -.02em; }
.kc-rw span { font-size: .7rem; color: var(--muted, #5f6b5c); text-transform: uppercase; letter-spacing: .1em; }
.kc-sci { font-style: italic; color: var(--muted, #5f6b5c); font-size: .9rem; margin-top: 2px; }
.kc-intro { color: var(--ink, #1f2a1f); line-height: 1.5; margin: 12px 0 14px; font-size: .95rem; }
.kc-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.kc-fact { background: #fff; border: 1px solid var(--line, rgba(31,42,31,.08)); border-radius: 14px; padding: 10px 12px;
           animation: kc-fade .6s ease both; }
.kc-fact b { display: block; font-size: 1.3rem; color: var(--navy, #1b2a4a); letter-spacing: -.02em; font-weight: 600; }
.kc-fact span { font-size: .8rem; color: var(--muted, #5f6b5c); line-height: 1.35; display: block; margin-top: 2px; }
.kc-ref { color: var(--lime, #6a9e12); font-weight: 600; font-size: .72rem; white-space: nowrap; }
.kc-sec { margin: 26px 0 10px; display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
          flex-wrap: wrap; }
.kc-sec b { font-size: 1.1rem; color: var(--ink, #1f2a1f); font-weight: 600; letter-spacing: -.01em; }
.kc-sec span { color: var(--muted, #5f6b5c); font-size: .85rem; }
.kc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.kc-var { background: #fff; border: 1px solid var(--line, rgba(31,42,31,.08)); border-radius: 16px; padding: 12px;
          transition: transform .18s, box-shadow .18s; animation: kc-fade .5s ease both; }
.kc-var:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(31,42,31,.13); }
.kc-var.fit { border-color: #cfe5a8; box-shadow: 0 4px 14px rgba(106,158,18,.10); }
.kc-var.off { opacity: .66; }
.kc-pic { height: 70px; display: flex; align-items: center; justify-content: center; border-radius: 12px;
          background: radial-gradient(circle at 50% 60%, #ffffff 0%, #f1f5ea 75%); margin-bottom: 8px; }
.kc-pic svg { height: 62px; width: auto; }
.kc-vname { font-weight: 600; color: var(--navy, #1b2a4a); font-size: 1.02rem; }
.kc-vname small { color: var(--muted, #5f6b5c); font-weight: 400; margin-left: 4px; font-size: .78rem; }
.kc-tags { margin: 4px 0 2px; }
.kc-tag { display: inline-block; white-space: nowrap; font-size: .7rem; border-radius: 999px; padding: 2px 8px; margin: 3px 4px 0 0;
          background: #eef1ea; color: var(--muted, #5f6b5c); }
.kc-tag.fit { background: #e3f1cc; color: #3f6b0c; font-weight: 600; }
.kc-tag.iron { background: #f7e1dc; color: #8f2f22; font-weight: 600; }
.kc-stats { display: flex; gap: 16px; margin: 8px 0 6px; }
.kc-stats div b { display: block; font-size: 1rem; color: var(--ink, #1f2a1f); font-weight: 600; }
.kc-stats div span { font-size: .7rem; color: var(--muted, #5f6b5c); }
.kc-traits { font-size: .78rem; color: var(--muted, #5f6b5c); line-height: 1.45; }
.kc-note { font-size: .78rem; color: var(--muted, #5f6b5c); margin-top: 10px; line-height: 1.45; }
.kc-row2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }
.kc-box { background: #fff; border: 1px solid var(--line, rgba(31,42,31,.08)); border-radius: 16px; padding: 14px 16px; }
.kc-box b { display: block; font-size: .95rem; color: var(--ink, #1f2a1f); margin-bottom: 6px; }
.kc-box p { margin: 0; font-size: .85rem; color: var(--muted, #5f6b5c); line-height: 1.5; }
.kc-box.sns { background: linear-gradient(135deg, #eef6df 0%, #ffffff 70%); border-color: #d7e8b9; }
.kc-src { font-size: .74rem; color: var(--muted, #5f6b5c); margin-top: 16px; line-height: 1.55; }
.kc-src a { color: var(--muted, #5f6b5c); }
@keyframes kc-fade { from { opacity: 0; translate: 0 8px; } to { opacity: 1; translate: 0 0; } }
@media (max-width: 860px) { .kc-top, .kc-row2 { grid-template-columns: 1fr; } }
@media (max-width: 520px) { .kc-facts { grid-template-columns: 1fr; } .kc-name { font-size: 1.7rem; } }
</style>
"""

CLIMBING_FROM_M = 1700   # the bean scene shows climbing beans at high altitude (RAB's split), bush beans below


def _num(v) -> str:
    return f"{float(v):g}"


def _span(values) -> str:
    """[20, 30] reads '20 to 30'; [56] reads '56'."""
    values = list(values or [])
    return " to ".join(_num(v) for v in values)


def _ref(numbers: dict, source) -> str:
    keys = cp.source_keys(source)
    return '<span class="kc-ref">' + " ".join(f"[{numbers[k]}]" for k in keys if k in numbers) + "</span>"


def _bean_traits(v: dict, prof: dict) -> str:
    names = prof["diseases"]
    seed = prof.get("seed_labels", {}).get(v.get("seed"), "")
    parts = [f"{seed} seed." if seed else ""]
    if v.get("resistant"):
        parts.append("Resists " + _and([names[c] for c in v["resistant"]]) + ".")
    if v.get("tolerant"):
        parts.append("Tolerates " + _and([names[c] for c in v["tolerant"]]) + ".")
    return " ".join(p for p in parts if p)


def _and(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _card(crop: str, v: dict, prof: dict, numbers: dict, uid: str, delay: float) -> str:
    cls = "kc-var " + ("fit" if v["fits"] is True else "off" if v["fits"] is False else "")
    tags, stats, traits = [], [], ""
    if v["fit_note"]:
        tags.append(f'<span class="kc-tag{" fit" if v["fits"] else ""}">{v["fit_note"]}</span>')
    if crop == "beans":
        tags.append(f'<span class="kc-tag">{v["habit"].capitalize()}</span>')
        tags.append('<span class="kc-tag iron">Iron rich</span>')
        stats = [(_span(v["yield_t_ha"]), "t/ha"), (_num(v["iron_ppm"]), "ppm iron")]
        traits = _bean_traits(v, prof)
    elif crop == "irish_potato":
        tags.append(f'<span class="kc-tag">Late blight: {v["late_blight"].lower()}</span>')
        if v.get("yield_t_ha"):
            stats.append((_span(v["yield_t_ha"]), "t/ha"))
        if v.get("maturity_days"):
            stats.append((_span(v["maturity_days"]), "days"))
        skin = prof["skin_labels"].get(v["skin"], "")
        flesh = prof["flesh_labels"].get(v["flesh"], "")
        traits = f"{skin}, {flesh}, {v['shape'].lower()}."
        if v.get("heat_tolerant"):
            traits += " Heat tolerant."
        if v.get("use"):
            traits += f" For {v['use'].lower()}."
        if v.get("where"):
            traits += f" Best in the {v['where'].lower()}."
    elif crop == "maize":
        if v.get("zone_label"):
            tags.append(f'<span class="kc-tag">{v["zone_label"]}</span>')
        if v.get("trial_yield_t_ha"):
            stats.append((_num(v["trial_yield_t_ha"]), "t/ha in trials"))
        if v.get("maturity_days"):
            stats.append((_span(v["maturity_days"]), "days"))
        traits = ". ".join(v.get("traits", [])) + "."
    else:
        if v.get("grain") == "long":
            tags.append('<span class="kc-tag">Long grain</span>')
        traits = ". ".join(v.get("notes", [])) + "."
    year = f'<small>{v["released"]}</small>' if v.get("released") else ""
    stat_html = "".join(f"<div><b>{a}</b><span>{b}</span></div>" for a, b in stats)
    return (f'<div class="{cls.strip()}" style="animation-delay:{delay:.2f}s">'
            f'<div class="kc-pic">{art.variety_picture(crop, v, uid)}</div>'
            f'<div class="kc-vname">{v["name"]}{year}</div><div class="kc-tags">{"".join(tags)}</div>'
            + (f'<div class="kc-stats">{stat_html}</div>' if stats else "")
            + f'<div class="kc-traits">{traits} {_ref(numbers, v["source"])}</div></div>')


def _scene_caption(crop: str, climbing: bool, first: Optional[dict]) -> str:
    if crop == "beans":
        return "Climbing beans on stakes" if climbing else "Bush beans"
    if crop == "irish_potato":
        return f"Potato plants on ridges, with tubers coloured like {first['name']}" if first else "Potato plants on ridges"
    if crop == "maize":
        return "Maize with its tassels and cobs"
    return "Rice in a marshland"


def profile_html(crop: str, elevation_m: Optional[float], uid: str) -> tuple:
    """The HTML of one crop's profile, cut at the varieties so the page can fold the extra ones away:
    (top part with the first variety cards, the other cards or '', bottom part, how many cards are folded)."""
    prof = cp.for_field(crop, elevation_m)
    numbers = {s["key"]: i + 1 for i, s in enumerate(prof["sources"])}
    climbing = elevation_m is None or elevation_m >= CLIMBING_FROM_M
    first = next((v for v in prof["varieties"] if v["fits"]), prof["varieties"][0] if prof["varieties"] else None)
    scene = art.scene(crop, f"{uid}-scene", climbing=climbing,
                      skin=(first or {}).get("skin") if crop == "irish_potato" else None,
                      title=f"Drawing of {prof['name'].lower()}")
    facts = "".join(f'<div class="kc-fact" style="animation-delay:{.15 + i * .12:.2f}s"><b>{f["big"]}</b>'
                    f'<span>{f["text"]} {_ref(numbers, f["source"])}</span></div>' for i, f in enumerate(prof["facts"]))
    names_ref = _ref(numbers, "nisr_2025")
    top = (f'<div class="kc-top"><div>{scene}<div class="kc-cap">{_scene_caption(crop, climbing, first)}. '
           'Drawing, not a photo.</div></div>'
           f'<div><div class="kc-name">{prof["name"]}</div>'
           f'<div class="kc-rw"><b>{prof["kinyarwanda"]}</b><span>in Kinyarwanda</span>{names_ref}</div>'
           f'<div class="kc-sci">{prof["scientific"]}</div><div class="kc-intro">{prof["intro"]}</div>'
           f'<div class="kc-facts">{facts}</div></div></div>')
    if crop == "rice" or elevation_m is None:
        sub = f"{len(prof['varieties'])} varieties"
    else:
        sub = f"{prof['fitting']} of {len(prof['varieties'])} fit a field at {elevation_m:,.0f} m"
    head = f'<div class="kc-sec"><b>Varieties for Rwanda</b><span>{sub}</span></div>'
    cards = [_card(crop, v, prof, numbers, f"{uid}-v{i}", .05 + min(i, 8) * .06) for i, v in enumerate(prof["varieties"])]
    shown = prof.get("shown_first") or len(cards)
    zone_note = ("" if crop == "rice" else
                 " Altitude zones in metres follow RAB's split for maize: mid altitude 900 to 1,700 m, high altitude "
                 "above 1,700 m.")
    note = f'<div class="kc-note">{prof["varieties_note"]}{zone_note}</div>'
    grid = f'{head}<div class="kc-grid">{"".join(cards[:shown])}</div>'
    extra = f'<div class="kc-grid">{"".join(cards[shown:])}</div>' if len(cards) > shown else ""
    watch = prof["watch_out"]
    inputs = prof["inputs"]
    bottom = (note + '<div class="kc-row2">'
              f'<div class="kc-box"><b>Watch out for</b><p>{watch["text"]} {_ref(numbers, watch["source"])} '
              'For a check from a leaf photo, open Diagnosis in the top bar.</p></div>'
              f'<div class="kc-box sns"><b>{inputs["title"]}</b><p>{inputs["text"]} {_ref(numbers, inputs["source"])}'
              '</p></div></div>'
              '<div class="kc-src"><b>Sources</b><br>'
              + "<br>".join(f'[{numbers[s["key"]]}] {s["text"]} <a href="{s["url"]}" target="_blank" '
                            'rel="noopener">Link</a>' for s in prof["sources"])
              + "</div>")
    return top + grid, extra, bottom, len(cards) - shown


def _one(crop: str, elevation_m: Optional[float], uid: str) -> None:
    top, extra, bottom, hidden = profile_html(crop, elevation_m, uid)
    html(top)
    if extra:
        with st.expander(f"{hidden} more varieties"):
            html(extra)
    html(bottom)


def show(crops: list, elevation_m: Optional[float] = None) -> None:
    """The profile of the recommended crop, or one tab per crop when the map gives two."""
    html(art.CSS + CSS)
    names = [cp.load()["crops"][c]["name"] for c in crops]
    title = f"Get to know {names[0].lower() if names[0] != 'Irish potato' else 'Irish potato'}" if len(crops) == 1 \
        else "Get to know both crops"
    html(f'<div class="card-head">{title} <b>···</b></div>')
    if len(crops) == 1:
        _one(crops[0], elevation_m, f"kc-{crops[0]}")
        return
    for tab, crop in zip(st.tabs([f"{n}  ({cp.load()['crops'][c]['kinyarwanda']})" for n, c in zip(names, crops)]), crops):
        with tab:
            _one(crop, elevation_m, f"kc-{crop}")
