"""Shared look of the dashboard pages: crop-row background, frosted panel, top navigation bar
(pill buttons, current page filled), cards and form widgets.

app.py calls apply(pages, current) once, before running the page. The sidebar stays for the
settings (data source, Evaluation view). Styles hook onto keyed containers and widgets
(key=... gives the element a `st-key-<key>` class):
  card_*    frosted card            pills_*   radio drawn as pill buttons
"""
import streamlit as st

# Crop rows background with a faint wordmark on top, as one CSS background (no extra elements)
WORDMARK = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 300'>"
    "<text x='800' y='255' text-anchor='middle' font-family='Inter,Arial,sans-serif' font-size='290' "
    "font-weight='500' letter-spacing='-10' fill='white' fill-opacity='0.16'>Agri DSS</text></svg>\")"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root { --lime: #6a9e12; --lime-soft: #a8d05a; --ink: #1f2a1f; --navy: #1b2a4a; --muted: #5f6b5c;
        --glass: rgba(255,255,255,.62); --line: rgba(31,42,31,.08); }
.stApp {
  background: WORDMARK top center / 100% auto no-repeat,
              radial-gradient(ellipse at 50% 40%, rgba(0,0,0,0) 30%, rgba(0,0,0,.45) 100%),
              repeating-linear-gradient(90deg, #1d4a20 0 7px, #2e6b2c 7px 11px, #163a18 11px 15px);
}
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stPageLink"] p,
.stApp [data-testid="stWidgetLabel"] p, .stApp button p { font-family: 'Inter', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] button, [data-testid="stHeader"] a, [data-testid="stHeader"] span { color: #fff !important; }
[data-testid="stMainBlockContainer"] {
  width: min(1240px, calc(100% - 64px)); max-width: none; margin: clamp(56px, 9vw, 130px) auto 48px; padding: 28px 40px 40px;
  background: linear-gradient(160deg, #f6f8f3 0%, #e9eee3 100%);
  border-radius: 28px; box-shadow: 0 30px 80px rgba(0,0,0,.35);
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { width: calc(100% - 32px); margin: 56px 16px 24px; padding: 20px 16px 28px; border-radius: 20px; }
}
/* top bar */
.brand { display: flex; align-items: center; gap: 10px; font-weight: 600; color: var(--ink); font-size: 1.1rem; padding-top: 6px; }
.brand-mark { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
              background: linear-gradient(140deg, var(--lime-soft), var(--lime)); font-size: 16px; }
/* nav links drawn as pill buttons; the current page (its own keyed container) is filled */
.st-key-nav { width: fit-content; margin: 0 auto; }
.st-key-nav a { background: #fff; border: 1px solid var(--line); border-radius: 999px; padding: 7px 18px;
                box-shadow: 0 2px 6px rgba(31,42,31,.08); transition: transform .15s, box-shadow .15s, border-color .15s; }
.st-key-nav a:hover { background: #fff; border-color: var(--lime); transform: translateY(-1px);
                      box-shadow: 0 6px 14px rgba(31,42,31,.14); }
.st-key-nav a p { font-weight: 500; }
.st-key-nav_current a, .st-key-nav_current a:hover { background: var(--ink); border-color: var(--ink); }
.st-key-nav_current a p { color: #fff !important; }
@media (max-width: 640px) {
  .st-key-nav { flex-wrap: nowrap; gap: 4px; }
  .st-key-nav a { padding: 6px 10px; }
  .st-key-nav a p { font-size: .82rem; }
}
.pilot { text-align: right; padding-top: 10px; color: var(--muted); font-size: .85rem; }
.pilot span { background: #fff; border: 1px solid var(--line); border-radius: 999px; padding: 6px 12px; }
/* hero */
.eyebrow { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: .9rem; margin-top: 28px; }
.eyebrow::before { content: ""; width: 8px; height: 8px; border-radius: 2px; background: var(--lime); }
.hero-title { font-size: clamp(2.3rem, 4.2vw, 3.4rem); line-height: 1.02; letter-spacing: -.045em;
              font-weight: 500; color: var(--ink); margin: 14px 0 16px; }
.hero-title em { font-style: normal; color: var(--lime); }
.hero-sub { color: var(--muted); max-width: 34rem; line-height: 1.55; }
/* cards */
[class*="st-key-card_"] { background: var(--glass); border: 1px solid rgba(255,255,255,.8); border-radius: 20px;
                          padding: 20px 20px 16px; box-shadow: 0 10px 30px rgba(31,42,31,.08); height: 100%; }
[class*="st-key-card_"] a { font-weight: 500; }
.card-head { display: flex; justify-content: space-between; align-items: center; color: var(--ink); font-weight: 500; }
.card-head b { letter-spacing: .15em; color: var(--muted); font-weight: 600; }
.card-title { font-size: 1.65rem; letter-spacing: -.035em; color: var(--navy); font-weight: 500; line-height: 1.15; margin: 4px 0 10px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0; }
.chip { font-size: .78rem; padding: 5px 11px; border-radius: 8px; background: #fff; border: 1px solid var(--line); color: var(--ink); }
.chip.on { background: var(--lime); border-color: var(--lime); color: #fff; }
.step-n { display: inline-grid; place-items: center; width: 24px; height: 24px; margin-right: 8px; border-radius: 7px;
          background: #fff; border: 1px solid var(--line); color: var(--lime); font-size: .78rem; font-weight: 600; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; color: var(--ink); }
.badge i { font-style: normal; background: var(--lime); color: #fff; border-radius: 5px; padding: 1px 5px; font-size: .7rem; }
.foot { color: var(--muted); font-size: .8rem; text-align: center; margin-top: 18px; }
/* form widgets */
[class*="st-key-pills_"] [role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
[class*="st-key-pills_"] [role="radiogroup"] label { background: #fff; border: 1px solid var(--line); border-radius: 999px;
    padding: 7px 16px 7px 12px; margin: 0; box-shadow: 0 2px 6px rgba(31,42,31,.06); cursor: pointer; }
[class*="st-key-pills_"] [role="radiogroup"] label:has(input:checked) { background: var(--ink); border-color: var(--ink); }
[class*="st-key-pills_"] [role="radiogroup"] label:has(input:checked) p { color: #fff; }
[data-testid="stBaseButton-primary"] { background: var(--ink); border-color: var(--ink); border-radius: 999px; padding: 10px 22px; }
[data-testid="stBaseButton-primary"]:hover { background: #33443a; border-color: #33443a; }
[data-testid="stBaseButton-primary"]:disabled { background: #c9d1c4; border-color: #c9d1c4; }
[data-testid="stFileUploaderDropzone"] { background: #fff; border: 1px dashed var(--lime-soft); border-radius: 14px; }
[data-testid="stExpander"] details { background: rgba(255,255,255,.7); border-color: var(--line); border-radius: 14px; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background-color: #fff !important; border-radius: 12px; }
[data-testid="stImage"] img { border-radius: 14px; }
/* headings and bordered containers written by the pages with plain Streamlit */
[data-testid="stMainBlockContainer"] h1, [data-testid="stMainBlockContainer"] h2, [data-testid="stMainBlockContainer"] h3 {
  font-family: 'Inter', sans-serif; color: var(--ink); letter-spacing: -.03em; font-weight: 500; }
[data-testid="stMainBlockContainer"] h2 { font-size: clamp(1.9rem, 3.4vw, 2.6rem); }
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] { background: var(--glass); border-radius: 18px; }
[data-testid="stSidebar"] { background: #eef1ea; }
</style>
""".replace("WORDMARK", WORDMARK)


def html(markup: str) -> None:
    """Raw HTML/CSS/SVG. st.html sanitizes styles and SVG away, so this goes through st.markdown;
    blank lines are dropped because markdown would end the HTML block at one."""
    st.markdown("\n".join(line for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def apply(pages: dict, current) -> None:
    """Page styles plus the top bar: brand, one pill button per page (current page filled), pilot badge.
    `pages` maps a short button label to its st.Page; `current` is the page st.navigation returned."""
    html(CSS)
    brand_col, nav_col, pilot_col = st.columns([1, 3, 1], vertical_alignment="center")
    with brand_col:
        html('<div class="brand"><div class="brand-mark">🌱</div>Agri DSS</div>')
    with nav_col:
        with st.container(key="nav", horizontal=True, gap="small"):
            for label, page in pages.items():
                if page.url_path == current.url_path:
                    with st.container(key="nav_current", width="content"):
                        st.page_link(page, label=label)
                else:
                    st.page_link(page, label=label)
    with pilot_col:
        html('<div class="pilot"><span>📍 Rwanda pilot</span></div>')


def hero(eyebrow: str, title: str, sub: str) -> None:
    """Page heading; `title` may wrap a word in <em> to colour it."""
    html(f'<div class="eyebrow">{eyebrow}</div><div class="hero-title">{title}</div><p class="hero-sub">{sub}</p>')
