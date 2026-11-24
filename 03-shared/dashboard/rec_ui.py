"""Parts of the crop recommendation pages (Map, Model scores, Crop and fertiliser) in the shared look of ui_theme.py.

The pages put their content in frosted cards (containers keyed card_*), number their steps and show a result the
same way as the Diagnosis page: a coloured badge, the answer as the card title, then a bar. The badge and the bar
use the same class names and colours as the Diagnosis page, so an answer looks the same on every page. All text
passed here comes from the pages themselves, never from what a user typed.
"""
from __future__ import annotations

from typing import Optional

from ui_theme import html

CSS = """
<style>
.result-badge { display: inline-block; font-size: .78rem; font-weight: 600; border-radius: 999px; padding: 4px 12px; }
.result-badge.ok { background: #e3f1cc; color: #3f6b0c; }
.result-badge.alert { background: #fbe7c6; color: #8a5300; }
.result-badge.unsure { background: #e1e8f5; color: var(--navy, #1b2a4a); }
.result-badge.retake { background: #f8d9d3; color: #9b2c1a; }
.confidence { height: 10px; border-radius: 999px; background: #fff; border: 1px solid var(--line, rgba(31,42,31,.08));
              overflow: hidden; margin: 6px 0 4px; }
.confidence > div { height: 100%; border-radius: 999px;
                    background: linear-gradient(90deg, var(--lime-soft, #a8d05a), var(--lime, #6a9e12)); }
.muted { color: var(--muted, #5f6b5c); font-size: .85rem; }
.place { color: var(--muted, #5f6b5c); font-size: .9rem; margin: 4px 0 12px; }
</style>
"""

# Confidence level of the answer: colour of the badge (the Diagnosis page's classes) and its words
LEVEL_BADGE = {"high": ("ok", "High"), "medium": ("alert", "Medium"), "low": ("retake", "Low")}


def styles() -> None:
    html(CSS)


def step(n: int, text: str) -> None:
    """Numbered step heading inside a card, as on the Diagnosis page."""
    html(f'<div class="card-head"><span><span class="step-n">{n}</span>{text}</span></div>')


def head(text: str) -> None:
    """Card heading, as the 'Result' heading of the Diagnosis page."""
    html(f'<div class="card-head">{text} <b>···</b></div>')


def place(text: str) -> None:
    html(f'<div class="place">{text}</div>')


def answer(crop_name: str, level: str, share: Optional[float], note: str) -> None:
    """The one answer: confidence badge, the crop as the card title, the tested hit rate as a bar, then the note."""
    cls, words = LEVEL_BADGE[level]
    bar = f'<div class="confidence"><div style="width:{share:.0%}"></div></div>' if share is not None else ""
    html(f'<span class="result-badge {cls}">{words} confidence</span>'
         f'<div class="card-title" style="margin-top:12px">{crop_name}</div>{bar}'
         f'<div class="muted">{note}</div>')
