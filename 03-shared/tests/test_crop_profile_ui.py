"""Tests for the crop profile drawings and HTML (crop_art.py, crop_ui.py). Run from the repo root:
python -m pytest 03-shared/tests -q"""
import re
import sys
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASH))
pytest.importorskip("streamlit")
import crop_art as art  # noqa: E402
import crop_ui  # noqa: E402

CROPS = ("beans", "irish_potato", "maize", "rice")
ANIMATED = re.compile(r'<[^>]*class="kc-(?:grow|sway|pop|pop-c|sun|ripple|cloud|drop)"[^>]*>')


def _all_pictures():
    out = [art.scene(c, f"t-{c}", climbing=climbing) for c in CROPS for climbing in (True, False)]
    for crop in CROPS:
        prof = crop_ui.cp.for_field(crop, 2000)
        out += [art.variety_picture(crop, v, f"t-{crop}-{i}") for i, v in enumerate(prof["varieties"])]
    return out


def test_animated_parts_never_carry_their_own_transform():
    """A CSS animation would replace an SVG transform attribute and throw the part to the corner."""
    for svg in _all_pictures():
        for tag in ANIMATED.findall(svg):
            assert "transform=" not in tag, tag


def test_two_scenes_on_one_page_share_no_ids():
    ids = [set(re.findall(r'id="([^"]+)"', art.scene(c, f"kc-{c}"))) for c in ("beans", "maize")]
    assert ids[0] and ids[1] and not ids[0] & ids[1]


def test_every_variety_has_a_picture_in_its_own_colours():
    for crop in CROPS:
        for v in crop_ui.cp.for_field(crop, 2000)["varieties"]:
            assert art.variety_picture(crop, v, "x").startswith("<svg")
    assert art.SKIN["red"][0] in art.potato_cut("red", "yellow") and art.FLESH["yellow"] in art.potato_cut("red", "yellow")
    assert art.SEED["salmon"][0] in art.bean_seeds("s", "salmon")


def test_potato_scene_takes_the_skin_of_the_first_variety_that_fits():
    top, _, _, _ = crop_ui.profile_html("irish_potato", 2284, "t")
    assert art.SKIN["red"][0] in top and "tubers coloured like Kinigi" in top       # Kinigi comes first


def test_bean_scene_shows_climbing_beans_high_and_bush_beans_lower():
    assert "Climbing beans on stakes" in crop_ui.profile_html("beans", 2284, "a")[0]
    assert "Bush beans" in crop_ui.profile_html("beans", 1620, "b")[0]


def test_profile_has_names_fitting_varieties_and_numbered_sources():
    top, extra, bottom, hidden = crop_ui.profile_html("irish_potato", 2284, "t")
    assert "Ibirayi" in top and "Solanum tuberosum" in top and "15 of 15 fit a field at 2,284 m" in top
    assert top.count('class="kc-var fit"') == 8 and hidden == 7 and extra.count("kc-var") == 7
    assert "Smart Nkunganire System" in bottom and "[1] D. Borus et al." in bottom
    listed = bottom.count('rel="noopener">Link</a>')
    body = top + extra + bottom.split('<div class="kc-src">')[0]
    assert {int(n) for n in re.findall(r"\[(\d+)\]", body)} == set(range(1, listed + 1))    # every source is cited


@pytest.mark.parametrize("crop", CROPS)
def test_no_long_dashes_or_arrows_in_what_the_page_shows(crop):
    text = "".join(crop_ui.profile_html(crop, 1600, "t")[:3])
    for ch in ("—", "–", "→", "⇒"):
        assert ch not in text
