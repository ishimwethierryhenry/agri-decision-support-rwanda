"""Drawings for the crop profiles on the map page: one animated scene per crop and one small picture per variety,
drawn in the variety's real colours from the profiles file (bean seed colour, potato skin and flesh, maize kernel
type, rice grain length).

Every picture is an original SVG drawing, so the page works offline and needs no photo licence. The animations
are CSS only (plants grow, leaves sway, season B rain falls, pods and tubers appear) and they stop for anyone whose
system asks for reduced motion. Element ids carry a prefix per picture, so two scenes on one page never clash.

A CSS animation replaces an SVG transform attribute, so an animated group never has its own transform: it sits
inside a positioned group instead.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

W, H = 400, 250

CSS = """
<style>
.kc-scene { width: 100%; height: auto; display: block; border-radius: 18px; box-shadow: 0 8px 24px rgba(31,42,31,.10); }
.kc-grow { transform-box: fill-box; transform-origin: 50% 100%; animation: kc-grow 1.9s cubic-bezier(.2,.8,.2,1) both; }
.kc-sway { transform-box: fill-box; transform-origin: 50% 100%; animation: kc-sway 5.5s ease-in-out infinite; }
.kc-pop { transform-box: fill-box; transform-origin: 50% 0%; animation: kc-pop .8s cubic-bezier(.3,1.5,.5,1) both; }
.kc-pop-c { transform-box: fill-box; transform-origin: 50% 50%; animation: kc-pop .8s cubic-bezier(.3,1.5,.5,1) both; }
.kc-drop { animation: kc-drop 1.5s linear infinite; }
.kc-cloud { animation: kc-drift 16s ease-in-out infinite alternate; }
.kc-sun { transform-box: fill-box; transform-origin: 50% 50%; animation: kc-glow 4s ease-in-out infinite; }
.kc-ripple { transform-box: fill-box; transform-origin: 50% 50%; animation: kc-ripple 3.4s ease-out infinite; }
@keyframes kc-grow { from { transform: scaleY(.03); opacity: 0; } 25% { opacity: 1; } to { transform: scaleY(1); opacity: 1; } }
@keyframes kc-sway { 0%, 100% { transform: rotate(-1.6deg); } 50% { transform: rotate(1.6deg); } }
@keyframes kc-pop { from { transform: scale(0); opacity: 0; } to { transform: scale(1); opacity: 1; } }
@keyframes kc-drop { from { transform: translateY(-8px); opacity: 0; } 15% { opacity: .85; } to { transform: translateY(104px); opacity: 0; } }
@keyframes kc-drift { from { transform: translateX(-10px); } to { transform: translateX(16px); } }
@keyframes kc-glow { 0%, 100% { transform: scale(1); opacity: .45; } 50% { transform: scale(1.14); opacity: .8; } }
@keyframes kc-ripple { from { transform: scale(.35); opacity: .8; } to { transform: scale(1.7); opacity: 0; } }
@media (prefers-reduced-motion: reduce) {
  .kc-scene *, .kc-var, .kc-fact { animation: none !important; }
}
</style>
"""


# ----------------------------------------------------------------------------------------------- small helpers
def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _smooth(points: list) -> str:
    """A smooth path through the points (Catmull-Rom turned into cubic Bezier segments)."""
    d = f"M{_f(points[0][0])} {_f(points[0][1])}"
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i else points[i]
        p1, p2 = points[i], points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{_f(c1[0])} {_f(c1[1])} {_f(c2[0])} {_f(c2[1])} {_f(p2[0])} {_f(p2[1])}"
    return d


def _at(x: float, y: float, angle: float, inner: str, cls: str = "", delay: Optional[float] = None) -> str:
    """`inner` drawn at (x, y) turned by `angle` degrees; an animation class goes on an inner group."""
    if cls:
        style = f' style="animation-delay:{delay:.2f}s"' if delay is not None else ""
        inner = f'<g class="{cls}"{style}>{inner}</g>'
    return f'<g transform="translate({_f(x)} {_f(y)}) rotate({_f(angle)})">{inner}</g>'


def _leaf(length: float, width: float, fill: str, vein: str) -> str:
    """A pointed leaf from (0, 0) upward."""
    l, w = length, width
    return (f'<path d="M0 0 C{_f(w)} {_f(-l * .25)} {_f(w * .8)} {_f(-l * .78)} 0 {_f(-l)} '
            f'C{_f(-w * .8)} {_f(-l * .78)} {_f(-w)} {_f(-l * .25)} 0 0Z" fill="{fill}"/>'
            f'<path d="M0 0 L0 {_f(-l * .9)}" stroke="{vein}" stroke-width=".8" fill="none" opacity=".7"/>')


def _leaflet(length: float, fill: str) -> str:
    """A broad bean leaflet (wide near the base, short tip) from (0, 0) upward."""
    l, w = length, length * .5
    return (f'<path d="M0 0 C{_f(w * 1.2)} {_f(-l * .08)} {_f(w * 1.0)} {_f(-l * .72)} 0 {_f(-l)} '
            f'C{_f(-w * 1.0)} {_f(-l * .72)} {_f(-w * 1.2)} {_f(-l * .08)} 0 0Z" fill="{fill}"/>'
            f'<path d="M0 0 L0 {_f(-l * .88)}" stroke="#3f7322" stroke-width=".8" fill="none" opacity=".6"/>')


def _trifoliate(size: float, shade: int = 0) -> str:
    """A bean leaf: three broad leaflets on a short stalk, from (0, 0) upward."""
    fills = ("#5f9e34", "#6aab3c", "#558f2e")
    fill = fills[shade % 3]
    side = size * .84
    return (f'<path d="M0 0 L0 {_f(-size * .35)}" stroke="#4d8429" stroke-width="1.3"/>'
            + _at(0, -size * .35, -58, _leaflet(side, fill))
            + _at(0, -size * .35, 58, _leaflet(side, fill))
            + _at(0, -size * .35, 0, _leaflet(size, fill)))


def _pod(length: float) -> str:
    """A bean pod hanging from (0, 0)."""
    l = length
    return (f'<path d="M-1.5 0 C4 {_f(l * .3)} 4.5 {_f(l * .7)} 1 {_f(l)} C-.5 {_f(l * 1.04)} -3 {_f(l * .98)} '
            f'-3.2 {_f(l * .88)} C-1 {_f(l * .62)} -1.8 {_f(l * .3)} -3.5 0Z" fill="#8cc152" stroke="#5f8f2f" '
            'stroke-width=".7"/>')


def _bean_flower() -> str:
    return ('<ellipse cx="-2.2" cy="0" rx="2.6" ry="3.4" fill="#e9dcf5"/><ellipse cx="2.2" cy="0" rx="2.6" ry="3.4" '
            'fill="#f6f0fb"/><circle cx="0" cy="1.4" r="1.2" fill="#c8a8e0"/>')


def _drops(xs: Iterable[float], y: float) -> str:
    out = []
    for i, x in enumerate(xs):
        out.append(f'<line class="kc-drop" style="animation-delay:{(i * .37) % 1.5:.2f}s" x1="{_f(x)}" '
                   f'y1="{_f(y)}" x2="{_f(x - 1.5)}" y2="{_f(y + 8)}" stroke="#7fb3d1" stroke-width="1.6" '
                   'stroke-linecap="round"/>')
    return "".join(out)


# ----------------------------------------------------------------------------------------------- backdrop
def _backdrop(u: str, ground: str) -> str:
    """Sky, sun, a rain cloud of season B and the terraced hills of Rwanda; `ground` is drawn on top."""
    mid_hill = ("M0 160 C45 140 95 138 140 150 C190 164 240 140 290 136 C335 132 370 146 400 152 "
                "L400 250 L0 250Z")
    terraces = "".join(
        f'<path d="M0 {160 + 9 * k} C45 {140 + 9 * k} 95 {138 + 9 * k} 140 {150 + 9 * k} C190 {164 + 9 * k} '
        f'240 {140 + 9 * k} 290 {136 + 9 * k} C335 {132 + 9 * k} 370 {146 + 9 * k} 400 {152 + 9 * k}" '
        'stroke="#86b357" stroke-width="1.3" fill="none" opacity=".75"/>' for k in range(1, 6))
    cloud = ('<g class="kc-cloud"><ellipse cx="86" cy="50" rx="34" ry="14" fill="#fff"/>'
             '<ellipse cx="66" cy="54" rx="22" ry="11" fill="#fff"/><ellipse cx="108" cy="55" rx="24" ry="10" '
             'fill="#f1f5f7"/><ellipse cx="86" cy="60" rx="38" ry="7" fill="#e4ebef"/>'
             + _drops([56, 64, 72, 80, 88, 96, 104, 112, 120, 60, 76, 92, 108], 66) + "</g>")
    small_cloud = ('<g class="kc-cloud" style="animation-delay:-6s"><ellipse cx="228" cy="34" rx="22" ry="8" '
                   'fill="#fff" opacity=".9"/><ellipse cx="244" cy="38" rx="16" ry="6" fill="#fff" '
                   'opacity=".9"/></g>')
    return (f'<defs><linearGradient id="{u}-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#cfe7f0"/>'
            f'<stop offset="1" stop-color="#f4f7ee"/></linearGradient>'
            f'<clipPath id="{u}-clip"><rect width="{W}" height="{H}" rx="18"/></clipPath>'
            f'<clipPath id="{u}-hill"><path d="{mid_hill}"/></clipPath></defs>'
            f'<g clip-path="url(#{u}-clip)"><rect width="{W}" height="{H}" fill="url(#{u}-sky)"/>'
            '<circle class="kc-sun" cx="338" cy="50" r="30" fill="#ffe7a1"/><circle cx="338" cy="50" r="17" '
            'fill="#ffcc4a"/>' + small_cloud + cloud
            + '<path d="M0 132 C50 104 105 100 150 118 C190 134 225 112 270 100 C320 88 365 104 400 116 '
              'L400 250 L0 250Z" fill="#bcd89b"/>'
            + f'<path d="{mid_hill}" fill="#9cc66f"/><g clip-path="url(#{u}-hill)">{terraces}</g>'
            + ground + "</g>")


def _soil(g: float) -> str:
    return (f'<path d="M0 {g} Q100 {g - 4} 200 {g} T400 {g} L400 250 L0 250Z" fill="#8b5b37"/>'
            f'<path d="M0 {g + 11} Q100 {g + 7} 200 {g + 11} T400 {g + 11}" stroke="#774c2d" stroke-width="2" '
            f'fill="none"/><path d="M0 {g + 24} Q100 {g + 20} 200 {g + 24} T400 {g + 24}" stroke="#774c2d" '
            'stroke-width="2" fill="none" opacity=".7"/>')


# ----------------------------------------------------------------------------------------------- the four crops
def _beans_climbing(g: float) -> str:
    out = []
    for i, (x, top, lean) in enumerate(((112, 66, -4), (200, 54, 3), (288, 70, -2))):
        d0 = i * .3
        stake = (f'<line x1="{x}" y1="{g + 3}" x2="{x + lean}" y2="{top}" stroke="#8b6a43" stroke-width="4" '
                 'stroke-linecap="round"/>')
        pts = []
        for k in range(13):
            t = k / 12
            y = g - t * (g - top - 4)
            pts.append((x + lean * t + 6.5 * math.sin(t * math.pi * 4.2), y))
        vine = f'<path d="{_smooth(pts)}" stroke="#4f8a2a" stroke-width="2.3" fill="none" stroke-linecap="round"/>'
        leaves, extras = [], []
        for j, t in enumerate((.16, .3, .44, .58, .72, .86)):
            px = x + lean * t + 6.5 * math.sin(t * math.pi * 4.2)
            py = g - t * (g - top - 4)
            side = -1 if j % 2 else 1
            leaves.append(_at(px, py, side * (58 + 6 * (j % 3)), _trifoliate(19 - j * .8, j)))
        for j, t in enumerate((.28, .5, .66)):
            px = x + lean * t + 6.5 * math.sin(t * math.pi * 4.2)
            py = g - t * (g - top - 4)
            extras.append(_at(px + (4 if j % 2 else -4), py, 8 if j % 2 else -8, _pod(19 - j), "kc-pop",
                              2.0 + d0 + j * .25))
        for j, t in enumerate((.8, .94)):
            px = x + lean * t + 6.5 * math.sin(t * math.pi * 4.2)
            py = g - t * (g - top - 4)
            extras.append(_at(px + 5, py, 0, _bean_flower(), "kc-pop-c", 2.4 + d0 + j * .2))
        plant = (f'<g class="kc-grow" style="animation-delay:{d0:.2f}s"><g class="kc-sway" '
                 f'style="animation-delay:{-d0 * 3:.2f}s">{vine}{"".join(leaves)}{"".join(extras)}</g></g>')
        out.append(stake + plant)
    return _soil(g) + "".join(out)


def _beans_bush(g: float) -> str:
    out = []
    for i, x in enumerate((84, 164, 244, 324)):
        d0 = i * .22
        cx, cy = x, g - 12
        leaves, extras = [], []
        for k in range(9):
            a = -84 + k * 21
            px = cx + 25 * math.sin(math.radians(a))
            py = cy - 23 * math.cos(math.radians(a))
            leaves.append(f'<path d="M{x} {g} Q{_f(x + (px - x) * .2)} {_f(g - 18)} {_f(px)} {_f(py)}" '
                          'stroke="#4d8429" stroke-width="1.5" fill="none"/>'
                          + _at(px, py, a, _trifoliate(17 + (k % 3) * 1.5, k)))
        for j, (dx, dy, a) in enumerate(((-13, -10, -16), (5, -12, 10), (16, -7, 22))):
            extras.append(_at(x + dx, g + dy, a, _pod(15), "kc-pop", 2.0 + d0 + j * .2))
        extras.append(_at(x - 18, g - 26, 0, _bean_flower(), "kc-pop-c", 2.5 + d0))
        out.append(f'<g class="kc-grow" style="animation-delay:{d0:.2f}s"><g class="kc-sway" '
                   f'style="animation-delay:{-d0 * 3:.2f}s">{"".join(leaves)}{"".join(extras)}</g></g>')
    return _soil(g) + "".join(out)


def _potato_leaf(length: float) -> str:
    """A compound potato leaf from (0, 0) upward: leaflets in pairs and one at the tip."""
    out = [f'<path d="M0 0 L0 {_f(-length)}" stroke="#4d8429" stroke-width="1.4"/>']
    for k, t in enumerate((.35, .6, .82)):
        s = 8.5 - k * 1.2
        out.append(_at(0, -length * t, -62, _leaf(s, s * .55, "#5b9a31", "#3f7322")))
        out.append(_at(0, -length * t, 62, _leaf(s, s * .55, "#64a538", "#3f7322")))
    out.append(_at(0, -length, 0, _leaf(9.5, 5.4, "#5b9a31", "#3f7322")))
    return "".join(out)


def _potato_flower() -> str:
    petals = "".join(_at(0, 0, a, '<ellipse cx="0" cy="-3.4" rx="2.3" ry="3.4" fill="#fbf8ff"/>')
                     for a in range(0, 360, 72))
    return petals + '<circle cx="0" cy="0" r="1.6" fill="#f2c522"/>'


def _tuber(rx: float, ry: float, skin: str, edge: str, spots: bool) -> str:
    out = (f'<ellipse cx="0" cy="0" rx="{_f(rx)}" ry="{_f(ry)}" fill="{skin}" stroke="{edge}" stroke-width=".8"/>'
           f'<path d="M{_f(-rx * .35)} {_f(-ry * .2)} q1.5 -1.4 3 0 M{_f(rx * .25)} {_f(ry * .15)} q1.5 -1.4 3 0" '
           f'stroke="{edge}" stroke-width=".9" fill="none"/>'
           f'<ellipse cx="{_f(-rx * .3)}" cy="{_f(-ry * .4)}" rx="{_f(rx * .35)}" ry="{_f(ry * .22)}" fill="#fff" '
           'opacity=".25"/>')
    if spots:
        out += (f'<circle cx="{_f(rx * .45)}" cy="{_f(-ry * .25)}" r="1.4" fill="#7d4a8f"/>'
                f'<circle cx="{_f(-rx * .1)}" cy="{_f(ry * .45)}" r="1.2" fill="#7d4a8f"/>')
    return out


def _potato(g: float, skin: str) -> str:
    col, edge, spots = SKIN.get(skin, SKIN["red"])
    soil = (f'<path d="M0 {g} L400 {g} L400 250 L0 250Z" fill="#7d5132"/>'
            f'<path d="M0 {g + 30} Q100 {g + 26} 200 {g + 30} T400 {g + 30} L400 250 L0 250Z" fill="#6c4529"/>')
    out = [soil]
    for i, x in enumerate((106, 200, 294)):
        d0 = i * .3
        ridge = f'<path d="M{x - 42} {g} Q{x} {g - 15} {x + 42} {g}Z" fill="#8b5b37"/>'
        stems, leaves = [], []
        for j, (dx, h, a) in enumerate(((-14, 46, -18), (0, 58, -4), (13, 50, 14))):
            sx, sy = x + dx * .3, g - 8
            tip = (x + dx + h * math.sin(math.radians(a)) * .4, g - 8 - h)
            stems.append(f'<path d="M{_f(sx)} {_f(sy)} Q{_f(x + dx)} {_f(sy - h * .5)} {_f(tip[0])} {_f(tip[1])}" '
                         'stroke="#4d8429" stroke-width="2" fill="none"/>')
            for k, t in enumerate((.35, .62, .88)):
                px = sx + (tip[0] - sx) * t
                py = sy + (tip[1] - sy) * t
                side = -1 if (j + k) % 2 else 1
                leaves.append(_at(px, py, side * (48 + 10 * k) + a * .5, _potato_leaf(17 - k * 2)))
            leaves.append(_at(tip[0], tip[1] - 2, 0, _potato_flower(), "kc-pop-c", 2.3 + d0 + j * .2))
        tubers, stolons = [], []
        for j, (dx, dy, rx, ry) in enumerate(((-24, 22, 11, 8), (2, 34, 13, 9), (26, 20, 10, 7.5))):
            tx, ty = x + dx, g + dy
            stolons.append(f'<path d="M{x} {g - 2} Q{_f((x + tx) / 2)} {_f(g + dy * .2)} {_f(tx)} {_f(ty - ry * .6)}" '
                           'stroke="#c9b28a" stroke-width="1.1" fill="none" opacity=".9"/>')
            tubers.append(_at(tx, ty, (j - 1) * 14, _tuber(rx, ry, col, edge, spots), "kc-pop-c", 2.2 + d0 + j * .25))
        out.append(f'{ridge}{"".join(stolons)}{"".join(tubers)}'
                   f'<g class="kc-grow" style="animation-delay:{d0:.2f}s"><g class="kc-sway" '
                   f'style="animation-delay:{-d0 * 3:.2f}s">{"".join(stems)}{"".join(leaves)}</g></g>')
    return "".join(out)


def _maize_blade(length: float, droop: float, fill: str) -> str:
    """A long maize leaf from the stalk at (0, 0), arching out to the right and down."""
    l = length
    return (f'<path d="M0 0 C{_f(l * .35)} {_f(-l * .32)} {_f(l * .75)} {_f(-l * .2)} {_f(l)} {_f(droop)} '
            f'C{_f(l * .72)} {_f(-l * .12)} {_f(l * .35)} {_f(-l * .2)} 0 5Z" fill="{fill}"/>')


def _cob() -> str:
    """A cob in its husk with silk, hanging from (0, 0)."""
    return ('<path d="M0 0 C7 4 9 16 5 28 C3 33 -3 33 -5 28 C-9 16 -7 4 0 0Z" fill="#9ccc5a" stroke="#6f9e3a" '
            'stroke-width=".8"/><path d="M-1 2 C-6 10 -6 20 -3 28" stroke="#6f9e3a" stroke-width=".8" fill="none"/>'
            '<path d="M0 0 C-2 -5 -5 -9 -9 -12 M0 0 C1 -6 0 -10 -2 -15 M0 0 C3 -5 6 -8 9 -10" stroke="#b07a3a" '
            'stroke-width="1" fill="none"/>')


def _tassel() -> str:
    out = ['<path d="M0 0 L0 -16" stroke="#cfa64a" stroke-width="1.6"/>']
    for a, l in ((-38, 13), (-20, 15), (0, 17), (22, 15), (40, 12)):
        out.append(_at(0, -6, a, f'<path d="M0 0 Q2 {-l * .5} 0 {-l}" stroke="#d8b04f" stroke-width="1.3" '
                                 'fill="none"/>'))
    return "".join(out)


def _maize(g: float) -> str:
    out = [_soil(g)]
    for i, (x, h) in enumerate(((116, 150), (204, 166), (292, 146))):
        d0 = i * .3
        top = g - h
        stalk = (f'<path d="M{x} {g} C{x - 2} {g - h * .4} {x + 3} {g - h * .75} {x + 1} {top}" stroke="#6f9f3a" '
                 'stroke-width="5" fill="none" stroke-linecap="round"/>')
        blades = []
        for j, t in enumerate((.12, .27, .42, .57, .72, .86)):
            y = g - h * t
            side = 1 if j % 2 == 0 else -1
            length = 62 - j * 6
            fill = "#6aa83a" if j % 2 else "#5f9e34"
            blade = _maize_blade(length, 10 + j * 2, fill)
            blades.append(_at(x, y, 0 if side > 0 else 0, blade if side > 0 else
                              f'<g transform="scale(-1 1)">{blade}</g>'))
        cob = _at(x + 4, g - h * .5, -18, _cob(), "kc-pop", 2.1 + d0)
        tassel = _at(x + 1, top, 0, _tassel())
        out.append(f'<g class="kc-grow" style="animation-delay:{d0:.2f}s"><g class="kc-sway" '
                   f'style="animation-delay:{-d0 * 3:.2f}s">{stalk}{"".join(blades)}{cob}{tassel}</g></g>')
    return "".join(out)


def _rice(u: str, g: float) -> str:
    water = (f'<defs><linearGradient id="{u}-water" x1="0" y1="0" x2="0" y2="1"><stop offset="0" '
             f'stop-color="#bfe3ef"/><stop offset="1" stop-color="#86c3d9"/></linearGradient></defs>'
             f'<path d="M0 {g - 6} L400 {g - 6} L400 {g + 2} L0 {g + 2}Z" fill="#7aa64c"/>'
             f'<rect x="0" y="{g}" width="400" height="{H - g}" fill="url(#{u}-water)"/>'
             + "".join(f'<path d="M{x} {y} q12 -3 24 0" stroke="#fff" stroke-width="1.2" fill="none" opacity=".6"/>'
                       for x, y in ((30, g + 22), (150, g + 34), (270, g + 18), (330, g + 40), (90, g + 44))))
    out = [water]
    for i, x in enumerate((52, 112, 172, 232, 292, 352)):
        d0 = i * .2
        blades = []
        for j in range(9):
            a = -34 + j * 8.5
            h = 58 + (j % 3) * 9
            tip = (x + h * math.sin(math.radians(a)), g - h * math.cos(math.radians(a)))
            blades.append(f'<path d="M{x} {g + 4} Q{_f(x + (tip[0] - x) * .35)} {_f(g - h * .55)} {_f(tip[0])} '
                          f'{_f(tip[1])}" stroke="{"#5e9e33" if j % 2 else "#6db23c"}" stroke-width="2.2" '
                          'fill="none" stroke-linecap="round"/>')
        panicles = []
        for j, (a, h) in enumerate(((-12, 66), (6, 72), (20, 62))):
            base = (x + h * .9 * math.sin(math.radians(a)), g - h * .9)
            grains = "".join(_at(k * 4.2, k * k * .55, 35 + k * 6,
                                 '<ellipse cx="0" cy="0" rx="2.8" ry="1.3" fill="#e2c35f"/>') for k in range(1, 7))
            panicles.append(_at(base[0], base[1], a * .6, f'<path d="M0 0 Q14 2 24 20" stroke="#b9a24a" '
                                                         f'stroke-width="1.2" fill="none"/>{grains}',
                                "kc-pop", 2.0 + d0 + j * .2))
        ripple = (f'<ellipse class="kc-ripple" style="animation-delay:{(i * .55) % 3.4:.2f}s" cx="{x}" cy="{g + 6}" '
                  'rx="20" ry="4" stroke="#fff" stroke-width="1.2" fill="none"/>')
        out.append(ripple + f'<g class="kc-grow" style="animation-delay:{d0:.2f}s"><g class="kc-sway" '
                   f'style="animation-delay:{-d0 * 3:.2f}s">{"".join(blades)}{"".join(panicles)}</g></g>')
    return "".join(out)


def scene(crop: str, uid: str, climbing: bool = True, skin: Optional[str] = None, title: str = "") -> str:
    """The animated scene for a crop. Beans: climbing beans on stakes or bush beans. Potato: the tubers take the
    skin colour of the first variety that fits the field."""
    if crop == "beans":
        ground = _beans_climbing(212) if climbing else _beans_bush(214)
    elif crop == "irish_potato":
        ground = _potato(184, skin or "red")
    elif crop == "maize":
        ground = _maize(214)
    elif crop == "rice":
        ground = _rice(uid, 196)
    else:
        ground = _soil(212)
    label = f"<title>{title}</title>" if title else ""
    return (f'<svg class="kc-scene" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">{label}'
            f'{_backdrop(uid, ground)}</svg>')


# ----------------------------------------------------------------------------------------------- variety pictures
SEED = {"white": ("#f3eee2", "#cfc4ab", None), "red": ("#b3262f", "#7d1820", None),
        "dark_red": ("#6b1521", "#4a0d16", None), "salmon": ("#e8967b", "#c06f55", None),
        "red_mottled": ("#b4343b", "#7d2026", "mottle"), "sugar": ("#efe1c2", "#c7b38c", "streak")}
SKIN = {"red": ("#b3453d", "#7f2c26", False), "cream": ("#e3cd99", "#b39b66", False),
        "light_pink": ("#e8a79c", "#b9776c", False), "white_purple": ("#efe8dc", "#b7ab97", True)}
FLESH = {"yellow": "#f2cb45", "light_yellow": "#f5df97", "cream": "#f1e3c0", "white_cream": "#f8f1dd"}

_KIDNEY = ("M-13 -1 C-13 -8 -6 -10 0 -9.5 C7 -9 13 -6 13.5 0 C14 6 9 9 3 8.6 C-1 8.3 -2 5.6 -5.5 6 "
           "C-9 6.4 -13 5 -13 -1Z")


def _seed(uid: str, colour: str) -> str:
    base, edge, pattern = SEED.get(colour, SEED["white"])
    extra = ""
    if pattern == "mottle":
        extra = "".join(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="#f0d8b2" opacity=".92"/>'
                        for x, y, rx, ry in ((-7, -3, 4, 2.4), (2, 2.5, 5, 2.2), (7.5, -4.5, 3, 1.9),
                                             (-2, -6.5, 3, 1.5), (-9, 3, 2.4, 1.6)))
    elif pattern == "streak":
        extra = ('<path d="M-11 -4 C-5 -8 3 -7 10 -3 M-12 2 C-5 -1 4 -1 12 3 M-6 6 C-1 4 5 5 9 7" stroke="#b5382f" '
                 'stroke-width="1.5" fill="none" stroke-linecap="round"/><circle cx="-4" cy="-1" r="1" '
                 'fill="#b5382f"/><circle cx="5" cy="-5.5" r=".9" fill="#b5382f"/>')
    hilum = "#d8ccb2" if colour in ("white", "sugar") else "#f4ecdc"
    clip = f'<clipPath id="{uid}"><path d="{_KIDNEY}"/></clipPath>'
    return (f'<defs>{clip}</defs><path d="{_KIDNEY}" fill="{base}" stroke="{edge}" stroke-width=".9"/>'
            f'<g clip-path="url(#{uid})">{extra}</g>'
            f'<ellipse cx="-2" cy="6.3" rx="2.6" ry="1.1" fill="{hilum}"/>'
            '<ellipse cx="-4" cy="-5" rx="5" ry="1.9" fill="#fff" opacity=".35"/>')


def bean_seeds(uid: str, colour: str) -> str:
    seeds = "".join(_at(x, y, a, _seed(f"{uid}-{i}", colour))
                    for i, (x, y, a) in enumerate(((24, 31, -18), (50, 24, 12), (74, 34, -6))))
    return f'<svg viewBox="6 9 84 40" width="96" height="46" xmlns="http://www.w3.org/2000/svg">{seeds}</svg>'


def potato_cut(skin: str, flesh: str) -> str:
    col, edge, spots = SKIN.get(skin, SKIN["cream"])
    inside = FLESH.get(flesh, FLESH["cream"])
    whole = _at(30, 30, -12, _tuber(17, 13, col, edge, spots))
    half = _at(70, 30, 8, f'<ellipse cx="0" cy="0" rx="15" ry="11.5" fill="{col}" stroke="{edge}" stroke-width=".8"/>'
                          f'<ellipse cx="0" cy="0" rx="13.2" ry="9.8" fill="{inside}"/>'
                          '<ellipse cx="0" cy="0" rx="9.4" ry="6.8" stroke="#000" stroke-opacity=".08" fill="none"/>'
                          '<ellipse cx="-4" cy="-4" rx="5" ry="1.8" fill="#fff" opacity=".35"/>')
    return f'<svg viewBox="9 12 80 36" width="96" height="43" xmlns="http://www.w3.org/2000/svg">{whole}{half}</svg>'


def maize_cob(uid: str, kernel: Optional[str]) -> str:
    kernels = []
    for row in range(-2, 3):
        for col in range(-6, 7):
            x, y = col * 4.6 + (1.2 if row % 2 else 0), row * 4.2
            k = f'<ellipse cx="{_f(x)}" cy="{_f(y)}" rx="2.3" ry="2" fill="#f0d67a" stroke="#d9b955" stroke-width=".4"/>'
            if kernel == "flint":
                k += f'<circle cx="{_f(x - .7)}" cy="{_f(y - .7)}" r=".7" fill="#fff" opacity=".7"/>'
            elif kernel == "semi_dent":
                k += f'<path d="M{_f(x - 1)} {_f(y - 1.2)} h2" stroke="#b9963e" stroke-width=".6"/>'
            kernels.append(k)
    cob = (f'<defs><clipPath id="{uid}"><ellipse cx="0" cy="0" rx="30" ry="11"/></clipPath></defs>'
           f'<ellipse cx="0" cy="0" rx="30" ry="11" fill="#e8c95f"/><g clip-path="url(#{uid})">{"".join(kernels)}</g>'
           '<path d="M-30 0 C-22 -14 -6 -15 2 -11 C-8 -8 -18 -4 -30 0Z" fill="#8dbb4f"/>'
           '<path d="M-30 0 C-22 13 -8 14 0 11 C-10 8 -19 5 -30 0Z" fill="#6f9e3a"/>'
           '<path d="M30 0 C35 -2 38 -5 41 -9 M30 1 C36 1 40 0 44 -3 M29 2 C34 4 38 6 41 9" stroke="#a9743b" '
           'stroke-width="1" fill="none"/>')
    return (f'<svg viewBox="8 8 84 44" width="96" height="50" xmlns="http://www.w3.org/2000/svg">'
            f'{_at(46, 30, -16, cob)}</svg>')


def rice_panicle(grain: Optional[str]) -> str:
    rx, ry = (4.6, 1.25) if grain == "long" else (3.3, 1.6)
    stem = '<path d="M14 54 C20 34 34 16 54 10 C66 7 76 12 82 22" stroke="#b9a24a" stroke-width="1.4" fill="none"/>'
    pts = ((30, 30, 40), (38, 22, 30), (47, 15, 18), (57, 10, 6), (66, 10, -10), (74, 14, -28), (80, 20, -45))
    grains = "".join(_at(x + dx, y + dy, a + da, f'<ellipse cx="0" cy="0" rx="{rx}" ry="{ry}" fill="#e2c35f" '
                                                 'stroke="#c3a142" stroke-width=".5"/>')
                     for x, y, a in pts for dx, dy, da in ((0, 0, 0), (2.5, 3.5, 25)))
    loose = "".join(_at(x, y, a, f'<ellipse cx="0" cy="0" rx="{rx}" ry="{ry}" fill="#efe0b0" stroke="#cdb57a" '
                                 'stroke-width=".5"/>') for x, y, a in ((60, 48, 10), (72, 50, -20), (82, 46, 35)))
    return f'<svg viewBox="0 0 96 58" width="96" height="58" xmlns="http://www.w3.org/2000/svg">{stem}{grains}{loose}</svg>'


def variety_picture(crop: str, variety: dict, uid: str) -> str:
    if crop == "beans":
        return bean_seeds(uid, variety.get("seed", "white"))
    if crop == "irish_potato":
        return potato_cut(variety.get("skin", "cream"), variety.get("flesh", "cream"))
    if crop == "maize":
        return maize_cob(uid, variety.get("kernel"))
    return rice_panicle(variety.get("grain"))
