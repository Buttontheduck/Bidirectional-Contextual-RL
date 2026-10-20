"""Figure 1 (overview), hand-designed: (a) pilots -> flight data -> GRU encoder -> battery estimate b_hat_t (offline);
(b) environment -> frozen encoder -> PPO -> environment (online). Writes figures/out_person/fig1_overview.{svg,png,pdf}.

Text is placed run by run with Roboto's glyph widths (subscripts and hats are drawn, not left to the renderer), so the SVG,
the PNG (resvg) and the PDF (svglib + reportlab, fonts embedded) look the same. Needs resvg-py, svglib and font-roboto:
    pip install --target <dir> resvg-py svglib font-roboto ; PYLIBS=<dir> python figures/fig1_overview.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

PYLIBS = os.environ.get("PYLIBS")
if PYLIBS:
    sys.path.insert(0, PYLIBS)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import font_roboto  # noqa: E402  (the font files)

FONTS = Path(font_roboto.__file__).parent / "files"
OUT = Path(__file__).resolve().parent / "out_person"
W, H = 1400, 690

INK, INK2, MUTED, HAIR, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f6f7f9"
BLUE, BLUE_DARK, BLUE_TINT, BLUE_LINE = "#2a78d6", "#1c5cab", "#e8f1fc", "#b9d3f3"
VIOLET, VIOLET_TINT = "#6250d6", "#efedfc"
PLANE, TRAIL, GROUND = "#1f2a44", "#7d8db0", "#a3c293"

FACES = {("normal", "normal"): "Roboto-Regular", ("normal", "italic"): "Roboto-Italic", ("500", "normal"): "Roboto-Medium",
         ("bold", "normal"): "Roboto-Bold", ("bold", "italic"): "Roboto-BoldItalic", ("500", "italic"): "Roboto-MediumItalic"}
for face in FACES.values():
    pdfmetrics.registerFont(TTFont(face, str(FONTS / f"{face}.ttf")))

parts: list[str] = []


def add(svg: str) -> None:
    parts.append(svg)


def width(text: str, size: float, weight="normal", style="normal") -> float:
    return pdfmetrics.stringWidth(text, FACES[(weight, style)], size)


def text(x, y, s, size=13, weight="normal", style="normal", color=INK, anchor="start", opacity=1.0):
    """One run of text; leading and trailing spaces become offsets (SVG would drop them), so runs line up."""
    w = width(s, size, weight, style)
    x0 = x - w / 2 if anchor == "middle" else x - w if anchor == "end" else x
    lead = len(s) - len(s.lstrip(" "))
    x0 += width(" " * lead, size, weight, style)
    s = s.strip(" ").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    add(f'<text x="{x0:.1f}" y="{y:.1f}" xml:space="preserve" font-family="Roboto" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{color}" fill-opacity="{opacity}">{s}</text>')
    return w


def rich(x, y, runs, size=13, color=INK, anchor="start"):
    """Runs of (text, options): options may set style, weight, color, sub (a subscript) and hat (a hat over the run)."""
    def run_width(t, o):
        return width(t, size * (0.7 if o.get("sub") else 1), o.get("weight", "normal"), o.get("style", "normal"))
    total = sum(run_width(t, o) for t, o in runs)
    cx = x - total / 2 if anchor == "middle" else x - total if anchor == "end" else x
    for t, o in runs:
        sub = o.get("sub")
        s = size * (0.7 if sub else 1)
        w = text(cx, y + (0.28 * size if sub else 0), t, s, o.get("weight", "normal"), o.get("style", "normal"), o.get("color", color))
        if o.get("hat"):
            hx, hy = cx + w * (0.62 if o.get("style") == "italic" else 0.5), y - 0.93 * size
            add(f'<path d="M{hx - 0.19 * size:.1f} {hy + 0.15 * size:.1f} L{hx:.1f} {hy:.1f} L{hx + 0.19 * size:.1f} {hy + 0.15 * size:.1f}" '
                f'fill="none" stroke="{o.get("color", color)}" stroke-width="{0.075 * size:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
        cx += w
    return total


I = {"style": "italic"}
SUB = {"style": "italic", "sub": True}
HAT = {"style": "italic", "hat": True}


def card(x, y, w, h, stroke=HAIR, fill="#ffffff", sw=1.2, r=16):
    add(f'<rect x="{x + 2}" y="{y + 4}" width="{w}" height="{h}" rx="{r}" fill="#0b0b0b" fill-opacity="0.045"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def arrowhead(x, y, angle, color=INK2, size=9):
    a = math.radians(angle)
    p = [(x, y), (x - size * math.cos(a) + size * 0.55 * math.sin(a), y - size * math.sin(a) - size * 0.55 * math.cos(a)),
         (x - size * math.cos(a) - size * 0.55 * math.sin(a), y - size * math.sin(a) + size * 0.55 * math.cos(a))]
    add(f'<polygon points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in p)}" fill="{color}"/>')


def arrow(x0, y0, x1, y1, color=INK2, sw=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a = math.degrees(math.atan2(y1 - y0, x1 - x0))
    ex, ey = x1 - 7 * math.cos(math.radians(a)), y1 - 7 * math.sin(math.radians(a))
    add(f'<line x1="{x0}" y1="{y0}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"{d}/>')
    arrowhead(x1, y1, a, color)


def flow_arrow(x0, x1, y, label_runs=None, color=INK2):
    """A stage-to-stage arrow, with an optional label above it."""
    arrow(x0, y, x1, y, color, 2.2)
    if label_runs:
        rich((x0 + x1) / 2, y - 10, label_runs, size=12, color=INK2, anchor="middle")


# ----------------------------------------------------------------------------------------------- canvas and bands
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
add('<defs>'
    '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c3d9f3"/><stop offset="1" stop-color="#f3f8fd"/></linearGradient>'
    '<linearGradient id="gru" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f1f7fe"/><stop offset="1" stop-color="#dceafb"/></linearGradient>'
    '<linearGradient id="ppo" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f6f5fe"/><stop offset="1" stop-color="#e6e2fb"/></linearGradient>'
    '<linearGradient id="fill" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#1c5cab"/><stop offset="1" stop-color="#4f93e6"/></linearGradient>'
    '<clipPath id="envclip"><rect x="30" y="388" width="470" height="236" rx="16"/></clipPath>'
    '</defs>')
add(f'<rect x="12" y="12" width="{W - 24}" height="312" rx="20" fill="{BAND}"/>')
add(f'<rect x="12" y="336" width="{W - 24}" height="342" rx="20" fill="{BAND}"/>')


def row_title(y, letter, title, subtitle):
    add(f'<circle cx="44" cy="{y - 6}" r="15" fill="{INK}"/>')
    text(44, y - 0.5, letter, 15, "bold", color="#ffffff", anchor="middle")
    w = text(70, y, title, 17, "bold")
    text(70 + w + 12, y, subtitle, 13.5, color=INK2)


# =============================================================================================== (a) offline
row_title(46, "a", "Offline: learn to read the battery", "many pilots fly; the encoder learns to infer the hidden battery from what it sees")

# --- stage 1: pilots
card(30, 72, 336, 236)
text(48, 100, "Pilots", 15, "bold")
text(98, 100, "11 flying styles, scripted and PPO", 12, color=INK2)
PILOTS = [  # name, throttle sketch in a 32 x 16 box (top = full throttle)
    ("Probe & hold", "M0 3 H9 V11 H32"), ("Steady below cap", "M0 10 H32"),
    ("Random hold", "M0 8 H8 V12 H16 V4 H24 V9 H32"),
    ("Shaky", "M0 9 L3 5 L6 12 L9 6 L12 11 L15 4 L18 10 L21 6 L24 12 L27 5 L30 10 L32 8"),
    ("Wavy", "M0 8 C3 2 7 2 10 8 C13 14 17 14 20 8 C23 2 27 2 30 8 L32 10"),
    ("Hold speed", "M0 3 C6 3 8 10 12 10 L15 9 L18 11 L21 9 L24 11 L27 9 L32 10"),
    ("Cruise", "M0 5 C6 5 8 11 12 11 H32"), ("Endurance", "M0 4 L5 4 L7 13 H32"),
    ("Memoryless PPO", "M0 2 H22 L24 14 H32"), ("Good PPO, replayed", "M0 2 H8 L12 9 C18 10 24 10 32 12"),
    ("Style switching", "M0 10 H14 L16 4 L18 12 L20 4 L22 12 L24 4 L26 12 L28 4 L30 12 L32 6"),
]
for k, (name, sketch) in enumerate(PILOTS):
    col, row = divmod(k, 6)
    px, py = 44 + col * 160, 116 + row * 31
    add(f'<rect x="{px}" y="{py}" width="154" height="25" rx="12.5" fill="#f2f5fa" stroke="#dde4ee" stroke-width="1"/>')
    add(f'<g transform="translate({px + 9},{py + 4.5})"><rect x="-2" y="-1" width="36" height="18" rx="4" fill="#ffffff"/>'
        f'<path d="{sketch}" fill="none" stroke="{BLUE_DARK}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></g>')
    text(px + 51, py + 17, name, 11.5, color=INK)
text(352, 296, "sketch: throttle over a flight", 10, style="italic", color=MUTED, anchor="end")

flow_arrow(372, 410, 190, [("fly", {})])

# --- stage 2: flight data
for dx, dy, op in ((16, -10, 0.55), (8, -5, 0.8)):
    add(f'<rect x="{416 + dx}" y="{72 + dy}" width="232" height="236" rx="16" fill="#ffffff" stroke="{HAIR}" stroke-opacity="{op}"/>')
card(416, 72, 232, 236)
text(434, 100, "Flight data", 15, "bold")
text(527, 100, "19,000 flights", 12, color=INK2)
PLOT_X0, PLOT_X1 = 466, 632
for k, (label_runs, path, dash, color) in enumerate([
        ([("y", I)], "M466 156 C500 150 520 132 556 130 C586 128 610 128 632 129", None, "#5b6b8c"),
        ([("v", I), ("x", SUB)], "M466 196 L472 189 L478 193 L484 184 L490 188 L496 180 L502 185 L508 177 L514 181 L520 175 L526 180 "
                                 "L532 172 L538 177 L544 171 L550 176 L556 170 L562 175 L568 169 L574 174 L580 168 L586 173 L592 168 "
                                 "L598 172 L604 167 L610 171 L616 166 L622 170 L632 167", None, "#5b6b8c"),
        ([("b", I), ("t", SUB)], "M466 222 C500 226 540 236 580 246 C600 251 616 255 632 258", "5 4", INK)]):
    top = 116 + k * 50
    add(f'<rect x="{PLOT_X0 - 4}" y="{top}" width="{PLOT_X1 - PLOT_X0 + 8}" height="40" rx="6" fill="#fafbfc"/>')
    add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round"'
        + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    rich(432, top + 25, label_runs, size=13, color=INK2)
# lock: the battery is a training label only
add('<g transform="translate(436,250)"><rect x="0" y="6" width="12" height="9" rx="2" fill="#52514e"/>'
    '<path d="M2.5 6 V3.5 A3.5 3.5 0 0 1 9.5 3.5 V6" fill="none" stroke="#52514e" stroke-width="1.6"/></g>')
rich(452, 296, [("inputs ", {}), ("o", I), ("t", SUB), (", ", {}), ("a", I), ("t−1", SUB), ("   label ", {}), ("b", I), ("t", SUB)],
     size=11.5, color=INK2)

flow_arrow(656, 694, 190, [("train", {})])

# --- stage 3: GRU encoder (unrolled)
card(700, 72, 312, 236, stroke=BLUE_LINE, sw=1.4)
text(718, 100, "GRU encoder", 15, "bold", color=BLUE_DARK)
text(818, 100, "reads the flight step by step", 12, color=INK2)
CELLS = [(758, "t−1"), (848, "t"), (938, "t+1")]
arrow(712, 188, 731, 188, BLUE, 1.8)
text(712, 180, "h", 12, style="italic", color=BLUE_DARK)
for k, (cx, sub) in enumerate(CELLS):
    add(f'<rect x="{cx - 27}" y="{166}" width="54" height="44" rx="10" fill="url(#gru)" stroke="{BLUE}" stroke-width="1.6"/>')
    text(cx, 193, "GRU", 12, "500", color=BLUE_DARK, anchor="middle")
    if k < 2:
        arrow(cx + 27, 188, cx + 63, 188, BLUE, 1.8)
    else:
        add(f'<line x1="{cx + 27}" y1="188" x2="{cx + 52}" y2="188" stroke="{BLUE}" stroke-width="1.8" stroke-dasharray="3 3"/>')
    arrow(cx, 248, cx, 212, INK2, 1.6)
    rich(cx, 266, [("x", I), (sub, SUB)], size=13, color=INK2, anchor="middle")
    arrow(cx, 165, cx, 141, BLUE, 1.6)
    add(f'<circle cx="{cx}" cy="128" r="13" fill="#ffffff" stroke="{BLUE}" stroke-width="1.6"/>')
    rich(cx - 1, 133, [("b", HAT)], size=13, color=BLUE_DARK, anchor="middle")
rich(856, 296, [("x", I), ("t", SUB), (" = (", {}), ("o", I), ("t", SUB), (", ", {}), ("a", I), ("t−1", SUB), (")", {})],
     size=11.5, color=INK2, anchor="middle")

flow_arrow(1020, 1058, 190, [("predict", {})])

# --- stage 4: the battery estimate
card(1064, 72, 306, 236)
rich(1082, 100, [("Battery estimate ", {"weight": "bold"}), ("b", {"style": "italic", "weight": "bold", "hat": True}),
                 ("t", {"style": "italic", "weight": "bold", "sub": True})], size=15)
# battery glyph: estimate (blue fill) and the true level (dashed line)
add(f'<rect x="1098" y="116" width="22" height="9" rx="2.5" fill="{INK2}"/>')
add(f'<rect x="1084" y="123" width="50" height="128" rx="10" fill="#ffffff" stroke="{INK2}" stroke-width="2"/>')
add('<rect x="1089" y="190" width="40" height="56" rx="6" fill="url(#fill)"/>')
add(f'<line x1="1080" y1="196" x2="1138" y2="196" stroke="{INK}" stroke-width="1.6" stroke-dasharray="4 3"/>')
# estimate vs truth over a flight
add('<rect x="1150" y="122" width="206" height="128" rx="8" fill="#fafbfc"/>')
add(f'<path d="M1158 140 C1220 154 1290 186 1350 226" fill="none" stroke="{INK}" stroke-width="1.8" stroke-dasharray="5 4"/>')
add(f'<path d="M1158 186 C1172 186 1182 185 1190 180 C1198 172 1203 158 1210 154 C1250 164 1300 192 1350 226" fill="none" '
    f'stroke="{BLUE}" stroke-width="2.4" stroke-linecap="round"/>')
add(f'<circle cx="1206" cy="157" r="3.5" fill="{BLUE}"/>')
text(1216, 142, "probe: the battery shows", 10.5, style="italic", color=INK2)
add(f'<line x1="1158" y1="262" x2="1178" y2="262" stroke="{BLUE}" stroke-width="2.4"/>')
rich(1183, 266, [("estimate ", {}), ("b", HAT), ("t", SUB)], size=11.5, color=INK2)
add(f'<line x1="1262" y1="262" x2="1282" y2="262" stroke="{INK}" stroke-width="1.8" stroke-dasharray="5 4"/>')
rich(1287, 266, [("true ", {}), ("b", I), ("t", SUB)], size=11.5, color=INK2)
rich(1217, 296, [("loss  (", {}), ("b", HAT), ("t", SUB), (" − ", {}), ("b", I), ("t", SUB), (")²", {})], size=12.5, color=INK, anchor="middle")

# =============================================================================================== (b) online
row_title(370, "b", "Online: PPO flies on the frozen encoder", "the policy sees only the encoder's estimate, never the battery")

# --- environment
card(30, 388, 470, 236)
add('<g clip-path="url(#envclip)">'
    '<rect x="30" y="388" width="470" height="236" fill="url(#sky)"/>'
    f'<rect x="30" y="592" width="470" height="32" fill="{GROUND}"/>'
    + "".join(f'<line x1="{x}" y1="592" x2="{x - 8}" y2="604" stroke="#86a877" stroke-width="2"/>' for x in range(52, 500, 26))
    + '<line x1="30" y1="414" x2="500" y2="414" stroke="#8a9ab8" stroke-width="1.2" stroke-dasharray="6 5"/>'
    f'<path d="M52 586 C120 584 200 560 262 512 C280 498 292 492 300 488" fill="none" stroke="{TRAIL}" stroke-width="2" stroke-dasharray="2 6" stroke-linecap="round"/>'
    '</g>')
text(492, 409, "ceiling", 10.5, style="italic", color="#5d6c8a", anchor="end")
# the plane, pitched up
add(f'<g transform="translate(330,476) rotate(-12)">'
    f'<path d="M-50 0 C-44 -10 28 -11 46 -3 C52 -1 51 4 46 5 L-42 7 C-49 7 -52 3 -50 0 Z" fill="{PLANE}"/>'
    f'<path d="M-50 -1 L-57 -21 L-45 -21 L-34 -3 Z" fill="{PLANE}"/>'
    f'<path d="M-8 3 L16 3 L-1 23 L-15 23 Z" fill="#33446a"/>'
    f'<path d="M24 -6 C30 -8 37 -7 41 -4 L26 -2 Z" fill="#c9daf2"/>'
    f'<line x1="56" y1="0" x2="98" y2="0" stroke="{INK2}" stroke-width="2.2" stroke-linecap="round"/>'
    f'<polygon points="106,0 96,-5.5 96,5.5" fill="{INK2}"/>'
    '</g>')
rich(418, 446, [("thrust ≤ cap(", {}), ("b", I), ("t", SUB), (")", {})], size=11.5, color=INK2, anchor="middle")
# label and the hidden battery
add('<rect x="44" y="424" width="132" height="28" rx="14" fill="#ffffff" fill-opacity="0.92"/>')
text(110, 443, "BatteryPlane", 13, "bold", anchor="middle")
add('<rect x="44" y="458" width="186" height="28" rx="14" fill="#ffffff" fill-opacity="0.92"/>')
add(f'<rect x="56" y="465" width="24" height="14" rx="3" fill="#ffffff" stroke="{MUTED}" stroke-width="1.6"/>'
    f'<rect x="80" y="469" width="3" height="6" rx="1" fill="{MUTED}"/>')
text(68, 476.5, "?", 11, "bold", color=MUTED, anchor="middle")
rich(92, 477, [("battery ", {}), ("b", I), ("t", SUB), (" is hidden", {})], size=12, color=INK2)
text(44, 528, "every action drains the battery;", 11.5, style="italic", color="#3f4a60")
text(44, 544, "the battery caps the thrust", 11.5, style="italic", color="#3f4a60")

# --- environment -> encoder, and the observation skip to PPO
flow_arrow(504, 590, 512, [("o", I), ("t", SUB), (", ", {}), ("a", I), ("t−1", SUB)])
add(f'<path d="M502 420 C650 392 800 392 908 448" fill="none" stroke="{INK2}" stroke-width="2" stroke-linecap="round"/>')
arrowhead(913, 451, 30, INK2)
rich(706, 395, [("o", I), ("t", SUB)], size=12, color=INK2, anchor="middle")

# --- frozen encoder
card(596, 432, 222, 150, stroke=BLUE_LINE, sw=1.4)
text(612, 458, "GRU encoder", 14, "bold", color=BLUE_DARK)
add(f'<rect x="724" y="443" width="82" height="22" rx="11" fill="{BLUE_TINT}"/>')
add(f'<g transform="translate(740,454)" stroke="{BLUE}" stroke-width="1.5" stroke-linecap="round">'
    + "".join(f'<line x1="0" y1="0" x2="{6.5 * math.cos(math.radians(a)):.2f}" y2="{6.5 * math.sin(math.radians(a)):.2f}"/>' for a in range(0, 360, 60))
    + '</g>')
text(752, 458.5, "frozen", 11.5, "500", color=BLUE_DARK)
add(f'<rect x="675" y="494" width="64" height="44" rx="10" fill="url(#gru)" stroke="{BLUE}" stroke-width="1.6"/>')
text(707, 521, "GRU", 12, "500", color=BLUE_DARK, anchor="middle")
add(f'<path d="M729 494 C744 470 670 470 685 494" fill="none" stroke="{BLUE}" stroke-width="1.6"/>')
arrowhead(686, 493, 60, BLUE, 7)
rich(752, 486, [("h", I), ("t", SUB)], size=12, color=BLUE_DARK)
arrow(600, 516, 673, 516, INK2, 1.6)
arrow(741, 516, 814, 516, BLUE, 1.6)
text(707, 566, "same weights as in (a)", 10.5, style="italic", color=MUTED, anchor="middle")

flow_arrow(822, 908, 512, [("b", HAT), ("t", SUB)], color=BLUE)

# --- PPO agent
card(914, 404, 290, 196, stroke="#c9c2f2", sw=1.4)
text(930, 430, "PPO agent", 14, "bold", color=VIOLET)
text(1006, 430, "trained by reinforcement learning", 11.5, color=INK2)
for k, (name, formula) in enumerate([("Actor", [("π(", {}), ("a", I), ("t", SUB), (" | ", {}), ("o", I), ("t", SUB), (", ", {}),
                                                ("b", HAT), ("t", SUB), (")", {})]),
                                     ("Critic", [("V(", {}), ("o", I), ("t", SUB), (", ", {}), ("b", HAT), ("t", SUB), (")", {})])]):
    top = 444 + k * 72
    add(f'<rect x="930" y="{top}" width="258" height="60" rx="11" fill="url(#ppo)"/>')
    text(946, top + 24, name, 13, "bold", color=VIOLET)
    rich(946, top + 46, formula, size=13.5, color=INK)
    # a small network glyph
    nodes = [[(1112, top + 18), (1112, top + 42)], [(1138, top + 12), (1138, top + 30), (1138, top + 48)], [(1164, top + 30)]]
    for left, right in zip(nodes[:-1], nodes[1:]):
        for a in left:
            for b in right:
                add(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{VIOLET}" stroke-opacity="0.35" stroke-width="1"/>')
    for layer in nodes:
        for nx, ny in layer:
            add(f'<circle cx="{nx}" cy="{ny}" r="4" fill="#ffffff" stroke="{VIOLET}" stroke-width="1.4"/>')

# --- reward (training only) and the action back to the environment
add(f'<path d="M500 606 L906 584" fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="5 4"/>')
arrowhead(912, 583.7, -3, MUTED, 8)
rich(706, 616, [("reward ", {}), ("r", I), ("t", SUB), ("  (training only)", {})], size=11.5, color=MUTED, anchor="middle")
add(f'<path d="M1204 512 H1270 Q1290 512 1290 532 V640 Q1290 658 1270 658 H285 Q265 658 265 640 V632" fill="none" '
    f'stroke="{INK2}" stroke-width="2.2" stroke-linecap="round"/>')
arrowhead(265, 627, -90, INK2)
rich(778, 650, [("action ", {}), ("a", I), ("t", SUB), (" = (cruise throttle, lift)", {})], size=12, color=INK2, anchor="middle")

# ----------------------------------------------------------------------------------------------- write
OUT.mkdir(exist_ok=True)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n' + "\n".join(parts) + "\n</svg>\n")
(OUT / "fig1_overview.svg").write_text(svg)

import resvg_py  # noqa: E402
from reportlab.graphics import renderPDF  # noqa: E402
from svglib import fonts as svgfonts  # noqa: E402
from svglib.svglib import svg2rlg  # noqa: E402

png = resvg_py.svg_to_bytes(svg_path=str(OUT / "fig1_overview.svg"), font_dirs=[str(FONTS)], zoom=3, skip_system_fonts=True)
(OUT / "fig1_overview.png").write_bytes(bytes(png))
for (weight, style), face in FACES.items():
    svgfonts.register_font("Roboto", str(FONTS / f"{face}.ttf"), weight=weight, style=style, rlgFontName=face)
renderPDF.drawToFile(svg2rlg(str(OUT / "fig1_overview.svg")), str(OUT / "fig1_overview.pdf"))
print("written", OUT / "fig1_overview.svg", "(.png, .pdf)")
