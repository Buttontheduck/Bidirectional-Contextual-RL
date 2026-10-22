"""Figure 1, paper-style diagram: (a) offline supervised encoder training (pilots -> dataset -> GRU encoder f_phi ->
b_hat_t, loss on the privileged label b_t); (b) PPO with the frozen encoder (environment -> concat [o_t, a_{t-1}] ->
f_phi -> [o_t, b_hat_t] -> pi_theta, V_psi -> a_t -> environment, with a z^-1 delay). Writes
figures/out_person/fig1_diagram.{svg,png,pdf}. Labels in Roboto, mathematics in STIX (Times-like italics, as in the
ICML body text); every run is placed with the fonts' glyph widths, so SVG, PNG (resvg) and PDF (svglib) agree.
    PYLIBS=<dir with resvg-py, svglib, font-roboto> python figures/fig1_diagram.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

PYLIBS = os.environ.get("PYLIBS")
if PYLIBS:
    sys.path.insert(0, PYLIBS)
import matplotlib  # noqa: E402  (only for its bundled STIX fonts)
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402

import font_roboto  # noqa: E402

ROBOTO = Path(font_roboto.__file__).parent / "files"
STIX = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
OUT = Path(__file__).resolve().parent / "out_person"
W, H = 1000, 454

INK, INK2, MUTED, LINE, FRAME = "#111111", "#4a4a46", "#8e8c85", "#4a4a46", "#b9b7af"
NEUTRAL = "#f4f4f1"
BLUE, BLUE_DARK, BLUE_TINT = "#2a78d6", "#1c5cab", "#eaf2fc"
VIOLET, VIOLET_DARK, VIOLET_TINT = "#6250d6", "#4a3aa7", "#f1effc"

# family, weight, style -> (reportlab face, file, svg family)
FACES = {
    ("sans", "normal", "normal"): ("Roboto-Regular", ROBOTO / "Roboto-Regular.ttf", "Roboto"),
    ("sans", "500", "normal"): ("Roboto-Medium", ROBOTO / "Roboto-Medium.ttf", "Roboto"),
    ("sans", "bold", "normal"): ("Roboto-Bold", ROBOTO / "Roboto-Bold.ttf", "Roboto"),
    ("sans", "normal", "italic"): ("Roboto-Italic", ROBOTO / "Roboto-Italic.ttf", "Roboto"),
    ("math", "normal", "normal"): ("STIX-Regular", STIX / "STIXGeneral.ttf", "STIXGeneral"),
    ("math", "normal", "italic"): ("STIX-Italic", STIX / "STIXGeneralItalic.ttf", "STIXGeneral"),
    ("math", "bold", "normal"): ("STIX-Bold", STIX / "STIXGeneralBol.ttf", "STIXGeneral"),
    ("math", "bold", "italic"): ("STIX-BoldItalic", STIX / "STIXGeneralBolIta.ttf", "STIXGeneral"),
}
for face, path, _ in FACES.values():
    pdfmetrics.registerFont(TTFont(face, str(path)))

parts: list[str] = []


def add(svg):
    parts.append(svg)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run_width(t, fam, weight, style, size):
    return pdfmetrics.stringWidth(t, FACES[(fam, weight, style)][0], size)


def put(x, y, t, size, fam="sans", weight="normal", style="normal", color=INK):
    """One text run; leading/trailing spaces become offsets. Returns its advance width."""
    w = run_width(t, fam, weight, style, size)
    lead = len(t) - len(t.lstrip(" "))
    x0 = x + run_width(" " * lead, fam, weight, style, size)
    add(f'<text x="{x0:.2f}" y="{y:.2f}" xml:space="preserve" font-family="{FACES[(fam, weight, style)][2]}" font-size="{size:.2f}" '
        f'font-weight="{weight}" font-style="{style}" fill="{color}">{esc(t.strip(" "))}</text>')
    return w


def S(t, **o):    # sans
    return (t, {"fam": "sans", **o})


def M(t, **o):    # math italic
    return (t, {"fam": "math", "style": "italic", **o})


def R(t, **o):    # math upright
    return (t, {"fam": "math", "style": "normal", **o})


def rich(x, y, runs, size=13, color=INK, anchor="start"):
    """Runs from S/M/R; options sub, sup (smaller, lowered / raised), hat, color, weight."""
    def geom(o):
        k = 0.68 if (o.get("sub") or o.get("sup")) else 1.0
        dy = 0.26 * size if o.get("sub") else (-0.42 * size if o.get("sup") else 0)
        return size * k, dy

    def wid(t, o):
        s, _ = geom(o)
        return run_width(t, o["fam"], o.get("weight", "normal"), o.get("style", "normal"), s)
    total = sum(wid(t, o) for t, o in runs)
    cx = x - total / 2 if anchor == "middle" else x - total if anchor == "end" else x
    for t, o in runs:
        s, dy = geom(o)
        c = o.get("color", color)
        w = put(cx, y + dy, t, s, o["fam"], o.get("weight", "normal"), o.get("style", "normal"), c)
        if o.get("hat"):
            hx, hy = cx + w * (0.6 if o.get("style") == "italic" else 0.5), y + dy - 0.86 * s
            add(f'<path d="M{hx - 0.2 * s:.2f} {hy + 0.15 * s:.2f} L{hx:.2f} {hy:.2f} L{hx + 0.2 * s:.2f} {hy + 0.15 * s:.2f}" fill="none" '
                f'stroke="{c}" stroke-width="{0.07 * s:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
        cx += w
    return total


def box(x, y, w, h, fill=NEUTRAL, stroke=LINE, sw=1.1, r=7, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def head(x, y, angle, color=LINE, size=7.5):
    a = math.radians(angle)
    pts = [(x, y), (x - size * math.cos(a) + 0.5 * size * math.sin(a), y - size * math.sin(a) - 0.5 * size * math.cos(a)),
           (x - size * math.cos(a) - 0.5 * size * math.sin(a), y - size * math.sin(a) + 0.5 * size * math.cos(a))]
    add(f'<polygon points="{" ".join(f"{px:.2f},{py:.2f}" for px, py in pts)}" fill="{color}"/>')


def path(d, color=LINE, sw=1.3, dash=None, end=None, end_angle=0):
    """A polyline/path; `end` = (x, y) gets an arrowhead pointing along end_angle (degrees)."""
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{ds}/>')
    if end:
        head(*end, end_angle, color)


def arrow(x0, y0, x1, y1, color=LINE, sw=1.3, dash=None):
    a = math.degrees(math.atan2(y1 - y0, x1 - x0))
    ex, ey = x1 - 6 * math.cos(math.radians(a)), y1 - 6 * math.sin(math.radians(a))
    path(f"M{x0} {y0} L{ex:.2f} {ey:.2f}", color, sw, dash, (x1, y1), a)


def dot(x, y, color=LINE, r=2.6):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')


def concat(x, y, r=9):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#ffffff" stroke="{LINE}" stroke-width="1.2"/>')
    rich(x, y + 4.6, [R("‖")], size=13, color=INK2, anchor="middle")


def frame(y, h, label_runs):
    box(8, y, W - 16, h, fill="none", stroke=FRAME, sw=1.1, r=10, dash="5 4")
    rich(22, y + 22, label_runs, size=13.5)


add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# =============================================================================================== (a)
frame(8, 212, [S("(a)", weight="bold"), S("  Offline: supervised encoder training", weight="bold")])

# pilots
box(22, 52, 238, 136)
rich(34, 72, [S("Pilot policies", weight="bold"), S("  11 flying styles", color=INK2)], size=12)
NAMES = ["Probe & hold", "Steady below cap", "Random hold", "Shaky", "Wavy", "Hold speed",
         "Cruise", "Endurance", "PPO, memoryless", "PPO, replayed", "Style switching"]
for k, name in enumerate(NAMES):
    col, row = divmod(k, 6)
    x, y = 40 + col * 112, 94 + row * 16
    dot(x - 6, y - 3.8, MUTED, 2.1)
    put(x, y, name, 11.5, color=INK2)

arrow(260, 120, 318, 120)
rich(289, 113, [S("roll out", style="italic", color=INK2)], size=11.5, anchor="middle")

# dataset (a cylinder)
cx, top, bot, rx, ry = 370, 72, 154, 50, 12
add(f'<path d="M{cx - rx} {top} V{bot} A{rx} {ry} 0 0 0 {cx + rx} {bot} V{top}" fill="{NEUTRAL}" stroke="{LINE}" stroke-width="1.1"/>')
add(f'<ellipse cx="{cx}" cy="{top}" rx="{rx}" ry="{ry}" fill="#ffffff" stroke="{LINE}" stroke-width="1.1"/>')
rich(cx, 112, [M("𝒟")], size=24, anchor="middle")
rich(cx, 136, [M("o"), M("t", sub=True), R(", "), M("a"), M("t", sub=True), R("−1", sub=True), R(", "), M("b"), M("t", sub=True)],
     size=12.5, anchor="middle")
rich(cx - 16, 186, [S("19,000 flights", color=INK2)], size=11.5, anchor="middle")

arrow(420, 120, 466, 120)
rich(443, 113, [M("x"), M("t", sub=True)], size=13, anchor="middle")

# encoder: one block, as in (b)
box(468, 88, 222, 64, fill=BLUE_TINT, stroke=BLUE, sw=1.3)
rich(579, 115, [S("Encoder ", weight="bold", color=BLUE_DARK), M("f", color=BLUE_DARK), M("φ", sub=True, color=BLUE_DARK),
                S("  GRU", color=INK2)], size=13.5, anchor="middle")
rich(579, 138, [S("trained on ", color=INK2), M("𝒟", color=INK2)], size=11.5, anchor="middle")

arrow(690, 120, 742, 120, BLUE, 1.4)
rich(716, 113, [M("b", hat=True, color=BLUE_DARK), M("t", sub=True, color=BLUE_DARK)], size=13.5, anchor="middle")

# loss
box(744, 90, 230, 60, fill="#ffffff", stroke=LINE, sw=1.2)
rich(859, 126, [M("ℒ"), R("("), M("φ"), R(") = "), R("Σ"), M("t", sub=True), R(" ("), M("b", hat=True), M("t", sub=True), R(" − "),
                M("b"), M("t", sub=True), R(")"), R("2", sup=True)], size=16, anchor="middle")
# the privileged label, and the gradient back to the encoder
path(f"M{cx + 30} {bot + 9.6:.1f} V206 H859 V156", MUTED, 1.2, dash="4 3", end=(859, 150), end_angle=-90)
rich(690, 201, [S("labels ", style="italic", color=MUTED), M("b", color=MUTED), M("t", sub=True, color=MUTED),
                S("  (privileged, training only)", style="italic", color=MUTED)], size=11.5, anchor="middle")
path("M859 90 V48 Q859 40 851 40 H587 Q579 40 579 48 V82", BLUE, 1.2, dash="4 3", end=(579, 88), end_angle=90)
rich(719, 34, [R("∇", color=BLUE_DARK), M("φ", sub=True, color=BLUE_DARK), M("ℒ", color=BLUE_DARK)], size=13, anchor="middle")

# =============================================================================================== (b)
# The same columns and shapes as (a): environment | encoder f_phi (frozen) | policy, with o_t over the top and the
# action back along the bottom.
frame(228, 218, [S("(b)", weight="bold"), S("  Online: PPO with the frozen encoder", weight="bold")])
TOP, ROW = 278, 357                        # block top, the main row (between the actor and the critic)

# environment (the Pilots column)
box(22, TOP, 238, 136)
rich(34, TOP + 20, [S("Environment", weight="bold"), S("  BatteryPlane", color=INK2)], size=12)
for k, runs in enumerate([[S("battery "), M("b"), M("t", sub=True), S(" is hidden")],
                          [S("every action drains "), M("b"), M("t", sub=True)],
                          [M("b"), M("t", sub=True), S(" caps the thrust")]]):
    y = TOP + 56 + k * 28
    dot(34, y - 4, MUTED, 2.1)
    rich(42, y, [(t, {**o, "color": INK2}) for t, o in runs], size=12.5)

arrow(260, ROW, 466, ROW)
rich(363, ROW - 8, [M("o"), M("t", sub=True), R(", "), M("a"), M("t", sub=True), R("−1", sub=True)], size=13, anchor="middle")

# the encoder of (a), frozen: one block
box(468, ROW - 32, 222, 64, fill=BLUE_TINT, stroke=BLUE, sw=1.3)
rich(579, ROW - 5, [S("Encoder ", weight="bold", color=BLUE_DARK), M("f", color=BLUE_DARK), M("φ", sub=True, color=BLUE_DARK),
                    S("  GRU", color=INK2)], size=13.5, anchor="middle")
add(f'<g transform="translate(546,{ROW + 14})" stroke="{BLUE}" stroke-width="1.1" stroke-linecap="round">'
    + "".join(f'<line x1="0" y1="0" x2="{5 * math.cos(math.radians(a)):.2f}" y2="{5 * math.sin(math.radians(a)):.2f}"/>'
              for a in range(0, 360, 60)) + "</g>")
rich(555, ROW + 18, [S("frozen: ", color=INK2), M("φ", color=INK2), S(" from (a)", color=INK2)], size=11.5)

arrow(690, ROW, 742, ROW, BLUE, 1.4)
rich(716, ROW - 7, [M("b", hat=True, color=BLUE_DARK), M("t", sub=True, color=BLUE_DARK)], size=13.5, anchor="middle")

# PPO: actor and critic
PPO_TOP, PPO_BOTTOM = TOP + 8, TOP + 136
box(744, PPO_TOP, 230, PPO_BOTTOM - PPO_TOP, fill=VIOLET_TINT, stroke=VIOLET, sw=1.3)
rich(756, PPO_TOP + 18, [S("PPO", weight="bold", color=VIOLET_DARK), S("  trained on reward ", color=INK2), M("r", color=INK2),
                         M("t", sub=True, color=INK2)], size=11.5)
ACTOR, CRITIC = (PPO_TOP + 28, PPO_TOP + 68), (PPO_TOP + 78, PPO_TOP + 118)
for (y0, y1), name, formula in ((ACTOR, "Actor", [M("π"), M("θ", sub=True), R("("), M("a"), M("t", sub=True), R(" | "), M("o"),
                                                  M("t", sub=True), R(", "), M("b", hat=True), M("t", sub=True), R(")")]),
                                (CRITIC, "Critic", [M("V"), M("ψ", sub=True), R("("), M("o"), M("t", sub=True), R(", "),
                                                    M("b", hat=True), M("t", sub=True), R(")")])):
    box(756, y0, 206, y1 - y0, fill="#ffffff", stroke=VIOLET, sw=1.0, r=5)
    rich(766, (y0 + y1) / 2 + 4.5, [S(name, weight="bold", color=VIOLET_DARK)], size=11.5)
    rich(888, (y0 + y1) / 2 + 5.5, formula, size=14.5, anchor="middle")

# the observation over the top, the action from the actor back along the bottom
path(f"M141 {TOP} V270 Q141 262 149 262 H851 Q859 262 859 270 V{PPO_TOP - 6}", LINE, 1.3, end=(859, PPO_TOP), end_angle=90)
rich(500, 257, [M("o"), M("t", sub=True)], size=13, anchor="middle")
ay = (ACTOR[0] + ACTOR[1]) / 2
path(f"M962 {ay} H976 Q984 {ay} 984 {ay + 8} V422 Q984 430 976 430 H149 Q141 430 141 422 V{TOP + 142}", LINE, 1.3,
     end=(141, TOP + 136), end_angle=-90)
rich(363, 425, [S("action ", color=INK2), M("a"), M("t", sub=True)], size=12.5, anchor="middle")

# ----------------------------------------------------------------------------------------------- write
OUT.mkdir(exist_ok=True)
(OUT / "fig1_diagram.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
                                      + "\n".join(parts) + "\n</svg>\n")

import resvg_py  # noqa: E402
from reportlab.graphics import renderPDF  # noqa: E402
from svglib import fonts as svgfonts  # noqa: E402
from svglib.svglib import svg2rlg  # noqa: E402

png = resvg_py.svg_to_bytes(svg_path=str(OUT / "fig1_diagram.svg"), font_dirs=[str(ROBOTO), str(STIX)], zoom=4, skip_system_fonts=True)
(OUT / "fig1_diagram.png").write_bytes(bytes(png))
for (fam, weight, style), (face, file, family) in FACES.items():
    svgfonts.register_font(family, str(file), weight=weight, style=style, rlgFontName=face)
renderPDF.drawToFile(svg2rlg(str(OUT / "fig1_diagram.svg")), str(OUT / "fig1_diagram.pdf"))
print("written", OUT / "fig1_diagram.svg", "(.png, .pdf)")
