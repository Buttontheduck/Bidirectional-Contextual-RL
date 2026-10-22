"""The paper figures in the author's chosen style, from figures/data/ (figures/collect_data.py), into figures/out_person/.
Roboto for text (font_roboto, if importable; else matplotlib's default) and STIX for mathematics, as in figure 1.

Figure 2, the headline (final test, seeds 2000-2002), as two single-column figures with statistics over training seeds
instead of every run: the mean and its 95% bootstrap confidence interval (percentile, 20,000 resamples, fixed generator).
    2a  plain PPO at batch 256: no battery information, the frozen GRU (ours), the true battery, the ideal estimate
    2b  Contextual PPO, whose encoder is trained end to end (experiment A's xLSTM, joint training J1's GRU), next to its
        own no-battery and true-battery runs; both panels share one return axis

    PYTHONPATH=<dir with font_roboto> python figures/make_figures_person.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "out_person"
WIDTH = 7.0

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OURS, OURS_DARK, NONE, FAIL = "#2a78d6", "#1c5cab", "#cfcdc6", "#b5b3ac"
# Figure 3: no battery in a dark red, the ideal estimate in green (validated with blue: colour-vision separation 13.9)
RED, GREEN = "#a8322d", "#3aa655"
PURPLE = "#7a2e8e"           # the true battery (validated with red, blue and green, all pairs)

try:
    import font_roboto

    for ttf in (Path(font_roboto.__file__).parent / "files").glob("Roboto-*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    FAMILY = "Roboto"
except ImportError:
    FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FAMILY, "font.size": 8, "mathtext.fontset": "stix", "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "figure.facecolor": "white", "savefig.dpi": 300,
    "pdf.fonttype": 42, "svg.fonttype": "none"})

RNG_SEED, RESAMPLES = 0, 20000


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(name)


def mean_ci(x, rng):
    x = np.asarray(x, float)
    boot = rng.choice(x, (RESAMPLES, len(x))).mean(1)
    return x.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def bars(ax, rows, value_fmt, label_width=0.0):
    """Horizontal bars with 95% confidence intervals; rows: (label, sublabel, (mean, lo, hi), style)."""
    for k, (label, sublabel, (m, lo, hi), style) in enumerate(rows):
        y = -k
        if style == "reference":
            ax.barh(y, m, height=0.56, color="white", edgecolor=INK2, linewidth=1.1, zorder=2)
        else:
            ax.barh(y, m, height=0.56, color={"ours": OURS, "none": NONE, "fail": FAIL}[style], edgecolor="none", zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, linewidth=1.1, zorder=3, solid_capstyle="butt")
        for end in (lo, hi):
            ax.plot([end, end], [y - 0.1, y + 0.1], color=INK, linewidth=1.1, zorder=3)
        ax.text(max(hi, m, 0) + 0.02 * np.ptp(ax.get_xlim()), y, value_fmt.format(m).replace("-", "−"), va="center", ha="left", fontsize=8,
                fontweight="bold" if style == "ours" else "normal", color=OURS_DARK if style == "ours" else INK)
        ax.text(-label_width, y + 0.07, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8,
                fontweight="bold" if style == "ours" else "normal", color=OURS_DARK if style == "ours" else INK)
        ax.text(-label_width, y - 0.05, sublabel, transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=6.8, color=MUTED)
    ax.set_yticks([])
    ax.set_ylim(-len(rows) + 0.45, 0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig2_single(name, title, rows):
    """One headline figure, single-column width; both share the return axis (-60 to 360) so they compare side by side."""
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    ax.set_xlim(-60, 360)
    bars(ax, rows, "{:.0f}", label_width=0.03)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("Final-test return (mean, 95% CI)")
    ax.set_title(title, loc="left", x=-0.92, pad=10)
    save(fig, name)


def fig2():
    s = load("final_test")["summary"]
    v = lambda key: s[key]["final_test_seed_scores"]
    rng = np.random.default_rng(RNG_SEED)
    n = lambda key: f"{len(v(key))} seeds"
    fig2_single("fig2a_plain_ppo", "Plain PPO with a frozen encoder", [
        ("No battery information", f"no memory · {n('0a_blind')}", mean_ci(v("0a_blind"), rng), "none"),
        ("Frozen GRU (ours)", f"GRU trained offline, then frozen · {n('4a_encoder (E1_gru_point_main_s1)')}",
         mean_ci(v("4a_encoder (E1_gru_point_main_s1)"), rng), "ours"),
        ("True battery", f"battery observed · {n('0a')}", mean_ci(v("0a"), rng), "reference"),
        ("Ideal estimate", f"Bayes filter · {n('2a_ideal')}", mean_ci(v("2a_ideal"), rng), "reference")])
    fig2_single("fig2b_end_to_end", "Contextual PPO, encoder trained end to end", [
        ("No battery information", f"no memory · {n('0c_blind')}", mean_ci(v("0c_blind"), rng), "none"),
        ("End-to-end xLSTM", f"experiment A · {n('A')}", mean_ci(v("A"), rng), "fail"),
        ("End-to-end GRU", f"joint training (J1) · {n('J1')}", mean_ci(v("J1"), rng), "fail"),
        ("True battery", f"battery observed · {n('0c')}", mean_ci(v("0c"), rng), "reference")])


# Figure 3: by starting battery (final test, plain PPO at batch 256); spread = 1 standard deviation across training seeds
# key, label, line colour (3a), bar fill (3b), line style, band opacity
AGENTS3 = [("0a_blind", "No battery information", RED, RED, "-", 0.14),
           ("4a_encoder (E1_gru_point_main_s1)", "Frozen GRU (ours)", OURS, OURS, "-", 0.16),
           ("0a", "True battery", PURPLE, PURPLE, "-", 0.10),
           ("2a_ideal", "Ideal estimate", GREEN, GREEN, "--", 0.12)]


def per_seed(key, field):
    cells = load("final_test")["by_b0_seed"][key]
    b0 = sorted(cells, key=float)
    return np.array([float(b) for b in b0]), np.array([cells[b][field] for b in b0], float)   # [battery, seed]


def fig3a(agents=AGENTS3, name="fig3a_return_by_battery"):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for key, label, line, _, style, band in agents:
        x, y = per_seed(key, "return")
        mean, sd = y.mean(1), y.std(1, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, color=line, alpha=band, linewidth=0, zorder=1)
        ax.plot(x, mean, style, color=line, linewidth=1.7, marker="o", markersize=3.6, markeredgecolor="white",
                markeredgewidth=0.6, label=label, zorder=4 if key == "2a_ideal" else 3)   # the dashed ideal estimate on top
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_xlim(0.18, 0.72)
    ax.set_xlabel("Starting battery")
    ax.set_ylabel("Final-test return")
    ax.spines["left"].set_visible(True)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5, handlelength=2.0,
              columnspacing=1.2)
    ax.text(0.99, 0.02, "mean ± 1 SD across 8 seeds", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=MUTED)
    save(fig, name)


def fig3b(agents=AGENTS3, name="fig3b_completion_by_battery"):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    width = 0.76 / len(agents)
    for k, (key, label, line, fill, style, _) in enumerate(agents):
        x, y = per_seed(key, "completed")
        mean, sd = 100 * y.mean(1), 100 * y.std(1, ddof=1)
        pos = np.arange(len(x)) + (k - (len(agents) - 1) / 2) * width
        ax.bar(pos, mean, width * 0.86, color=fill, edgecolor="none", label=label, zorder=2)
        for px in pos[mean == 0]:                                   # an empty bar would look missing
            ax.text(px, 1.5, "0", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        # 1 SD across seeds, kept inside 0-100%
        lo, hi = np.clip(mean - sd, 0, 100), np.clip(mean + sd, 0, 100)
        show = sd > 0
        ax.vlines(pos[show], lo[show], hi[show], color=INK2, linewidth=0.8, zorder=3)
        ax.hlines(np.r_[lo[show], hi[show]], np.r_[pos[show], pos[show]] - width * 0.18, np.r_[pos[show], pos[show]] + width * 0.18,
                  color=INK2, linewidth=0.8, zorder=3)
    ax.set_xticks(np.arange(6), ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
    ax.set_xlim(-0.55, 5.55)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Starting battery")
    ax.set_ylabel("Flights completed (%)")
    ax.spines["left"].set_visible(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5, handlelength=1.4,
              columnspacing=1.2)
    ax.text(0.01, 0.985, "mean ± 1 SD across 8 seeds", transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED)
    save(fig, name)


if __name__ == "__main__":
    fig2()
    fig3a()
    fig3b()
    # 3c, 3d: the same without the ideal estimate
    without_ideal = [a for a in AGENTS3 if a[0] != "2a_ideal"]
    fig3a(without_ideal, "fig3c_return_by_battery_no_ideal")
    fig3b(without_ideal, "fig3d_completion_by_battery_no_ideal")
