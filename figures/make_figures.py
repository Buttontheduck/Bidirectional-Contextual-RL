"""The paper figures, from figures/data/ (figures/collect_data.py). Writes figures/out/fig<k>_<name>.{pdf,png,svg}.

Headline: plain PPO with a frozen GRU trained offline on true-battery labels. Ablation: the same GRU trained on the ideal
estimate (T1). Every agent keeps one colour in every figure (categorical slots validated for colour-vision deficiency,
all pairs, light surface); references are in ink, agents without the battery or trained end to end in grey.

    python figures/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "out"
WIDTH = 7.0                                 # double-column width, inches

# Ink and chrome
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LIGHT = "#b5b3ac"                           # agents without the battery / trained end to end
# Entities -> colour, the same in every figure
LABELS, TEACHER, XLSTM = "#2a78d6", "#eb6834", "#1baf7a"
IDEAL, TRUE = INK, INK2                     # references: the ideal estimate (solid), the true battery (dashed)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "lines.linewidth": 1.6,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(name)


def panel(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left", pad=6)


def shared_legend(fig, entries, y=1.02, ncol=None):
    """One legend above the panels: entries of (label, color, linestyle)."""
    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.6, marker="o" if ls != ":" else None, markersize=3.5,
                      markeredgecolor="white", markeredgewidth=0.5) for _, c, ls in entries]
    fig.legend(handles, [e[0] for e in entries], loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(entries),
               handlelength=2.2, columnspacing=1.6)


def binned(curve, dt, every=5):
    """Per-step RMSE pooled over bins of `every` steps (weighted by the flights still flying); times at bin centres."""
    rmse, n = np.array(curve["rmse"]), np.array(curve["flights"], float)
    k = len(rmse) // every * every
    sq = (rmse[:k] ** 2 * n[:k]).reshape(-1, every).sum(1) / n[:k].reshape(-1, every).sum(1)
    t = (np.arange(k).reshape(-1, every).mean(1) + 0.5) * dt
    return t, np.sqrt(sq)


def mean_curve(curves, dt):
    ts, ys = zip(*(binned(c, dt) for c in curves))
    k = min(len(t) for t in ts)
    return ts[0][:k], np.mean([y[:k] for y in ys], axis=0)


def dot_rows(ax, rows, xlabel, value_fmt="{:.0f}", show_values=True):
    """A dot plot: each row's values as dots, their mean as a tick, the mean written at the right. rows: dicts with
    label, values, color, and optional group (a header row before it)."""
    y, ticks, labels = 0, [], []
    for row in rows:
        if row.get("group"):
            y -= 0.35
            ax.text(-0.01, y, row["group"], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7.5, fontweight="bold", color=INK)
            y -= 0.85
        values = np.asarray(row["values"], float)
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, y + jitter, s=16, color=row["color"], alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([values.mean()] * 2, [y - 0.32, y + 0.32], color=row["color"] if row["color"] != LIGHT else MUTED,
                linewidth=2.2, zorder=4, solid_capstyle="round")
        if show_values:
            ax.text(1.01, y, value_fmt.format(values.mean()), transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, color=INK)
        ticks.append(y)
        labels.append(row["label"])
        y -= 1
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK2)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


# ----------------------------------------------------------------------------------------------- figure 1: method
def fig1():
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 40)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge=AXIS, face=SURFACE, weight="bold"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=1.2,
                                    edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8, fontweight=weight, color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.35)

    def arrow(p, q, text=None, color=INK2, rad=0.0, dy=1.4):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=color,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
        if text:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=6.8, color=INK2)

    ax.text(0.5, 39.5, "Two-stage training (ours)", fontsize=8.5, fontweight="bold", color=INK, va="top")
    box(1, 22, 21, 13.5, "Pilot flights", "11 flying styles,\n19,000 flights;\ntrue battery recorded")
    box(27.5, 22, 24, 13.5, "Offline GRU encoder", "reads (o_t, a_t-1);\ntrained to predict b_t\n(the true battery)", edge=LABELS)
    box(57, 22, 18, 13.5, "Freeze", "the GRU's estimate\nof the battery, b̂_t")
    box(81, 22, 18, 13.5, "PPO policy", "acts on (o_t, b̂_t);\nnever sees b_t")
    arrow((22.6, 28.7), (27.4, 28.7))
    arrow((51.6, 28.7), (56.4, 28.7))
    arrow((75.6, 28.7), (80.4, 28.7))
    ax.text(39.5, 20.2, "ablation: target = the ideal estimate instead of b_t", ha="center", va="top", fontsize=6.8,
            color=TEACHER, fontweight="bold")

    ax.text(0.5, 15.2, "End-to-end training (fails: the autogenic loop)", fontsize=8.5, fontweight="bold", color=INK, va="top")
    nodes = [(12.5, 7, "Policy flies one\nnarrow style"), (37.5, 7, "Battery rarely shows\nin the flights"),
             (62.5, 7, "Encoder learns a clock\ninstead of the battery"), (87.5, 7, "Estimate adds nothing:\npolicy ignores it")]
    for x, y, text in nodes:
        ax.add_patch(FancyBboxPatch((x - 10.5, y - 4.0), 21, 8.0, boxstyle="round,pad=0.3,rounding_size=1.2", linewidth=1.0,
                                    edgecolor=AXIS, facecolor="white"))
        ax.text(x, y, text, ha="center", va="center", fontsize=6.8, color=INK2, linespacing=1.3)
    for (x0, y0, _), (x1, y1, _) in zip(nodes[:-1], nodes[1:]):
        arrow((x0 + 11.0, y0), (x1 - 11.0, y1))
    # Back to the start, below the boxes
    ax.add_patch(FancyArrowPatch((87.5, 2.6), (12.5, 2.6), arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=INK2,
                                 connectionstyle="arc3,rad=-0.06", shrinkA=0, shrinkB=0))
    save(fig, "fig1_method")


# ----------------------------------------------------------------------------------------------- figure 2: headline
def fig2():
    s = load("final_test")["summary"]
    v = lambda key: s[key]["final_test_seed_scores"]
    rows = [
        {"group": "No battery information", "label": "PPO, batch 256", "values": v("0a_blind"), "color": LIGHT},
        {"label": "PPO, batch 1024", "values": v("0b_blind"), "color": LIGHT},
        {"label": "Contextual PPO", "values": v("0c_blind"), "color": LIGHT},
        {"group": "Memory, trained end to end", "label": "Contextual PPO + xLSTM (A, 3 seeds)", "values": v("A"), "color": LIGHT},
        {"label": "Contextual PPO + GRU, joint (J1, 5 seeds)", "values": v("J1"), "color": LIGHT},
        {"group": "Memory, two-stage (ours)", "label": "PPO + frozen GRU, batch 256", "values": v("4a_encoder (E1_gru_point_main_s1)"), "color": LABELS},
        {"label": "PPO + frozen GRU, batch 1024", "values": v("4b_encoder (E1_gru_point_main_s1)"), "color": LABELS},
        {"group": "References", "label": "PPO + true battery, batch 256", "values": v("0a"), "color": TRUE},
        {"label": "PPO + true battery, batch 1024", "values": v("0b"), "color": TRUE},
        {"label": "Contextual PPO + true battery", "values": v("0c"), "color": TRUE},
        {"label": "PPO + ideal estimate, batch 256", "values": v("2a_ideal"), "color": IDEAL},
        {"label": "PPO + ideal estimate, batch 1024", "values": v("2b_ideal"), "color": IDEAL},
    ]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.78, 3.9))
    dot_rows(ax, rows, "Final-test return (seeds 2000–2002; dots: training seeds, tick: mean)")
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(1.01, 0.62, "mean", transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED, va="center")
    save(fig, "fig2_final_test")


# ----------------------------------------------------------------------------------------------- figure 3: by battery
AGENTS_256 = [("0a_blind", "No battery information", LIGHT, "-"), ("4a_encoder (E1_gru_point_main_s1)", "Frozen GRU (ours)", LABELS, "-"),
              ("0a", "True battery", TRUE, "--"), ("2a_ideal", "Ideal estimate", IDEAL, "-")]


def fig3():
    by = load("final_test")["by_b0"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for key, label, color, style in AGENTS_256:
        b0 = sorted(by[key], key=float)
        x = [float(k) for k in b0]
        ret = [by[key][k]["return"] for k in b0]
        done = [100 * by[key][k]["completed"] for k in b0]
        for ax, y in ((a, ret), (b, done)):
            ax.plot(x, y, style, color=color if color != LIGHT else MUTED, marker="o", markersize=4, label=label,
                    markeredgecolor="white", markeredgewidth=0.6)
        offset = {"Ideal estimate": 22, "True battery": -22}.get(label, 0)
        a.text(0.715, ret[-1] + offset, label, va="center", fontsize=7, color=INK2)
    panel(a, "a", "Return by starting battery (PPO, batch 256)")
    a.set_xlabel("Starting battery")
    a.set_ylabel("Mean final-test return")
    a.set_xlim(0.18, 0.86)
    a.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    a.axhline(0, color=AXIS, linewidth=0.8)
    panel(b, "b", "Flights completed")
    b.set_xlabel("Starting battery")
    b.set_ylabel("Completed (%)")
    b.set_ylim(-5, 105)
    b.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    fig.tight_layout(w_pad=2.5)
    shared_legend(fig, [(label, color if color != LIGHT else MUTED, style) for _, label, color, style in AGENTS_256])
    save(fig, "fig3_by_battery")


# ----------------------------------------------------------------------------------------------- figure 4: offline
def offline_panel(ax, curves, entities, letter, title, label_offsets=None):
    """Error against flight time (mean over seeds), each line labelled at its end."""
    dt = curves["dt"]
    for key, label, color, style in entities:
        t, y = binned(curves[key], dt) if isinstance(curves[key], dict) else mean_curve(curves[key], dt)
        keep = t <= 34.5
        ax.plot(t[keep], y[keep], style, color=color, label=label, linewidth=1.6 if key != "clock" else 1.4)
        ax.text(35.4, y[keep][-1] + (label_offsets or {}).get(key, 0), label, va="center", fontsize=6.8, color=INK2,
                clip_on=False)
    panel(ax, letter, title)
    ax.set_xlabel("Time into the flight (s)")
    ax.set_ylabel("Battery error (RMSE)")
    ax.set_xlim(0, 35)
    ax.set_ylim(0.06, 0.16)


def fig4():
    curves, train = load("offline_curves"), load("training_curves")
    arch = load("architecture")
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    entities = [("clock", "Clock only", MUTED, ":"), ("xlstm", "xLSTM", XLSTM, "-"), ("gru_labels", "GRU", LABELS, "-"),
                ("ideal", "Ideal", IDEAL, "-")]
    offline_panel(a, curves, entities, "a", "Offline error over a flight (pilot test flights)",
                  {"xlstm": 0.0035, "gru_labels": -0.0025})
    for run in train["gru_labels"]:
        b.plot(np.arange(1, len(run["validation_rmse"]) + 1), run["validation_rmse"], color=LABELS, linewidth=1.0, alpha=0.9)
    for run in train["xlstm"]:
        b.plot(np.arange(1, len(run["validation_rmse"]) + 1), run["validation_rmse"], color=XLSTM, linewidth=1.3)
    b.axhline(train["clock_validation"], color=MUTED, linestyle=":", linewidth=1.4)
    b.axhline(train["floor_validation"], color=IDEAL, linewidth=1.2)
    b.text(40.6, train["clock_validation"], "Clock only", va="center", fontsize=6.8, color=INK2)
    b.text(40.6, train["floor_validation"], "Ideal", va="center", fontsize=6.8, color=INK2)
    b.annotate("the 'click'", xy=(10.5, 0.114), xytext=(16, 0.132), fontsize=7, color=INK2,
               arrowprops={"arrowstyle": "-|>", "color": INK2, "linewidth": 0.8, "mutation_scale": 7})
    panel(b, "b", "Validation error per training pass")
    b.set_xlabel("Training pass")
    b.set_ylabel("Battery error (RMSE)")
    b.set_xlim(0, 40)
    b.set_ylim(0.08, 0.15)
    fig.tight_layout(w_pad=4.5)
    shared_legend(fig, [("Clock only", MUTED, ":"), ("xLSTM (3 seeds)", XLSTM, "-"), ("GRU, true-battery labels (5 seeds)", LABELS, "-"),
                        ("Ideal estimator", IDEAL, "-")])
    save(fig, "fig4_offline")


# ----------------------------------------------------------------------------------------------- figure 5: closed loop
def bias_panel(ax, group, color, letter, title):
    """Bias by starting battery of the frozen GRU's estimate and of the ideal estimate on the same PPO flights."""
    b0 = sorted(group["by_b0"], key=float)
    x = [float(k) for k in b0]
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.plot(x, [group["by_b0"][k]["ideal_bias"] for k in b0], color=IDEAL, marker="o", markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.6, label="Ideal estimate, same flights")
    ax.plot(x, [group["by_b0"][k]["encoder_bias"] for k in b0], color=color, marker="o", markersize=4,
            markeredgecolor="white", markeredgewidth=0.6, label="Frozen GRU's estimate")
    panel(ax, letter, title)
    ax.set_xlabel("Starting battery")
    ax.set_ylabel("Bias: estimate − true battery")
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_ylim(-0.16, 0.23)


def swap_panel(ax, swap, groups, letter, title):
    for y, (key, label, color) in enumerate(groups):
        seeds = swap[key]
        own = np.array([seeds[s]["own"] for s in sorted(seeds)])
        true = np.array([seeds[s]["true_battery"] for s in sorted(seeds)])
        yy = -y
        ax.scatter(own, [yy + 0.12] * len(own), s=7, color=color, alpha=0.35, linewidth=0)
        ax.scatter(true, [yy - 0.12] * len(true), s=7, color=TRUE, alpha=0.35, linewidth=0)
        ax.add_patch(FancyArrowPatch((own.mean(), yy), (true.mean(), yy), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.3, color=INK2, shrinkA=4, shrinkB=4, zorder=3))
        ax.scatter([own.mean()], [yy], s=34, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter([true.mean()], [yy], s=34, facecolor="white", edgecolor=TRUE, linewidth=1.4, zorder=4)
        ax.text(1.01, yy, f"{true.mean() - own.mean():+.0f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5)
    ax.set_yticks([-y for y in range(len(groups))], [g[1] for g in groups])
    ax.tick_params(axis="y", length=0, labelcolor=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-len(groups) + 0.4, 0.6)
    ax.set_xlabel("Validation return (filled: own encoder; hollow: true battery instead)")
    panel(ax, letter, title)


def fig5():
    loop, swap = load("closed_loop"), load("swap")
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
    bias_panel(a, loop["E1_gru_point_main_s1|256"], LABELS, "a", "On its policies' own flights (batch 256)")
    a.legend(loc="upper right")
    groups = [("4a_encoder (E1_gru_point_main_s1)", "GRU s1, batch 256", LABELS), ("4b_encoder (E1_gru_point_main_s1)", "GRU s1, batch 1024", LABELS),
              ("4b_encoder (E1_gru_point_main_s0)", "GRU s0, batch 1024", LABELS), ("4b_encoder (E1_gru_point_main_s4)", "GRU s4, batch 1024", LABELS)]
    swap_panel(b, swap, groups, "b", "Swap test: the true battery in place of the estimate")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_closed_loop")


# ----------------------------------------------------------------------------------------------- figure 6: ablation
def fig6():
    curves, s, by, loop, swap = load("offline_curves"), load("final_test")["summary"], load("final_test")["by_b0"], load("closed_loop"), load("swap")
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 7.6), gridspec_kw={"width_ratios": [1, 1.12]})
    (a, b), (c, d), (e, f) = axes
    offline_panel(a, curves, [("clock", "Clock only", MUTED, ":"), ("gru_labels", "Labels", LABELS, "-"),
                              ("gru_teacher", "Teacher", TEACHER, "-"), ("ideal", "Ideal", IDEAL, "-")],
                  "a", "Offline error over a flight", {"gru_labels": 0.003, "gru_teacher": -0.0005, "ideal": -0.003})
    # (b) final test, every encoder seed, with the references of each batch size
    v = lambda key: s[key]["final_test_seed_scores"]
    rows = [
        {"group": "Batch 256", "label": "Labels, encoder s1", "values": v("4a_encoder (E1_gru_point_main_s1)"), "color": LABELS},
        {"label": "Teacher, encoder s3", "values": v("4a_encoder (E1_gru_teacher_point_main_s3)"), "color": TEACHER},
        {"group": "Batch 1024", "label": "Labels, encoder s0", "values": v("4b_encoder (E1_gru_point_main_s0)"), "color": LABELS},
        {"label": "Labels, encoder s1", "values": v("4b_encoder (E1_gru_point_main_s1)"), "color": LABELS},
        {"label": "Labels, encoder s4", "values": v("4b_encoder (E1_gru_point_main_s4)"), "color": LABELS},
        {"label": "Teacher, encoder s0", "values": v("4b_encoder (E1_gru_teacher_point_main_s0)"), "color": TEACHER},
        {"label": "Teacher, encoder s2", "values": v("4b_encoder (E1_gru_teacher_point_main_s2)"), "color": TEACHER},
        {"label": "Teacher, encoder s3", "values": v("4b_encoder (E1_gru_teacher_point_main_s3)"), "color": TEACHER},
    ]
    dot_rows(b, rows, "Final-test return")
    for lo, hi, true, ideal in ((-0.9, -2.6, s["0a"]["final_test_mean_reward"], s["2a_ideal"]["final_test_mean_reward"]),
                                (-3.8, -9.7, s["0b"]["final_test_mean_reward"], s["2b_ideal"]["final_test_mean_reward"])):
        b.plot([true, true], [lo, hi], "--", color=TRUE, linewidth=1.0, zorder=2)
        b.plot([ideal, ideal], [lo, hi], color=IDEAL, linewidth=1.0, zorder=2)
    panel(b, "b", "PPO with the frozen GRU (all encoder seeds)")
    # (c) return by starting battery
    for key, label, color, style in [("2a_ideal", "Ideal estimate", IDEAL, "-"), ("0a", "True battery", TRUE, "--"),
                                     ("4a_encoder (E1_gru_point_main_s1)", "Labels", LABELS, "-"),
                                     ("4a_encoder (E1_gru_teacher_point_main_s3)", "Teacher", TEACHER, "-")]:
        b0 = sorted(by[key], key=float)
        c.plot([float(k) for k in b0], [by[key][k]["return"] for k in b0], style, color=color, marker="o", markersize=3.5,
               markeredgecolor="white", markeredgewidth=0.6)
    c.axhline(0, color=AXIS, linewidth=0.8)
    c.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    c.set_xlabel("Starting battery")
    c.set_ylabel("Mean final-test return")
    panel(c, "c", "By starting battery (batch 256)")
    # (d) swap test
    groups = [("4a_encoder (E1_gru_point_main_s1)", "Labels s1, 256", LABELS), ("4b_encoder (E1_gru_point_main_s1)", "Labels s1, 1024", LABELS),
              ("4b_encoder (E1_gru_point_main_s0)", "Labels s0, 1024", LABELS), ("4b_encoder (E1_gru_point_main_s4)", "Labels s4, 1024", LABELS),
              ("4a_encoder (E1_gru_teacher_point_main_s3)", "Teacher s3, 256", TEACHER), ("4b_encoder (E1_gru_teacher_point_main_s0)", "Teacher s0, 1024", TEACHER),
              ("4b_encoder (E1_gru_teacher_point_main_s2)", "Teacher s2, 1024", TEACHER), ("4b_encoder (E1_gru_teacher_point_main_s3)", "Teacher s3, 1024", TEACHER)]
    swap_panel(d, swap, groups, "d", "Swap test (validation flights)")
    d.set_xlabel("Return: own encoder (filled) → true battery (hollow)")
    # (e, f) closed-loop bias, each on its own policies' flights
    bias_panel(e, loop["E1_gru_point_main_s1|256"], LABELS, "e", "Bias on the labels GRU's own PPO flights")
    bias_panel(f, loop["E1_gru_teacher_point_main_s3|256"], TEACHER, "f", "Bias on the teacher GRU's own PPO flights")
    f.set_ylabel("")
    fig.tight_layout(w_pad=2.0, h_pad=2.6)
    shared_legend(fig, [("GRU, true-battery labels", LABELS, "-"), ("GRU, ideal-estimate teacher", TEACHER, "-"),
                        ("Ideal estimate", IDEAL, "-"), ("True battery", TRUE, "--"), ("Clock only", MUTED, ":")], y=1.0)
    save(fig, "fig6_ablation_target")


# ----------------------------------------------------------------------------------------------- figure 7: architecture
def fig7():
    arch = load("architecture")
    colors = {"xlstm": XLSTM, "gru_labels": LABELS, "gru_teacher": TEACHER}
    rows = [{"label": r["label"], "values": r["values"], "color": colors[r["kind"]]} for r in arch["rows"]]
    rows[0]["group"] = "True-battery labels"
    rows[-1]["group"] = "Ideal-estimate teacher"
    fig, ax = plt.subplots(figsize=(WIDTH * 0.72, 2.9))
    dot_rows(ax, rows, "Offline test error (RMSE, pilot test flights)", value_fmt="{:.4f}")
    for x, label, style in ((arch["clock"], "clock only", ":"), (arch["floor"], "ideal estimator", "-")):
        ax.axvline(x, color=MUTED if style == ":" else IDEAL, linestyle=style, linewidth=1.2, zorder=1)
        ax.text(x + (0.0007 if style == "-" else -0.0007), 0.75, label, ha="left" if style == "-" else "right", va="bottom",
                fontsize=7, color=INK2)
    ax.set_xlim(0.08, 0.15)
    ax.set_ylim(ax.get_ylim()[0], 1.4)
    save(fig, "fig7_ablation_architecture")


# ----------------------------------------------------------------------------------------------- figure 8: end to end
def fig8():
    jg = load("joint_and_grid")
    names = {"J0": "J0: A's budget, GRU", "J1": "J1: 16× encoder training", "J2": "J2: J1 + entropy bonus",
             "J3": "J3: J2 + Gaussian head", "J4": "J4: J3 + curiosity", "J5": "J5: J1 + warm-up"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    rows = [{"label": names[k], "values": jg["joint"][k], "color": LIGHT} for k in names]
    rows[0]["group"] = "Joint training (GRU)"
    rows.append({"group": "References", "label": "Experiment A rerun (xLSTM)", "values": jg["references"]["A"], "color": LIGHT})
    rows.append({"label": "No context (memoryless)", "values": jg["references"]["0c_blind"], "color": LIGHT})
    rows.append({"label": "True battery", "values": jg["references"]["0c"], "color": TRUE})
    dot_rows(a, rows, "Validation late score (1.5M–2M steps)")
    a.axvline(np.mean(jg["references"]["0c_blind"]), color=MUTED, linestyle=":", linewidth=1.1, zorder=1)
    panel(a, "a", "Contextual PPO trained end to end")
    labels = {"A": "A: point estimate", "B": "B: A + speed head", "C": "C: Gaussian (β = 0)", "D": "D: Gaussian (β = 0.5)",
              "E": "E: Gaussian, mean only", "F": "F: C + speed head"}
    rows = [{"label": labels[k], "values": jg["grid"][k], "color": LIGHT} for k in labels]
    dot_rows(b, rows, "Final validation return")
    b.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    panel(b, "b", "Encoder-head grid A–F (unscaled rewards)")
    fig.tight_layout(w_pad=3.0)
    save(fig, "fig8_end_to_end")


if __name__ == "__main__":
    for make in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        make()
