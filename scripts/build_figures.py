"""Build the manuscript figures from the archived simulation estimates."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
BLUE = "#415b73"
TEAL = "#146b61"
INK = "#263641"
GREY = "#92a0ab"


def read_summary(folder):
    with (EXPERIMENTS / folder / "summary.csv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=340, facecolor="white")
    plt.close(fig)


def timing_diagram(output_dir):
    fig, ax = plt.subplots(figsize=(6.7, 1.48))
    fig.subplots_adjust(left=.023, right=.977, top=.96, bottom=.19)
    ax.set_xlim(13.8, 32.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for left, right, label, fill in (
        (14.5, 21.5, "Committed rounds\n15–21", "#c6d6e4"),
        (21.5, 28.5, "Nominal lookahead\n22–28", "#e5edf4"),
        (28.5, 31.5, "Extra wait\n29–31", "#d0e8df"),
    ):
        ax.add_patch(Rectangle((left, .27), right - left, .37, facecolor=fill, edgecolor="white", linewidth=.8))
        ax.text((left + right) / 2, .452, label, ha="center", va="center", fontsize=9.5, color=INK)
    endpoints = (
        (21.5, "Commit cut", INK, "center"),
        (28.5, "Deadline 28", BLUE, "right"),
        (31.5, "Deadline 31", TEAL, "right"),
    )
    for x, label, color, alignment in endpoints:
        ax.vlines(x, .21, .765, color, linewidth=1.1)
        ax.text(x, .84, label, ha=alignment, va="bottom", fontsize=8.5, color=color, clip_on=False)
    ax.annotate("Reports stamped 15–17 can arrive here",
                xy=(30, .25), xycoords="data", xytext=(27.9, .012), textcoords="data",
                arrowprops={"arrowstyle": "-", "color": TEAL, "linewidth": .8},
                ha="center", va="top", fontsize=8.6, color=TEAL)
    save(fig, output_dir, "figure_1")


def paired_interval(axis, position, row, *, marker, color, scale=1e6, outline=False):
    value, low, high = (scale * float(row[k]) for k in ("estimate", "ci_low", "ci_high"))
    axis.errorbar(
        value, position, xerr=[[value - low], [high - value]], fmt=marker, color=color,
        linewidth=1.45, capsize=3.4, markersize=5.5, markerfacecolor="white" if outline else color,
        markeredgewidth=1.2, zorder=3,
    )


def format_forest(axis):
    axis.axvline(0, color=GREY, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
    axis.grid(axis="x", color="#e3eaf0", linewidth=.72)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=3, width=.75)


def matched_wait(output_dir):
    rows = {r["metric"]: r for r in read_summary("matched_wait/results") if r["group"] == "population"}
    fig, axis = plt.subplots(figsize=(6.7, 2.22))
    fig.subplots_adjust(left=.432, right=.985, top=.94, bottom=.29)
    cases = (
        ("selective_vs_random", "Selective alerts vs. random waiting"),
        ("selective_vs_syndrome", "Selective alerts vs. syndrome-only"),
        ("both_vs_lookahead", "Add late alerts to lookahead"),
    )
    for index, (metric, _) in enumerate(cases):
        paired_interval(axis, index, rows[metric], marker="D" if index == 2 else "o",
                        color=TEAL if index == 2 else BLUE)
    axis.set(
        xlim=(-.09, 1.11), ylim=(2.53, -.56), yticks=range(len(cases)),
        yticklabels=[label for _, label in cases], xlabel="Fewer logical failures per million memory trials",
    )
    axis.set_xticks(np.arange(0, 1.01, .2))
    format_forest(axis)
    save(fig, output_dir, "figure_2")


def distance_nine(output_dir):
    fig, panels = plt.subplots(1, 2, figsize=(6.7, 2.75), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=.207, right=.977, hspace=.2, wspace=.22, bottom=.32, top=.84)
    runs = (read_summary("distance9/results"), read_summary("distance9/replication"))
    colors = (BLUE, TEAL)
    marks = ("o", "D")
    for a, scope in zip(panels, ("global", "local"), strict=True):
        for index, run in enumerate(runs):
            rows = {r["timing"]: r for r in run
                    if r["scope"] == scope and r["group"] == "population" and r["metric"] == "both_vs_lookahead"}
            offset = -.13 if index == 0 else .13
            for row, timing in enumerate(("fixed", "jitter")):
                paired_interval(a, row + offset, rows[timing], marker=marks[index], color=colors[index],
                                outline=index == 0)
        a.set(ylim=(1.47, -.47), yticks=[0, 1])
        format_forest(a)
    panels[0].set(title="(a) Global bursts", xlim=(-.075, 1.29), xticks=[0, .4, .8, 1.2],
                  yticklabels=["14 rounds", "12–16 rounds"])
    panels[1].set(title="(b) Localized bursts", xlim=(-.036, .052), xticks=[-.03, 0, .03],
                  yticklabels=[])
    panels[0].tick_params(axis="y", pad=6)
    panels[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.text(.57, .187, "Fewer logical failures per million memory trials", ha="center", color=INK, fontsize=9.5)
    handles = [
        Line2D([0], [0], marker=m, color=c, markerfacecolor="white" if i == 0 else c, markersize=5.5, linewidth=1.3)
        for i, (m, c) in enumerate(zip(marks, colors, strict=True))
    ]
    fig.legend(handles, ("Original", "Independent replication"), loc="lower center", bbox_to_anchor=(.57, .008),
               ncol=2, frameon=False, columnspacing=1.45, handlelength=1.8, fontsize=9)
    save(fig, output_dir, "figure_3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced" / "figures")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#81919e", "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    })
    timing_diagram(output_dir)
    matched_wait(output_dir)
    distance_nine(output_dir)
    print(f"Wrote three manuscript figures to {output_dir}")


if __name__ == "__main__":
    main()
