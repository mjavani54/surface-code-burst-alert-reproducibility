"""Plot the paired failure reductions and the measured wait rates."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from write_results import read_csv


def plot(folder, output):
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    rows = {(row["metric"], row["group"]): row for row in read_csv(folder / "summary.csv")}
    plt.rcParams.update({
        "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8b99a4", "text.color": "#20323f",
        "axes.labelcolor": "#344a5a", "xtick.color": "#344a5a", "ytick.color": "#344a5a",
    })
    fig, (gain_axis, wait_axis) = plt.subplots(1, 2, figsize=(10.3, 4.05))
    fig.subplots_adjust(left=.15, right=.98, bottom=.23, top=.74, wspace=.54)
    fig.suptitle("Selective waiting for delayed burst alerts", y=.965, fontsize=16, weight="semibold")
    fig.text(.5, .875, "Distance 7  ·  14-round alert delay  ·  3-round deadline extension",
             ha="center", fontsize=10, color="#435968")
    fixed, selective = "#526578", "#17665d"
    for y, name, color in ((0, "always_wait_gain", fixed), (1, "selective_wait_gain", selective)):
        row = rows[name, "population"]
        value, low, high = [1e6 * float(row[key]) for key in ("gain", "ci_low", "ci_high")]
        gain_axis.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o",
                           color=color, capsize=4, linewidth=1.6, markersize=6)
    gain_axis.axvline(0, color="#81909c", linestyle="--", linewidth=.9)
    gain_axis.set(
        title="Gain versus the fixed deadline", xlabel="Fewer logical failures per million trials",
        yticks=[0, 1], yticklabels=["Always wait", "Selective wait"], ylim=(1.57, -.57), xlim=(-.11, 1.48),
    )

    row = rows["selective_wait_rate", "ordinary"]
    rate, low, high = [100 * float(row[key]) for key in ("gain", "ci_low", "ci_high")]
    budget = 100 * meta["ordinary_wait_target"]
    wait_axis.axvspan(0, budget, color="#e8f2ef", zorder=0)
    wait_axis.axvline(budget, color="#526578", linestyle="--", linewidth=1.1)
    wait_axis.errorbar(rate, 0, xerr=[[rate - low], [high - rate]], fmt="D", color=selective, markersize=6)
    wait_axis.text(rate, -.23, f"{rate:.3f}%", color=selective, ha="center")
    wait_axis.annotate("1% limit", (budget, .27), xytext=(-6, 0), textcoords="offset points",
                       color=fixed, ha="right", fontsize=9.5)
    wait_axis.set(title="Selective waiting on ordinary trials", xlabel="Trials with an extended deadline (%)",
                  yticks=[], xlim=(0, 1.2), xticks=np.arange(0, 1.21, .2), ylim=(.57, -.57))
    for ax in (gain_axis, wait_axis):
        ax.grid(axis="x", color="#dfe8ed", linewidth=.8)
        ax.set_axisbelow(True)
    fig.text(.5, .035, "Pointwise 95% intervals  ·  Overall gain assumes a burst in 0.1% of memory trials",
             ha="center", color="#435968", fontsize=9)
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=.15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output)
