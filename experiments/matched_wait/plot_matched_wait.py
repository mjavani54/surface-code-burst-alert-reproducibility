"""Plot the paired policy comparisons at the matched wait budget."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from write_results import read_csv


def error_interval(rows, group, metric, scale):
    row = rows[group, "", metric]
    point, low, high = [scale * float(row[k]) for k in ("estimate", "ci_low", "ci_high")]
    return point, (point - low, high - point)


def plot(folder, output):
    rows = {(r["group"], r["value"], r["metric"]): r for r in read_csv(folder / "summary.csv")}
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8e9ca6", "text.color": "#24343e", "axes.labelcolor": "#304653",
        "xtick.color": "#304653", "ytick.color": "#304653", "axes.titlecolor": "#24343e",
    })
    fig, (gain, delay) = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=.245, right=.973, bottom=.25, top=.74, wspace=.70)
    fig.suptitle("Delayed alerts and syndrome lookahead at matched wait budgets",
                 y=.956, fontsize=16.5, weight="semibold")
    fig.text(.5, .865, "Distance 7 · 84 syndrome rounds · Three additional rounds per extended commit",
             ha="center", fontsize=10, color="#435968")
    green, gray, accent = "#166d63", "#526c80", "#9b5b33"
    comparisons = (
        ("selective_vs_random", "vs. random alert waiting", green),
        ("selective_vs_syndrome", "vs. syndrome-only lookahead", green),
        ("selective_vs_lookahead", "vs. lookahead with pre-deadline alerts", green),
        ("both_vs_lookahead", "add late alerts to lookahead", accent),
    )
    for y, (metric, _, color) in enumerate(comparisons):
        value, err = error_interval(rows, "population", metric, 1e6)
        gain.errorbar(value, y, xerr=np.asarray(err)[:, None], fmt="o", capsize=3, color=color, linewidth=1.6)
    gain.axvline(0, color="#80919d", linestyle="--", linewidth=1)
    gain.set(title="Paired logical error differences", xlabel="Fewer failures per million memory trials",
             yticks=range(len(comparisons)), yticklabels=[label for _, label, _ in comparisons], ylim=(3.6, -.6))

    for y, name in enumerate(("trigger", "random")):
        value, err = error_interval(rows, "population", f"delay_{name}", 1)
        delay.errorbar(value, y, xerr=np.asarray(err)[:, None], fmt="D", capsize=3,
                       color=green if name == "trigger" else gray, linewidth=1.6)
    delay.set(title="Added wait", xlabel="Mean added rounds per memory trial",
              yticks=(0, 1), yticklabels=("Trigger", "Random"), ylim=(1.65, -.65))
    for axis in (gain, delay):
        axis.grid(axis="x", color="#dfe7ec")
        axis.set_axisbelow(True)
    fig.text(.5, .075, "First three rows compare selective alert waiting with each control. The last row compares combined inputs with lookahead.",
             ha="center", fontsize=8.6, color="#435968")
    fig.text(.5, .04, "Pointwise 95% intervals · Population estimates assume a 0.1% burst probability per memory trial.",
             ha="center", fontsize=8.6, color="#435968")
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output)
