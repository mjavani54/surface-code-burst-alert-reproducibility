"""Plot the matched-budget effect of late alerts for all four scenarios."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from write_results import read_csv


def interval(rows, key, scale):
    result = rows[key]
    value, lo, hi = [scale * float(result[k]) for k in ("estimate", "ci_low", "ci_high")]
    return value, np.array([[value - lo], [hi - value]])


def plot(folder, output):
    rows = {(r["scope"], r["timing"], r["group"], r["metric"]): r for r in read_csv(folder / "summary.csv")}
    plt.rcParams.update({
        "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8d9ca6", "text.color": "#20323f", "axes.labelcolor": "#304653",
        "xtick.color": "#304653", "ytick.color": "#304653", "axes.titlecolor": "#20323f",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.204, right=.965, bottom=.252, top=.745, wspace=.54)
    fig.suptitle("Delayed burst alerts in a distance-9 surface code", y=.963, fontsize=17, weight="semibold")
    fig.text(.5, .875, "84 syndrome rounds · Three extra rounds per extended deadline",
             ha="center", fontsize=10.5, color="#435968")
    colors = {"fixed": "#50687e", "jitter": "#13746a"}
    labels = []
    for y, (scope, timing) in enumerate((s, t) for s in ("global", "local") for t in ("fixed", "jitter")):
        label = "Global" if scope == "global" else "Localized"
        labels.append(f"{label}, {'14 rounds' if timing == 'fixed' else '12–16 rounds'}")
        for ax, group, scale in zip(axes, ("population", "burst"), (1e6, 100), strict=True):
            value, err = interval(rows, (scope, timing, group, "both_vs_lookahead"), scale)
            ax.errorbar(value, y, xerr=err, color=colors[timing], fmt="o" if timing == "fixed" else "D",
                        capsize=3.5, linewidth=1.65, zorder=3)
    axes[0].set(title="All trials at 0.1% burst probability", xlabel="Fewer failures per million memory trials",
                yticks=range(4), yticklabels=labels)
    axes[1].set(title="Trials with a burst", xlabel="Fewer failures (percentage points)",
                yticks=range(4), yticklabels=[])
    for ax in axes:
        ax.set(ylim=(3.65, -.65))
        ax.axvline(0, color="#8293a0", linestyle="--", linewidth=1)
        ax.axhline(1.5, color="#d9e2e7", linewidth=.9)
        ax.grid(axis="x", color="#e0e7eb")
        ax.set_axisbelow(True)
    fig.text(.5, .103, "Each comparison adds late alerts to the same syndrome lookahead on the same trials.",
             ha="center", fontsize=9, color="#435968")
    fig.text(.5, .057, "Error bars show paired pointwise 95% intervals.",
             ha="center", fontsize=9, color="#435968")
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output)
