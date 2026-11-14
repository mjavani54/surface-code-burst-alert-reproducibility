"""Plot the main delay and logical error comparisons for the rolling study."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from write_results import read_csv


COLORS = {"always": "#53677a", "transferred": "#b36226", "budgeted": "#187568"}


def interval(rows, group, value, metric, scale):
    row = rows[group, value, metric]
    mean, low, high = (scale * float(row[k]) for k in ("estimate", "ci_low", "ci_high"))
    return mean, (mean - low, high - mean)


def plot(folder, output):
    rows = {(r["group"], r["value"], r["metric"]): r for r in read_csv(folder / "summary.csv")}
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    labels = {"always": "Always wait"}
    labels.update({name: f"Threshold {meta[f'threshold_{name}']}" for name in ("transferred", "budgeted")})
    plt.rcParams.update({
        "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8d9da9", "text.color": "#20323f", "axes.labelcolor": "#344a5a",
        "xtick.color": "#344a5a", "ytick.color": "#344a5a", "axes.titlecolor": "#20323f",
    })
    fig, panels = plt.subplots(2, 2, figsize=(11.5, 7.7), gridspec_kw={"height_ratios": [1, 1.06]})
    fig.subplots_adjust(left=.16, right=.965, bottom=.14, top=.825, hspace=.68, wspace=.52)
    fig.suptitle("Selective waiting across an 84-round surface-code memory",
                 y=.975, fontsize=17, weight="semibold")
    fig.text(.5, .922, "Distance 7 · Nine eligible commits · Three extra rounds per extended deadline",
             ha="center", fontsize=10.5, color="#435968")
    gain, wait, phase_gain, phase_wait = panels.flat
    for ax in panels.flat:
        ax.set_axisbelow(True)
        ax.grid(color="#e0e8ec", axis="x" if ax in (gain, wait) else "y")
    names = ("always", "transferred", "budgeted")
    for y, name in enumerate(names):
        mean, errors = interval(rows, "population", "", f"gain_{name}", 1e6)
        gain.errorbar(mean, y, xerr=np.array(errors)[:, None], fmt="o", color=COLORS[name],
                      linewidth=1.6, capsize=3, zorder=3)
    gain.axvline(0, color="#80909b", linewidth=1, linestyle="--")
    gain.set(title="Logical error reduction", xlabel="Fewer failures per million memory trials",
             yticks=range(len(names)), yticklabels=[labels[n] for n in names], ylim=(2.6, -.6))

    wait.axvspan(0, 1, color="#e9f3ee", zorder=0)
    wait.axvline(1, color="#80909b", linewidth=1, linestyle="--")
    for y, name in enumerate(names[1:]):
        mean, errors = interval(rows, "ordinary", "", f"wait_{name}", 100)
        wait.errorbar(mean, y, xerr=np.array(errors)[:, None], fmt="D", color=COLORS[name],
                      linewidth=1.6, capsize=3, zorder=3)
        offset = (0, 9) if mean > 1.5 else (-5, 9)
        wait.annotate(f"{mean:.2f}%", (mean, y), xytext=offset, textcoords="offset points",
                      ha="center" if mean > 1.5 else "right", color=COLORS[name], fontsize=10)
    wait.set(title="Extra waits without a burst", xlabel="Memory trials with any added wait (%)",
             yticks=range(2), yticklabels=[labels[n] for n in names[1:]], ylim=(1.6, -.6), xlim=(0, 6.15))

    phases = np.arange(1, 8)
    for ax in (phase_gain, phase_wait):
        ax.axvspan(.5, 3.5, color="#e9f3ee", zorder=0)
        ax.set(xlim=(.5, 7.5), xticks=phases, xlabel="True alert arrival after the nominal deadline (rounds)")
    mean, errors = zip(*(interval(rows, "arrival_phase", str(p), "gain_budgeted", 100) for p in phases))
    phase_gain.axhline(0, color="#80909b", linewidth=1)
    phase_gain.errorbar(phases, mean, yerr=np.array(errors).T, fmt="o", color=COLORS["budgeted"],
                        capsize=3, linewidth=1.6)
    phase_gain.set(title=f"{labels['budgeted']} gain on burst trials", ylabel="Fewer failures (percentage points)")
    means, errors = zip(*(interval(rows, "arrival_phase", str(p), "wait_budgeted", 100) for p in phases))
    phase_wait.bar(phases, means, yerr=np.array(errors).T, capsize=2,
                   color=["#187568"] * 3 + ["#9bb6b5"] * 4, width=.65)
    phase_wait.set(title=f"{labels['budgeted']} waits on burst trials", ylabel="Memory trials with any added wait (%)",
                   ylim=(0, 104))
    fig.text(.5, .062, "Shading marks the 1% wait limit (top right) and arrivals within the extension (bottom).",
             ha="center", fontsize=9, color="#435968")
    fig.text(.5, .032, "Pointwise 95% intervals · Population gains assume a 0.1% burst probability per memory trial.",
             ha="center", fontsize=9, color="#435968")
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output)
