"""Aggregate paired error estimates and trial-level wait distributions."""

from dataclasses import asdict

import numpy as np

from rolling_policy import BURST_PRIOR, EXTRA_WAIT, onsets_by_phase
from write_results import PairedDifference, paired_gain


POLICIES = ("fixed", "always", "transferred", "budgeted")
METRICS = (
    *(f"failure_{p}" for p in POLICIES),
    *(f"gain_{p}" for p in POLICIES[1:]),
    "margin_transferred", "margin_budgeted",
    "difference_transferred_vs_always", "difference_budgeted_vs_always",
    "wait_transferred", "wait_budgeted", "delay_transferred", "delay_budgeted",
)


def group_weights(windows, onsets):
    weight = 1 / len(onsets)
    groups = {
        ("ordinary", ""): {0: 1.0},
        ("burst", ""): dict.fromkeys(onsets, weight),
        ("population", ""): {0: 1 - BURST_PRIOR, **dict.fromkeys(onsets, BURST_PRIOR * weight)},
    }
    by_onset = onsets_by_phase(windows, onsets)
    for i, group in enumerate(("arrival_phase", "nominal_deadline")):
        for value in sorted({item[i] for item in by_onset.values()}):
            selected = [onset for onset in onsets if by_onset[onset][i] == value]
            groups[group, value] = dict.fromkeys(selected, 1 / len(selected))
    for within in (True, False):
        selected = [onset for onset in onsets if (by_onset[onset][0] <= EXTRA_WAIT) == within]
        groups["arrival_band", "within_extension" if within else "after_extension"] = dict.fromkeys(
            selected, 1 / len(selected)
        )
    other_onsets = [onset for onset in onsets if by_onset[onset][1] != 28]
    groups["sensitivity", "excluding_round_28"] = dict.fromkeys(other_onsets, 1 / len(other_onsets))
    for onset in onsets:
        groups["onset", onset] = {onset: 1.0}
    return groups


def summarize(moment_rows, hist_rows, windows, onsets):
    strata = {}
    for row in moment_rows:
        key = int(row["onset"]), row["metric"]
        aggregate = strata.setdefault(key, asdict(PairedDifference()))
        for field in aggregate:
            aggregate[field] += type(aggregate[field])(row[field])
    hist = {}
    for row in hist_rows:
        key = int(row["onset"]), row["policy"], int(row["added_rounds"])
        hist[key] = hist.get(key, 0) + int(row["trials"])
    groups = group_weights(windows, onsets)
    summary, distributions = [], []
    delays = sorted({key[-1] for key in hist})
    for (group, value), weights in groups.items():
        for metric in METRICS:
            moments = [strata[onset, metric] for onset in weights]
            estimate = paired_gain(moments, weights.values())
            summary.append({
                "group": group, "value": value, "metric": metric, "estimate": estimate.pop("gain"),
                "test_trials": sum(row["shots"] for row in moments), **estimate,
            })
        if group not in ("ordinary", "burst", "population", "arrival_phase", "arrival_band"):
            continue
        for policy in POLICIES:
            for delay in delays:
                rate = sum(
                    weight * hist.get((onset, policy, delay), 0) / strata[onset, "wait_budgeted"]["shots"]
                    for onset, weight in weights.items()
                )
                distributions.append({
                    "group": group, "value": value, "policy": policy, "added_rounds": delay,
                    "probability": rate,
                })
    return summary, distributions


def wait_metrics(n, waits, extra):
    delays = {name: mask.sum(axis=1) * extra for name, mask in waits.items()}
    values = {f"wait_{name}": delay > 0 for name, delay in delays.items()}
    values.update({f"delay_{name}": delay for name, delay in delays.items()})
    histograms = {"fixed": np.array([n]), "always": np.array([0] * waits["budgeted"].shape[1] + [n])}
    for name, delay in delays.items():
        histograms[name] = np.bincount(delay // extra, minlength=waits[name].shape[1] + 1)
    return values, histograms
