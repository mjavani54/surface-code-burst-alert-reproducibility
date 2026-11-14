"""Paired logical error estimates and matched delay accounting."""

from dataclasses import asdict

import numpy as np

from rolling_policy import BURST_PRIOR, EXTRA_WAIT, onsets_by_phase
from write_results import PairedDifference, paired_gain


CONTRASTS = {
    "gain_always": ("fixed_alert", "always_alert"),
    "gain_random": ("fixed_alert", "random_alert"),
    "gain_selective": ("fixed_alert", "selective_alert"),
    "gain_syndrome": ("fixed_syndrome", "selective_syndrome"),
    "gain_lookahead": ("fixed_alert", "selective_lookahead"),
    "gain_both": ("fixed_alert", "selective_both"),
    "selective_vs_random": ("random_alert", "selective_alert"),
    "selective_vs_syndrome": ("selective_syndrome", "selective_alert"),
    "selective_vs_lookahead": ("selective_lookahead", "selective_alert"),
    "both_vs_lookahead": ("selective_lookahead", "selective_both"),
    "both_vs_selective": ("selective_alert", "selective_both"),
    "fixed_alert_vs_syndrome": ("fixed_syndrome", "fixed_alert"),
}
POLICIES = tuple(dict.fromkeys(p for pair in CONTRASTS.values() for p in pair))
LATENCY = ("trigger", "random")
METRICS = (
    *(f"failure_{p}" for p in POLICIES),
    *CONTRASTS,
    "recovery_margin",
    *(f"{stat}_{p}" for p in LATENCY for stat in ("wait", "delay")),
    "delay_difference",
)


def wait_values(trigger, random, probabilities):
    trigger_delay = EXTRA_WAIT * trigger.sum(axis=1)
    random_delay = EXTRA_WAIT * random.sum(axis=1)
    expected_random = EXTRA_WAIT * probabilities.sum()
    values = {
        "wait_trigger": trigger_delay > 0,
        "wait_random": random_delay > 0,
        "delay_trigger": trigger_delay,
        "delay_random": random_delay,
        "delay_difference": trigger_delay - expected_random,
    }
    histograms = {
        "trigger": np.bincount(trigger_delay // EXTRA_WAIT, minlength=trigger.shape[1] + 1),
        "random": np.bincount(random_delay // EXTRA_WAIT, minlength=random.shape[1] + 1),
    }
    return values, histograms


def group_weights(windows, onsets):
    groups = {
        ("ordinary", ""): {0: 1.0},
        ("burst", ""): dict.fromkeys(onsets, 1 / len(onsets)),
        ("population", ""): {0: 1 - BURST_PRIOR, **dict.fromkeys(onsets, BURST_PRIOR / len(onsets))},
    }
    by_onset = onsets_by_phase(windows, onsets)
    for i, group in enumerate(("arrival_phase", "nominal_deadline")):
        for value in sorted({pair[i] for pair in by_onset.values()}):
            subset = [o for o in onsets if by_onset[o][i] == value]
            groups[group, str(value)] = dict.fromkeys(subset, 1 / len(subset))
    for within in (True, False):
        subset = [o for o in onsets if (by_onset[o][0] <= EXTRA_WAIT) == within]
        groups["arrival_band", "within_extension" if within else "after_extension"] = dict.fromkeys(
            subset, 1 / len(subset)
        )
    for onset in onsets:
        groups["onset", str(onset)] = {onset: 1.0}
    return groups


def summarize(moment_rows, hist_rows, windows, onsets):
    strata, hist = {}, {}
    for row in moment_rows:
        key = int(row["onset"]), row["metric"]
        aggregate = strata.setdefault(key, asdict(PairedDifference()))
        for field in aggregate:
            aggregate[field] += type(aggregate[field])(row[field])
    for row in hist_rows:
        key = int(row["onset"]), row["schedule"], int(row["added_rounds"])
        hist[key] = hist.get(key, 0) + int(row["trials"])
    summary, delay = [], []
    for (group, value), weights in group_weights(windows, onsets).items():
        for metric in METRICS:
            moments = [strata[o, metric] for o in weights]
            estimate = paired_gain(moments, weights.values())
            summary.append({
                "group": group, "value": value, "metric": metric, "estimate": estimate.pop("gain"),
                "test_trials": sum(r["shots"] for r in moments), **estimate,
            })
        if group not in ("ordinary", "burst", "population", "arrival_band"):
            continue
        for name in LATENCY:
            for rounds in sorted({key[2] for key in hist}):
                probability = sum(
                    weight * hist.get((o, name, rounds), 0) / strata[o, f"wait_{name}"]["shots"]
                    for o, weight in weights.items()
                )
                delay.append({
                    "group": group, "value": value, "schedule": name, "added_rounds": rounds,
                    "probability": probability,
                })
    return summary, delay
