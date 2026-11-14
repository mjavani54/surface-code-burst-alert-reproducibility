"""First and second moments for paired logical error and waiting metrics."""

from dataclasses import asdict

from robust_design import FIXED_DELAY, PRIOR, SCOPES, TIMINGS, arrival_weights
from robust_policy import POLICIES
from write_results import PairedDifference, paired_gain


CONTRASTS = {
    "gain_always": ("fixed", "always"),
    "gain_alert": ("fixed", "alert"),
    "gain_lookahead": ("fixed", "lookahead"),
    "alert_vs_lookahead": ("lookahead", "alert"),
    "both_vs_lookahead": ("lookahead", "both"),
    "both_vs_alert": ("alert", "both"),
}
TIMING_METRICS = (
    *(f"failure_{p}" for p in POLICIES),
    *CONTRASTS, "recovery_margin", "wait", "delay",
)
METRICS = (
    *(f"{timing}:{metric}" for timing in TIMINGS for metric in TIMING_METRICS),
    *(f"jitter_minus_fixed:{metric}" for metric in ("both_vs_lookahead", "alert_vs_lookahead", "wait")),
)


def cohorts(scope, onsets, windows, eligible):
    ordinary = (0 if scope == 1 else 2, 0)
    yield "ordinary", {ordinary: 1.0}
    yield "burst", {(scope, o): 1 / len(onsets) for o in onsets}
    yield "population", {ordinary: 1 - PRIOR, **{(scope, o): PRIOR / len(onsets) for o in onsets}}
    weights = arrival_weights(windows, onsets, eligible, FIXED_DELAY)
    eligible_onsets = [o for o, w in weights.items() if w == 1]
    other_onsets = [o for o, w in weights.items() if w == 0]
    yield "fixed_arrival_within", {(scope, o): 1 / len(eligible_onsets) for o in eligible_onsets}
    yield "fixed_arrival_outside", {(scope, o): 1 / len(other_onsets) for o in other_onsets}
    for o in onsets:
        yield f"onset_{o}", {(scope, o): 1.0}


def summarize(rows, onsets, windows, eligible):
    strata = {}
    for row in rows:
        key = (int(row["scope"]), int(row["onset"])), row["metric"]
        aggregate = strata.setdefault(key, asdict(PairedDifference()))
        for field, previous in aggregate.items():
            aggregate[field] = previous + type(previous)(row[field])
    summary = []
    for scope, name in enumerate(SCOPES, 1):
        for group, weights in cohorts(scope, onsets, windows, eligible):
            for metric in METRICS:
                moments = [strata[key, metric] for key in weights]
                estimate = paired_gain(moments, weights.values())
                timing, comparison = metric.split(":")
                summary.append({
                    "scope": name, "group": group, "timing": timing, "metric": comparison,
                    "estimate": estimate.pop("gain"), "test_trials": sum(r["shots"] for r in moments),
                    **estimate,
                })
    return summary
