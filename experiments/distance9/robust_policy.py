"""Causal syndrome scores and alert evidence with variable reporting times."""

import numpy as np

from robust_decoder import Decision
from robust_design import FALSE_ALARM, PRIOR, SENSITIVITY


def alert_log_likelihood(arrivals, onsets, deadline, delays):
    # Before the last possible arrival, silence is right-censored evidence.
    arrival_probability = np.mean(onsets[:, None] + np.asarray(delays) <= deadline, axis=1)
    silence = np.log((1 - SENSITIVITY * arrival_probability) /
                     (1 - FALSE_ALARM * arrival_probability))
    if arrivals is None:
        return silence
    return np.where(arrivals <= deadline, np.log(SENSITIVITY / FALSE_ALARM), silence)


class Scores:
    """Use the v0.8 onset likelihoods with only the syndrome visible at each stop."""

    def __init__(self, scorer, prefix):
        self.scorer = scorer
        self.onsets = np.asarray(scorer.onsets)
        self.candidates = np.r_[0, self.onsets]
        self.prefix = prefix
        self.prior = np.log(PRIOR / ((1 - PRIOR) * len(self.onsets)))
        duration = scorer.duration + 1
        counts = prefix[:, duration:] - prefix[:, :-duration]
        self.full = np.column_stack([
            scorer.ratios[o, duration][counts[:, i]] for i, o in enumerate(self.onsets)
        ]) + self.prior

    def at(self, stop, selected=slice(None)):
        complete, visible = np.searchsorted(self.onsets, (stop - self.scorer.duration, stop), side="right")
        partial = [
            self.scorer.ratios[self.onsets[i], visible - i][
                self.prefix[selected, visible] - self.prefix[selected, i]
            ] + self.prior for i in range(complete, visible)
        ]
        full = self.full[selected, :complete]
        return np.column_stack((np.zeros(len(full)), full, *partial))

    def choices(self, table, arrivals, deadline, delays):
        indices = self.onsets[:table.shape[1] - 1]
        received = arrivals[:, :len(indices)] if arrivals is not None else None
        evidence = alert_log_likelihood(received, indices, deadline, delays)
        updated = table.copy()
        updated[:, 1:] += evidence
        return self.candidates[updated.argmax(axis=1)]


POLICIES = ("fixed", "always", "alert", "lookahead", "both")


def decisions(scores, windows, longer, eligible, trigger, arrivals, delays, nominal, extended):
    never = np.zeros(len(scores.full), dtype=bool)
    by_window = dict(zip(eligible, trigger.T, strict=True))
    plans = {name: [] for name in POLICIES}
    for index, (early, late) in enumerate(zip(windows, longer, strict=True)):
        table = nominal[index]
        fixed = scores.choices(table, arrivals, early.deadline, delays)
        plans["fixed"].append(Decision(fixed, never))
        if index not in by_window:
            for name in POLICIES[1:]:
                plans[name].append(Decision(fixed, never))
            continue
        wait = by_window[index]
        always = scores.choices(table, arrivals, late.deadline, delays)
        plans["always"].append(Decision(always, never))
        plans["alert"].append(Decision(np.where(wait, always, fixed), never))
        lookahead, both = fixed.copy(), fixed.copy()
        if wait.any():
            received = arrivals[wait] if arrivals is not None else None
            lookahead[wait] = scores.choices(extended[index], received, early.deadline, delays)
            both[wait] = scores.choices(extended[index], received, late.deadline, delays)
        plans["lookahead"].append(Decision(lookahead, wait))
        plans["both"].append(Decision(both, wait))
    return plans
