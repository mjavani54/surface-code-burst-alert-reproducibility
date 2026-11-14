"""Causal alert deadlines and the syndrome trigger for a rolling memory."""

from dataclasses import dataclass, replace

import numpy as np

from window_decoder import schedule


DISTANCE = 7
ROUNDS = 84
COMMIT_WIDTH = LOOKAHEAD = 7
BURST_DURATION = 3
ALERT_DELAY = 14
EXTRA_WAIT = 3
BACKGROUND_RATE = 0.003
BURST_RATE = 0.020
BURST_PRIOR = 0.001
SENSITIVITY = 0.90
FALSE_ALARM = 0.0001
TRANSFER_THRESHOLD = 28
CALIBRATION_BUDGET = 0.009
WAIT_TARGET = 0.01
RECOVERY_TARGET = 0.80


def study_schedule():
    windows = schedule(ROUNDS, COMMIT_WIDTH, LOOKAHEAD)
    # Include complete seven-round onset blocks for the interior deadlines.
    # The first block starts at round 8, and the last alert arrives by readout.
    onsets = tuple(range(8, ROUNDS - ALERT_DELAY + 1))
    later = []
    eligible = {}
    for index, window in enumerate(windows):
        deadline = min(window.deadline + EXTRA_WAIT, ROUNDS)
        candidates = tuple(o for o in onsets if window.deadline < o + ALERT_DELAY <= deadline)
        later.append(replace(window, deadline=deadline) if candidates else window)
        if candidates:
            if (index + 1 == len(windows) or deadline >= windows[index + 1].deadline
                    or max(candidates) + BURST_DURATION > window.deadline):
                raise ValueError("invalid rolling deadline or unavailable trigger data")
            eligible[index] = candidates
    return windows, tuple(later), onsets, eligible


def count_prefix(counts):
    prefix = np.zeros((len(counts), counts.shape[1] + 1), dtype=np.int32)
    np.cumsum(counts, axis=1, out=prefix[:, 1:])
    return prefix


def trigger_scores(prefix, first, eligible):
    """Largest four-slice detection count for arrivals during each extension."""
    scores = []
    for candidates in eligible.values():
        columns = np.asarray(candidates) - first
        totals = prefix[:, columns + BURST_DURATION + 1] - prefix[:, columns]
        scores.append(totals.max(axis=1))
    return np.column_stack(scores)


def calibrate_threshold(histogram, budget=CALIBRATION_BUDGET):
    """Choose the smallest integer threshold inside the trial-level wait budget."""
    tails = np.append(histogram[::-1].cumsum()[::-1], 0)
    threshold = int(np.flatnonzero(tails <= budget * histogram.sum())[0])
    return threshold, int(tails[threshold])


def conditional_false_flags(rng, shots, candidates, probability):
    """Sample independent Bernoulli flags conditional on at least one positive."""
    positions = np.arange(candidates)
    weights = probability * (1 - probability)**positions
    mass = weights.sum()
    first = rng.choice(candidates, shots, p=weights / mass)
    flags = (rng.random((shots, candidates)) < probability) & (positions > first[:, None])
    flags[np.arange(shots), first] = True
    return float(mass), flags


@dataclass(frozen=True)
class EvidenceWindow:
    old: int
    arrived: int
    soon: int
    visible: int
    complete: int


def best_candidate(score, onset, candidates, first):
    """Merge maxima in onset order, so ties keep the earliest hypothesis."""
    if candidates.shape[1] == 0:
        return score, onset
    local = candidates.argmax(axis=1)
    candidate_score = candidates[np.arange(len(score)), local]
    wins = candidate_score > score
    return np.where(wins, candidate_score, score), np.where(wins, first + local, onset)


class RollingEvidence:
    """Reuse settled onset scores across commits with the same decoder as v0.6."""

    def __init__(self, scorer, windows, later, prior=BURST_PRIOR):
        self.scorer = scorer
        self.onsets = np.asarray(scorer.onsets)
        if not np.all(np.diff(self.onsets) == 1):
            raise ValueError("onset hypotheses must be consecutive")
        self.first = int(self.onsets[0])
        self.prior_odds = np.log(prior / ((1 - prior) * len(self.onsets)))
        self.plans = []
        old = 0
        for window, waiting in zip(windows, later, strict=True):
            arrived, soon = np.searchsorted(
                self.onsets + ALERT_DELAY, (window.deadline, waiting.deadline), side="right"
            )
            complete, visible = np.searchsorted(
                self.onsets, (window.stop - BURST_DURATION, window.stop), side="right"
            )
            if not old <= arrived <= soon <= complete <= visible:
                raise ValueError("an alert arrived before its syndrome score was complete")
            self.plans.append(EvidenceWindow(old, arrived, soon, visible, complete))
            old = arrived

    def scores(self, prefix):
        scorer = self.scorer
        duration = scorer.duration + 1
        full_counts = prefix[:, duration:] - prefix[:, :-duration]
        full = np.column_stack([
            scorer.ratios[onset, duration][full_counts[:, i]]
            for i, onset in enumerate(self.onsets)
        ]) + self.prior_odds
        pending = []
        for plan in self.plans:
            recent = full[:, plan.arrived:plan.complete]
            if plan.visible > plan.complete:
                partial = np.column_stack([
                    scorer.ratios[self.onsets[i], plan.visible - i][prefix[:, plan.visible] - prefix[:, i]]
                    for i in range(plan.complete, plan.visible)
                ]) + self.prior_odds
                recent = np.concatenate((recent, partial), axis=1)
            pending.append(recent)
        return full, pending

    def choices(self, full, pending, flags):
        yes = np.log(SENSITIVITY / FALSE_ALARM)
        no = np.log((1 - SENSITIVITY) / (1 - FALSE_ALARM))
        evidence = np.full((1, len(self.onsets)), no) if flags is None else np.where(flags, yes, no)
        previous_score = np.zeros(len(full))
        previous_pick = np.zeros(len(full), dtype=int)
        fixed, later = [], []
        for plan, recent in zip(self.plans, pending, strict=True):
            old = slice(plan.old, plan.arrived)
            previous_score, previous_pick = best_candidate(
                previous_score, previous_pick, full[:, old] + evidence[:, old], self.first + plan.old
            )
            _, early_pick = best_candidate(previous_score, previous_pick, recent, self.first + plan.arrived)
            fixed.append(early_pick)
            if plan.soon == plan.arrived:
                later.append(early_pick)
                continue
            n = plan.soon - plan.arrived
            score, pick = best_candidate(
                previous_score, previous_pick,
                recent[:, :n] + evidence[:, plan.arrived:plan.soon], self.first + plan.arrived
            )
            _, late_pick = best_candidate(score, pick, recent[:, n:], self.first + plan.soon)
            later.append(late_pick)
        return fixed, later


def onsets_by_phase(windows, onsets):
    """Map each true onset to its arrival phase and preceding nominal deadline."""
    by_onset = {}
    for onset in onsets:
        arrival = onset + ALERT_DELAY
        deadline = max(w.deadline for w in windows if w.deadline < arrival)
        by_onset[onset] = (arrival - deadline, deadline)
    return by_onset
