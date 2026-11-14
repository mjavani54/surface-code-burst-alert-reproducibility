"""Prespecified circuit settings, alert timing and test seeds."""

from dataclasses import replace
import numpy as np
import stim

from rolling_policy import EXTRA_WAIT, ROUNDS, conditional_false_flags
from window_decoder import schedule


DISTANCE = 9
COMMIT_WIDTH = LOOKAHEAD = 7
BURST_LENGTH = 3
THRESHOLD_BUDGET = .009
FALSE_ALARM, SENSITIVITY, PRIOR = .0001, .90, .001
BACKGROUND, BURST = .003, .020
FIXED_DELAY = (14,)
JITTER = (12, 13, 14, 15, 16)
PATCH_CENTERS = ((5, 5), (13, 5), (5, 13), (13, 13))
PATCH_RADIUS = 3
SCOPES = ("global", "local")
TIMINGS = ("fixed", "jitter")


def study_schedule():
    windows = schedule(ROUNDS, COMMIT_WIDTH, LOOKAHEAD)
    onsets = tuple(range(8, 71))
    eligible = tuple(i for i, w in enumerate(windows) if 14 <= w.deadline <= 77)
    longer = tuple(replace(w, stop=w.stop + EXTRA_WAIT, deadline=w.deadline + EXTRA_WAIT)
                   if i in eligible else w for i, w in enumerate(windows))
    return windows, longer, onsets, eligible


def hot_patches():
    """Four compact patches in the coordinates of Stim's rotated layout."""
    qubits = stim.Circuit.generated("surface_code:rotated_memory_x", distance=DISTANCE,
                                    rounds=ROUNDS).get_final_qubit_coordinates()
    return tuple(frozenset(q for q, (x, y) in qubits.items() if
                           max(abs(x - cx), abs(y - cy)) <= PATCH_RADIUS)
                 for cx, cy in PATCH_CENTERS)


def trigger_candidates(onsets, deadline, delays):
    return tuple(o for o in onsets if any(deadline < o + d <= deadline + EXTRA_WAIT for d in delays))


def trigger_matrix(prefix, first, windows, onsets, eligible, delays):
    scores = []
    for index in eligible:
        candidates = trigger_candidates(onsets, windows[index].deadline, delays)
        if not candidates:
            scores.append(np.zeros(len(prefix), dtype=int))
            continue
        columns = np.asarray(candidates) - first
        last = columns + BURST_LENGTH + 1
        if max(candidates) + BURST_LENGTH > windows[index].deadline:
            raise ValueError("the trigger needs unavailable syndrome data")
        scores.append((prefix[:, last] - prefix[:, columns]).max(axis=1))
    return np.column_stack(scores)


def threshold_from_histogram(histogram):
    tail = np.r_[np.cumsum(histogram[::-1])[::-1], 0]
    return int(np.flatnonzero(tail <= THRESHOLD_BUDGET * histogram.sum())[0])


def alert_cases(onset, onsets, count, alert_rng, jitter_rng):
    """Pair the positive flags across timing models and jitter each timestamp independently."""
    if onset:
        p = np.where(np.asarray(onsets) == onset, SENSITIVITY, FALSE_ALARM)
        flag_cases = ((1.0, alert_rng.random((count, len(onsets))) < p),)
    else:
        mass, flags = conditional_false_flags(alert_rng, count, len(onsets), FALSE_ALARM)
        flag_cases = ((1 - mass, None), (mass, flags))
    for weight, flags in flag_cases:
        if flags is None:
            yield weight, (None, None)
            continue
        base = np.broadcast_to(onsets, flags.shape)
        actual = np.where(flags, base + FIXED_DELAY[0], 127).astype(np.int8)
        varied = actual.copy()
        varied[flags] = base[flags] + jitter_rng.integers(JITTER[0], JITTER[-1] + 1, flags.sum())
        yield weight, (actual, varied)


def arrival_weights(windows, onsets, eligible, delays):
    deadlines = [windows[i].deadline for i in eligible]
    return {
        onset: sum(any(t < onset + delay <= t + EXTRA_WAIT for t in deadlines) for delay in delays) / len(delays)
        for onset in onsets
    }


def seed_plan(seed, onsets, burst_shots, ordinary_shots, ordinary_chunks):
    """Reserve independent streams for calibration and every tested stratum."""
    graph, trigger, test = np.random.SeedSequence(seed).spawn(3)
    graph_seeds = graph.generate_state(1 + 2 * len(onsets), dtype=np.uint64).tolist()
    trigger_seed = int(trigger.generate_state(1, dtype=np.uint64)[0])
    labels = [(0, 0, i) for i in range(ordinary_chunks)]
    labels += [(scope, onset, 0) for scope in (1, 2) for onset in onsets]
    streams = test.generate_state(3 * len(labels), dtype=np.uint64).reshape(-1, 3).tolist()
    size, extra = divmod(ordinary_shots, ordinary_chunks)
    jobs = [
        (scope, onset, chunk, size + (chunk < extra) if scope == 0 else burst_shots, *seeds)
        for (scope, onset, chunk), seeds in zip(labels, streams, strict=True)
    ]
    return graph_seeds, trigger_seed, jobs


def local_background_jobs(jobs):
    """Reuse each burst-free sample and alert realization with the local decoder."""
    return [(2, *job[1:]) for job in jobs if job[0] == 0]
