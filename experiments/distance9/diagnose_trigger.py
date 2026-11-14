"""Estimate how often the trigger catches an alert that could arrive during a wait."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

import numpy as np

from burst_alert import SyndromeScores
from robust_design import (
    BURST_LENGTH, EXTRA_WAIT, FIXED_DELAY, JITTER, SCOPES, TIMINGS, hot_patches, study_schedule, trigger_matrix,
)
from robust_graphs import make_circuit
from rolling_policy import count_prefix
from window_decoder import detector_rounds
from write_results import PairedDifference, paired_gain, write_csv


def arrival_capture(trigger, deadlines, onset, delays):
    possible = np.array([[(t < onset + d <= t + EXTRA_WAIT) for d in delays] for t in deadlines])
    # Each possible reporting time belongs to at most one extension.
    return (trigger @ possible.sum(axis=1)) / len(delays), float(possible.sum() / len(delays))


def measure_onset(job, *, per_patch, batch, scorer, thresholds, windows, onsets, eligible):
    scope, onset, seed = job
    patches = (None,) if scope == 1 else hot_patches()
    shots = per_patch * 4 // len(patches)
    seeds = np.random.SeedSequence(seed).generate_state(len(patches), dtype=np.uint64).tolist()
    deadlines = [windows[i].deadline for i in eligible]
    moments = [PairedDifference() for _ in TIMINGS]
    for patch, sub_seed in zip(patches, seeds, strict=True):
        sampler = make_circuit(onset, patch).compile_detector_sampler(seed=sub_seed)
        for offset in range(0, shots, batch):
            n = min(batch, shots - offset)
            packed = sampler.sample(n, bit_packed=True)
            prefix = count_prefix(scorer.packed_counts(packed))
            for i, delays in enumerate((FIXED_DELAY, JITTER)):
                trigger = trigger_matrix(prefix, scorer.first, windows, onsets, eligible, delays) >= thresholds[i]
                joint, _ = arrival_capture(trigger, deadlines, onset, delays)
                moments[i].add(joint)
    return [
        {"scope": SCOPES[scope - 1], "onset": onset, "timing": timing, "seed": seed,
         "eligible_probability": arrival_capture(np.zeros((1, len(deadlines)), bool), deadlines, onset, delays)[1],
         **asdict(moment)}
        for timing, delays, moment in zip(TIMINGS, (FIXED_DELAY, JITTER), moments, strict=True)
    ]


def aggregate(rows):
    summary = []
    for scope in SCOPES:
        for timing in TIMINGS:
            matching = [row for row in rows if row["scope"] == scope and row["timing"] == timing]
            weights = [1 / len(matching)] * len(matching)
            eligibility = sum(float(row["eligible_probability"]) for row in matching) / len(matching)
            paired = paired_gain(matching, weights)
            summary.append({
                "scope": scope, "timing": timing, "test_trials": sum(int(row["shots"]) for row in matching),
                "eligible_probability": eligibility, "capture_given_eligible": paired["gain"] / eligibility,
                "ci_low": paired["ci_low"] / eligibility, "ci_high": paired["ci_high"] / eligibility,
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--seed", type=int, default=2026111314)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shots-per-patch", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.seed < 0 or args.shots_per_patch < 2 or min(args.workers, args.batch_size) < 1:
        parser.error("invalid seed, worker count, or sample size")
    meta = json.loads((args.folder / "run.json").read_text(encoding="utf-8"))
    windows, _, onsets, eligible = study_schedule()
    calibration_and_test = {*meta["calibration_seeds"], meta["trigger_seed"],
                            *(seed for job in meta["test_jobs"] for seed in job[4:])}
    seeds = np.random.SeedSequence(args.seed).generate_state(2 * len(onsets), dtype=np.uint64).tolist()
    if calibration_and_test.intersection(seeds):
        raise ValueError("diagnostic seeds overlap with the main experiment")
    labels = [(s, o) for s in (1, 2) for o in onsets]
    jobs = [(*label, seed) for label, seed in zip(labels, seeds, strict=True)]
    scorer = SyndromeScores(detector_rounds(make_circuit(0)), onsets, BURST_LENGTH)
    measure = partial(measure_onset, per_patch=args.shots_per_patch, batch=args.batch_size, scorer=scorer,
                      thresholds=meta["thresholds"], windows=windows, onsets=onsets, eligible=eligible)
    if args.workers == 1:
        completed = list(map(measure, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            completed = list(pool.map(measure, jobs))
    rows = [row for part in completed for row in part]
    for name, values in (("trigger_capture_by_onset", rows), ("trigger_capture", aggregate(rows))):
        path = args.folder / f"{name}.csv"
        temp = path.with_suffix(".tmp")
        write_csv(temp, values)
        temp.replace(path)


if __name__ == "__main__":
    main()
