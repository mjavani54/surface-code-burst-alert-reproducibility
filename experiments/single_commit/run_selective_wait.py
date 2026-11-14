"""Compare fixed, always-wait, and syndrome-triggered alert deadlines."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from burst_alert import SyndromeScores, burst_flags, choices, ordinary_flags
from surface_code import memory_circuit
from window_decoder import ForwardDecoder, detector_rounds, graph_edges, schedule
from write_results import PairedDifference, paired_gain, write_csv


DISTANCE, ROUNDS, COMMIT_WIDTH, LOOKAHEAD = 7, 42, 7, 7
ONSETS = tuple(range(15, 22))
DURATION, ALERT_DELAY, COMMIT_ROUND, EXTRA_WAIT = 3, 14, 21, 3
BACKGROUND_RATE, BURST_RATE, BURST_PRIOR = 0.003, 0.020, 0.001
SENSITIVITY, FALSE_ALARM = 0.90, 0.0001
CALIBRATION_WAIT_BUDGET, TEST_WAIT_BUDGET, RECOVERY_TARGET = 0.009, 0.01, 0.80
METRICS = ("always_wait_gain", "selective_wait_gain", "foregone_gain",
           "recovery_margin", "selective_wait_rate")


def deadlines():
    windows = schedule(ROUNDS, COMMIT_WIDTH, LOOKAHEAD)
    index, = (i for i, window in enumerate(windows) if window.commit == COMMIT_ROUND)
    fixed = windows[index]
    waiting = replace(fixed, deadline=fixed.deadline + EXTRA_WAIT)
    relevant = tuple(onset for onset in ONSETS if fixed.deadline < onset + ALERT_DELAY <= waiting.deadline)
    if any(onset + DURATION > fixed.deadline for onset in relevant):
        raise ValueError("the trigger would require syndrome data after the fixed deadline")
    return windows, index, waiting, relevant


def trigger_score(counts, relevant):
    """Largest four-slice detection count across the candidate onsets."""
    start = min(ONSETS)
    return np.maximum.reduce([
        counts[:, onset - start:onset - start + DURATION + 1].sum(axis=1)
        for onset in relevant
    ])


def threshold_for_budget(histogram, budget):
    """Smallest inclusive threshold satisfying the empirical wait budget."""
    tails = np.append(histogram[::-1].cumsum()[::-1], 0)
    threshold = int(np.flatnonzero(tails <= budget * histogram.sum())[0])
    return threshold, int(tails[threshold])


def calibrate_trigger(circuit, scorer, relevant, shots, batch_size, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    histogram = np.zeros(int(scorer.slices[-1] - scorer.slices[0]) + 1, dtype=np.int64)
    for start in range(0, shots, batch_size):
        packed = sampler.sample(min(batch_size, shots - start), bit_packed=True)
        values = trigger_score(scorer.packed_counts(packed), relevant)
        histogram += np.bincount(values, minlength=len(histogram))
    threshold, waited = threshold_for_budget(histogram, CALIBRATION_WAIT_BUDGET)
    return threshold, waited, histogram.tolist()


def paired_decoding_gain(decoder, packed, actual, fixed, later_choice, index):
    """Skip shots for which the two deadlines select the same graph."""
    changes = fixed[index] != later_choice
    gain = np.zeros(len(packed), dtype=np.int8)
    if changes.any():
        size = len(decoder.geometry[0].times)
        syndromes = np.unpackbits(packed[changes], axis=1, count=size, bitorder="little")
        observed = actual[changes]
        early = [selected[changes] for selected in fixed]
        late = early.copy()
        late[index] = later_choice[changes]
        early_failures = (decoder.decode(syndromes, early) != observed).astype(np.int8)
        late_failures = (decoder.decode(syndromes, late) != observed).astype(np.int8)
        gain[changes] = early_failures - late_failures
    return gain


def simulate_stratum(item, *, batch_size, circuits, times, edges, scorer, threshold):
    onset, chunk, shots, syndrome_seed, alert_seed = item
    sampler = circuits[onset].compile_detector_sampler(seed=syndrome_seed)
    rng = np.random.default_rng(alert_seed)
    windows, index, waiting_window, relevant = deadlines()
    decoder = ForwardDecoder(times, windows, edges)
    moments = {metric: PairedDifference() for metric in METRICS}
    for start in range(0, shots, batch_size):
        count = min(batch_size, shots - start)
        packed, observable = sampler.sample(count, separate_observables=True, bit_packed=True)
        actual = observable[:, 0] & 1
        counts = scorer.packed_counts(packed)
        waits = trigger_score(counts, relevant) >= threshold
        scores = scorer.score_counts(counts, windows, BURST_PRIOR)
        flag_cases = (
            burst_flags(count, ONSETS, onset, rng, SENSITIVITY, FALSE_ALARM)
            if onset else ordinary_flags(count, len(ONSETS), rng, FALSE_ALARM)
        )
        gain_always = np.zeros(count)
        for weight, flags in flag_cases:
            evidence = dict(flags=flags, delay=ALERT_DELAY, sensitivity=SENSITIVITY, false_alarm=FALSE_ALARM)
            fixed = choices(scores, ONSETS, windows, **evidence)
            later_choice, = choices([scores[index]], ONSETS, (waiting_window,), **evidence)
            gain_always += weight * paired_decoding_gain(decoder, packed, actual, fixed, later_choice, index)
        gain_selective = gain_always * waits
        moments["always_wait_gain"].add(gain_always)
        moments["selective_wait_gain"].add(gain_selective)
        moments["foregone_gain"].add(gain_always * ~waits)
        moments["recovery_margin"].add(gain_selective - RECOVERY_TARGET * gain_always)
        moments["selective_wait_rate"].add(waits)
    print(f"Onset {onset}, chunk {chunk}: {shots:,} trials", flush=True)
    return [{"true_onset": onset, "chunk": chunk, "metric": metric, **asdict(moment)}
            for metric, moment in moments.items()]


def group_estimates(rows):
    summary = []
    for metric in METRICS:
        strata = {}
        for row in rows:
            if row["metric"] != metric:
                continue
            entry = strata.setdefault(int(row["true_onset"]), asdict(PairedDifference()))
            for field, value in entry.items():
                entry[field] = value + type(value)(row[field])
        ordinary = strata[0]
        burst = [strata[onset] for onset in ONSETS]
        n = len(burst)
        groups = [
            ("burst", burst, [1 / n] * n),
            ("ordinary", [ordinary], [1.0]),
            ("population", [ordinary, *burst], [1 - BURST_PRIOR, *([BURST_PRIOR / n] * n)]),
        ]
        for group, relevant, weights in groups:
            summary.append({"metric": metric, "group": group, **paired_gain(relevant, weights)})
    return summary


def seed_plan(args):
    calibration, trigger, burst, ordinary = np.random.SeedSequence(args.seed).spawn(4)
    calibration_seeds = calibration.generate_state(1 + len(ONSETS), dtype=np.uint64).tolist()
    trigger_seed = int(trigger.generate_state(1, dtype=np.uint64)[0])
    burst_seeds = burst.generate_state(2 * len(ONSETS), dtype=np.uint64).tolist()
    ordinary_seeds = ordinary.generate_state(2 * args.ordinary_chunks, dtype=np.uint64).tolist()
    quotient, remainder = divmod(args.ordinary_shots, args.ordinary_chunks)
    ordinary_jobs = [(0, i, quotient + (i < remainder), *ordinary_seeds[2*i:2*i+2])
                     for i in range(args.ordinary_chunks)]
    burst_jobs = [(onset, 0, args.burst_shots, *burst_seeds[2*i:2*i+2])
                  for i, onset in enumerate(ONSETS)]
    return calibration_seeds, trigger_seed, burst_jobs + ordinary_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026111310)
    parser.add_argument("--calibration-shots", type=int, default=80_000)
    parser.add_argument("--trigger-shots", type=int, default=200_000)
    parser.add_argument("--burst-shots", type=int, default=120_000, help="per burst onset")
    parser.add_argument("--ordinary-shots", type=int, default=24_000_000)
    parser.add_argument("--ordinary-chunks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if (min(args.workers, args.batch_size, args.ordinary_chunks) < 1 or args.seed < 0
            or min(args.calibration_shots, args.trigger_shots, args.burst_shots, args.ordinary_shots) < 2
            or args.ordinary_chunks > args.ordinary_shots):
        parser.error("invalid shot counts, seed, or worker settings")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("choose an empty output directory")

    start = perf_counter()
    calibration, trigger_seed, items = seed_plan(args)
    circuits = {
        onset: memory_circuit(DISTANCE, ROUNDS, "x", BACKGROUND_RATE, BURST_RATE, onset, DURATION)
        for onset in (0, *ONSETS)
    }
    times = detector_rounds(circuits[0])
    coords = circuits[0].get_detector_coordinates()
    if any(circuit.get_detector_coordinates() != coords for circuit in circuits.values()):
        raise ValueError("all circuits must share detector coordinates")
    if len(circuits[0].shortest_graphlike_error()) != DISTANCE:
        raise ValueError("incorrect surface-code distance")
    edges = {onset: graph_edges(circuit, times) for onset, circuit in circuits.items()}
    scorer = SyndromeScores(times, ONSETS, DURATION)
    scorer.fit(circuits, args.calibration_shots, calibration, args.batch_size)
    windows, index, waiting_window, relevant = deadlines()
    threshold, calibration_waits, histogram = calibrate_trigger(
        circuits[0], scorer, relevant, args.trigger_shots, args.batch_size, trigger_seed
    )
    print(f"Trigger set to {threshold} detections. Calibration waits: "
          f"{calibration_waits:,}/{args.trigger_shots:,}.", flush=True)

    simulate = partial(
        simulate_stratum, batch_size=args.batch_size, circuits=circuits, times=times,
        edges=edges, scorer=scorer, threshold=threshold,
    )
    if args.workers == 1:
        strata = list(map(simulate, items))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(items))) as pool:
            strata = list(pool.map(simulate, items))
    rows = [row for stratum in strata for row in stratum]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "paired_moments.csv", rows)
    write_csv(args.output / "summary.csv", group_estimates(rows))
    meta = {
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "distance": DISTANCE, "rounds": ROUNDS, "commit_width": COMMIT_WIDTH, "lookahead": LOOKAHEAD,
        "basis": "x", "onsets": ONSETS, "burst_duration": DURATION, "alert_delay": ALERT_DELAY,
        "background_rate": BACKGROUND_RATE, "burst_rate": BURST_RATE, "burst_prior": BURST_PRIOR,
        "alert_sensitivity": SENSITIVITY, "false_alarm_per_candidate": FALSE_ALARM,
        "windows": [asdict(w) for w in windows],
        "target_window": index, "waiting_window": asdict(waiting_window),
        "trigger_onsets": relevant, "trigger_threshold": threshold,
        "trigger_calibration_budget": CALIBRATION_WAIT_BUDGET, "trigger_histogram": histogram,
        "trigger_calibration_waits": calibration_waits, "ordinary_wait_target": TEST_WAIT_BUDGET,
        "recovery_target": RECOVERY_TARGET, "calibration_seeds": calibration, "trigger_seed": trigger_seed,
        "test_chunks": [
            {"onset": onset, "chunk": chunk, "shots": shots, "syndrome_seed": seed, "alert_seed": alert_seed}
            for onset, chunk, shots, seed, alert_seed in items
        ],
        "versions": {
            "platform": platform.platform(), "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "stim", "pymatching")},
        },
    }
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Completed in {perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
