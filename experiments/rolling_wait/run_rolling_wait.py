"""Test four alert-deadline policies on an 84-round distance-seven memory."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from burst_alert import SyndromeScores
from rolling_policy import (
    ALERT_DELAY, BACKGROUND_RATE, BURST_DURATION, BURST_PRIOR, BURST_RATE, CALIBRATION_BUDGET,
    COMMIT_WIDTH, DISTANCE, EXTRA_WAIT, FALSE_ALARM, LOOKAHEAD, RECOVERY_TARGET, ROUNDS, SENSITIVITY,
    TRANSFER_THRESHOLD, WAIT_TARGET, RollingEvidence, calibrate_threshold, conditional_false_flags,
    count_prefix, study_schedule, trigger_scores,
)
from rolling_results import METRICS, POLICIES, summarize, wait_metrics
from surface_code import memory_circuit
from window_decoder import ForwardDecoder, detector_rounds, graph_edges
from write_results import PairedDifference, write_csv


def seed_plan(args, onsets):
    calibration, trigger, burst, ordinary = np.random.SeedSequence(args.seed).spawn(4)
    calibration_seeds = calibration.generate_state(len(onsets) + 1, dtype=np.uint64).tolist()
    trigger_seed = int(trigger.generate_state(1, dtype=np.uint64)[0])
    burst_seeds = burst.generate_state(2 * len(onsets), dtype=np.uint64).reshape(-1, 2).tolist()
    ordinary_seeds = ordinary.generate_state(2 * args.ordinary_chunks, dtype=np.uint64).reshape(-1, 2).tolist()
    jobs = [(onset, 0, args.burst_shots, *seeds, True) for onset, seeds in zip(onsets, burst_seeds, strict=True)]
    size, remainder = divmod(args.ordinary_shots, args.ordinary_chunks)
    jobs += [
        (0, chunk, size + (chunk < remainder), *seeds, chunk < args.ordinary_failure_chunks)
        for chunk, seeds in enumerate(ordinary_seeds)
    ]
    return calibration_seeds, trigger_seed, jobs


def calibration_histograms(circuit, onset, scorer, shots, seed, batch_size):
    """Count the syndrome prefix distributions used to score each onset."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    onsets = (onset,) if onset else scorer.onsets
    histograms = {}
    for candidate in onsets:
        offset = candidate - scorer.first
        for length in range(1, BURST_DURATION + 2):
            width = scorer.slices[offset + length] - scorer.slices[offset]
            histograms[candidate, length] = np.zeros(width + 1, dtype=np.int64)
    for start in range(0, shots, batch_size):
        packed = sampler.sample(min(shots - start, batch_size), bit_packed=True)
        prefix = count_prefix(scorer.packed_counts(packed))
        for candidate in onsets:
            offset = candidate - scorer.first
            for length in range(1, BURST_DURATION + 2):
                hist = histograms[candidate, length]
                total = prefix[:, offset + length] - prefix[:, offset]
                hist += np.bincount(total, minlength=len(hist))
    return histograms


def prepare_hypothesis(item, *, times, coords, scorer, shots, batch_size):
    onset, seed = item
    circuit = memory_circuit(DISTANCE, ROUNDS, "x", BACKGROUND_RATE, BURST_RATE, onset, BURST_DURATION)
    if circuit.get_detector_coordinates() != coords:
        raise ValueError("detector coordinates changed with burst onset")
    hist = calibration_histograms(circuit, onset, scorer, shots, seed, batch_size)
    return onset, circuit, graph_edges(circuit, times), hist


def trigger_calibration(circuit, scorer, eligible, shots, seed, batch_size):
    sampler = circuit.compile_detector_sampler(seed=seed)
    histogram = np.zeros(1 + (BURST_DURATION + 1) * np.diff(scorer.slices).max(), dtype=np.int64)
    for start in range(0, shots, batch_size):
        packed = sampler.sample(min(batch_size, shots - start), bit_packed=True)
        prefix = count_prefix(scorer.packed_counts(packed))
        maxima = trigger_scores(prefix, scorer.first, eligible).max(axis=1)
        histogram += np.bincount(maxima, minlength=len(histogram))
    return histogram


def decode_reusing(decoder, syndromes, picks, references):
    """Reuse a prediction whenever the complete sequence of graph choices matches."""
    if not references:
        return decoder.decode(syndromes, picks)
    unresolved = np.ones(len(syndromes), dtype=bool)
    logical = np.empty(len(syndromes), dtype=np.uint8)
    for other_picks, other_logical in references:
        same = unresolved.copy()
        for pick, other in zip(picks, other_picks, strict=True):
            same &= pick == other
        logical[same] = other_logical[same]
        unresolved[same] = False
        if not unresolved.any():
            return logical
    logical[unresolved] = decoder.decode(syndromes[unresolved], [pick[unresolved] for pick in picks])
    return logical


def different_choices(fixed, later):
    changes = np.zeros(len(fixed[0]), dtype=bool)
    for early, late in zip(fixed, later, strict=True):
        changes |= early != late
    return changes


def policy_picks(fixed, later, waits, eligible):
    schedules = {"fixed": fixed, "always": later}
    for name, mask in waits.items():
        picks = fixed.copy()
        for column, index in enumerate(eligible):
            picks[index] = np.where(mask[:, column], later[index], fixed[index])
        schedules[name] = picks
    return schedules


class Simulation:
    def __init__(self, circuits, edges, times, scorer, windows, later, eligible, threshold, batch_size):
        self.circuits = circuits
        self.scorer = scorer
        self.evidence = RollingEvidence(scorer, windows, later)
        self.eligible = eligible
        self.threshold = threshold
        self.batch_size = batch_size
        self.decoder = ForwardDecoder(times, windows, edges)
        self.num_detectors = len(times)

    def batch(self, packed, actual, onset, rng, measure_failure):
        n = len(packed)
        prefix = count_prefix(self.scorer.packed_counts(packed))
        trigger = trigger_scores(prefix, self.scorer.first, self.eligible)
        waits = {"transferred": trigger >= TRANSFER_THRESHOLD, "budgeted": trigger >= self.threshold}
        full, pending = self.evidence.scores(prefix)
        flag_shape = (n, len(self.evidence.onsets))
        if onset:
            probabilities = np.where(self.evidence.onsets == onset, SENSITIVITY, FALSE_ALARM)
            cases = ((1.0, rng.random(flag_shape) < probabilities),)
        else:
            mass, flags = conditional_false_flags(rng, *flag_shape, FALSE_ALARM)
            cases = ((1 - mass, None), (mass, flags))
        failures = {p: np.zeros(n) for p in POLICIES} if measure_failure else {}
        gains = {p: np.zeros(n) for p in POLICIES[1:]}
        references = []
        for weight, flags in cases:
            fixed, later = self.evidence.choices(full, pending, flags)
            if measure_failure:
                selected = packed
                labels = actual
                selected_waits = waits
            else:
                changes = different_choices(fixed, later)
                if not changes.any():
                    continue
                selected = packed[changes]
                labels = actual[changes]
                fixed = [choice[changes] for choice in fixed]
                later = [choice[changes] for choice in later]
                selected_waits = {name: mask[changes] for name, mask in waits.items()}
            syndromes = np.unpackbits(selected, axis=1, count=self.num_detectors, bitorder="little")
            picks = policy_picks(fixed, later, selected_waits, self.eligible)
            errors = {}
            for policy, choices in picks.items():
                logical = decode_reusing(self.decoder, syndromes, choices, references)
                errors[policy] = (logical != labels).astype(np.int8)
                references.append((choices, logical))
            if measure_failure:
                for policy in POLICIES:
                    failures[policy] += weight * errors[policy]
                for policy in POLICIES[1:]:
                    gains[policy] += weight * (errors["fixed"] - errors[policy])
            else:
                for policy in POLICIES[1:]:
                    gains[policy][changes] += weight * (errors["fixed"] - errors[policy])
                references.clear()
        values, histogram = wait_metrics(n, waits, EXTRA_WAIT)
        values.update({f"failure_{name}": value for name, value in failures.items()})
        values.update({f"gain_{name}": value for name, value in gains.items()})
        for name in waits:
            values[f"margin_{name}"] = values[f"gain_{name}"] - RECOVERY_TARGET * values["gain_always"]
            values[f"difference_{name}_vs_always"] = gains[name] - gains["always"]
        return values, histogram, {name: mask.sum(axis=0) for name, mask in waits.items()}

    def run_job(self, item):
        onset, chunk, shots, syndrome_seed, alert_seed, measure_failure = item
        start_time = perf_counter()
        sampler = self.circuits[onset].compile_detector_sampler(seed=syndrome_seed)
        rng = np.random.default_rng(alert_seed)
        moments = {
            metric: PairedDifference() for metric in METRICS
            if measure_failure or not metric.startswith("failure_")
        }
        delay_counts = {policy: None for policy in POLICIES}
        commit_counts = {name: np.zeros(len(self.eligible), dtype=np.int64) for name in POLICIES[2:]}
        for start in range(0, shots, self.batch_size):
            n = min(self.batch_size, shots - start)
            packed, actual = sampler.sample(n, separate_observables=True, bit_packed=True)
            values, histogram, commit_waits = self.batch(packed, actual[:, 0] & 1, onset, rng, measure_failure)
            for metric, moment in moments.items():
                moment.add(values[metric])
            for policy, counts in histogram.items():
                if delay_counts[policy] is None:
                    delay_counts[policy] = counts
                else:
                    delay_counts[policy] += counts
            for name, counts in commit_waits.items():
                commit_counts[name] += counts
        key = {"onset": onset, "chunk": chunk}
        rows = [{**key, "metric": metric, **asdict(moment)} for metric, moment in moments.items()]
        delays = [
            {**key, "policy": policy, "added_rounds": i * EXTRA_WAIT, "trials": int(count)}
            for policy, counts in delay_counts.items() for i, count in enumerate(counts)
        ]
        commits = [
            {**key, "policy": policy, "window_index": index, "waits": int(count), "trials": shots}
            for policy, counts in commit_counts.items()
            for index, count in zip(self.eligible, counts, strict=True)
        ]
        print(f"Onset {onset:2d}, chunk {chunk:02d}: {shots:,} trials in {perf_counter()-start_time:.1f}s", flush=True)
        return rows, delays, commits


_simulation = None


def init_worker(settings):
    global _simulation
    _simulation = Simulation(*settings)


def run_worker(item):
    return _simulation.run_job(item)


def make_metadata(args, windows, later, onsets, eligible, calibration, trigger_seed, histogram, threshold, waited, jobs):
    return {
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "distance": DISTANCE, "basis": "x", "rounds": ROUNDS, "commit_width": COMMIT_WIDTH,
        "lookahead": LOOKAHEAD, "burst_duration": BURST_DURATION, "alert_delay": ALERT_DELAY,
        "extra_wait": EXTRA_WAIT, "background_rate": BACKGROUND_RATE, "burst_rate": BURST_RATE,
        "burst_prior": BURST_PRIOR, "alert_sensitivity": SENSITIVITY, "false_alarm_per_candidate": FALSE_ALARM,
        "onsets": onsets, "windows": [asdict(w) for w in windows], "later_windows": [asdict(w) for w in later],
        "eligible": eligible, "threshold_transferred": TRANSFER_THRESHOLD, "threshold_budgeted": threshold,
        "calibration_budget": CALIBRATION_BUDGET, "wait_target": WAIT_TARGET, "recovery_target": RECOVERY_TARGET,
        "calibration_histogram": histogram.tolist(), "calibration_waits": waited,
        "calibration_seeds": calibration, "trigger_seed": trigger_seed,
        "test_chunks": [
            dict(onset=onset, chunk=chunk, shots=shots, syndrome_seed=seed, alert_seed=alert_seed,
                 measure_failure=measure_failure)
            for onset, chunk, shots, seed, alert_seed, measure_failure in jobs
        ],
        "versions": {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "stim", "pymatching")},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026111311)
    parser.add_argument("--calibration-shots", type=int, default=60_000, help="per onset hypothesis")
    parser.add_argument("--trigger-shots", type=int, default=200_000)
    parser.add_argument("--burst-shots", type=int, default=40_000, help="per onset in the test")
    parser.add_argument("--ordinary-shots", type=int, default=6_000_000)
    parser.add_argument("--ordinary-chunks", type=int, default=24)
    parser.add_argument("--ordinary-failure-chunks", type=int, default=2,
                        help="ordinary chunks to decode in full to estimate absolute logical failure rates")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if (args.seed < 0 or min(args.workers, args.batch_size, args.ordinary_chunks) < 1
            or min(args.calibration_shots, args.trigger_shots, args.burst_shots, args.ordinary_shots) < 2
            or args.ordinary_chunks > args.ordinary_shots // 2
            or not 1 <= args.ordinary_failure_chunks <= args.ordinary_chunks):
        parser.error("invalid shot counts, seed, or worker settings")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("choose an empty output directory")
    start = perf_counter()
    windows, later, onsets, eligible = study_schedule()
    calibration, trigger_seed, jobs = seed_plan(args, onsets)
    circuit = memory_circuit(DISTANCE, ROUNDS, "x", BACKGROUND_RATE, BURST_RATE)
    if len(circuit.shortest_graphlike_error()) != DISTANCE:
        raise ValueError("incorrect surface-code distance")
    times = detector_rounds(circuit)
    scorer = SyndromeScores(times, onsets, BURST_DURATION)
    prepare = partial(prepare_hypothesis, times=times, coords=circuit.get_detector_coordinates(),
                      scorer=scorer, shots=args.calibration_shots, batch_size=args.batch_size)
    hypotheses = list(zip((0, *onsets), calibration, strict=True))
    if args.workers == 1:
        prepared = list(map(prepare, hypotheses))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(hypotheses))) as pool:
            prepared = list(pool.map(prepare, hypotheses))
    circuits, edges = {}, {}
    background = prepared[0][-1]
    for onset, circuit, graph, hist in prepared:
        circuits[onset], edges[onset] = circuit, graph
        if onset:
            for key, counts in hist.items():
                scorer.ratios[key] = np.log((counts + 0.5) / (background[key] + 0.5))
    histogram = trigger_calibration(circuits[0], scorer, eligible, args.trigger_shots, trigger_seed, args.batch_size)
    threshold, waited = calibrate_threshold(histogram)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = make_metadata(args, windows, later, onsets, eligible, calibration, trigger_seed, histogram, threshold, waited, jobs)
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Thresholds frozen at {TRANSFER_THRESHOLD} and {threshold} detections. "
          f"Budgeted calibration waits: {waited:,}/{args.trigger_shots:,}. "
          f"Calibration took {perf_counter()-start:.1f}s.", flush=True)
    settings = (circuits, edges, times, scorer, windows, later, eligible, threshold, args.batch_size)
    if args.workers == 1:
        simulation = Simulation(*settings)
        completed = [simulation.run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), initializer=init_worker,
                                 initargs=(settings,)) as pool:
            completed = list(pool.map(run_worker, jobs))
    moments, histograms, commits = ([row for part in column for row in part] for column in zip(*completed, strict=True))
    summary, delay = summarize(moments, histograms, windows, onsets)
    for name, rows in (("paired_moments", moments), ("delay_counts", histograms), ("commit_waits", commits),
                       ("summary", summary), ("delay_distribution", delay)):
        write_csv(args.output / f"{name}.csv", rows)
    print(f"Completed in {perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
