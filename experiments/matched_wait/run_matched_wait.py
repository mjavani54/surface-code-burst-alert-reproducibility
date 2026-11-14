"""Compare alert waiting, random waiting, and syndrome lookahead at matched delay."""

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
from matched_policy import CommitDecoder, PolicyEvidence, frozen_design, lookahead_windows, random_probabilities
from matched_results import CONTRASTS, LATENCY, METRICS, POLICIES, summarize, wait_values
from rolling_policy import (
    ALERT_DELAY, BACKGROUND_RATE, BURST_DURATION, BURST_PRIOR, BURST_RATE, DISTANCE, EXTRA_WAIT,
    FALSE_ALARM, RECOVERY_TARGET, ROUNDS, SENSITIVITY, WAIT_TARGET, conditional_false_flags, count_prefix,
    study_schedule, trigger_scores,
)
from surface_code import memory_circuit
from window_decoder import detector_rounds, graph_edges
from write_results import PairedDifference, write_csv


def seed_plan(args, onsets):
    burst, ordinary = np.random.SeedSequence(args.seed).spawn(2)
    burst_seeds = burst.generate_state(3 * len(onsets), dtype=np.uint64).reshape(-1, 3).tolist()
    ordinary_seeds = ordinary.generate_state(3 * args.ordinary_chunks, dtype=np.uint64).reshape(-1, 3).tolist()
    jobs = [(onset, 0, args.burst_shots, *seeds) for onset, seeds in zip(onsets, burst_seeds, strict=True)]
    size, remainder = divmod(args.ordinary_shots, args.ordinary_chunks)
    jobs += [(0, chunk, size + (chunk < remainder), *seeds) for chunk, seeds in enumerate(ordinary_seeds)]
    return jobs


def calibration_histograms(circuit, onset, scorer, shots, seed, batch_size):
    """Reconstruct the v0.7.0 likelihood tables with the original sampling plan."""
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
        for (candidate, length), hist in histograms.items():
            offset = candidate - scorer.first
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


def local_edges(graphs, windows, longer, times):
    """Keep only the graph edges each matching window can access."""
    stops = [(index, False, w) for index, w in enumerate(windows)]
    stops += [(index, True, long) for index, long in enumerate(longer) if long != windows[index]]
    sorted_graphs = {}
    for onset, edges in graphs.items():
        ordered = sorted(enumerate(edges), key=lambda item: first_node(item[1]))
        nodes = [first_node(edge) for _, edge in ordered]
        sorted_graphs[onset] = np.asarray(nodes), ordered
    slices = {}
    for index, extended, w in stops:
        first, end = np.searchsorted(times, [w.start, w.stop + 1])
        window_edges = {}
        known = {}
        for onset, (nodes, ordered) in sorted_graphs.items():
            start, stop = np.searchsorted(nodes, (first, end))
            # The original insertion order fixes deterministic matching tie behavior.
            local = tuple(edge for _, edge in sorted(ordered[start:stop]))
            window_edges[onset] = known.setdefault(local, (onset, local))
        slices[index, extended] = window_edges
        if w == longer[index]:
            slices[index, True] = window_edges
    return slices


def first_node(edge):
    return edge.first if edge.second is None else min(edge.first, edge.second)


class Simulation:
    def __init__(self, circuits, edges, times, scorer, windows, later, longer, eligible,
                 threshold, probabilities, batch_size):
        self.circuits = circuits
        self.scorer = scorer
        self.evidence = PolicyEvidence(scorer, windows, later, longer, eligible)
        self.eligible = eligible
        self.threshold = threshold
        self.probabilities = probabilities
        self.batch_size = batch_size
        self.decoder = CommitDecoder(times, windows, longer, edges)
        self.num_detectors = len(times)

    def batch(self, packed, actual, onset, alert_rng, random_rng):
        n = len(packed)
        syndromes = np.unpackbits(packed, axis=1, count=self.num_detectors, bitorder="little")
        prefix = count_prefix(self.scorer.counts(syndromes))
        trigger = trigger_scores(prefix, self.scorer.first, self.eligible) >= self.threshold
        random = random_rng.random(trigger.shape) < self.probabilities
        full, pending = self.evidence.alert.scores(prefix)
        syndrome = self.evidence.nominal_syndrome(full, pending)
        tables = self.evidence.extended_scores(prefix, full, trigger)
        onsets = self.evidence.alert.onsets
        if onset:
            probabilities = np.where(onsets == onset, SENSITIVITY, FALSE_ALARM)
            cases = ((1.0, alert_rng.random((n, len(onsets))) < probabilities),)
        else:
            mass, flags = conditional_false_flags(alert_rng, n, len(onsets), FALSE_ALARM)
            cases = ((1 - mass, None), (mass, flags))
        failures = {p: np.zeros(n) for p in POLICIES}
        contrasts = {name: np.zeros(n) for name in CONTRASTS}
        for weight, flags in cases:
            policies = self.evidence.decisions(full, pending, syndrome, tables, trigger, random, flags)
            # Syndrome-only trajectories are reused for the two false-alert strata.
            if flags is not None and not onset:
                for name in ("fixed_syndrome", "selective_syndrome"):
                    del policies[name]
            reference = policies.pop("fixed_alert")
            fixed, others = self.decoder.alternatives(syndromes, reference, policies)
            errors = {"fixed_alert": (fixed != actual).astype(np.int8)}
            errors.update({name: (logical != actual).astype(np.int8) for name, logical in others.items()})
            if not onset:
                if flags is None:
                    syndrome_errors = {name: errors[name] for name in ("fixed_syndrome", "selective_syndrome")}
                errors.update(syndrome_errors)
            for name, failure in errors.items():
                failures[name] += weight * failure
            for name, (base, target) in CONTRASTS.items():
                contrasts[name] += weight * (errors[base] - errors[target])
        values, histograms = wait_values(trigger, random, self.probabilities)
        values.update({f"failure_{p}": value for p, value in failures.items()})
        values.update(contrasts)
        values["recovery_margin"] = contrasts["gain_selective"] - RECOVERY_TARGET * contrasts["gain_always"]
        return values, histograms, {"trigger": trigger.sum(axis=0), "random": random.sum(axis=0)}

    def run_job(self, item):
        onset, chunk, shots, syndrome_seed, alert_seed, random_seed = item
        start_time = perf_counter()
        sampler = self.circuits[onset].compile_detector_sampler(seed=syndrome_seed)
        alert_rng = np.random.default_rng(alert_seed)
        random_rng = np.random.default_rng(random_seed)
        moments = {metric: PairedDifference() for metric in METRICS}
        delay_counts = {name: np.zeros(len(self.eligible) + 1, dtype=np.int64) for name in LATENCY}
        commit_counts = {name: np.zeros(len(self.eligible), dtype=np.int64) for name in LATENCY}
        for start in range(0, shots, self.batch_size):
            n = min(shots - start, self.batch_size)
            packed, actual = sampler.sample(n, separate_observables=True, bit_packed=True)
            values, histograms, commits = self.batch(packed, actual[:, 0] & 1, onset, alert_rng, random_rng)
            for metric, moment in moments.items():
                moment.add(values[metric])
            for name in LATENCY:
                delay_counts[name] += histograms[name]
                commit_counts[name] += commits[name]
        key = {"onset": onset, "chunk": chunk}
        rows = [{**key, "metric": metric, **asdict(moment)} for metric, moment in moments.items()]
        delays = [
            {**key, "schedule": name, "added_rounds": i * EXTRA_WAIT, "trials": int(count)}
            for name, counts in delay_counts.items() for i, count in enumerate(counts)
        ]
        commits = [
            {**key, "schedule": name, "window_index": i, "waits": int(count), "trials": shots}
            for name, counts in commit_counts.items() for i, count in zip(self.eligible, counts, strict=True)
        ]
        print(f"Onset {onset:2d}, chunk {chunk:02d}: {shots:,} trials in {perf_counter()-start_time:.1f}s", flush=True)
        return rows, delays, commits


_simulation = None


def init_worker(settings):
    global _simulation
    _simulation = Simulation(*settings)


def run_worker(item):
    return _simulation.run_job(item)


def check_test_seeds(jobs, prior):
    used = {*prior["calibration_seeds"], prior["trigger_seed"]}
    used.update(row[name] for row in prior["test_chunks"] for name in ("syndrome_seed", "alert_seed"))
    seeds = [seed for job in jobs for seed in job[3:]]
    if len(set(seeds)) != len(seeds) or used.intersection(seeds):
        raise ValueError("test seeds overlap with each other or the earlier experiment")


def metadata(args, windows, later, longer, onsets, eligible, threshold, probabilities, jobs, prior):
    return {
        "settings": {k: v for k, v in vars(args).items() if k != "output"},
        "distance": DISTANCE, "basis": "x", "rounds": ROUNDS, "burst_duration": BURST_DURATION,
        "alert_delay": ALERT_DELAY, "extra_wait": EXTRA_WAIT, "background_rate": BACKGROUND_RATE,
        "burst_rate": BURST_RATE, "burst_prior": BURST_PRIOR, "alert_sensitivity": SENSITIVITY,
        "false_alarm_per_candidate": FALSE_ALARM, "threshold": threshold, "wait_target": WAIT_TARGET,
        "recovery_target": RECOVERY_TARGET, "random_wait_probabilities": probabilities.tolist(),
        "random_expected_delay": EXTRA_WAIT * float(probabilities.sum()),
        "onsets": onsets, "eligible": eligible, "windows": [asdict(w) for w in windows],
        "later_windows": [asdict(w) for w in later], "longer_windows": [asdict(w) for w in longer],
        "calibration_seed": prior["settings"]["seed"], "calibration_seeds": prior["calibration_seeds"],
        "test_chunks": [
            dict(onset=o, chunk=c, shots=n, syndrome_seed=s, alert_seed=a, random_seed=r)
            for o, c, n, s, a, r in jobs
        ],
        "versions": {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "stim", "pymatching")},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026111312)
    parser.add_argument("--burst-shots", type=int, default=40_000, help="test trials per onset")
    parser.add_argument("--ordinary-shots", type=int, default=3_000_000)
    parser.add_argument("--ordinary-chunks", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if (args.seed < 0 or min(args.workers, args.batch_size, args.ordinary_chunks) < 1
            or args.burst_shots < 2 or args.ordinary_shots < 2 * args.ordinary_chunks):
        parser.error("invalid shot counts, seed, or worker settings")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("choose an empty output directory")
    start = perf_counter()
    prior = frozen_design()
    windows, later, onsets, eligible = study_schedule()
    longer = lookahead_windows(windows, eligible)
    jobs = seed_plan(args, onsets)
    check_test_seeds(jobs, prior)
    threshold = prior["threshold_budgeted"]
    probabilities = random_probabilities(eligible, onsets)
    circuit = memory_circuit(DISTANCE, ROUNDS, "x", BACKGROUND_RATE, BURST_RATE)
    if len(circuit.shortest_graphlike_error()) != DISTANCE:
        raise ValueError("incorrect surface-code distance")
    times = detector_rounds(circuit)
    scorer = SyndromeScores(times, onsets, BURST_DURATION)
    prepare = partial(prepare_hypothesis, times=times, coords=circuit.get_detector_coordinates(),
                      scorer=scorer, shots=prior["settings"]["calibration_shots"],
                      batch_size=prior["settings"]["batch_size"])
    hypotheses = list(zip((0, *onsets), prior["calibration_seeds"], strict=True))
    if args.workers == 1:
        prepared = list(map(prepare, hypotheses))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(hypotheses))) as pool:
            prepared = list(pool.map(prepare, hypotheses))
    circuits, edges = {}, {}
    background = prepared[0][-1]
    for onset, hypothesis, graph, hist in prepared:
        circuits[onset], edges[onset] = hypothesis, graph
        if onset:
            for key, counts in hist.items():
                scorer.ratios[key] = np.log((counts + 0.5) / (background[key] + 0.5))
    edges = local_edges(edges, windows, longer, times)
    del prepared
    args.output.mkdir(parents=True, exist_ok=True)
    meta = metadata(args, windows, later, longer, onsets, eligible, threshold, probabilities, jobs, prior)
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen threshold {threshold}. Expected random wait {meta['random_expected_delay']:.6f} "
          f"added rounds per trial. Preparation took {perf_counter()-start:.1f}s.", flush=True)
    settings = (circuits, edges, times, scorer, windows, later, longer, eligible, threshold, probabilities, args.batch_size)
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
