"""Run the distance-nine burst-scope and alert-delay factorial experiment."""

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

from robust_decoder import CommitDecoder
from robust_design import (
    BACKGROUND, BURST, BURST_LENGTH, DISTANCE, EXTRA_WAIT, FALSE_ALARM, FIXED_DELAY, JITTER,
    PATCH_CENTERS, PATCH_RADIUS, PRIOR, ROUNDS, SCOPES, SENSITIVITY, THRESHOLD_BUDGET, TIMINGS,
    alert_cases, hot_patches, local_background_jobs, seed_plan, study_schedule, threshold_from_histogram, trigger_matrix,
)
from robust_graphs import SyndromeCalibration, build_edges, make_circuit
from robust_policy import POLICIES, Scores, decisions
from robust_results import CONTRASTS, METRICS, summarize
from rolling_policy import RECOVERY_TARGET, count_prefix
from write_results import PairedDifference, write_csv


def fit_burst(item, *, scorer, shots, batch):
    scope, onset, seed = item
    if scope == 1:
        return scorer.histograms(make_circuit(onset), seed, shots, (onset,), batch)
    patches = hot_patches()
    sub_seeds = np.random.SeedSequence(seed).generate_state(len(patches), dtype=np.uint64).tolist()
    result = None
    for patch, sub_seed in zip(patches, sub_seeds, strict=True):
        result = scorer.histograms(make_circuit(onset, patch), sub_seed, shots // len(patches), (onset,),
                                   batch, result)
    return result


def fit_trigger(circuit, scorer, windows, onsets, eligible, shots, seed, batch):
    sampler = circuit.compile_detector_sampler(seed=seed)
    limit = 1 + (BURST_LENGTH + 1) * np.diff(scorer.slices).max()
    histograms = np.zeros((2, limit), dtype=np.int64)
    for start in range(0, shots, batch):
        packed = sampler.sample(min(batch, shots - start), bit_packed=True)
        prefix = count_prefix(scorer.packed_counts(packed))
        for i, delays in enumerate((FIXED_DELAY, JITTER)):
            scores = trigger_matrix(prefix, scorer.first, windows, onsets, eligible, delays)
            histograms[i] += np.bincount(scores.max(axis=1), minlength=limit)
    return histograms


class Simulation:
    def __init__(self, background, times, contexts, scorers, windows, longer, onsets, eligible, thresholds, batch_size):
        self.background = background
        self.scorers = scorers
        self.windows, self.longer = windows, longer
        self.onsets, self.eligible = onsets, eligible
        self.thresholds = thresholds
        self.num_detectors = len(times)
        self.batch_size = batch_size
        self.decoders = [CommitDecoder(times, windows, longer, edges) for edges in contexts]

    def batch(self, packed, actual, scope, onset, alert_rng, jitter_rng):
        scorer = self.scorers[max(0, scope - 1)]
        decoder = self.decoders[max(0, scope - 1)]
        n = len(packed)
        syndrome = np.unpackbits(packed, axis=1, count=self.num_detectors, bitorder="little")
        prefix = count_prefix(scorer.counts(syndrome))
        scores = Scores(scorer, prefix)
        nominal = [scores.at(w.stop) for w in self.windows]
        by_timing = {}
        values = {}
        for timing, delays, threshold in zip(TIMINGS, (FIXED_DELAY, JITTER), self.thresholds, strict=True):
            trigger = trigger_matrix(prefix, scorer.first, self.windows, self.onsets, self.eligible, delays) >= threshold
            extended = {
                i: scores.at(self.longer[i].stop, trigger[:, col])
                for col, i in enumerate(self.eligible) if trigger[:, col].any()
            }
            by_timing[timing] = (trigger, extended, {p: np.zeros(n) for p in POLICIES},
                                 {name: np.zeros(n) for name in CONTRASTS})
            values[f"{timing}:wait"] = trigger.any(axis=1)
            values[f"{timing}:delay"] = EXTRA_WAIT * trigger.sum(axis=1)

        for weight, arrivals in alert_cases(onset, self.onsets, n, alert_rng, jitter_rng):
            schedules = {}
            for timing, received, delays in zip(TIMINGS, arrivals, (FIXED_DELAY, JITTER), strict=True):
                trigger, extended, _, _ = by_timing[timing]
                plans = decisions(scores, self.windows, self.longer, self.eligible, trigger, received,
                                  delays, nominal, extended)
                schedules.update({(timing, policy): plan for policy, plan in plans.items()})
            key = ("fixed", "fixed")
            reference = schedules.pop(key)
            logical, outputs = decoder.alternatives(syndrome, reference, schedules)
            outputs[key] = logical
            for timing, (_, _, failures, contrasts) in by_timing.items():
                failed = {p: (outputs[timing, p] != actual).astype(np.int8) for p in POLICIES}
                for p, errors in failed.items():
                    failures[p] += weight * errors
                for metric, (base, target) in CONTRASTS.items():
                    contrasts[metric] += weight * (failed[base] - failed[target])

        for timing, (_, _, failures, contrasts) in by_timing.items():
            values.update({f"{timing}:failure_{p}": failure for p, failure in failures.items()})
            values.update({f"{timing}:{metric}": gain for metric, gain in contrasts.items()})
            values[f"{timing}:recovery_margin"] = contrasts["gain_alert"] - RECOVERY_TARGET * contrasts["gain_always"]
        for metric in ("both_vs_lookahead", "alert_vs_lookahead", "wait"):
            values[f"jitter_minus_fixed:{metric}"] = values[f"jitter:{metric}"].astype(float) - values[f"fixed:{metric}"]
        return values

    def job(self, item, output):
        scope, onset, chunk, shots, syndrome_seed, alert_seed, jitter_seed = item
        start_time = perf_counter()
        if not onset:
            generators = [(self.background, shots, syndrome_seed)]
        elif scope == 1:
            generators = [(make_circuit(onset), shots, syndrome_seed)]
        else:
            patches = hot_patches()
            seeds = np.random.SeedSequence(syndrome_seed).generate_state(len(patches), dtype=np.uint64).tolist()
            generators = [(make_circuit(onset, patch), shots // len(patches), seed)
                          for patch, seed in zip(patches, seeds, strict=True)]
        alert_rng = np.random.default_rng(alert_seed)
        jitter_rng = np.random.default_rng(jitter_seed)
        moments = {metric: PairedDifference() for metric in METRICS}
        for circuit, count, seed in generators:
            sampler = circuit.compile_detector_sampler(seed=seed)
            for offset in range(0, count, self.batch_size):
                n = min(self.batch_size, count - offset)
                packed, actual = sampler.sample(n, separate_observables=True, bit_packed=True)
                values = self.batch(packed, actual[:, 0] & 1, scope, onset, alert_rng, jitter_rng)
                for metric, moment in moments.items():
                    moment.add(values[metric])
        rows = [{"scope": scope, "onset": onset, "chunk": chunk, "metric": metric, **asdict(moment)}
                for metric, moment in moments.items()]
        path = chunk_path(output, item)
        temp = path.with_suffix(".tmp")
        write_csv(temp, rows)
        temp.replace(path)
        print(f"Scope {scope}, onset {onset:2d}, chunk {chunk:02d}: {shots:,} trials in "
              f"{perf_counter()-start_time:.1f}s", flush=True)


def chunk_path(output, item):
    return output / "chunks" / f"{item[0]}_{item[1]:02d}_{item[2]:02d}.csv"


_simulation = None


def init_worker(settings):
    global _simulation
    _simulation = Simulation(*settings)


def run_worker(item, output):
    _simulation.job(item, output)


def write_results(output, jobs, onsets, windows, eligible):
    from write_results import read_csv

    rows = [row for job in jobs for row in read_csv(chunk_path(output, job))]
    write_csv(output / "paired_moments.csv", rows)
    write_csv(output / "summary.csv", summarize(rows, onsets, windows, eligible))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026111313)
    parser.add_argument("--burst-shots", type=int, default=40_000, help="test trials per onset and scope")
    parser.add_argument("--ordinary-shots", type=int, default=1_800_000)
    parser.add_argument("--ordinary-chunks", type=int, default=18)
    parser.add_argument("--calibration-shots", type=int, default=32_000)
    parser.add_argument("--trigger-shots", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="continue a partial run with the same simulation settings")
    args = parser.parse_args()
    if (args.seed < 0 or min(args.workers, args.batch_size, args.ordinary_chunks) < 1
            or args.ordinary_shots < 2 * args.ordinary_chunks or args.trigger_shots < 2
            or any(n < 4 or n % 4 for n in (args.burst_shots, args.calibration_shots))):
        parser.error("invalid shot counts, seed, or worker settings")
    if args.output.exists() and not args.output.is_dir():
        parser.error("output must be a directory")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("choose a new output directory or use --resume with the same settings")
    start = perf_counter()
    windows, longer, onsets, eligible = study_schedule()
    graph_seeds, trigger_seed, jobs = seed_plan(
        args.seed, onsets, args.burst_shots, args.ordinary_shots, args.ordinary_chunks
    )
    local_background = local_background_jobs(jobs)
    settings = {k: v for k, v in vars(args).items() if k not in ("output", "workers", "resume")}
    run = args.output / "run.json"
    if args.resume:
        old = json.loads(run.read_text(encoding="utf-8"))
        if old["settings"] != settings or old["versions"]["stim"] != version("stim"):
            parser.error("resume requires the original simulation settings and Stim version")
    background, times, contexts = build_edges(windows, longer, onsets)
    calibration = SyndromeCalibration(onsets, times)
    base = calibration.histograms(background, graph_seeds[0], args.calibration_shots, onsets, args.batch_size)
    labels = [(scope, onset) for scope in (1, 2) for onset in onsets]
    items = [(*label, seed) for label, seed in zip(labels, graph_seeds[1:], strict=True)]
    fit = partial(fit_burst, scorer=calibration, shots=args.calibration_shots, batch=args.batch_size)
    if args.workers == 1:
        burst = list(map(fit, items))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            burst = list(pool.map(fit, items))
    scorers = [SyndromeCalibration(onsets, times).scorer for _ in SCOPES]
    for (scope, _), histograms in zip(labels, burst, strict=True):
        for key, hist in histograms.items():
            scorers[scope - 1].ratios[key] = np.log((hist + .5) / (base[key] + .5))
    histograms = fit_trigger(background, scorers[0], windows, onsets, eligible,
                             args.trigger_shots, trigger_seed, args.batch_size)
    thresholds = [threshold_from_histogram(hist) for hist in histograms]
    meta = {
        "settings": settings, "distance": DISTANCE, "rounds": ROUNDS, "basis": "x", "burst_duration": BURST_LENGTH,
        "background": BACKGROUND, "burst_rate": BURST, "prior": PRIOR, "false_alarm": FALSE_ALARM,
        "sensitivity": SENSITIVITY, "extra_wait": EXTRA_WAIT, "scopes": SCOPES, "timings": TIMINGS,
        "fixed_delay": FIXED_DELAY, "jitter": JITTER, "patch_centers": PATCH_CENTERS, "patch_radius": PATCH_RADIUS,
        "patch_qubits": [sorted(p) for p in hot_patches()], "trigger_budget": THRESHOLD_BUDGET,
        "recovery_target": RECOVERY_TARGET, "thresholds": thresholds,
        "trigger_histograms": histograms.tolist(), "onsets": onsets, "eligible": eligible,
        "windows": [asdict(w) for w in windows], "longer_windows": [asdict(w) for w in longer],
        "calibration_seeds": graph_seeds, "trigger_seed": trigger_seed, "test_jobs": jobs,
        "local_background_jobs": local_background,
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
                     **{name: version(name) for name in ("numpy", "stim", "pymatching")}},
    }
    if args.resume:
        # Earlier partial runs recorded the distinct syndrome streams only.
        old.setdefault("local_background_jobs", list(map(list, local_background)))
        if old != json.loads(json.dumps(meta)):
            parser.error("the reconstructed run differs from the original")
    (args.output / "chunks").mkdir(parents=True, exist_ok=True)
    run.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared in {perf_counter()-start:.1f}s. Frozen thresholds: {thresholds}. "
          f"Calibration waits: {[int(h[t:].sum()) for h, t in zip(histograms, thresholds, strict=True)]}.", flush=True)
    params = (background, times, contexts, scorers, windows, longer, onsets, eligible, thresholds, args.batch_size)
    all_jobs = jobs + local_background
    pending = [job for job in all_jobs if not chunk_path(args.output, job).exists()]
    if args.workers == 1:
        simulation = Simulation(*params)
        for item in pending:
            simulation.job(item, args.output)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)), initializer=init_worker,
                                 initargs=(params,)) as pool:
            list(pool.map(partial(run_worker, output=args.output), pending))
    write_results(args.output, all_jobs, onsets, windows, eligible)
    print(f"Completed in {perf_counter()-start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
