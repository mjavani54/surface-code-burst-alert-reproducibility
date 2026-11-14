"""Verify the pretest calibration, paired moments, and stratified summaries."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from diagnose_trigger import aggregate
from robust_design import (
    BACKGROUND, BURST, BURST_LENGTH, DISTANCE, EXTRA_WAIT, FALSE_ALARM, FIXED_DELAY, JITTER,
    PATCH_CENTERS, PATCH_RADIUS, PRIOR, ROUNDS, SCOPES, SENSITIVITY, THRESHOLD_BUDGET, TIMINGS,
    hot_patches, local_background_jobs, seed_plan, study_schedule,
)
from robust_design import threshold_from_histogram
from robust_results import CONTRASTS, METRICS, cohorts
from rolling_policy import RECOVERY_TARGET
from write_results import read_csv


def require(test, message):
    if not test:
        raise ValueError(message)


def close(got, want, tol=1e-12):
    require(math.isclose(float(got), want, rel_tol=1e-10, abs_tol=tol), f"expected {want}, found {got}")


def verify(folder):
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    windows, longer, onsets, eligible = study_schedule()
    require(meta["windows"] == [vars(w) for w in windows] and meta["longer_windows"] == [vars(w) for w in longer],
            "incorrect syndrome windows")
    require(meta["onsets"] == list(onsets) and meta["eligible"] == list(eligible), "incorrect onset support")
    require(meta["distance"] == DISTANCE and meta["patch_qubits"] == [sorted(p) for p in hot_patches()],
            "incorrect local noise setting")
    for key, value in (
        ("rounds", ROUNDS), ("basis", "x"), ("burst_duration", BURST_LENGTH), ("extra_wait", EXTRA_WAIT),
        ("fixed_delay", list(FIXED_DELAY)), ("jitter", list(JITTER)), ("patch_radius", PATCH_RADIUS),
        ("patch_centers", list(map(list, PATCH_CENTERS))), ("scopes", list(SCOPES)), ("timings", list(TIMINGS)),
    ):
        require(meta[key] == value, f"incorrect {key}")
    for key, value in (
        ("background", BACKGROUND), ("burst_rate", BURST), ("prior", PRIOR), ("false_alarm", FALSE_ALARM),
        ("sensitivity", SENSITIVITY), ("trigger_budget", THRESHOLD_BUDGET), ("recovery_target", RECOVERY_TARGET),
    ):
        close(meta[key], value)
    settings = meta["settings"]
    graph, trigger, jobs = seed_plan(settings["seed"], onsets, settings["burst_shots"],
                                     settings["ordinary_shots"], settings["ordinary_chunks"])
    require(meta["calibration_seeds"] == graph and meta["trigger_seed"] == trigger, "incorrect calibration seeds")
    require(meta["test_jobs"] == list(map(list, jobs)), "incorrect test seed plan")
    replay = local_background_jobs(jobs)
    require(meta["local_background_jobs"] == list(map(list, replay)), "incorrect shared burst-free replay")
    seeds = [*graph, trigger, *(seed for job in jobs for seed in job[4:])]
    require(len(set(seeds)) == len(seeds), "a seed was reused")
    for values, threshold in zip(meta["trigger_histograms"], meta["thresholds"], strict=True):
        hist = np.asarray(values)
        require(hist.sum() == settings["trigger_shots"] and (hist >= 0).all(), "invalid calibration histogram")
        require(threshold_from_histogram(hist) == threshold, "incorrect frozen threshold")
    expected = {job[:3]: job[3] for job in jobs + replay}
    entries = defaultdict(dict)
    for row in read_csv(folder / "paired_moments.csv"):
        key = tuple(int(row[k]) for k in ("scope", "onset", "chunk"))
        n, changed = (int(row[k]) for k in ("shots", "changed_shots"))
        total, square = (float(row[k]) for k in ("gain_sum", "gain_square_sum"))
        metric = row["metric"]
        require(metric in METRICS and metric not in entries[key] and n == expected[key], "incorrect moment key")
        bound = (EXTRA_WAIT * len(eligible)) if metric.endswith(":delay") else (2 if metric.startswith("jitter_minus") else 1)
        require(0 <= changed <= n and total**2 / n - 1e-8 <= square <= changed * bound**2 + 1e-8,
                "invalid moments")
        entries[key][metric] = (n, total, square, changed)
    require(entries.keys() == expected.keys(), "missing chunks")
    for job in replay:
        _, onset, chunk = job[:3]
        for timing in TIMINGS:
            for metric in ("wait", "delay"):
                key = f"{timing}:{metric}"
                require(entries[0, onset, chunk][key] == entries[2, onset, chunk][key],
                        "shared burst-free trials yielded different waits")
    combined = defaultdict(lambda: defaultdict(list))
    for (scope, onset, _), data in entries.items():
        require(set(data) == set(METRICS), "missing chunk metrics")
        for timing in TIMINGS:
            def moment(metric):
                return data[f"{timing}:{metric}"]

            for metric, (base, target) in CONTRASTS.items():
                close(moment(metric)[1], moment(f"failure_{base}")[1] - moment(f"failure_{target}")[1], 1e-8)
            close(moment("recovery_margin")[1],
                  moment("gain_alert")[1] - RECOVERY_TARGET * moment("gain_always")[1], 1e-8)
            _, wait, square, changed = moment("wait")
            require(wait == square == changed and EXTRA_WAIT * wait <= moment("delay")[1], "invalid wait moments")
        for metric in ("both_vs_lookahead", "alert_vs_lookahead", "wait"):
            close(data[f"jitter_minus_fixed:{metric}"][1], data[f"jitter:{metric}"][1] - data[f"fixed:{metric}"][1], 1e-8)
        for metric, values in data.items():
            combined[scope, onset][metric].append(values)

    rows = read_csv(folder / "summary.csv")
    saved = {(r["scope"], r["group"], r["timing"], r["metric"]): r for r in rows}
    require(len(saved) == len(rows), "duplicate summary row")
    checked = 0
    for scope, name in enumerate(SCOPES, 1):
        for group, weights in cohorts(scope, onsets, windows, eligible):
            for metric in METRICS:
                shots = changes = estimate = variance = 0
                for key, weight in weights.items():
                    n, total, square, changed = map(sum, zip(*combined[key][metric], strict=True))
                    estimate += weight * total / n
                    variance += weight**2 * max(0, square - total**2 / n) / (n * (n - 1))
                    shots += n
                    changes += changed
                se = math.sqrt(variance)
                row = saved[name, group, *metric.split(":")]
                for key, value in (("test_trials", shots), ("changed_shots", changes), ("estimate", estimate),
                                   ("paired_se", se), ("ci_low", estimate - 1.96 * se), ("ci_high", estimate + 1.96 * se)):
                    close(row[key], value)
                checked += 1
    require(checked == len(saved), "unexpected summary rows")
    diagnostics = folder / "trigger_capture.csv"
    if diagnostics.exists():
        per_onset = read_csv(folder / "trigger_capture_by_onset.csv")
        diagnostic_keys = {(r["scope"], r["timing"], int(r["onset"])) for r in per_onset}
        require(diagnostic_keys == {(s, t, o) for s in SCOPES for t in TIMINGS for o in onsets}
                and len(per_onset) == len(diagnostic_keys), "incomplete or duplicate trigger diagnostics")
        diagnostic_seeds = {int(r["seed"]) for r in per_onset}
        require(len(diagnostic_seeds) == 2 * len(onsets) and not diagnostic_seeds.intersection(seeds),
                "invalid diagnostic seeds")
        for got, want in zip(read_csv(diagnostics), aggregate(per_onset), strict=True):
            for key in ("scope", "timing"):
                require(got[key] == want[key], "incorrect diagnostic grouping")
            for key in ("test_trials", "eligible_probability", "capture_given_eligible", "ci_low", "ci_high"):
                close(got[key], want[key])
    print(f"Verified {checked:,} estimates from {sum(job[3] for job in jobs):,} distinct syndrome trials.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
