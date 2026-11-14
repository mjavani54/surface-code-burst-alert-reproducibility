"""Verify the frozen inputs, paired moments, and matched latency records."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from matched_policy import frozen_design, lookahead_windows, random_probabilities
from matched_results import CONTRASTS, LATENCY, METRICS, POLICIES, group_weights
from rolling_policy import BURST_PRIOR, EXTRA_WAIT, RECOVERY_TARGET, WAIT_TARGET, study_schedule
from run_matched_wait import check_test_seeds, seed_plan
from write_results import read_csv


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, tolerance=1e-12):
    require(math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=tolerance),
            f"expected {expected}, found {actual}")


def verify(folder):
    prior = frozen_design()
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    windows, later, onsets, eligible = study_schedule()
    require(meta["onsets"] == list(onsets) and meta["eligible"] == prior["eligible"], "incorrect onset schedule")
    for key, schedule in (("windows", windows), ("later_windows", later),
                          ("longer_windows", lookahead_windows(windows, eligible))):
        require(meta[key] == [vars(w) for w in schedule], f"incorrect {key}")
    require(meta["threshold"] == prior["threshold_budgeted"], "incorrect threshold")
    require(meta["calibration_seeds"] == prior["calibration_seeds"], "incorrect frozen calibration")
    for field, value in (("wait_target", WAIT_TARGET), ("recovery_target", RECOVERY_TARGET), ("burst_prior", BURST_PRIOR)):
        close(meta[field], value)
    probabilities = random_probabilities(eligible, onsets)
    np.testing.assert_allclose(meta["random_wait_probabilities"], probabilities, rtol=0, atol=1e-14)
    expected_random = EXTRA_WAIT * probabilities.sum()
    close(meta["random_expected_delay"], expected_random)
    jobs = seed_plan(argparse.Namespace(**meta["settings"]), onsets)
    stored = [
        tuple(row[k] for k in ("onset", "chunk", "shots", "syndrome_seed", "alert_seed", "random_seed"))
        for row in meta["test_chunks"]
    ]
    require(jobs == stored, "incorrect test seed plan")
    check_test_seeds(jobs, prior)
    sizes = {(o, c): n for o, c, n, *_ in jobs}
    moments = defaultdict(dict)
    delay_limit = EXTRA_WAIT * len(eligible)
    for row in read_csv(folder / "paired_moments.csv"):
        key = int(row["onset"]), int(row["chunk"])
        metric, shots, changed = row["metric"], int(row["shots"]), int(row["changed_shots"])
        total, square = float(row["gain_sum"]), float(row["gain_square_sum"])
        require(shots == sizes[key] and metric in METRICS and metric not in moments[key], "invalid paired moment key")
        bound = delay_limit if metric.startswith("delay_") else 1
        require(0 <= changed <= shots and total**2 / shots - 1e-8 <= square <= changed * bound**2 + 1e-8,
                "invalid paired moment")
        moments[key][metric] = (shots, total, square, changed)
    require(moments.keys() == sizes.keys(), "missing paired test chunks")
    bins = defaultdict(dict)
    for row in read_csv(folder / "delay_counts.csv"):
        key = int(row["onset"]), int(row["chunk"]), row["schedule"]
        rounds, n = int(row["added_rounds"]), int(row["trials"])
        require(key[:2] in sizes and key[2] in LATENCY and rounds not in bins[key], "invalid delay bin key")
        require(0 <= rounds <= delay_limit and rounds % EXTRA_WAIT == 0 and n >= 0, "invalid delay bin")
        bins[key][rounds] = n
    commits = defaultdict(dict)
    for row in read_csv(folder / "commit_waits.csv"):
        key = int(row["onset"]), int(row["chunk"]), row["schedule"]
        index, n, shots = int(row["window_index"]), int(row["waits"]), int(row["trials"])
        require(key[2] in LATENCY and index in eligible and index not in commits[key], "invalid commit key")
        require(shots == sizes[key[:2]] and 0 <= n <= shots, "invalid commit counts")
        commits[key][index] = n
    combined = defaultdict(lambda: defaultdict(list))
    trial_counts, delay_counts = defaultdict(int), defaultdict(int)
    for (onset, chunk), data in moments.items():
        require(set(data) == set(METRICS), "incomplete test chunk")
        n = sizes[onset, chunk]
        trial_counts[onset] += n
        for name, (base, target) in CONTRASTS.items():
            total = data[f"failure_{base}"][1] - data[f"failure_{target}"][1]
            close(data[name][1], total, tolerance=1e-8)
        close(data["recovery_margin"][1], data["gain_selective"][1] - RECOVERY_TARGET * data["gain_always"][1])
        for schedule in LATENCY:
            key = onset, chunk, schedule
            hist = bins[key]
            require(set(hist) == set(range(0, delay_limit + 1, EXTRA_WAIT)), "missing delay bins")
            require(sum(hist.values()) == n, "delay histogram loses trials")
            waits = n - hist[0]
            first = sum(r * count for r, count in hist.items())
            second = sum(r * r * count for r, count in hist.items())
            require(data[f"wait_{schedule}"][1:] == (waits, waits, waits), "wait metric differs from histogram")
            require(data[f"delay_{schedule}"][1:] == (first, second, waits), "delay metric differs from histogram")
            require(commits[key].keys() == eligible.keys() and EXTRA_WAIT * sum(commits[key].values()) == first,
                    "per-commit waits differ from the histogram")
            for rounds, count in hist.items():
                delay_counts[onset, schedule, rounds] += count
            if schedule == "trigger":
                close(data["delay_difference"][1], first - n * expected_random)
                close(data["delay_difference"][2], second - 2 * expected_random * first + n * expected_random**2)
        for metric, values in data.items():
            combined[metric][onset].append(values)

    groups = group_weights(windows, onsets)
    rows = read_csv(folder / "summary.csv")
    summary = {(r["group"], r["value"], r["metric"]): r for r in rows}
    require(len(rows) == len(summary) == len(groups) * len(METRICS), "incorrect number of summary rows")
    for (group, value), weights in groups.items():
        for metric in METRICS:
            mean = variance = n_all = changed = 0
            for onset, weight in weights.items():
                n, total, square, nonzero = map(sum, zip(*combined[metric][onset], strict=True))
                mean += weight * total / n
                variance += weight**2 * max(0, square - total**2 / n) / (n * (n - 1))
                n_all += n
                changed += nonzero
            se = math.sqrt(variance)
            row = summary[group, value, metric]
            for key, want in (("estimate", mean), ("paired_se", se), ("ci_low", mean - 1.96 * se),
                              ("ci_high", mean + 1.96 * se), ("test_trials", n_all), ("changed_shots", changed)):
                close(row[key], want)
    rows = read_csv(folder / "delay_distribution.csv")
    saved = {(r["group"], r["value"], r["schedule"], int(r["added_rounds"])): r["probability"] for r in rows}
    checked = 0
    for key, weights in groups.items():
        if key[0] not in ("ordinary", "burst", "population", "arrival_band"):
            continue
        for schedule in LATENCY:
            for rounds in range(0, delay_limit + 1, EXTRA_WAIT):
                want = sum(w * delay_counts[o, schedule, rounds] / trial_counts[o] for o, w in weights.items())
                close(saved[*key, schedule, rounds], want)
                checked += 1
    require(checked == len(saved) == len(rows), "incorrect number of aggregate delay bins")
    print(f"Verified {len(summary):,} estimates from {sum(sizes.values()):,} paired test trials.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
