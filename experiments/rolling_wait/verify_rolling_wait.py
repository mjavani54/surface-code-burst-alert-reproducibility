"""Check the run manifest, paired moments, and trial-level delay distributions."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from rolling_policy import (
    BURST_PRIOR, CALIBRATION_BUDGET, EXTRA_WAIT, RECOVERY_TARGET, TRANSFER_THRESHOLD, WAIT_TARGET,
    calibrate_threshold, study_schedule,
)
from rolling_results import METRICS, POLICIES, group_weights
from run_rolling_wait import seed_plan
from write_results import read_csv


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def check_close(got, expected, abs_tol=1e-12):
    require(math.isclose(float(got), expected, abs_tol=abs_tol, rel_tol=1e-10),
            f"expected {expected}, found {got}")


def manifest(folder):
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    windows, later, onsets, eligible = study_schedule()
    require(meta["windows"] == [vars(w) for w in windows], "incorrect windows")
    require(meta["later_windows"] == [vars(w) for w in later], "incorrect extended deadlines")
    require(meta["onsets"] == list(onsets), "incorrect onset set")
    require(meta["eligible"] == {str(i): list(o) for i, o in eligible.items()}, "incorrect eligible windows")
    settings = argparse.Namespace(**meta["settings"])
    seeds, trigger_seed, jobs = seed_plan(settings, onsets)
    saved_jobs = [
        tuple(row[k] for k in ("onset", "chunk", "shots", "syndrome_seed", "alert_seed", "measure_failure"))
        for row in meta["test_chunks"]
    ]
    require((seeds, trigger_seed, jobs) == (meta["calibration_seeds"], meta["trigger_seed"], saved_jobs),
            "the test or calibration seeds differ from the run plan")
    streams = [*seeds, trigger_seed, *(seed for job in jobs for seed in job[3:5])]
    require(len(set(streams)) == len(streams), "a seed was reused")
    hist = np.array(meta["calibration_histogram"], dtype=np.int64)
    require((hist >= 0).all() and hist.sum() == settings.trigger_shots, "invalid calibration histogram")
    require(calibrate_threshold(hist) == (meta["threshold_budgeted"], meta["calibration_waits"]),
            "invalid calibrated threshold")
    require(meta["threshold_transferred"] == TRANSFER_THRESHOLD, "incorrect transferred threshold")
    for name, value in (("wait_target", WAIT_TARGET), ("calibration_budget", CALIBRATION_BUDGET),
                        ("recovery_target", RECOVERY_TARGET), ("burst_prior", BURST_PRIOR)):
        check_close(meta[name], value)
    return jobs, windows, onsets, eligible, meta["threshold_budgeted"]


def verify(folder):
    jobs, windows, onsets, eligible, threshold = manifest(folder)
    expected = {(onset, chunk): (n, measured) for onset, chunk, n, _, _, measured in jobs}
    entries = defaultdict(dict)
    for row in read_csv(folder / "paired_moments.csv"):
        key = (int(row["onset"]), int(row["chunk"]))
        n, measured = expected[key]
        metric = row["metric"]
        require(metric in METRICS and metric not in entries[key], "unexpected or duplicate metric")
        require(int(row["shots"]) == n, "incorrect moment trial count")
        changed = int(row["changed_shots"])
        total, square = float(row["gain_sum"]), float(row["gain_square_sum"])
        bound = EXTRA_WAIT * len(eligible) if metric.startswith("delay_") else 1
        require(0 <= changed <= n and total**2 / n - 1e-8 <= square <= changed * bound**2 + 1e-8,
                "invalid first or second moment")
        entries[key][metric] = (n, total, square, changed)
    require(entries.keys() == expected.keys(), "missing test chunks")
    hist = defaultdict(dict)
    for row in read_csv(folder / "delay_counts.csv"):
        key = (int(row["onset"]), int(row["chunk"]), row["policy"])
        delay, count = int(row["added_rounds"]), int(row["trials"])
        require(key[:2] in expected and key[2] in POLICIES and delay not in hist[key],
                "unexpected or duplicate delay bin")
        require(0 <= delay <= EXTRA_WAIT * len(eligible) and delay % EXTRA_WAIT == 0 and count >= 0,
                "invalid delay bin")
        hist[key][delay] = count
    commits = defaultdict(dict)
    for row in read_csv(folder / "commit_waits.csv"):
        key = (int(row["onset"]), int(row["chunk"]), row["policy"])
        window = int(row["window_index"])
        n, count = int(row["trials"]), int(row["waits"])
        require(key[2] in POLICIES[2:] and window in eligible and window not in commits[key],
                "unexpected or duplicate commit count")
        require(n == expected[key[:2]][0] and 0 <= count <= n, "incorrect commit trial count")
        commits[key][window] = count

    totals = defaultdict(lambda: defaultdict(list))
    trial_delays = defaultdict(lambda: defaultdict(int))
    trial_counts = defaultdict(int)
    for (onset, chunk), data in entries.items():
        n, measured = expected[onset, chunk]
        require(set(data) == {m for m in METRICS if measured or not m.startswith("failure_")},
                "incomplete chunk")
        trial_counts[onset] += n
        gain_always = data["gain_always"][1]
        for policy in POLICIES[1:]:
            gain = data[f"gain_{policy}"][1]
            if measured:
                # The two unpaired totals can each be several thousand while
                # their difference is near zero. Allow for cancellation.
                check_close(gain, data["failure_fixed"][1] - data[f"failure_{policy}"][1], abs_tol=1e-8)
            if policy in POLICIES[2:]:
                check_close(data[f"margin_{policy}"][1], gain - RECOVERY_TARGET * gain_always)
                check_close(data[f"difference_{policy}_vs_always"][1], gain - gain_always)
        for policy in POLICIES:
            bins = hist[onset, chunk, policy]
            require(sum(bins.values()) == n, "missing delay histogram mass")
            for delay, count in bins.items():
                trial_delays[onset, policy][delay] += count
            if policy in POLICIES[2:]:
                first = sum(delay * count for delay, count in bins.items())
                second = sum(delay**2 * count for delay, count in bins.items())
                waits = n - bins[0]
                require(data[f"wait_{policy}"][1:] == (waits, waits, waits), "wait count disagrees with histogram")
                require(data[f"delay_{policy}"][1:] == (first, second, waits), "delay moments disagree with histogram")
                by_commit = commits[onset, chunk, policy]
                require(by_commit.keys() == eligible.keys() and EXTRA_WAIT * sum(by_commit.values()) == first,
                        "commit counts disagree with the delay histogram")
            else:
                delay = 0 if policy == "fixed" else EXTRA_WAIT * len(eligible)
                require(bins[delay] == n, "reference delays must be constant")
        low, high = ("transferred", "budgeted") if threshold >= TRANSFER_THRESHOLD else ("budgeted", "transferred")
        require(all(commits[onset, chunk, low][w] >= commits[onset, chunk, high][w] for w in eligible),
                "the stricter threshold waited more often at a commit")
        for metric, moments in data.items():
            totals[metric][onset].append(moments)

    groups = group_weights(windows, onsets)
    saved = {(row["group"], row["value"], row["metric"]): row for row in read_csv(folder / "summary.csv")}
    require(len(saved) == len(groups) * len(METRICS), "incorrect summary row count")
    for (group, value), weights in groups.items():
        for metric in METRICS:
            n_all = mean = variance = changed = 0
            for onset, weight in weights.items():
                n, total, square, nonzero = map(sum, zip(*totals[metric][onset], strict=True))
                mean += weight * total / n
                variance += weight**2 * max(0, square - total**2 / n) / (n * (n - 1))
                n_all += n
                changed += nonzero
            row = saved[group, str(value), metric]
            se = math.sqrt(variance)
            for field, number in (("estimate", mean), ("paired_se", se), ("ci_low", mean - 1.96 * se),
                                  ("ci_high", mean + 1.96 * se), ("test_trials", n_all), ("changed_shots", changed)):
                check_close(row[field], number)
    distributions = read_csv(folder / "delay_distribution.csv")
    delay_groups = {
        key for key in groups if key[0] in ("ordinary", "burst", "population", "arrival_phase", "arrival_band")
    }
    require(len(distributions) == len(delay_groups) * len(POLICIES) * (len(eligible) + 1),
            "incorrect number of delay bins")
    checked = set()
    for row in distributions:
        group, value, policy = row["group"], row["value"], row["policy"]
        value = int(value) if value.isdecimal() else value
        delay = int(row["added_rounds"])
        key = (group, value, policy, delay)
        require((group, value) in delay_groups and policy in POLICIES and key not in checked
                and 0 <= delay <= EXTRA_WAIT * len(eligible) and delay % EXTRA_WAIT == 0,
                "unexpected or duplicate aggregate delay bin")
        checked.add(key)
        weights = groups[group, value]
        mass = sum(w * trial_delays[onset, policy][delay] / trial_counts[onset] for onset, w in weights.items())
        check_close(row["probability"], mass)
    print(f"Verified {len(saved):,} estimates and delay counts from {sum(trial_counts.values()):,} paired test trials.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
