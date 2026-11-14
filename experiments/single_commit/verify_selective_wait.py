"""Check calibration, seed streams, and paired estimates from the result files."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from run_selective_wait import (
    BURST_PRIOR, CALIBRATION_WAIT_BUDGET, METRICS, ONSETS, RECOVERY_TARGET,
    deadlines, seed_plan, threshold_for_budget,
)
from write_results import read_csv
import numpy as np


def close(actual, expected):
    if not math.isclose(float(actual), expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"expected {expected}, found {actual}")


def verify(folder):
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    settings = argparse.Namespace(**meta["settings"])
    calibration, trigger_seed, test_items = seed_plan(settings)
    saved_items = [
        (row["onset"], row["chunk"], row["shots"], row["syndrome_seed"], row["alert_seed"])
        for row in meta["test_chunks"]
    ]
    if (calibration, trigger_seed, test_items) != (meta["calibration_seeds"], meta["trigger_seed"], saved_items):
        raise ValueError("saved seed streams do not match the run seed")
    seeds = [*calibration, trigger_seed, *(seed for item in test_items for seed in item[-2:])]
    if len(seeds) != len(set(seeds)):
        raise ValueError("calibration and test streams overlap")

    windows, index, waiting, onsets = deadlines()
    if ([vars(w) for w in windows], index, vars(waiting), onsets) != (
        meta["windows"], meta["target_window"], meta["waiting_window"], tuple(meta["trigger_onsets"])
    ):
        raise ValueError("saved commit windows do not match the experiment")
    hist = np.array(meta["trigger_histogram"], dtype=np.int64)
    if hist.sum() != settings.trigger_shots or (hist < 0).any():
        raise ValueError("invalid trigger calibration histogram")
    threshold, waits = threshold_for_budget(hist, CALIBRATION_WAIT_BUDGET)
    if (threshold, waits) != (meta["trigger_threshold"], meta["trigger_calibration_waits"]):
        raise ValueError("trigger threshold does not match the calibration budget")
    close(meta["recovery_target"], RECOVERY_TARGET)
    close(meta["burst_prior"], BURST_PRIOR)

    chunks = {(onset, chunk): shots for onset, chunk, shots, _, _ in test_items}
    collected = defaultdict(dict)
    for row in read_csv(folder / "paired_moments.csv"):
        key = (int(row["true_onset"]), int(row["chunk"]))
        n = int(row["shots"])
        total, square = (float(row[k]) for k in ("gain_sum", "gain_square_sum"))
        changed = int(row["changed_shots"])
        metric = row["metric"]
        if n != chunks[key] or not 0 <= changed <= n:
            raise ValueError("invalid shot count")
        if not total**2 / n - 1e-9 <= square <= changed + 1e-9:
            raise ValueError("invalid paired second moment")
        if metric in collected[key]:
            raise ValueError("duplicate metric")
        collected[key][metric] = (n, total, square, changed)
    if collected.keys() != chunks.keys():
        raise ValueError("missing or unexpected test chunks")

    cohorts = defaultdict(lambda: defaultdict(list))
    for (onset, _), entries in collected.items():
        if set(entries) != set(METRICS):
            raise ValueError("incomplete test chunk")
        _, sum_always, square_always, changed_always = entries["always_wait_gain"]
        _, sum_selective, square_selective, changed_selective = entries["selective_wait_gain"]
        _, sum_foregone, square_foregone, changed_foregone = entries["foregone_gain"]
        _, sum_margin, square_margin, _ = entries["recovery_margin"]
        n, waits, square_waits, changed_waits = entries["selective_wait_rate"]
        close(sum_always, sum_selective + sum_foregone)
        close(square_always, square_selective + square_foregone)
        close(changed_always, changed_selective + changed_foregone)
        close(sum_margin, sum_selective - RECOVERY_TARGET * sum_always)
        close(square_margin, (1 - RECOVERY_TARGET)**2 * square_selective + RECOVERY_TARGET**2 * square_foregone)
        if not (waits == square_waits == changed_waits and changed_selective <= waits <= n - changed_foregone):
            raise ValueError("the selective policy differs from its chosen deadline")
        for metric, values in entries.items():
            cohorts[metric][onset].append(values)

    q = meta["burst_prior"]
    burst_weights = {onset: 1 / len(ONSETS) for onset in ONSETS}
    groups = {"ordinary": {0: 1.0}, "burst": burst_weights,
              "population": {0: 1 - q, **{onset: q*w for onset, w in burst_weights.items()}}}
    expected = {}
    for metric, strata in cohorts.items():
        for group, weights in groups.items():
            mean = variance = changed = 0
            for onset, weight in weights.items():
                n, total, square, nonzero = [sum(values) for values in zip(*strata[onset], strict=True)]
                mean += weight * total / n
                variance += weight**2 * max(0, square - total**2 / n) / (n * (n - 1))
                changed += nonzero
            se = math.sqrt(variance)
            expected[metric, group] = (mean, se, mean - 1.96*se, mean + 1.96*se, changed)
    saved = read_csv(folder / "summary.csv")
    if len(saved) != len(expected):
        raise ValueError("unexpected summary size")
    for row in saved:
        check = expected.pop((row["metric"], row["group"]))
        fields = ("gain", "paired_se", "ci_low", "ci_high", "changed_shots")
        for field, value in zip(fields, check, strict=True):
            close(row[field], value)
    print(f"Verified {len(saved)} estimates from {sum(chunks.values()):,} shared test trials.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
