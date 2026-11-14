"""Paired moment accumulation, stratified estimates, and CSV I/O."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass
class PairedDifference:
    shots: int = 0
    gain_sum: float = 0.0
    gain_square_sum: float = 0.0
    changed_shots: int = 0

    def add(self, differences: np.ndarray):
        self.shots += len(differences)
        self.gain_sum += float(differences.sum())
        self.gain_square_sum += float(np.square(differences).sum())
        self.changed_shots += int(np.count_nonzero(differences))


def paired_gain(strata, weights):
    gain = math.fsum(
        w * float(row["gain_sum"]) / int(row["shots"])
        for row, w in zip(strata, weights, strict=True)
    )
    variance = []
    changed = 0
    for row, w in zip(strata, weights, strict=True):
        if not w:
            continue
        n = int(row["shots"])
        total = float(row["gain_sum"])
        sum_squares = float(row["gain_square_sum"])
        variance.append(w**2 * max(0, sum_squares - total**2 / n) / ((n - 1) * n))
        changed += int(row["changed_shots"])
    se = math.sqrt(math.fsum(variance))
    return {
        "gain": gain, "paired_se": se, "ci_low": gain - 1.96 * se,
        "ci_high": gain + 1.96 * se, "changed_shots": changed,
    }


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))
