"""Fit a syndrome-based onset score and add timestamped alert evidence."""

from itertools import product

import numpy as np
import stim

from window_decoder import Window


class SyndromeScores:
    def __init__(self, times: np.ndarray, onsets: tuple[int, ...], duration: int):
        self.onsets = onsets
        self.duration = duration
        self.first = min(onsets)
        self.last = max(onsets) + duration
        self.slices = np.searchsorted(times, np.arange(self.first, self.last + 2))
        self.ratios: dict[tuple[int, int], np.ndarray] = {}

    def counts(self, syndromes: np.ndarray, offset: int = 0) -> np.ndarray:
        return np.column_stack([
            syndromes[:, first - offset:last - offset].sum(axis=1, dtype=np.int32)
            for first, last in zip(self.slices[:-1], self.slices[1:], strict=True)
        ])

    def packed_counts(self, packed: np.ndarray) -> np.ndarray:
        first_byte = self.slices[0] // 8
        last_byte = (self.slices[-1] + 7) // 8
        bits = np.unpackbits(packed[:, first_byte:last_byte], axis=1, bitorder="little")
        return self.counts(bits, offset=8 * first_byte)

    def fit(self, circuits: dict[int, stim.Circuit], shots: int, seeds: list[int], batch_size: int) -> None:
        def calibration_counts(onset, seed):
            sampler = circuits[onset].compile_detector_sampler(seed=seed)
            return np.concatenate([
                self.counts(sampler.sample(min(batch_size, shots - start)))
                for start in range(0, shots, batch_size)
            ])

        background = calibration_counts(0, seeds[0])
        for onset, seed in zip(self.onsets, seeds[1:], strict=True):
            burst = calibration_counts(onset, seed)
            start = onset - self.first
            normal_prefix = background[:, start:start + self.duration + 1].cumsum(axis=1)
            burst_prefix = burst[:, start:start + self.duration + 1].cumsum(axis=1)
            for j in range(self.duration + 1):
                max_count = self.slices[start + j + 1] - self.slices[start]
                normal_hist = np.bincount(normal_prefix[:, j], minlength=max_count + 1) + 0.5
                burst_hist = np.bincount(burst_prefix[:, j], minlength=max_count + 1) + 0.5
                # Both histograms have the same total and the same smoothing.
                self.ratios[(onset, j + 1)] = np.log(burst_hist / normal_hist)

    def for_windows(self, syndromes: np.ndarray, windows: tuple[Window, ...], burst_prior: float):
        return self.score_counts(self.counts(syndromes), windows, burst_prior)

    def score_counts(self, counts: np.ndarray, windows: tuple[Window, ...], burst_prior: float):
        prior_odds = np.log(burst_prior / ((1 - burst_prior) * len(self.onsets)))
        scores = []
        for window in windows:
            table = np.full((len(counts), len(self.onsets) + 1), prior_odds)
            table[:, 0] = 0.0
            for index, onset in enumerate(self.onsets, 1):
                visible = min(self.duration + 1, window.stop - onset + 1)
                if visible <= 0:
                    continue
                start = onset - self.first
                total = counts[:, start:start + visible].sum(axis=1)
                table[:, index] += self.ratios[(onset, visible)][total]
            scores.append(table)
        return scores


def choices(scores, onsets, windows, *, flags=None, delay=0, sensitivity=0.90, false_alarm=0.0001):
    candidates = np.array((0, *onsets))
    if flags is None:
        return [candidates[table.argmax(axis=1)] for table in scores]
    log_positive = np.log(sensitivity / false_alarm)
    log_negative = np.log((1 - sensitivity) / (1 - false_alarm))
    evidence = np.where(flags, log_positive, log_negative)
    selected = []
    for table, window in zip(scores, windows, strict=True):
        updated = table.copy()
        visible = np.array(onsets) + delay <= window.deadline
        updated[:, 1:] += evidence * visible
        selected.append(candidates[updated.argmax(axis=1)])
    return selected


def exact_onset_choices(windows, true_onset, shots):
    return [
        np.full(shots, true_onset if true_onset <= window.deadline else 0, dtype=int)
        for window in windows
    ]


def ordinary_flags(shots, n_candidates, rng, false_alarm):
    """Conditionally sample nonempty masks and weight them by their exact mass."""
    masks = np.array(list(product((False, True), repeat=n_candidates))[1:])
    sizes = masks.sum(axis=1)
    probabilities = false_alarm**sizes * (1 - false_alarm)**(n_candidates - sizes)
    positive_mass = probabilities.sum()
    drawn = rng.choice(len(masks), shots, p=probabilities / positive_mass)
    return ((1 - positive_mass, np.zeros((shots, n_candidates), dtype=bool)),
            (positive_mass, masks[drawn]))


def burst_flags(shots, onsets, true_onset, rng, sensitivity, false_alarm):
    probabilities = np.where(np.array(onsets) == true_onset, sensitivity, false_alarm)
    return ((1.0, rng.random((shots, len(onsets))) < probabilities),)
