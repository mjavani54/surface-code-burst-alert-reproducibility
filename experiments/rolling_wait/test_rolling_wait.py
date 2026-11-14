"""Regression checks for the rolling trigger, graph choices, and paired evaluation."""

from itertools import product
import unittest

import numpy as np

from burst_alert import SyndromeScores, choices
from rolling_policy import (
    ALERT_DELAY, BURST_DURATION, BURST_PRIOR, EXTRA_WAIT, RollingEvidence,
    calibrate_threshold, conditional_false_flags, count_prefix, onsets_by_phase, study_schedule, trigger_scores,
)
from run_rolling_wait import Simulation
from surface_code import memory_circuit
from window_decoder import detector_rounds, graph_edges


class RollingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows, cls.later, cls.onsets, cls.eligible = study_schedule()
        cls.circuit = memory_circuit(7, 84, "x", 0.003, 0.020)
        cls.times = detector_rounds(cls.circuit)
        cls.scorer = SyndromeScores(cls.times, cls.onsets, BURST_DURATION)
        cls.evidence = RollingEvidence(cls.scorer, cls.windows, cls.later)
        cls.sampler = cls.circuit.compile_detector_sampler(seed=987)
        rng = np.random.default_rng(813)
        for onset, length in product(cls.onsets, range(1, BURST_DURATION + 2)):
            first = onset - cls.scorer.first
            size = cls.scorer.slices[first + length] - cls.scorer.slices[first] + 1
            cls.scorer.ratios[onset, length] = rng.integers(-20, 21, size)

    def test_schedule_spans_all_arrival_phases_and_is_causal(self):
        phases = onsets_by_phase(self.windows, self.onsets)
        self.assertEqual([sum(phase == p for phase, _ in phases.values()) for p in range(1, 8)], [9] * 7)
        self.assertEqual([self.windows[i].deadline for i in self.eligible], list(range(21, 78, 7)))
        for i, (fixed, later) in enumerate(zip(self.windows, self.later, strict=True)):
            self.assertEqual((fixed.start, fixed.commit, fixed.stop), (later.start, later.commit, later.stop))
            if i in self.eligible:
                self.assertEqual(later.deadline - fixed.deadline, EXTRA_WAIT)
                self.assertLess(later.deadline, self.windows[i + 1].deadline)
                for onset in self.eligible[i]:
                    self.assertLess(fixed.deadline, onset + ALERT_DELAY)
                    self.assertLessEqual(onset + ALERT_DELAY, later.deadline)
                    self.assertLessEqual(onset + BURST_DURATION, fixed.deadline)
            else:
                self.assertEqual(fixed, later)

    def test_trigger_uses_exactly_the_expected_slices(self):
        counts = np.zeros((1, len(self.onsets) + BURST_DURATION), dtype=int)
        first = self.scorer.first
        for column, candidates in enumerate(self.eligible.values()):
            start, stop = min(candidates) - first, max(candidates) - first + BURST_DURATION + 1
            counts.fill(900)
            counts[0, start:stop] = [1, 2, 3, 4, 5, 6]
            self.assertEqual(trigger_scores(count_prefix(counts), first, self.eligible)[0, column], 18)
        self.assertEqual(calibrate_threshold(np.array([910, 78, 8, 4])), (3, 4))

    def test_graph_choices_match_original_scoring_for_both_deadlines(self):
        rng = np.random.default_rng(50)
        counts = self.scorer.packed_counts(self.sampler.sample(180, bit_packed=True))
        full, pending = self.evidence.scores(count_prefix(counts))
        for flags in (None, rng.random((len(counts), len(self.onsets))) < .04):
            received = np.zeros_like(full, dtype=bool) if flags is None else flags
            fixed, later = self.evidence.choices(full, pending, flags)
            for windows, got in ((self.windows, fixed), (self.later, later)):
                scores = self.scorer.score_counts(counts, windows, BURST_PRIOR)
                expected = choices(scores, self.onsets, windows, flags=received, delay=ALERT_DELAY)
                np.testing.assert_array_equal(got, expected)

    def test_paired_shortcut_matches_full_decoding(self):
        # Synthetic, near-threshold scores create graph disagreements even on an
        # ordinary trial. Decode with real matching graphs for each hypothesis.
        onsets = self.onsets[7:10]
        windows = self.windows[:4]
        later = self.later[:4]
        eligible = {2: onsets}
        scorer = SyndromeScores(self.times, onsets, BURST_DURATION)
        edges = {0: graph_edges(self.circuit, self.times)}
        for onset in onsets:
            for length in range(1, BURST_DURATION + 2):
                scorer.ratios[onset, length] = np.full(length * 48 + 1, 6.0)
            edges[onset] = graph_edges(memory_circuit(7, 84, "x", .003, .02, onset, 3), self.times)
        sim = Simulation({}, edges, self.times, scorer, windows, later, eligible, 32, 400)
        packed, actual = self.sampler.sample(400, separate_observables=True, bit_packed=True)
        actual = actual[:, 0] & 1
        seed = 21
        full, full_hist, full_commits = sim.batch(packed, actual, 0, np.random.default_rng(seed), True)
        fast, fast_hist, fast_commits = sim.batch(packed, actual, 0, np.random.default_rng(seed), False)
        self.assertGreater(np.count_nonzero(full["gain_always"]), 0)
        for name, values in fast.items():
            np.testing.assert_array_equal(values, full[name])
        for name, histogram in fast_hist.items():
            np.testing.assert_array_equal(histogram, full_hist[name])
        for name, counts in fast_commits.items():
            np.testing.assert_array_equal(counts, full_commits[name])

    def test_conditional_false_alert_masks(self):
        probability = .2
        mass, flags = conditional_false_flags(np.random.default_rng(14), 60_000, 3, probability)
        self.assertAlmostEqual(mass, 1 - (1 - probability)**3)
        self.assertTrue(flags.any(axis=1).all())
        np.testing.assert_allclose(mass * flags.mean(axis=0), probability, atol=.003)
        self.assertAlmostEqual(mass * flags.all(axis=1).mean(), probability**3, delta=.002)


if __name__ == "__main__":
    unittest.main()
