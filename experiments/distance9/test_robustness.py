"""Check causal reporting, matched waits, and local circuit construction."""

import unittest

import numpy as np

from burst_alert import SyndromeScores, choices
from robust_design import FIXED_DELAY, JITTER, arrival_weights, hot_patches, study_schedule, trigger_candidates
from robust_graphs import REFERENCE_ONSET, edge_table, make_circuit, translated_weights
from robust_policy import Scores, alert_log_likelihood, decisions
from rolling_policy import count_prefix
from window_decoder import detector_rounds


class RobustnessChecks(unittest.TestCase):
    def test_local_noise_preserves_ideal_measurements_and_detector_layout(self):
        base = make_circuit(0)
        patches = hot_patches()
        self.assertEqual([len(patch) for patch in patches], [25] * 4)
        self.assertEqual(len(set.union(*(set(p) for p in patches))), 100)
        local = make_circuit(27, patches[0])
        self.assertEqual(local.without_noise(), base.without_noise())
        self.assertEqual(local.get_detector_coordinates(), base.get_detector_coordinates())

    def test_delay_evidence_changes_only_after_an_observed_arrival_or_new_silence(self):
        onsets = np.array([10])
        late = np.array([[25], [127]])
        evidence = alert_log_likelihood(late, onsets, 21, JITTER)
        np.testing.assert_array_equal(evidence, np.zeros((2, 1)))
        evidence = alert_log_likelihood(late, onsets, 24, JITTER)
        partial_silence = np.log((1 - .9 * .6) / (1 - .0001 * .6))
        np.testing.assert_allclose(evidence, np.full((2, 1), partial_silence))
        evidence = alert_log_likelihood(late, onsets, 25, JITTER)
        self.assertAlmostEqual(evidence[0, 0], np.log(.9 / .0001))
        self.assertAlmostEqual(evidence[1, 0], np.log((1 - .9 * .8) / (1 - .0001 * .8)))

    def test_all_selective_policies_use_the_same_wait_decisions(self):
        windows, longer, onsets, eligible = study_schedule()
        times = detector_rounds(make_circuit(0))
        scorer = SyndromeScores(times, onsets, 3)
        rng = np.random.default_rng(13)
        for o in onsets:
            for length in range(1, 5):
                offset = o - scorer.first
                width = scorer.slices[offset + length] - scorer.slices[offset]
                scorer.ratios[o, length] = rng.integers(-14, 15, width + 1)
        counts = rng.integers(0, 25, (30, len(onsets) + 3))
        scores = Scores(scorer, count_prefix(counts))
        nominal = [scores.at(w.stop) for w in windows]
        flags = rng.random((len(counts), len(onsets))) < .04
        arrivals = np.where(flags, np.asarray(onsets) + 14, 127)
        wait = rng.random((len(counts), len(eligible))) < .45
        extended = {i: scores.at(longer[i].stop, wait[:, col]) for col, i in enumerate(eligible)}
        plans = decisions(scores, windows, longer, eligible, wait, arrivals, FIXED_DELAY, nominal, extended)
        baseline = choices(scorer.score_counts(counts, windows, .001), onsets, windows, flags=flags, delay=14)
        for i, decision in enumerate(plans["fixed"]):
            np.testing.assert_array_equal(decision.onset, baseline[i])
        for col, i in enumerate(eligible):
            np.testing.assert_array_equal(plans["lookahead"][i].extend, wait[:, col])
            np.testing.assert_array_equal(plans["both"][i].extend, wait[:, col])
            candidate = trigger_candidates(onsets, windows[i].deadline, JITTER)
            self.assertTrue(all(o + 3 <= windows[i].deadline for o in candidate))

    def test_reference_edge_translation_matches_direct_circuit(self):
        base = make_circuit(0)
        times = detector_rounds(base)
        initial = edge_table(base, times)
        for patch in (None, hot_patches()[-1]):
            reference = edge_table(make_circuit(REFERENCE_ONSET, patch), times)
            for onset in (8, 70):
                weights = translated_weights(initial, reference, times, 80, onset)
                self.assertEqual(initial | weights, edge_table(make_circuit(onset, patch), times))

    def test_arrival_window_and_censoring_fractions(self):
        w, _, onsets, eligible = study_schedule()
        fixed = arrival_weights(w, onsets, eligible, FIXED_DELAY)
        jitter = arrival_weights(w, onsets, eligible, JITTER)
        self.assertEqual(sum(fixed.values()), 27)
        self.assertAlmostEqual(sum(jitter.values()), 26.4)
        self.assertEqual(sum(o + d > 84 for o in onsets for d in JITTER), 3)


if __name__ == "__main__":
    unittest.main()
