"""Check causal evidence, matched budgets, and reuse of matching results."""

from itertools import product
import unittest

import numpy as np

from burst_alert import SyndromeScores, choices
from matched_policy import CommitDecoder, Decision, PolicyEvidence, frozen_design, lookahead_windows, random_probabilities
from matched_results import wait_values
from rolling_policy import ALERT_DELAY, BURST_DURATION, BURST_PRIOR, EXTRA_WAIT, count_prefix, study_schedule
from run_matched_wait import local_edges
from surface_code import memory_circuit
from window_decoder import ForwardDecoder, detector_rounds, graph_edges


class MatchedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows, cls.later, cls.onsets, cls.eligible = study_schedule()
        cls.longer = lookahead_windows(cls.windows, cls.eligible)
        cls.circuit = memory_circuit(7, 84, "x", .003, .02)
        cls.times = detector_rounds(cls.circuit)
        cls.scorer = SyndromeScores(cls.times, cls.onsets, BURST_DURATION)
        rng = np.random.default_rng(914)
        for onset, length in product(cls.onsets, range(1, BURST_DURATION + 2)):
            start = onset - cls.scorer.first
            size = cls.scorer.slices[start + length] - cls.scorer.slices[start] + 1
            cls.scorer.ratios[onset, length] = rng.integers(-20, 21, size)
        cls.evidence = PolicyEvidence(cls.scorer, cls.windows, cls.later, cls.longer, cls.eligible)

    def test_waits_share_deadlines_and_commit_boundaries(self):
        for index, (nominal, alert, long) in enumerate(zip(self.windows, self.later, self.longer, strict=True)):
            self.assertEqual((nominal.start, nominal.commit), (long.start, long.commit))
            self.assertEqual(nominal.stop, alert.stop)
            self.assertEqual(alert.deadline, long.deadline)
            if index in self.eligible:
                self.assertEqual(long.stop - nominal.stop, EXTRA_WAIT)
                self.assertLess(long.stop, self.windows[index + 1].deadline)
            else:
                self.assertEqual(nominal, long)

    def test_all_policy_choices_match_direct_scoring(self):
        rng = np.random.default_rng(140)
        counts = self.scorer.packed_counts(self.circuit.compile_detector_sampler(seed=44).sample(200, bit_packed=True))
        prefix = count_prefix(counts)
        full, pending = self.evidence.alert.scores(prefix)
        syndrome = self.evidence.nominal_syndrome(full, pending)
        trigger = rng.random((len(counts), len(self.eligible))) < .2
        random = rng.random(trigger.shape) < .01
        flags = rng.random((len(counts), len(self.onsets))) < .05
        tables = self.evidence.extended_scores(prefix, full, trigger)
        got = self.evidence.decisions(full, pending, syndrome, tables, trigger, random, flags)
        nominal = self.scorer.score_counts(counts, self.windows, BURST_PRIOR)
        long = self.scorer.score_counts(counts, self.longer, BURST_PRIOR)
        early = choices(nominal, self.onsets, self.windows, flags=flags, delay=ALERT_DELAY)
        late = choices(nominal, self.onsets, self.later, flags=flags, delay=ALERT_DELAY)
        short_syndrome = choices(nominal, self.onsets, self.windows)
        long_syndrome = choices(long, self.onsets, self.longer)
        early_long = choices(long, self.onsets, self.windows, flags=flags, delay=ALERT_DELAY)
        late_long = choices(long, self.onsets, self.longer, flags=flags, delay=ALERT_DELAY)
        by_window = dict(zip(self.eligible, range(len(self.eligible)), strict=True))
        for index in range(len(self.windows)):
            wait = trigger[:, by_window[index]] if index in by_window else np.zeros(len(counts), dtype=bool)
            draw = random[:, by_window[index]] if index in by_window else np.zeros(len(counts), dtype=bool)
            cases = {
                "fixed_alert": early[index], "always_alert": late[index],
                "random_alert": np.where(draw, late[index], early[index]),
                "selective_alert": np.where(wait, late[index], early[index]),
                "fixed_syndrome": short_syndrome[index],
                "selective_syndrome": np.where(wait, long_syndrome[index], short_syndrome[index]),
                "selective_lookahead": np.where(wait, early_long[index], early[index]),
                "selective_both": np.where(wait, late_long[index], early[index]),
            }
            for policy, picks in cases.items():
                decision = got[policy][index]
                np.testing.assert_array_equal(decision.onset, picks, err_msg=f"{policy} in window {index}")
                expected = wait if policy in ("selective_syndrome", "selective_lookahead", "selective_both") else False
                np.testing.assert_array_equal(decision.extend, expected)

    def test_window_reuse_matches_full_trajectory_decoding(self):
        rng = np.random.default_rng(17)
        source = memory_circuit(7, 84, "x", .003, .02, 17, 3)
        edges = {0: graph_edges(self.circuit, self.times), 17: graph_edges(source, self.times)}
        syndromes = source.compile_detector_sampler(seed=35).sample(32).astype(np.uint8)
        decoder = CommitDecoder(self.times, self.windows, self.longer,
                                local_edges(edges, self.windows, self.longer, self.times))
        baseline = [Decision(np.zeros(len(syndromes), dtype=int), np.zeros(len(syndromes), dtype=bool))
                    for _ in self.windows]
        picks = [rng.choice([0, 17], len(syndromes)) for _ in self.windows]
        wait = {i: rng.random(len(syndromes)) < .5 for i in tuple(self.eligible)[1:4]}
        masks = [wait.get(i, np.zeros(len(syndromes), dtype=bool)) for i in range(len(self.windows))]
        mixed = [Decision(onset, mask) for onset, mask in zip(picks, masks, strict=True)]
        reference, alt = decoder.alternatives(syndromes, baseline, {"mixed": mixed, "repeat": mixed})
        original = ForwardDecoder(self.times, self.windows, edges)
        np.testing.assert_array_equal(reference, original.decode(syndromes, [d.onset for d in baseline]))
        # For each fixed pattern, use the original decoder on exactly that schedule.
        wait_patterns = np.column_stack(masks)
        for pattern in np.unique(wait_patterns, axis=0):
            selected = np.all(wait_patterns == pattern, axis=1)
            windows = tuple(long if extend else short for long, short, extend in
                            zip(self.longer, self.windows, pattern, strict=True))
            original = ForwardDecoder(self.times, windows, edges)
            expected = original.decode(syndromes[selected], [d.onset[selected] for d in mixed])
            np.testing.assert_array_equal(alt["mixed"][selected], expected)
        np.testing.assert_array_equal(alt["repeat"], alt["mixed"])

    def test_frozen_random_wait_probabilities_and_latency(self):
        self.assertEqual(frozen_design()["threshold_budgeted"], 32)
        probabilities = random_probabilities(self.eligible, self.onsets)
        self.assertAlmostEqual(EXTRA_WAIT * probabilities.sum(), .027404083690476186)
        masks = np.array([[True, False, True], [False, True, False], [False, False, False]])
        stats, hist = wait_values(masks, masks, probabilities[:3])
        np.testing.assert_array_equal(stats["delay_trigger"], [6, 3, 0])
        np.testing.assert_array_equal(hist["random"], [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
