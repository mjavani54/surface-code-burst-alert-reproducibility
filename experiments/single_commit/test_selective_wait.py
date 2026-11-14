"""Test the cutoff, the calibrated trigger, and the paired decoder shortcut."""

import unittest

import numpy as np

from burst_alert import choices
from run_selective_wait import (
    ALERT_DELAY, ONSETS, deadlines, paired_decoding_gain, threshold_for_budget, trigger_score,
)
from surface_code import memory_circuit
from window_decoder import ForwardDecoder, detector_rounds, graph_edges, schedule


class SelectiveWaitChecks(unittest.TestCase):
    def test_only_the_target_deadline_changes(self):
        windows, index, waiting, relevant = deadlines()
        fixed = windows[index]
        self.assertEqual((fixed.start, fixed.commit, fixed.stop, fixed.deadline), (15, 21, 28, 28))
        self.assertEqual((waiting.start, waiting.commit, waiting.stop, waiting.deadline), (15, 21, 28, 31))
        self.assertLess(waiting.deadline, windows[index + 1].deadline)
        self.assertEqual(relevant, (15, 16, 17))
        scores = np.zeros((len(ONSETS), len(ONSETS) + 1))
        scores[:, 1:] = -1
        evidence = {"flags": np.eye(len(ONSETS), dtype=bool), "delay": ALERT_DELAY}
        np.testing.assert_array_equal(choices([scores], ONSETS, (fixed,), **evidence)[0], 0)
        np.testing.assert_array_equal(choices([scores], ONSETS, (waiting,), **evidence)[0],
                                      [15, 16, 17, 0, 0, 0, 0])

    def test_trigger_uses_only_counts_near_eligible_onsets(self):
        counts = np.array([[1, 2, 3, 4, 5, 6, 100, 100, 100], [3, 2, 1, 0, 0, 0, 99, 99, 99]])
        np.testing.assert_array_equal(trigger_score(counts, (15, 16, 17)), [18, 6])
        self.assertEqual(threshold_for_budget(np.array([910, 78, 8, 4]), .009), (3, 4))

    def test_paired_shortcut_matches_full_decoding(self):
        windows = schedule(12, 3, 3)
        circuits = {onset: memory_circuit(3, 12, "x", .003, .02, onset, 3) for onset in (0, 5)}
        times = detector_rounds(circuits[0])
        edges = {onset: graph_edges(circuit, times) for onset, circuit in circuits.items()}
        decoder = ForwardDecoder(times, windows, edges)
        packed, observable = circuits[5].compile_detector_sampler(seed=18).sample(
            100, separate_observables=True, bit_packed=True
        )
        syndromes = np.unpackbits(packed, axis=1, count=len(times), bitorder="little")
        actual = observable[:, 0] & 1
        early = [np.zeros(len(packed), dtype=int) for _ in windows]
        late = early.copy()
        late[1] = np.where(np.arange(len(packed)) % 2 == 0, 5, 0)
        difference = ((decoder.decode(syndromes, early) != actual).astype(np.int8)
                      - (decoder.decode(syndromes, late) != actual).astype(np.int8))
        np.testing.assert_array_equal(paired_decoding_gain(decoder, packed, actual, early, late[1], 1), difference)


if __name__ == "__main__":
    unittest.main()
