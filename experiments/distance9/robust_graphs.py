"""Build local matching graphs and calibrate syndrome onset scores."""

from dataclasses import replace

import numpy as np

from burst_alert import SyndromeScores
from robust_design import BACKGROUND, BURST, BURST_LENGTH, DISTANCE, ROUNDS, hot_patches
from rolling_policy import count_prefix
from surface_code import memory_circuit
from window_decoder import detector_rounds, graph_edges


REFERENCE_ONSET = 27


def make_circuit(onset, patch=None):
    return memory_circuit(DISTANCE, ROUNDS, "x", BACKGROUND, BURST, onset, BURST_LENGTH, patch)


class SyndromeCalibration:
    def __init__(self, onsets, times):
        self.scorer = SyndromeScores(times, onsets, BURST_LENGTH)

    def histograms(self, circuit, seed, shots, onsets, batch, result=None):
        s = self.scorer
        sizes = {
            (o, length): s.slices[o - s.first + length] - s.slices[o - s.first] + 1
            for o in onsets for length in range(1, BURST_LENGTH + 2)
        }
        if result is None:
            result = {key: np.zeros(size, dtype=np.int64) for key, size in sizes.items()}
        sampler = circuit.compile_detector_sampler(seed=seed)
        for start in range(0, shots, batch):
            packed = sampler.sample(min(batch, shots - start), bit_packed=True)
            prefix = count_prefix(s.packed_counts(packed))
            for (o, length), hist in result.items():
                col = o - s.first
                hist += np.bincount(prefix[:, col + length] - prefix[:, col], minlength=len(hist))
        return result


def edge_key(edge):
    if edge.second is None:
        return edge.first, -1, edge.logical
    return min(edge.first, edge.second), max(edge.first, edge.second), edge.logical


def edge_table(circuit, times):
    return {edge_key(edge): edge.weight for edge in graph_edges(circuit, times)}


def translated_weights(background, affected, times, sample_times, onset):
    """Translate interior circuit-level weight changes between stationary rounds."""
    changed = {key: weight for key, weight in affected.items() if background[key] != weight}
    delta = (onset - REFERENCE_ONSET) * sample_times
    if not all(times[a] >= REFERENCE_ONSET - 1 and times[a] <= REFERENCE_ONSET + BURST_LENGTH
               for a, _, _ in changed):
        raise ValueError("the reference burst affects unexpected rounds")
    return {(a + delta, b + delta if b != -1 else -1, logical): weight
            for (a, b, logical), weight in changed.items()}


def build_edges(windows, longer, onsets):
    background = make_circuit(0)
    times = detector_rounds(background)
    coords = background.get_detector_coordinates()
    if len(background.shortest_graphlike_error()) != DISTANCE:
        raise ValueError("incorrect surface-code distance")
    edges = graph_edges(background, times)
    table = {edge_key(e): e.weight for e in edges}
    ref = [make_circuit(REFERENCE_ONSET, patch) for patch in (None, *hot_patches())]
    affected = [edge_table(circuit, times) for circuit in ref]
    if any(item.keys() != table.keys() for item in affected):
        raise ValueError("a burst changed graph topology")
    steps = np.bincount(times)
    if not np.all(steps[2:ROUNDS] == steps[2]):
        raise ValueError("nonstationary interior detector count")
    if any(circuit.get_detector_coordinates() != coords for circuit in ref):
        raise ValueError("the noise model changed the detector layout")
    global_edges = {o: translated_weights(table, affected[0], times, int(steps[2]), o) for o in onsets}
    local_template = {}
    affected_keys = set().union(*(set(k for k, w in d.items() if table[k] != w) for d in affected[1:]))
    for key in affected_keys:
        # The alert reports the onset only. The decoder averages the edge error
        # probabilities over the four possible patch locations.
        probabilities = [1 / (1 + np.exp(d[key])) for d in affected[1:]]
        p = float(np.mean(probabilities))
        local_template[key] = float(np.log((1 - p) / p))
    local_edges = {o: translated_weights(table, local_template, times, int(steps[2]), o) for o in onsets}
    if any(not weights.keys() <= table.keys() for weights in (*global_edges.values(), *local_edges.values())):
        raise ValueError("translated burst changed the edge topology")
    graph_sets = [global_edges, local_edges]
    ordered = sorted(enumerate(edges), key=lambda item: edge_key(item[1])[0])
    first_nodes = np.asarray([edge_key(edge)[0] for _, edge in ordered])
    contexts = [{}, {}]
    for index, window in enumerate(windows):
        for extended, w in ((False, window), (True, longer[index])):
            if extended and w == window:
                for context in contexts:
                    context[index, True] = context[index, False]
                continue
            first, end = np.searchsorted(times, (w.start, w.stop + 1))
            left, right = np.searchsorted(first_nodes, (first, end))
            entries = tuple((edge_key(e), e) for _, e in sorted(ordered[left:right]))
            base = tuple(e for _, e in entries)
            for context, weights_by_onset in zip(contexts, graph_sets, strict=True):
                graphs = {0: (0, base)}
                for onset, weights in weights_by_onset.items():
                    if onset > w.stop + 1 or onset + BURST_LENGTH < w.start:
                        graphs[onset] = (0, base)
                        continue
                    changed = tuple(replace(e, weight=weights[k]) if k in weights else e for k, e in entries)
                    graphs[onset] = (0, base) if changed == base else (onset, changed)
                context[index, extended] = graphs
    return background, times, contexts
