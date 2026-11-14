"""Forward sliding-window MWPM with a logical parity and boundary carry."""

from dataclasses import dataclass

import numpy as np
import pymatching
import stim


@dataclass(frozen=True)
class Window:
    start: int
    commit: int
    stop: int
    deadline: int


def schedule(rounds: int, step: int, lookahead: int) -> tuple[Window, ...]:
    """Detector rounds are 1 through rounds + 1, including data readout."""
    if rounds < 1 or step < 1 or lookahead < 1:
        raise ValueError("rounds, commit width, and lookahead must be positive")
    windows = []
    start = 1
    while start + step + lookahead - 1 < rounds + 1:
        commit = start + step - 1
        stop = commit + lookahead
        windows.append(Window(start, commit, stop, stop))
        start = commit + 1
    windows.append(Window(start, rounds + 1, rounds + 1, rounds))
    return tuple(windows)


@dataclass(frozen=True)
class Edge:
    first: int
    second: int | None
    weight: float
    logical: bool


def graph_edges(circuit: stim.Circuit, detector_rounds: np.ndarray) -> list[Edge]:
    graph = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    edges = []
    for first, second, data in graph.edges():
        if second is not None:
            span = abs(detector_rounds[first] - detector_rounds[second])
            if span > 1:
                raise ValueError("the forward decoder requires edges spanning at most one round")
        edges.append(Edge(first, second, data["weight"], 0 in data["fault_ids"]))
    return edges


def detector_rounds(circuit: stim.Circuit) -> np.ndarray:
    coords = circuit.get_detector_coordinates()
    times = np.array([coords[index][2] for index in range(circuit.num_detectors)])
    if np.any(times != np.floor(times)) or np.any(np.diff(times) < 0):
        raise ValueError("expected nondecreasing integer detector times")
    return times.astype(int) + 1


class WindowGeometry:
    def __init__(self, window: Window, times: np.ndarray):
        self.window = window
        self.first, self.end = np.searchsorted(times, [window.start, window.stop + 1])
        self.carry_first, self.carry_end = np.searchsorted(times, [window.commit + 1, window.commit + 2])
        self.times = times

    def matcher(self, edges: list[Edge]) -> pymatching.Matching:
        """Encode only the committed logical and boundary effects as fault IDs."""
        graph = pymatching.Matching()
        start, stop, commit = self.window.start, self.window.stop, self.window.commit
        num_nodes = self.end - self.first
        for edge in edges:
            a, b = edge.first, edge.second
            ta = self.times[a]
            if b is None:
                if start <= ta <= stop:
                    logical_id = {0} if edge.logical and ta <= commit else set()
                    graph.add_boundary_edge(a - self.first, fault_ids=logical_id, weight=edge.weight)
                continue
            tb = self.times[b]
            if min(ta, tb) < start or min(ta, tb) > stop:
                continue
            if max(ta, tb) > stop:
                # Each crossing edge has a distinct future-boundary node.
                # Keep it separate from the real spatial boundary.
                inside = a if ta <= stop else b
                graph.add_edge(inside - self.first, num_nodes, weight=edge.weight)
                num_nodes += 1
                continue
            ids = {0} if edge.logical and min(ta, tb) <= commit else set()
            if min(ta, tb) <= commit < max(ta, tb):
                deferred = a if ta > commit else b
                ids.add(1 + deferred - self.carry_first)
            graph.add_edge(a - self.first, b - self.first, fault_ids=ids, weight=edge.weight)
        graph.set_boundary_nodes(set(range(self.end - self.first, num_nodes)))
        graph.ensure_num_fault_ids(1 + self.carry_end - self.carry_first)
        if graph.num_detectors != self.end - self.first:
            raise RuntimeError("window detector count does not match its syndrome slice")
        return graph


class ForwardDecoder:
    """Use the candidate graph selected at each window's decision deadline."""

    def __init__(self, times: np.ndarray, windows: tuple[Window, ...], edges_by_onset: dict[int, list[Edge]]):
        self.geometry = [WindowGeometry(window, times) for window in windows]
        self.edges_by_onset = edges_by_onset
        self.matchers: dict[tuple[int, int], pymatching.Matching] = {}

    def decode(self, syndromes: np.ndarray, choices: list[np.ndarray], *, trace=False):
        logical = np.zeros(len(syndromes), dtype=np.uint8)
        history = []
        carry = None
        for index, (geometry, candidates) in enumerate(zip(self.geometry, choices, strict=True)):
            local = syndromes[:, geometry.first:geometry.end].copy()
            if carry is not None:
                local[:, :carry.shape[1]] ^= carry
            outputs = np.empty(
                (len(syndromes), 1 + geometry.carry_end - geometry.carry_first), dtype=np.uint8
            )
            for candidate in np.unique(candidates):
                key = (index, int(candidate))
                if key not in self.matchers:
                    self.matchers[key] = geometry.matcher(self.edges_by_onset[int(candidate)])
                select = candidates == candidate
                outputs[select] = self.matchers[key].decode_batch(local[select])
            logical ^= outputs[:, 0]
            carry = outputs[:, 1:]
            if trace:
                history.append((logical.copy(), carry.copy()))
        return (logical, history) if trace else logical
