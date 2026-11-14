"""Forward matching for several policies sharing the same test shots."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from window_decoder import WindowGeometry


@dataclass(frozen=True)
class Decision:
    onset: np.ndarray
    extend: np.ndarray


class CommitDecoder:
    """Apply corrections with short or long syndrome windows at each commit."""

    def __init__(self, times, windows, longer, edges):
        self.geometry = [WindowGeometry(w, times) for w in windows]
        self.extended = [WindowGeometry(w, times) for w in longer]
        self.edges = edges
        self.matchers = OrderedDict()
        self.canonical = {}
        for key, graph in edges.items():
            ids = np.zeros(max(graph) + 1, dtype=int)
            for onset, (canonical, _) in graph.items():
                ids[onset] = canonical
            self.canonical[key] = ids

    def matcher(self, index, extended, onset, geometry):
        canonical, window_edges = self.edges[index, extended][int(onset)]
        key = (index, extended, canonical)
        if key not in self.matchers:
            if len(self.matchers) >= 48:
                self.matchers.popitem(last=False)
            self.matchers[key] = geometry.matcher(window_edges)
        self.matchers.move_to_end(key)
        return self.matchers[key]

    def step(self, index, syndromes, carry, decision):
        base = self.geometry[index]
        outputs = np.empty((len(syndromes), 1 + base.carry_end - base.carry_first), dtype=np.uint8)
        # Using the two geometries keeps the time cut and carry convention
        # identical to the rolling decoder.
        for extended, geometry in ((False, base), (True, self.extended[index])):
            selected = decision.extend == extended
            if not selected.any():
                continue
            ids = self.canonical[index, extended][decision.onset]
            for onset in np.unique(ids[selected]):
                chosen = selected & (ids == onset)
                local = syndromes[chosen, geometry.first:geometry.end].astype(np.uint8, copy=False)
                local[:, :carry.shape[1]] ^= carry[chosen]
                outputs[chosen] = self.matcher(index, extended, onset, geometry).decode_batch(local)
        return outputs

    def alternatives(self, syndromes, reference, policies):
        """Reuse each window's output when its inputs are identical."""
        n = len(syndromes)
        parity = np.zeros(n, dtype=np.uint8)
        carry = np.zeros((n, 0), dtype=np.uint8)
        alternate = {name: (parity.copy(), carry.copy()) for name in policies}
        for index, decision in enumerate(reference):
            outputs = self.step(index, syndromes, carry, decision)
            seen = [(decision, carry, outputs)]
            parity ^= outputs[:, 0]
            for name, decisions in policies.items():
                alt_parity, alt_carry = alternate[name]
                current = decisions[index]
                unresolved = np.ones(n, dtype=bool)
                alt_outputs = np.empty_like(outputs)
                for other, other_carry, previous in seen:
                    equal = (unresolved & (current.onset == other.onset) & (current.extend == other.extend)
                             & np.all(alt_carry == other_carry, axis=1))
                    alt_outputs[equal] = previous[equal]
                    unresolved[equal] = False
                    if not unresolved.any():
                        break
                if unresolved.any():
                    alt_outputs[unresolved] = self.step(
                        index, syndromes[unresolved], alt_carry[unresolved],
                        Decision(current.onset[unresolved], current.extend[unresolved]),
                    )
                alt_parity ^= alt_outputs[:, 0]
                alternate[name] = (alt_parity, alt_outputs[:, 1:])
                seen.append((current, alt_carry, alt_outputs))
            carry = outputs[:, 1:]
        return parity, {name: state[0] for name, state in alternate.items()}

