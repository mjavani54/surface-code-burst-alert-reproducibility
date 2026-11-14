"""Graph schedules for alert waiting and matched syndrome lookahead."""

from collections import OrderedDict
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from rolling_policy import (
    ALERT_DELAY, BURST_PRIOR, EXTRA_WAIT, FALSE_ALARM, ROUNDS, SENSITIVITY,
    RollingEvidence, best_candidate, study_schedule,
)
from window_decoder import WindowGeometry
from write_results import read_csv


FROZEN_DESIGN = Path(__file__).with_name("priority1_design.json")


def frozen_design():
    meta = json.loads(FROZEN_DESIGN.read_text(encoding="utf-8"))
    windows, _, onsets, eligible = study_schedule()
    if (meta["onsets"] != list(onsets) or meta["windows"] != [vars(w) for w in windows]
            or meta["eligible"] != {str(i): list(values) for i, values in eligible.items()}):
        raise ValueError("the saved Priority 1 schedule differs from the rolling model")
    return meta


def random_probabilities(eligible, onsets):
    """Use Priority 1 wait rates at each commit under the same trial mixture."""
    totals = {(onset, index): [0, 0] for onset in (0, *onsets) for index in eligible}
    for row in read_csv(FROZEN_DESIGN.with_name("priority1_commit_waits.csv")):
        if row["policy"] == "budgeted":
            total = totals[int(row["onset"]), int(row["window_index"])]
            total[0] += int(row["waits"])
            total[1] += int(row["trials"])

    def rate(onset, index):
        waits, trials = totals[onset, index]
        return waits / trials

    return np.array([
        (1 - BURST_PRIOR) * rate(0, i) + BURST_PRIOR / len(onsets) * sum(rate(o, i) for o in onsets)
        for i in eligible
    ])


def lookahead_windows(windows, eligible):
    """Spend the three extra rounds on syndrome data at the same commits."""
    result = list(windows)
    for index in eligible:
        window = windows[index]
        stop = window.stop + EXTRA_WAIT
        if stop > ROUNDS or stop >= windows[index + 1].deadline:
            raise ValueError("extended syndrome window overlaps the next deadline")
        result[index] = replace(window, stop=stop, deadline=window.deadline + EXTRA_WAIT)
    return tuple(result)


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
            for onset in np.unique(decision.onset[selected]):
                chosen = selected & (decision.onset == onset)
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


class PolicyEvidence:
    def __init__(self, scorer, windows, later, longer, eligible):
        self.scorer = scorer
        self.eligible = eligible
        self.alert = RollingEvidence(scorer, windows, later)
        self.windows = windows
        self.later = later
        self.longer = longer
        self.candidates = np.r_[0, self.alert.onsets]

    def nominal_syndrome(self, full, pending):
        """Choose the best visible onset without applying any alert evidence."""
        score = np.zeros(len(full))
        pick = np.zeros(len(full), dtype=int)
        start = 0
        choices = []
        for plan, recent in zip(self.alert.plans, pending, strict=True):
            stop = plan.complete
            score, pick = best_candidate(score, pick, full[:, start:stop], self.alert.first + start)
            _, choice = best_candidate(score, pick, recent[:, stop - plan.arrived:], self.alert.first + stop)
            choices.append(choice)
            start = stop
        return choices

    def extended_scores(self, prefix, full, waits):
        """Read the additional syndrome only on trials whose trigger requested it."""
        tables = {}
        for column, index in enumerate(self.eligible):
            selected = waits[:, column]
            if not selected.any():
                continue
            stop = self.longer[index].stop
            complete, visible = np.searchsorted(
                self.alert.onsets, (stop - self.scorer.duration, stop), side="right"
            )
            tail = [
                self.scorer.ratios[self.alert.onsets[i], visible - i][
                    prefix[selected, visible] - prefix[selected, i]
                ] + self.alert.prior_odds
                for i in range(complete, visible)
            ]
            tables[index] = np.column_stack((np.zeros(selected.sum()), full[selected, :complete], *tail))
        return tables

    def after_wait(self, table, flags, deadline):
        """Apply the alert evidence observed by an explicitly chosen deadline."""
        visible = np.searchsorted(self.alert.onsets + ALERT_DELAY, deadline, side="right")
        no = np.log((1 - SENSITIVITY) / (1 - FALSE_ALARM))
        yes = np.log(SENSITIVITY / FALSE_ALARM)
        evidence = no if flags is None else np.where(flags[:, :visible], yes, no)
        updated = table.copy()
        updated[:, 1:visible + 1] += evidence
        return self.candidates[updated.argmax(axis=1)]

    def decisions(self, full, pending, syndrome, tables, trigger, random, flags):
        fixed, later = self.alert.choices(full, pending, flags)
        never = np.zeros(len(full), dtype=bool)
        policies = {
            "fixed_alert": [Decision(onset, never) for onset in fixed],
            "always_alert": [Decision(onset, never) for onset in later],
            "random_alert": [Decision(onset, never) for onset in fixed],
            "selective_alert": [Decision(onset, never) for onset in fixed],
            "fixed_syndrome": [Decision(onset, never) for onset in syndrome],
            "selective_syndrome": [Decision(onset, never) for onset in syndrome],
            "selective_lookahead": [Decision(onset, never) for onset in fixed],
            "selective_both": [Decision(onset, never) for onset in fixed],
        }
        for column, index in enumerate(self.eligible):
            wait = trigger[:, column]
            policies["random_alert"][index] = Decision(np.where(random[:, column], later[index], fixed[index]), never)
            policies["selective_alert"][index] = Decision(np.where(wait, later[index], fixed[index]), never)
            if index not in tables:
                continue
            table = tables[index]
            received = None if flags is None else flags[wait]
            for name, base, extended in (
                ("selective_syndrome", syndrome[index], self.candidates[table.argmax(axis=1)]),
                ("selective_lookahead", fixed[index], self.after_wait(table, received, self.windows[index].deadline)),
                ("selective_both", fixed[index], self.after_wait(table, received, self.later[index].deadline)),
            ):
                picks = base.copy()
                picks[wait] = extended
                policies[name][index] = Decision(picks, wait)
        return policies
