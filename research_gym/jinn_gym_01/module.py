"""Executable external persona modules (exact rubric or learned head) and their decisions.

A PersonaModule turns a validated specification into a research action policy:

  score(a)  = P(train | audited tags of a) from the exact rubric (score_policy score) or
              from a trained head (softmax train probability), minus a lattice hold penalty
              when the action's phase is not reachable in one step from the task phase.
  keep(a)   = top lane is not reject (character_ldt_heads_v1 keep rule); abstain always kept.
  abstain   when the persona's evidence or abstention rule fires.
  choice    = argmax over host-permitted kept actions of  log prior(a) + STRENGTH * log score(a)
              (the Dual-LDT-01 sort_with_jinn projection; prior = Inky's distribution).

The module receives the host-permitted set as data and never computes it; host.execute is
still applied to its output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import host
from ._vendor.ldt_experiment import LANES, parse_policy, score_policy, vectorize
from .schema import (REJECT_MAXIMUM, TRAIN_MINIMUM, parse_policy_module, policy_round_trips,
                     spec_hash, write_modules)
from .tags import ABSTAIN, ALL_TAGS, METHOD_TAGS
from .tasks import Task

STRENGTH = 2.0
HOLD_PENALTY = 1.0  # log-score penalty for an action outside the persona's next lattice phases
FLOOR = 1e-4
PRIOR_EPS = 0.05


def host_flags(task: Task) -> dict:
    """Public evidence flags the host derives from the task's evidence text."""
    ev = " ".join(task.evidence).lower()
    return {"inputs_missing": "has not arrived" in ev, "failure_reported": ("fail" in ev or "misformalized" in ev),
            "evidence_items": len(task.evidence)}


def action_phase(tags) -> set:
    return {METHOD_TAGS[t] for t in tags if t in METHOD_TAGS}


# The exact rubric's persona score is its exact lattice lane, on the same scale as the
# learned head's P(train): the head is trained to predict these lanes. score_policy's raw
# score in [0, 1] is compressed (most actions land in 0.5-0.85), so using it directly would
# hand Inky's prior the decision and make arm C a scale artifact rather than a rubric.
EXACT_LANE_TRAIN_PROB = {"train": 0.9, "hold": 0.1, "reject": 0.01}


class RubricScorer:
    kind = "exact_rubric"

    def __init__(self, rubric_path: Path):
        self.policy = parse_policy(rubric_path)

    def raw(self, tags) -> dict:
        return score_policy(sorted(set(tags)), self.policy, allowed_universe=frozenset(ALL_TAGS),
                            train_minimum=TRAIN_MINIMUM, reject_maximum=REJECT_MAXIMUM)

    def __call__(self, tags) -> tuple[float, str]:
        lane = self.raw(tags)["lane"]
        return EXACT_LANE_TRAIN_PROB[lane], lane


class HeadScorer:
    kind = "learned_head"

    def __init__(self, head):
        self.head = head

    def __call__(self, tags) -> tuple[float, str]:
        p = self.head.probabilities(vectorize(sorted(set(tags)), self.head.feature_names))
        return float(p[0]), LANES[max(range(3), key=lambda i: p[i])]


@dataclass
class PersonaModule:
    persona: str
    spec: dict
    version: str
    scorer: Callable
    policy: dict
    files: dict = field(default_factory=dict)

    @property
    def spec_hash(self) -> str:
        return spec_hash(self.spec)

    @classmethod
    def compile(cls, spec: dict, version: str, directory: Path, scorer=None) -> "PersonaModule":
        rp, pp = write_modules(spec, version, directory)
        policy = parse_policy_module(pp.read_text(encoding="utf-8"))
        if not policy_round_trips(spec, policy):
            raise ValueError("policy module does not round-trip to the specification")
        rubric = RubricScorer(rp)
        return cls(spec["persona"], spec, version, scorer or rubric, policy,
                   {"rubric": str(rp), "policy": str(pp)})

    def with_scorer(self, scorer) -> "PersonaModule":
        return PersonaModule(self.persona, self.spec, self.version, scorer, self.policy, dict(self.files))

    def next_phases(self, phase: str) -> set:
        p = phase if phase in self.policy["states"] else self.policy["initial"]
        return {p} | {b for a, b in self.policy["transitions"] if a == p}

    def evaluate(self, task: Task) -> dict:
        """Persona-side scores for every action (no host input)."""
        nxt = self.next_phases(task.phase)
        flags = host_flags(task)
        out = {}
        for a in task.actions:
            s, lane = self.scorer(a.tags)
            phases = action_phase(a.tags)
            blocked = bool(phases) and not (phases & nxt)
            if "bounded_repair" in a.tags and self.policy["require_failure_log_for_repair"] and not flags["failure_reported"]:
                blocked = True
            u = math.log(max(s, FLOOR)) - (HOLD_PENALTY if blocked else 0.0)
            veto = lane == "reject" and a.action_id != ABSTAIN
            out[a.action_id] = {"score": s, "lane": lane, "u": u, "veto": veto, "lattice_blocked": blocked}
        return out

    def abstain_reason(self, task: Task, scores: Mapping, candidates: Sequence[str]) -> str | None:
        flags = host_flags(task)
        if self.policy["abstain_on_missing_inputs"] and flags["inputs_missing"]:
            return "missing_inputs"
        if flags["evidence_items"] < self.policy["min_evidence_items"]:
            return "insufficient_evidence_items"
        acts = [a for a in candidates if a != ABSTAIN]
        if not acts:
            return "no_kept_action"
        if max(scores[a]["score"] for a in acts) < self.policy["min_train_score"]:
            return "below_min_train_score"
        if self.policy["abstain_if_no_ranked_action"]:
            top = set(self.policy["ranking"][:3])
            if not any(task.action(a).tags & top for a in acts):
                return "no_ranked_action"
        return None


def prior_from_response(task: Task, parsed: dict | None) -> dict:
    """Inky's distribution over the menu, mixed with PRIOR_EPS uniform (strictly positive)."""
    ids = list(task.action_ids)
    n = len(ids)
    base = None
    if parsed:
        probs = parsed.get("probabilities")
        if isinstance(probs, dict):
            vals = {a: float(probs[a]) for a in ids if isinstance(probs.get(a), (int, float)) and probs[a] >= 0}
            if vals and sum(vals.values()) > 0:
                tot = sum(vals.values())
                base = {a: vals.get(a, 0.0) / tot for a in ids}
        if base is None and isinstance(parsed.get("ranking"), list):
            rk = [a for a in parsed["ranking"] if a in ids]
            if rk:
                w = {a: 2.0 ** -(rk.index(a)) if a in rk else 2.0 ** -len(rk) for a in ids}
                tot = sum(w.values())
                base = {a: w[a] / tot for a in ids}
        if base is None and parsed.get("choice") in ids:
            base = {a: (0.6 if a == parsed["choice"] else 0.4 / (n - 1)) for a in ids}
    if base is None:
        base = {a: 1.0 / n for a in ids}
    return {a: (1 - PRIOR_EPS) * base[a] + PRIOR_EPS / n for a in ids}


def decide(task: Task, prior: Mapping[str, float], modules: Sequence[PersonaModule],
           strength: float = STRENGTH) -> dict:
    """Decision of one module, or the product-of-experts composition of several."""
    snap = host.snapshot(task)
    host_ok = host.permitted(task, snap)
    evals = [m.evaluate(task) for m in modules]
    ids = task.action_ids
    u = {a: sum(e[a]["u"] for e in evals) for a in ids}
    veto = {a: any(e[a]["veto"] for e in evals) for a in ids}
    kept = [a for a in ids if not veto[a]]
    # Unconstrained persona proposal (ignores the host) for intervention accounting.
    logit = {a: math.log(prior[a]) + strength * u[a] for a in ids}
    proposal = max(kept, key=lambda a: (logit[a], -ids.index(a)))
    candidates = [a for a in kept if a in host_ok]
    if ABSTAIN not in candidates:
        candidates.append(ABSTAIN)
    reasons = [m.abstain_reason(task, e, candidates) for m, e in zip(modules, evals)]
    if len(modules) == 1:
        reason = reasons[0]
    else:  # composite abstains when every component's rule fires, or on missing inputs
        reason = ("missing_inputs" if "missing_inputs" in reasons else
                  ("all_components" if all(reasons) else None))
    if reason:
        choice = ABSTAIN
        probs = {a: (1.0 if a == ABSTAIN else 0.0) for a in candidates}
    else:
        peak = max(logit[a] for a in candidates)
        w = {a: math.exp(logit[a] - peak) for a in candidates}
        tot = sum(w.values())
        probs = {a: w[a] / tot for a in candidates}
        choice = max(candidates, key=lambda a: (probs[a], -ids.index(a)))
    gate = host.execute(task, choice)
    return {"choice": choice, "proposal_unconstrained": proposal,
            "proposal_unauthorized": proposal not in host_ok, "kept": kept, "abstain_reason": reason,
            "confidence": probs.get(gate["executed"], 0.0) if gate["executed"] else 0.0,
            "probabilities": probs, "executed": gate["executed"], "gate": gate,
            "persona_u": u, "scores": evals}
