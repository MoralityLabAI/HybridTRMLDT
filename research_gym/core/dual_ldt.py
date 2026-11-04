"""Dual-LDT-01: host-authorized Beast and learned Jinn preference sorter.

The Beast's symbolic evidence is an independent host input. Model logits never
grant permission. Human intervention changes the host snapshot and invalidates
old execution receipts. The pure functions below do not authenticate callers:
production integration must own evidence generation and execution outside the
untrusted proposer.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Callable, Mapping


@dataclass(frozen=True)
class ActionEvidence:
    """Host-reported authority and candidate semantic tag interpretations."""
    positive: bool
    negative: bool
    possible_tags: tuple[frozenset[str], ...]

    def __post_init__(self) -> None:
        if type(self.positive) is not bool or type(self.negative) is not bool:
            raise TypeError("authority evidence must be Boolean")
        tags = tuple(frozenset(x) for x in self.possible_tags)
        if any(not all(isinstance(t, str) for t in x) for x in tags):
            raise TypeError("tags must be strings")
        object.__setattr__(self, "possible_tags", tags)


@dataclass(frozen=True)
class ApprovalReceipt:
    action: str
    epoch: str
    snapshot_digest: str


@dataclass(frozen=True)
class BeastSnapshot:
    """Immutable host-owned policy/evidence snapshot for one decision context."""
    epoch: str
    actions: tuple[str, ...]
    forbidden_tags: frozenset[str]
    evidence: Mapping[str, ActionEvidence]
    halted: bool = False

    def __post_init__(self) -> None:
        if not self.epoch or not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("epoch and unique action universe required")
        if not all(isinstance(a, str) and a for a in self.actions):
            raise ValueError("invalid action")
        if type(self.halted) is not bool:
            raise TypeError("halted must be Boolean")
        if set(self.evidence) != set(self.actions):
            raise ValueError("every action needs an explicit evidence entry")
        if any(not isinstance(e, ActionEvidence) for e in self.evidence.values()):
            raise TypeError("evidence must be ActionEvidence")
        object.__setattr__(self, "forbidden_tags", frozenset(self.forbidden_tags))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def status(self, action: str) -> str:
        if action not in self.evidence:
            raise ValueError("unknown action")
        if self.halted:
            return "halted"
        e = self.evidence[action]
        if e.positive and e.negative:
            return "hold_conflicting_authority"
        if e.negative:
            return "reject_denied"
        if not e.positive:
            return "hold_missing_authority"
        if not e.possible_tags:
            return "hold_unknown_semantics"
        hit = [bool(tags & self.forbidden_tags) for tags in e.possible_tags]
        if all(hit):
            return "reject_forbidden"
        if any(hit):
            return "hold_ambiguous_semantics"
        return "authorized"

    def authorized(self) -> frozenset[str]:
        return frozenset(a for a in self.actions if self.status(a) == "authorized")

    def digest(self) -> str:
        payload = {
            "epoch": self.epoch, "actions": self.actions, "halted": self.halted,
            "forbidden_tags": sorted(self.forbidden_tags),
            "evidence": {
                a: {"positive": e.positive, "negative": e.negative,
                    "possible_tags": sorted([sorted(tags) for tags in e.possible_tags])}
                for a, e in sorted(self.evidence.items())
            },
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                      .encode("utf-8")).hexdigest()

    def receipt(self, action: str) -> ApprovalReceipt | None:
        if self.status(action) != "authorized":
            return None
        return ApprovalReceipt(action, self.epoch, self.digest())

    def verify_execution(
        self,
        receipt: ApprovalReceipt | None,
        host_checker: Callable[[str, str], bool],
    ) -> bool:
        """Recheck current epoch, evidence, and an independently owned host gate."""
        return bool(receipt is not None and receipt.epoch == self.epoch and
                    receipt.snapshot_digest == self.digest() and
                    receipt.action in self.authorized() and
                    host_checker(self.epoch, receipt.action) is True)


@dataclass(frozen=True)
class JinnHead:
    """Fixed real additive three-lane head; its weights are preference evidence."""
    feature_names: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    @classmethod
    def from_record(cls, record: Mapping) -> JinnHead:
        if tuple(record["classes"]) != ("train", "hold", "reject"):
            raise ValueError("unknown lane order")
        names = tuple(record["feature_names"])
        weights = tuple(tuple(float(v) for v in row) for row in record["weights"])
        bias = tuple(float(v) for v in record["bias"])
        if (len(set(names)) != len(names) or not names or len(weights) != 3 or
                len(bias) != 3 or any(len(row) != len(names) for row in weights) or
                not all(isfinite(v) for row in (*weights, bias) for v in row)):
            raise ValueError("malformed model")
        return cls(names, weights, bias)

    def logits(self, tags: frozenset[str]) -> tuple[float, float, float]:
        active = frozenset(tags)
        return tuple(
            self.bias[k] + sum(w for feature, w in zip(self.feature_names, self.weights[k])
                               if feature in active)
            for k in range(3)
        )

    def lane(self, tags: frozenset[str]) -> str:
        values = self.logits(tags)
        return ("train", "hold", "reject")[max(range(3), key=lambda i: values[i])]

    def train_log_probability(self, tags: frozenset[str]) -> float:
        values = self.logits(tags)
        top = max(values)
        return values[0] - top - log(sum(exp(v - top) for v in values))


@dataclass(frozen=True)
class JinnDecision:
    action: str | None
    probabilities: Mapping[str, float]
    original_probabilities: Mapping[str, float]
    permitted: frozenset[str]
    preference_tax: float | None
    epoch: str
    snapshot_digest: str


def sort_with_jinn(
    snapshot: BeastSnapshot, head: JinnHead,
    *,
    strength: float = 1.,
    baseline: Mapping[str, float] | None = None,
) -> JinnDecision:
    """Forward-KL projection of reference preference onto the Beast's safe set.

    The persona score is log P_train from the saved three-lane head.
    Multiple independently possible safe tag sets use the minimum score.
    This is an explicit conservative preference convention, not authorization.
    """
    if not isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    prior = {a: 1.0 for a in snapshot.actions} if baseline is None else dict(baseline)
    if set(prior) != set(snapshot.actions):
        raise ValueError("baseline must cover exactly the action universe")
    if any(not isfinite(p) or p <= 0 for p in prior.values()):
        raise ValueError("baseline must be positive and finite")
    logw = {}
    for action in snapshot.actions:
        interpretations = snapshot.evidence[action].possible_tags
        # Unknown semantics carry the least informative model score; this never
        # affects authorization and should not be interpreted as a moral rule.
        score = min((head.train_log_probability(t) for t in interpretations),
                    default=head.train_log_probability(frozenset()))
        logw[action] = log(prior[action]) + strength * score
    peak = max(logw.values())
    weights = {a: exp(v - peak) for a, v in logw.items()}
    denom = sum(weights.values())
    original = {a: weights[a] / denom for a in snapshot.actions}
    allowed = snapshot.authorized()
    safe_mass = sum(original[a] for a in allowed)
    if not allowed:
        return JinnDecision(None, {}, original, allowed, None,
                            snapshot.epoch, snapshot.digest())
    probabilities = {a: original[a] / safe_mass for a in snapshot.actions if a in allowed}
    winner = max(allowed, key=lambda a: (probabilities[a], -snapshot.actions.index(a)))
    return JinnDecision(winner, probabilities, original, allowed,
                        -log(safe_mass), snapshot.epoch, snapshot.digest())


def total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return sum(abs(left.get(k, 0.) - right.get(k, 0.))
               for k in set(left) | set(right)) / 2.
