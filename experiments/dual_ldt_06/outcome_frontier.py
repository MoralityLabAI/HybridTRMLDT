"""Exact persona-independent compression after host-certified plan evaluation.

A coverage certificate preserves every nonnegative linear preference on the
same feasible set. It is conditional on accurate host outcome/cost values.
Computing this cover requires evaluation of its input plans; no search-time
speedup is asserted. Recompute after policy/feasibility changes.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable


@dataclass(frozen=True)
class Outcome:
    plan_id: int
    features: tuple[int, ...]
    cost: int

    def __post_init__(self):
        if type(self.plan_id) is not int or self.plan_id < 0:
            raise ValueError('nonnegative integer plan identifier required')
        if not isinstance(self.features, tuple) or not self.features:
            raise ValueError('nonempty immutable feature vector required')
        if any(type(x) is not int for x in self.features) or type(self.cost) is not int:
            raise TypeError('exact integer numerators required')
        if self.cost < 0:
            raise ValueError('nonnegative cost required')

    @property
    def profile(self):
        return self.features + (-self.cost,)

    def score(self, weights):
        w = tuple(weights)
        if len(w) != len(self.features):
            raise ValueError('preference dimension mismatch')
        if any(type(x) not in (int, Fraction) or x < 0 for x in w):
            raise ValueError('nonnegative exact rational preference weights required')
        return sum((x*y for x, y in zip(w, self.features)), Fraction(0)) - self.cost


def dominates(a: Outcome, b: Outcome) -> bool:
    if len(a.features) != len(b.features):
        raise ValueError('mixed outcome dimensions')
    return all(x >= y for x, y in zip(a.profile, b.profile))


@dataclass(frozen=True)
class Cover:
    representatives: tuple[Outcome, ...]
    witnesses: tuple[tuple[int, int], ...]


def _records(items: Iterable[Outcome]) -> tuple[Outcome, ...]:
    records = tuple(items)
    if not records or any(not isinstance(x, Outcome) for x in records):
        raise ValueError('nonempty validated outcome records required')
    if len({x.plan_id for x in records}) != len(records):
        raise ValueError('duplicate plan identifier')
    if len({len(x.features) for x in records}) != 1:
        raise ValueError('mixed outcome dimensions')
    return tuple(sorted(records, key=lambda x: x.plan_id))


def compile_cover(items: Iterable[Outcome]) -> Cover:
    records = _records(items)
    unique = {}
    for item in records:
        unique.setdefault(item.profile, item)
    choices = tuple(unique.values())
    frontier = tuple(x for x in choices if not any(
        x.plan_id != y.plan_id and dominates(y, x) for y in choices))
    frontier = tuple(sorted(frontier, key=lambda x: x.plan_id))
    witnesses = tuple((x.plan_id, next(y.plan_id for y in frontier if dominates(y, x)))
                      for x in records)
    result = Cover(frontier, witnesses)
    if not check_cover(records, result):
        raise AssertionError('internal dominance-certificate failure')
    return result


def check_cover(items: Iterable[Outcome], certificate: Cover) -> bool:
    """Check complete coverage; minimality is unnecessary for soundness."""
    try:
        records = _records(items)
        if not isinstance(certificate, Cover) or not certificate.representatives:
            return False
        by_id = {r.plan_id: r for r in records}
        reps = {r.plan_id: r for r in certificate.representatives}
        if len(reps) != len(certificate.representatives):
            return False
        if any(by_id.get(i) != r for i, r in reps.items()):
            return False
        witnesses = dict(certificate.witnesses)
        if len(witnesses) != len(certificate.witnesses) or set(witnesses) != set(by_id):
            return False
        return all(j in reps and dominates(reps[j], by_id[i])
                   for i, j in witnesses.items())
    except (ValueError, TypeError, AttributeError):
        return False
