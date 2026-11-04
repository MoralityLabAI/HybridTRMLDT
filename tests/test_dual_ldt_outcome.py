from fractions import Fraction
from itertools import product
import random
import pytest
from experiments.dual_ldt_06.outcome_frontier import (
    Outcome, Cover, compile_cover, check_cover,
)


def test_full_small_preference_grid_and_random_plan_sets():
    rng = random.Random(1709)
    universe = list(product(range(3), repeat=3))
    for _ in range(250):
        points = rng.sample(universe, rng.randrange(1, 28))
        records = tuple(Outcome(i, (x, y), c) for i, (x, y, c) in enumerate(points))
        result = compile_cover(records)
        assert check_cover(records, result)
        for theta in product((0, Fraction(1, 3), 1, 4), repeat=2):
            assert max(x.score(theta) for x in records) == max(
                x.score(theta) for x in result.representatives)


def test_equal_vectors_share_canonical_representative():
    records = (Outcome(9, (4, 2), 1), Outcome(3, (4, 2), 1), Outcome(0, (1, 1), 2))
    result = compile_cover(records)
    assert tuple(x.plan_id for x in result.representatives) == (3,)
    assert dict(result.witnesses) == {0: 3, 3: 3, 9: 3}


def test_forged_or_incomplete_certificate_fails():
    records = (Outcome(0, (1, 0), 0), Outcome(1, (0, 1), 0))
    assert not check_cover(records, Cover((records[0],), ((0, 0), (1, 0))))
    assert not check_cover(records, Cover((records[0],), ((0, 0),)))
    assert not check_cover(records, Cover((Outcome(0, (9, 9), 0),), ((0, 0), (1, 0))))


def test_new_policy_needs_cover_recomputed():
    records = (Outcome(0, (4, 4), 0), Outcome(1, (3, 3), 1))
    old = compile_cover(records)
    remaining = (records[1],)
    assert not check_cover(remaining, old)
    updated = compile_cover(remaining)
    assert updated.representatives == remaining


def test_negative_weights_or_mixed_units_rejected():
    r = Outcome(0, (1, 2), 0)
    with pytest.raises(ValueError):
        r.score((-1, 1))
    with pytest.raises(ValueError):
        r.score((1.0, 1))
    with pytest.raises(ValueError):
        compile_cover((r, Outcome(1, (2,), 0)))
    with pytest.raises(ValueError):
        compile_cover((r, r))
    with pytest.raises(ValueError):
        compile_cover(())
    with pytest.raises(TypeError):
        Outcome(0, (1.0, 2), 0)
