import math
import random

import pytest

from research_gym.ldt_mesh import (
    CostModel,
    Factor,
    Mesh,
    MeshUnit,
    Router,
    StructuralRegret,
    objective_J,
    route_features,
)
from research_gym.ldt_mesh.units import K, U, UNIT_BITS, unit_mask


def _sensor(x: int, bits: list[int], kind: int) -> int:
    ones = sum((x >> b) & 1 for b in bits)
    if kind == 0:
        return ones % 2
    if kind == 1:
        return int(ones == len(bits))
    return int(2 * ones > len(bits))


def _random_factors(rng: random.Random, n: int, cross_pairs, rho: float) -> list[Factor]:
    x = rng.getrandbits(K)
    out = []
    for _ in range(n):
        if cross_pairs and rng.random() < rho:
            a, b = rng.choice(cross_pairs)
            ar = rng.choice([2, 3])
            na = rng.randint(1, ar - 1)
            bits = rng.sample(range(4 * a, 4 * a + 4), na) + rng.sample(range(4 * b, 4 * b + 4), ar - na)
        else:
            u = rng.randrange(U)
            bits = rng.sample(range(4 * u, 4 * u + 4), rng.choice([1, 2, 3]))
        kind = 0 if len(bits) == 1 else rng.randrange(3)
        e = rng.uniform(0.05, 0.25)
        y = _sensor(x, bits, kind) ^ int(rng.random() < e)
        out.append(Factor(sum(1 << b for b in bits), kind, y, e))
    return out


def _brute_force_parity(factors: list[Factor], q: int) -> float:
    """Independent reference: enumerate all 2^16 states with its own sensor code."""
    num = den = 0.0
    for s in range(1 << K):
        w = 1.0
        for f in factors:
            bits = [b for b in range(K) if (f.mask >> b) & 1]
            v = _sensor(s, bits, f.kind)
            p_on = f.e if v == 0 else 1 - f.e
            w *= p_on if f.y else 1 - p_on
        den += w
        if bin(s & q).count("1") % 2:
            num += w
    return num / den


def _random_query(rng: random.Random, pairs) -> int:
    if rng.random() < 0.5:
        u = rng.randrange(U)
        bits = rng.sample(range(4 * u, 4 * u + 4), 2)
    else:
        a, b = rng.choice(pairs)
        bits = [rng.randrange(4 * a, 4 * a + 4), rng.randrange(4 * b, 4 * b + 4)]
    return sum(1 << b for b in bits)


def test_exact_equals_brute_force_over_all_16_bits_on_random_episodes():
    for seed in range(5):
        rng = random.Random(seed)
        pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
        active = rng.sample(pairs, 2)
        factors = _random_factors(rng, 10, active, 0.5)
        mesh = Mesh()
        for f in factors:
            mesh.absorb(f)
        if seed % 2:
            mesh.merge(mesh.block_of(active[0][0]), mesh.block_of(active[0][1]))
        for _ in range(2):
            q = _random_query(rng, active)
            p, cost = mesh.exact(q)
            assert abs(p - _brute_force_parity(factors, q)) < 1e-9
            assert cost == mesh.exact_cost(q)[0]


def test_exact_enumerates_only_the_relevant_component():
    mesh = Mesh()
    mesh.absorb(Factor(0b11 << 0, 0, 1, 0.1))
    mesh.absorb(Factor(1 << 0 | 1 << 4, 0, 1, 0.1))   # joins units 0 and 1
    mesh.absorb(Factor(0b11 << 8, 0, 1, 0.1))         # unit 2, isolated
    q = 0b11
    cost, comp = mesh.exact_cost(q)
    assert sorted(u for b in comp for u in b) == [0, 1]
    assert cost == 16 ** 2 * 2
    full = mesh.exact_cost(0b11 | (0b11 << 8))
    assert full[0] > cost


def test_local_equals_exact_without_cross_factors():
    rng = random.Random(77)
    factors = _random_factors(rng, 12, [], 0.0)
    mesh = Mesh()
    for f in factors:
        mesh.absorb(f)
    assert not mesh.cross_factors()
    for u in range(U):
        for i in range(4):
            for j in range(i + 1, 4):
                q = (1 << (4 * u + i)) | (1 << (4 * u + j))
                assert abs(mesh.local(q)[0] - mesh.exact(q)[0]) < 1e-9
    q = (1 << 0) | (1 << 5) | (1 << 9)
    assert abs(mesh.local(q)[0] - mesh.exact(q)[0]) < 1e-9
    assert all(mesh.expose(b).info_lost == 0 for b in mesh.blocks)


def test_merge_moves_cross_factor_into_block_and_split_returns_it_to_pending():
    rng = random.Random(4)
    factors = _random_factors(rng, 8, [(0, 1)], 0.8)
    mesh = Mesh()
    for f in factors:
        mesh.absorb(f)
    q = 1 | (1 << 4)
    p_exact = mesh.exact(q)[0]
    merged = mesh.merge((0,), (1,))
    assert merged == (0, 1)
    assert not [bl for _, bl in mesh.cross_factors() if merged in bl]
    assert abs(mesh.local(q)[0] - p_exact) < 1e-9
    mesh.split(merged)
    assert mesh.block_of(0) == (0,) and mesh.block_of(1) == (1,)
    assert abs(mesh.exact(q)[0] - p_exact) < 1e-9


def test_cluster_bp_is_exact_on_a_single_cross_factor_marginals():
    mesh = Mesh()
    rng = random.Random(3)
    x = rng.getrandbits(K)
    for u in range(U):
        for b in range(4):
            bit = 4 * u + b
            mesh.absorb(Factor(1 << bit, 0, ((x >> bit) & 1) ^ int(rng.random() < 0.2), 0.2))
    mesh.absorb(Factor((1 << 1) | (1 << 5), 0, 1, 0.05))
    for bit in (1, 5):
        assert abs(mesh.bp(1 << bit, iters=40)[0] - mesh.exact(1 << bit)[0]) < 1e-6


SEVEN = ("local_state", "dependencies", "uncertainty", "operators",
         "cost_estimates", "verification", "info_lost")


def test_exposed_interface_contains_all_seven_fields():
    assert tuple(MeshUnit.__dataclass_fields__) == SEVEN
    rng = random.Random(5)
    mesh = Mesh()
    for f in _random_factors(rng, 12, [(0, 1)], 0.6):
        mesh.absorb(f)
    unit = mesh.expose(mesh.blocks[0])
    for name in SEVEN:
        assert hasattr(unit, name)
    assert unit.verification.status == "unverified"
    mesh.exact(1 | (1 << 4))
    assert mesh.expose(mesh.block_of(0)).verification.status == "verified"
    mesh.absorb(Factor(1, 0, 1, 0.1))
    assert mesh.expose(mesh.block_of(0)).verification.status == "stale"
    for b in mesh.blocks:
        mesh.expose(b).check()
    ent = mesh.expose(mesh.blocks[0]).uncertainty
    assert 0.0 <= ent <= UNIT_BITS * math.log(2)


def test_check_rejects_a_malformed_unit():
    mesh = Mesh()
    u = mesh.expose((0,))
    bad = MeshUnit(u.local_state, ((0,),), u.uncertainty, u.operators, u.cost_estimates,
                   u.verification, u.info_lost)
    with pytest.raises(ValueError):
        bad.check()


def _near_threshold_mesh(e: float) -> Mesh:
    """One unary factor: local P(bit 0 = 1) = 1 - e, so e=0.2 sits exactly on 0.8."""
    mesh = Mesh()
    mesh.absorb(Factor(1, 0, 1, e))
    return mesh


def test_router_picks_exact_within_margin_of_threshold_when_budget_allows():
    router = Router()
    d = router.route(_near_threshold_mesh(0.2), 1)
    assert d.features.margin < 1e-9
    assert d.op == "exact"
    d = router.route(_near_threshold_mesh(0.17), 1)   # p_local = 0.83, margin 0.03
    assert d.op == "exact"
    d = router.route(_near_threshold_mesh(0.23), 1)   # p_local = 0.77, margin 0.03
    assert d.op == "exact"


def test_router_keeps_local_far_from_thresholds_and_respects_the_budget():
    router = Router()
    assert router.route(_near_threshold_mesh(0.01), 1).op == "local"
    tight = Router(cost=CostModel(exact_budget=1))
    d = tight.route(_near_threshold_mesh(0.2), 1)
    assert "exact" not in d.J
    assert d.op != "exact"


def test_router_uses_bp_or_exact_when_cross_factors_carry_information():
    rng = random.Random(11)
    mesh = Mesh()
    for f in _random_factors(rng, 14, [(0, 1)], 0.7):
        mesh.absorb(f)
    q = 1 | (1 << 4)
    f = route_features(mesh, q)
    assert f.kl_hat_query > 0
    d = Router().route(mesh, q)
    assert d.op in ("bp", "exact")
    assert set(d.J) >= {"local", "exact", "abstain"}


def test_objective_J_terms_and_learned_regret_hook():
    mesh = _near_threshold_mesh(0.2)
    f = route_features(mesh, 1)
    cost = CostModel()
    base = objective_J(f, "local", StructuralRegret(), cost)
    assert base > objective_J(f, "local", StructuralRegret(), cost, launched=["local"])

    class AlwaysBP:
        def predict(self, features, op):
            return {"local": 1.0, "bp": 0.0, "exact": 1.0, "abstain": 1.0}[op]

    mesh.absorb(Factor(1 | (1 << 4), 0, 1, 0.1))
    assert Router(regret=AlwaysBP()).route(mesh, 1).op == "bp"
