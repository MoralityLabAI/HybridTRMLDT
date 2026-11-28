import numpy as np
import pytest
import torch

from research_gym.molt_goblin_02 import gym
from research_gym.molt_goblin_03.gym3 import (
    FAMILIES3, FINALS_ONLY_FAMILIES, MAX_T, NEW_FAMILIES, TaskSpec, score, verify_spec,
)

SPECS = [
    TaskSpec("local_map", 8),
    TaskSpec("cumsum3", 12, 0, 4),
    TaskSpec("proof_resolve", 16, 3),
    TaskSpec("experiment_portfolio", 5),
    TaskSpec("revision_retract", 20, 0, 4),
    TaskSpec("cex_repair", 4, 1, 2),
    TaskSpec("research_plan", 12, 3),
]
MAX_SPECS = [
    TaskSpec("local_map", 24),
    TaskSpec("cumsum3", 24, 0, 72),
    TaskSpec("proof_resolve", 96, 5),
    TaskSpec("experiment_portfolio", 6),
    TaskSpec("revision_retract", 24, 0, 72),
    TaskSpec("cex_repair", 5, 8, 80),
    TaskSpec("research_plan", 16, 4),
]
IDS = [s.family for s in SPECS]


def test_families3_contents():
    assert set(gym.FAMILIES) <= set(FAMILIES3)
    assert set(NEW_FAMILIES) <= set(FAMILIES3)
    assert len(NEW_FAMILIES) == 8 and "local_map3" in NEW_FAMILIES
    assert FINALS_ONLY_FAMILIES == ("research_plan",)
    assert NEW_FAMILIES["research_plan"].withheld is True


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_determinism(spec):
    fam = FAMILIES3[spec.family]
    a, b, c = fam.generate(spec, 32, 7), fam.generate(spec, 32, 7), fam.generate(spec, 32, 8)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.x, c.x)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_verify_spec_ok(spec):
    res = verify_spec(spec)
    assert res["ok"], res


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_solve_agrees_and_admissible_sound(spec):
    fam = FAMILIES3[spec.family]
    b = fam.generate(spec, 256, 3)
    assert int(b.x.max()) < fam.vocab and int(b.y.max()) < fam.n_out
    for i in range(256):
        sol = fam.solve(b.x[i].tolist())
        for t in range(b.x.shape[1]):
            assert (sol[t] is None) == (not bool(b.mask[i, t]))
            if sol[t] is not None:
                assert sol[t] == int(b.y[i, t])
    adm = fam.admissible(b.x)
    hit = adm.gather(-1, b.y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    assert bool((hit | ~b.mask).all())
    assert score(b, b.y.clamp(min=0))["exact"] == 1.0


@pytest.mark.parametrize("spec", [s for s in SPECS if s.distract == 0] +
                         [TaskSpec("cumsum3", 12), TaskSpec("revision_retract", 20), TaskSpec("cex_repair", 4, 1)],
                         ids=lambda s: s.family)
def test_causality(spec):
    fam = FAMILIES3[spec.family]
    b = fam.generate(spec, 64, 5)
    rng = np.random.default_rng(0)
    for i in range(64):
        base = b.x[i].tolist()
        full = fam.solve(base)
        scored = [t for t in range(len(base)) if full[t] is not None]
        t = int(rng.choice(scored))
        other = b.x[(i + 1) % 64, t].item()
        mut = list(base)
        mut[t] = other
        assert fam.solve(mut)[:t] == full[:t]


@pytest.mark.parametrize("spec", MAX_SPECS, ids=lambda s: s.family)
def test_max_sizes(spec):
    fam = FAMILIES3[spec.family]
    b = fam.generate(spec, 8, 1)
    assert b.x.shape[1] <= MAX_T
    assert verify_spec(spec)["T"] == b.x.shape[1]


def test_over_max_rejected():
    assert not verify_spec(TaskSpec("cumsum3", 24, 0, 73))["ok"]


BAD = [
    TaskSpec("local_map", 3), TaskSpec("local_map", 25), TaskSpec("cumsum3", 1), TaskSpec("cumsum3", 25),
    TaskSpec("proof_resolve", 8, 0), TaskSpec("experiment_portfolio", 2), TaskSpec("experiment_portfolio", 7),
    TaskSpec("revision_retract", 1), TaskSpec("cex_repair", 0), TaskSpec("cex_repair", 6),
    TaskSpec("cex_repair", 3, 9), TaskSpec("research_plan", 1, 2), TaskSpec("research_plan", 17, 2),
    TaskSpec("research_plan", 8, 0), TaskSpec("local_map", 8, 0, 0, "shuffled"), TaskSpec("local_map", 8, -1),
]


@pytest.mark.parametrize("spec", BAD, ids=lambda s: s.key())
def test_validate_rejects(spec):
    with pytest.raises(ValueError):
        FAMILIES3[spec.family].validate(spec)
    assert not verify_spec(spec)["ok"]


def test_validate_wrong_family():
    with pytest.raises(ValueError):
        NEW_FAMILIES["cumsum3"].validate(TaskSpec("local_map", 8))


def test_portfolio_optimal_closed_and_feasible():
    fam = NEW_FAMILIES["experiment_portfolio"]
    b = fam.generate(TaskSpec("experiment_portfolio", 6), 256, 11)
    for row, lab in zip(b.x.tolist(), b.y.tolist()):
        L = 6
        B = row[0] - 47
        items = [((t // 12) + 1, (t % 12) // 3, t % 3) for t in row[1 : 1 + L]]
        chosen = {j for j in range(L) if lab[1 + L + j] == 1}
        assert sum(items[j][0] for j in chosen) <= B
        for j in chosen:
            pre = items[j][2]
            assert pre == 0 or (j - pre >= 0 and (j - pre) in chosen)
        for j, (_, _, pre) in enumerate(items):
            assert pre == 0 or j - pre >= 0


def test_revision_retract_hand_example():
    fam = NEW_FAMILIES["revision_retract"]
    x = [1, 2, 28, 24, 28, 24, 28, 24, 28, 6 + 3, 28 + 1, 25, 28 + 1]
    assert fam.solve(x) == [None, None, 2, None, 1, None, 6, None, 6, None, 3, None, 6]
    assert fam.admissible(torch.tensor([x]))[0, 4].nonzero().flatten().tolist() == [1, 2, 6]


def test_cex_repair_bruteforce():
    fam = NEW_FAMILIES["cex_repair"]
    spec = TaskSpec("cex_repair", 5, 2, 3)
    b = fam.generate(spec, 256, 9)
    for row, lab in zip(b.x.tolist(), b.y.tolist()):
        pts = [((t - 25) // 5, (t - 25) % 5) for t in row if 25 <= t < 50]
        assert len({p[0] for p in pts[:2]}) == 2
        fits = [(a, c) for a in range(1, 5) for c in range(5) if all((a * px + c) % 5 == py for px, py in pts)]
        assert len(fits) == 1
        a, c = fits[0]
        assert (row[0] // 5, row[0] % 5) != (a, c)
        for t, tok in enumerate(row):
            if 50 <= tok < 55:
                assert lab[t] == (a * (tok - 50) + c) % 5
            else:
                assert lab[t] == -100


def test_research_plan_exact_when_no_prereqs():
    fam = NEW_FAMILIES["research_plan"]
    b = fam.generate(TaskSpec("research_plan", 16, 3), 64, 2)
    adm = fam.admissible(b.x)
    for row, lab, a in zip(b.x.tolist(), b.y.tolist(), adm):
        for t, tok in enumerate(row):
            if tok // 3 == 0:
                assert a[t].nonzero().flatten().tolist() == [tok % 3 + 1]
            assert lab[t] >= tok % 3 + 1
