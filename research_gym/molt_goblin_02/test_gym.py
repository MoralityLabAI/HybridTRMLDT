import numpy as np
import pytest
import torch

from research_gym.molt_goblin_02.gym import (
    DEV_FAMILIES, FAMILIES, MAX_T, WITHHELD_FAMILIES, TaskSpec, score, verify_spec,
)

SPECS = [
    TaskSpec("cumsum", 12, 0, 4),
    TaskSpec("pointer_chase", 6, 4, 3),
    TaskSpec("revision", 20, 2, 4),
    TaskSpec("revision", 20, 0, 4, "reordered"),
    TaskSpec("repair", 8, 2, 0, "flagged"),
    TaskSpec("repair", 8, 2, 0, "unflagged"),
    TaskSpec("proof_deps", 16, 3),
    TaskSpec("shallow_copy", 10),
    TaskSpec("bounded_walk", 10, 0, 4),
    TaskSpec("experiment_select", 5),
]
MAX_SPECS = [
    TaskSpec("cumsum", 60, 0, 36),
    TaskSpec("pointer_chase", 40, 8, 47),
    TaskSpec("pointer_chase", 40, 8, 47, "shuffled"),
    TaskSpec("revision", 60, 3, 36),
    TaskSpec("repair", 47, 5, 0, "flagged"),
    TaskSpec("proof_deps", 96, 5),
    TaskSpec("shallow_copy", 96),
    TaskSpec("bounded_walk", 60, 0, 36),
    TaskSpec("experiment_select", 6),
]
ids = [s.key() for s in SPECS]


def test_registry():
    assert set(DEV_FAMILIES + WITHHELD_FAMILIES) == set(FAMILIES)
    assert all(FAMILIES[f].withheld for f in WITHHELD_FAMILIES)
    assert not any(FAMILIES[f].withheld for f in DEV_FAMILIES)
    assert all(FAMILIES[f].describe() for f in FAMILIES)


@pytest.mark.parametrize("spec", SPECS, ids=ids)
def test_determinism(spec):
    f = FAMILIES[spec.family]
    a, b, c = f.generate(spec, 16, 7), f.generate(spec, 16, 7), f.generate(spec, 16, 8)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.x, c.x)


@pytest.mark.parametrize("spec", SPECS, ids=ids)
def test_verify_spec(spec):
    r = verify_spec(spec)
    assert r["ok"], r
    assert r["T"] <= MAX_T and r["majority_exact"] < 0.5


@pytest.mark.parametrize("spec", SPECS, ids=ids)
def test_solve_and_admissible(spec):
    f = FAMILIES[spec.family]
    b = f.generate(spec, 256, 3)
    adm = f.admissible(b.x)
    assert adm.shape == (256, b.x.shape[1], f.n_out)
    assert int(b.x.max()) < f.vocab and int(b.x.min()) >= 0
    for i in range(256):
        sol = f.solve(b.x[i].tolist())
        for t in range(b.x.shape[1]):
            if b.mask[i, t]:
                assert sol[t] == b.y[i, t].item()
                assert adm[i, t, b.y[i, t]]
            else:
                assert sol[t] is None


@pytest.mark.parametrize("spec", [s for s in SPECS if s.variant != "reordered"] + [
    TaskSpec("pointer_chase", 6, 4, 3, "shuffled")], ids=lambda s: s.key())
def test_causality(spec):
    f = FAMILIES[spec.family]
    spec = TaskSpec(spec.family, spec.length, spec.depth, spec.distract, spec.variant)
    if spec.family == "pointer_chase":
        spec = TaskSpec("pointer_chase", 6, 4, 3, "clean")
    b = f.generate(spec, 24, 5)
    rng = np.random.default_rng(0)
    for i in range(24):
        row = b.x[i].tolist()
        t = int(rng.integers(0, len(row)))
        mut = list(row)
        mut[t] = b.x[(i + 1) % 24, t].item()
        before, after = f.solve(row), f.solve(mut)
        assert before[:t] == after[:t]


def test_validate_rejects():
    bad = [
        TaskSpec("nope", 4),
        TaskSpec("cumsum", 0),
        TaskSpec("cumsum", 60, 0, 37),
        TaskSpec("cumsum", 4, 0, 0, "weird"),
        TaskSpec("pointer_chase", 4, 0),
        TaskSpec("pointer_chase", 4, 9),
        TaskSpec("pointer_chase", 40, 4, 48),
        TaskSpec("revision", 4, 2),
        TaskSpec("revision", 10, 0, 0, "bogus"),
        TaskSpec("repair", 4, 5, 0, "flagged"),
        TaskSpec("repair", 49, 1, 0, "flagged"),
        TaskSpec("repair", 4, 1, 0, "clean"),
        TaskSpec("proof_deps", 8, 0),
        TaskSpec("shallow_copy", 97),
        TaskSpec("bounded_walk", 60, 0, 37),
        TaskSpec("experiment_select", 3),
        TaskSpec("experiment_select", 7),
    ]
    for spec in bad:
        fam = FAMILIES.get(spec.family)
        if fam is None:
            assert not verify_spec(spec)["ok"]
            continue
        with pytest.raises(ValueError):
            fam.validate(spec)
        assert not verify_spec(spec)["ok"]


@pytest.mark.parametrize("spec", MAX_SPECS, ids=lambda s: s.key())
def test_max_size(spec):
    b = FAMILIES[spec.family].generate(spec, 4, 1)
    assert b.x.shape[1] <= MAX_T


def test_score():
    for spec in SPECS:
        f = FAMILIES[spec.family]
        b = f.generate(spec, 16, 2)
        pred = b.y.clamp(min=0)
        assert score(b, pred)["exact"] == 1.0
        logits = torch.nn.functional.one_hot(pred, f.n_out).float()
        r = score(b, logits)
        assert r["exact"] == 1.0 and r["token_acc"] == 1.0 and r["per_example"] == [1] * 16
        wrong = (pred + 1) % f.n_out
        assert score(b, wrong)["exact"] == 0.0


def test_shuffled_pointer_chase_labels_match_clean():
    f = FAMILIES["pointer_chase"]
    clean = f.generate(TaskSpec("pointer_chase", 6, 5, 2, "clean"), 32, 11)
    shuf = f.generate(TaskSpec("pointer_chase", 6, 5, 2, "shuffled"), 32, 11)
    assert torch.equal(clean.y, shuf.y)
    assert torch.equal(clean.x[:, 8:], shuf.x[:, 8:])
    assert torch.equal(clean.x[:, :8].sort(1).values, shuf.x[:, :8].sort(1).values)
    assert not torch.equal(clean.x[:, :8], shuf.x[:, :8])
