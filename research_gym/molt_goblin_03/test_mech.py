import torch

from .factory3 import Config, Net
from .gym3 import TaskSpec
from .mech import effective_rank, gradient_similarity, linear_cka, multitask_routing, specialization


def test_cka_and_rank_properties():
    a = torch.randn(200, 8)
    assert abs(linear_cka(a, a) - 1) < 1e-5
    assert abs(linear_cka(a, 3 * a) - 1) < 1e-5
    assert linear_cka(a, torch.randn(200, 8)) < 0.3
    assert effective_rank(torch.randn(500, 2) @ torch.randn(2, 16)) < 2.5
    assert effective_rank(torch.randn(500, 16)) > 12


def test_specialization_suite_runs():
    spec = TaskSpec("proof_resolve", 8, depth=2)
    for routing, tie in (("hrm", False), ("sparse_st", False), ("mix", True), ("sequential", False)):
        net = Net(Config(routing=routing, n_types=4, width=16, loops=2, tie_modules=tie), 64, 2)
        r = specialization(net, spec, 32, 3)
        assert len(r["ablate_module"]) == 4 and "permute_modules" in r and "effective_rank_final" in r
        if routing != "sequential":
            assert len(r["route_forced"]) == 4
        g = gradient_similarity(net, spec, 32, 3)
        assert (g == {}) == tie
    single = specialization(Net(Config(width=16, loops=2), 64, 2), spec, 32, 3)
    assert "ablate_module" not in single


def test_multitask_routing():
    panels = {"proof_resolve": {"train": TaskSpec("proof_resolve", 8, depth=2), "eval": {"dev": TaskSpec("proof_resolve", 8, depth=2)}},
              "cumsum3": {"train": TaskSpec("cumsum3", 6), "eval": {"dev": TaskSpec("cumsum3", 6)}}}
    r = multitask_routing(Config(routing="hrm", n_types=4, width=16, loops=2, steps=16), panels, 1, 8, 7, n_eval=32)
    assert set(r["families"]) == set(panels) and 0 <= r["family_module_mi_nats"] <= r["max_mi_nats"] + 1e-6
