import itertools

import pytest
import torch

from .factory import MEMORY, ROUTING, INTERVENTIONS, MoltConfig, MoltNet, count_params, loss_fn
from .reference_pilot import PrefixReasoner
from .gym import FAMILIES, TaskSpec
from .train import train_and_eval

PARITY = dict(width=32, heads=1, rel_bias=False, inject=False, loop_keys=False)


def _load_pilot(net: MoltNet, pilot: PrefixReasoner):
    net.embedding.weight.data.copy_(pilot.embedding.weight.data)
    net.head.load_state_dict(pilot.head.state_dict())
    blk = net.blocks[0]
    for name in ("q", "k", "v", "o", "norm", "ff"):
        getattr(blk, name).load_state_dict(getattr(pilot.block, name).state_dict())


@pytest.mark.parametrize("memory,arm", [("prefix", "prefix_kv"), ("prev", "residual")])
def test_pilot_parity(memory, arm):
    torch.manual_seed(3)
    pilot = PrefixReasoner()
    net = MoltNet(MoltConfig(memory=memory, loops=4, **PARITY), 7, 7)
    _load_pilot(net, pilot)
    x = torch.randint(0, 7, (5, 9))
    ref = pilot(x, 4, arm)
    got = net(x)["logits"][-1]
    assert torch.allclose(ref, got, atol=1e-5)
    y = x.cumsum(1).remainder(7)
    torch.nn.functional.cross_entropy(ref.flatten(0, 1), y.flatten()).backward()
    torch.nn.functional.cross_entropy(got.flatten(0, 1), y.flatten()).backward()
    for name in ("q", "k", "v", "o"):
        a = getattr(pilot.block, name).weight.grad
        b = getattr(net.blocks[0], name).weight.grad
        assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(pilot.embedding.weight.grad, net.embedding.weight.grad, atol=1e-5)


def test_one_pass_parity():
    torch.manual_seed(4)
    pilot = PrefixReasoner()
    net = MoltNet(MoltConfig(memory="prefix", loops=1, **PARITY), 7, 7)
    _load_pilot(net, pilot)
    x = torch.randint(0, 7, (3, 8))
    assert torch.allclose(pilot(x, 1, "one_pass"), net(x)["logits"][-1], atol=1e-6)


def test_prefix_stabilizes_and_is_causal():
    torch.manual_seed(5)
    net = MoltNet(MoltConfig(memory="prefix", loops=10, **PARITY), 7, 7).eval()
    x = torch.randint(0, 7, (2, 8))
    with torch.no_grad():
        a = net(x, 8)["logits"][-1]
        b = net(x, 10)["logits"][-1]
        assert torch.allclose(a, b, atol=1e-6)          # strict prefix DAG of depth 8
        x2 = x.clone()
        x2[:, 5] = (x2[:, 5] + 1) % 7
        c = net(x2, 8)["logits"][-1]
        assert torch.allclose(a[:, :5], c[:, :5], atol=1e-6)


@pytest.mark.parametrize("memory,routing", list(itertools.product(MEMORY, ROUTING)))
def test_every_cell_compiles_and_is_causal(memory, routing):
    n = 1 if routing == "single" else 3
    cfg = MoltConfig(memory=memory, routing=routing, n_types=n, loops=3, width=16, topk=4, rank=4,
                     conf_bias=True, halt="utility", loss="all")
    net = MoltNet(cfg, 8, 7)
    x = torch.randint(0, 8, (4, 7))
    y = torch.randint(0, 7, (4, 7))
    out = net(x)
    loss, per = loss_fn(cfg, out, y, torch.ones(4, 7))
    loss.backward()
    assert torch.isfinite(loss) and len(per) == 3
    net.eval()
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 4] = (x2[:, 4] + 1) % 8
        a, b = net(x)["logits"][-1], net(x2)["logits"][-1]
        assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5)
        for iv in INTERVENTIONS:
            assert torch.isfinite(net(x, intervene=iv, halt_eval=True, hard_eval=True)["logits"][-1]).all()


def test_update_rules_and_feedback():
    for update in ("direct", "damped", "rlt"):
        cfg = MoltConfig(memory="full", update=update, update_rank=4, error_feedback=True, width=16)
        out = MoltNet(cfg, 8, 7)(torch.randint(0, 8, (2, 6)))
        assert out["logits"][-1].shape == (2, 6, 7)


def test_lowrank_retains_fewer_bytes():
    x = torch.randint(0, 8, (2, 6))
    full = MoltNet(MoltConfig(memory="full", width=32), 8, 7)(x)["retained_bytes_per_example"]
    low = MoltNet(MoltConfig(memory="lowrank", rank=4, width=32), 8, 7)(x)["retained_bytes_per_example"]
    assert low * 8 == full


def test_schema_rejects():
    with pytest.raises(ValueError):
        MoltConfig.from_dict({"memory": "magic"})
    with pytest.raises(ValueError):
        MoltConfig.from_dict({"routing": "hrm", "n_types": 1})
    with pytest.raises(ValueError):
        MoltConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        MoltConfig.from_dict({"memory": "prefix", "error_feedback": True})


def test_untied_cycle_has_more_params():
    a = count_params(MoltNet(MoltConfig(), 8, 7))
    b = count_params(MoltNet(MoltConfig(routing="cycle", n_types=4), 8, 7))
    assert b > 3 * a // 2


def test_train_smoke():
    cfg = MoltConfig(steps=20, loops=2, width=16, halt="utility", routing="hrm", n_types=2)
    spec = TaskSpec("cumsum", 6)
    r = train_and_eval(cfg, "cumsum", spec, {"dev": spec, "ood": TaskSpec("cumsum", 10)}, seed=1, n_eval=32,
                       extrap_loops=4)
    assert r["finite"] and 0 <= r["summary"]["exact"] <= 1
    assert "dev|stale_swap" in r["evals"] and "ood@deep" in r["evals"] and "dev|hard_route" in r["evals"]
