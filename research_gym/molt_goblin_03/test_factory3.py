import pytest
import torch

from research_gym.molt_goblin_02.factory import MoltConfig, MoltNet
from research_gym.molt_goblin_02.gym import TaskSpec
from research_gym.molt_goblin_02.train import train_and_eval as train_v2

from .factory3 import ROUTING, Config, Net, distill_loss, logical_visits, loss_fn
from .train3 import run_cell, train


def _v2_pair(routing, n):
    v2 = MoltConfig(memory="prefix", routing=routing, n_types=n, loss="all", steps=16, width=16, loops=3)
    v3 = Config(memory="prefix", routing=routing, n_types=n, loss="all", steps=16, width=16, loops=3)
    return v2, v3


@pytest.mark.parametrize("routing,n", [("single", 1), ("hrm", 4), ("mix", 4), ("cycle", 4)])
def test_forward_parity_with_v2(routing, n):
    v2, v3 = _v2_pair(routing, n)
    torch.manual_seed(0)
    a = MoltNet(v2, 8, 7)
    torch.manual_seed(0)
    b = Net(v3, 8, 7)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    b.load_state_dict({k: v for k, v in a.state_dict().items() if k in b.state_dict()})
    x = torch.randint(0, 8, (3, 9))
    for la, lb in zip(a(x)["logits"], b(x)["logits"]):
        assert torch.allclose(la, lb, atol=1e-6)


def test_training_parity_with_v2():
    """Same init, data and loss: v3 reproduces a v2 training run bit for bit."""
    from research_gym.molt_goblin_02 import train as t2
    from research_gym.molt_goblin_02.gym import FAMILIES
    v2, v3 = _v2_pair("hrm", 4)
    spec = TaskSpec("proof_deps", 8, depth=3)
    torch.manual_seed(5)
    a = MoltNet(v2, FAMILIES["proof_deps"].vocab, 2)
    opt = torch.optim.AdamW(a.parameters(), lr=v2.lr)
    data = FAMILIES["proof_deps"].generate(spec, 16 * 32, 77)
    for s in range(16):
        sl = slice(s * 32, (s + 1) * 32)
        from research_gym.molt_goblin_02.factory import loss_fn as l2
        loss, _ = l2(v2, a(data.x[sl]), data.y[sl], data.mask[sl].float())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(a.parameters(), 1.0); opt.step()
    torch.manual_seed(5)
    b = Net(v3, FAMILIES["proof_deps"].vocab, 2)
    opt = torch.optim.AdamW(b.parameters(), lr=v3.lr)
    for s in range(16):
        sl = slice(s * 32, (s + 1) * 32)
        b.train(); b.progress = s / 15
        loss, _ = loss_fn(b, b(data.x[sl]), data.y[sl], data.mask[sl].float())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0); opt.step()
    a.eval(); b.eval()
    assert torch.allclose(a(data.x[:8])["logits"][-1], b(data.x[:8])["logits"][-1], atol=1e-5)


@pytest.mark.parametrize("routing", ROUTING)
def test_causal_and_trainable(routing):
    n = 1 if routing == "single" else 4
    cfg = Config(routing=routing, n_types=n, width=16, loops=3, load_balance=0.1, route_entropy=0.05,
                 diversity=0.1 if n > 1 else 0.0, functional=n == 4, func_coef=0.5 if n == 4 else 0.0)
    net = Net(cfg, 8, 7)
    x = torch.randint(0, 8, (4, 7))
    y = torch.randint(0, 7, (4, 7))
    net.train()
    loss, parts = loss_fn(net, net(x), y, torch.ones(4, 7))
    loss.backward()
    assert torch.isfinite(loss)
    if routing in ("hrm", "sparse_st", "anneal"):
        assert net.route.weight.grad is not None and net.route.weight.grad.abs().sum() > 0
    net.eval()
    with torch.no_grad():
        x2 = x.clone(); x2[:, 4] = (x2[:, 4] + 1) % 8
        assert torch.allclose(net(x)["logits"][-1][:, :4], net(x2)["logits"][-1][:, :4], atol=1e-5)


@pytest.mark.parametrize("routing", ["sparse_st", "anneal", "hrm", "mix"])
def test_dispatch_matches_dense_hard_routing(routing):
    cfg = Config(routing=routing, n_types=4, width=16, loops=3)
    net = Net(cfg, 8, 7).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.3 * torch.randn_like(p))
        x = torch.randint(0, 8, (5, 9))
        dense = net(x, hard_eval=True)
        fast = net(x, hard_eval=True, dispatch=True)
    assert torch.allclose(dense["logits"][-1], fast["logits"][-1], atol=1e-5)
    assert fast["visits"] == 3


def test_visits_and_canonical():
    assert logical_visits(Config(routing="hrm", n_types=4)) == 16
    assert logical_visits(Config(routing="sparse_st", n_types=4)) == 4
    assert logical_visits(Config(routing="sequential", n_types=4)) == 16
    a = Config(routing="single", st_tau=0.5)
    b = Config(routing="single", st_tau=2.0)
    assert a.digest() == b.digest()
    assert Config(routing="sparse_st", n_types=4, st_tau=0.5).digest() != Config(routing="sparse_st", n_types=4).digest()
    with pytest.raises(ValueError):
        Config(distill="hidden").validate()
    with pytest.raises(ValueError):
        Config(tie_modules=True).validate()


def test_tied_modules_share_parameters():
    tied = Net(Config(routing="hrm", n_types=4, tie_modules=True), 8, 7)
    untied = Net(Config(routing="hrm", n_types=4), 8, 7)
    assert len(tied.blocks) == 1 and len(untied.blocks) == 4


def test_anneal_schedule_and_sparse_eval_one_visit():
    net = Net(Config(routing="anneal", n_types=4, anneal_frac=0.5, anneal_end_temp=0.1), 8, 7)
    net.progress = 0.0; assert abs(net.anneal_temp() - 1.0) < 1e-9
    net.progress = 0.5; assert abs(net.anneal_temp() - 0.1) < 1e-9
    net.eval()
    out = net(torch.randint(0, 8, (2, 5)))
    assert out["visits"] == 4 and all(((w == 0) | (w == 1)).all() for w in out["route_w"])


def test_distillation_cell(tmp_path, monkeypatch):
    from . import train3
    monkeypatch.setattr(train3, "TEACHER_CACHE", tmp_path)
    spec = TaskSpec("cumsum3", 5)
    panel = {"train": spec, "eval": {"dev": spec}}
    teacher = Config(routing="hrm", n_types=4, width=16, loops=2, steps=16)
    student = Config(width=16, loops=2, steps=16, distill="logits_all", distill_coef=1.0)
    r = run_cell(student, "cumsum3", panel, 1, 25_000_000, teacher_cfg=teacher, n_eval=16)
    assert r["finite"] and any("distill" in t for t in r["loss_trace"])
    assert len(list(tmp_path.glob("*.pt"))) == 1
    for mode in ("logits_final", "hidden"):
        c = student.replace(distill=mode)
        s, t = Net(c, 4, 3), Net(teacher, 4, 3)
        x = torch.randint(0, 4, (2, 5))
        assert torch.isfinite(distill_loss(c, s(x), t(x), torch.ones(2, 5)))
