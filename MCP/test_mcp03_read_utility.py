"""MCP-03 read-aware utility targets and dual-head CPU smoke tests."""
from __future__ import annotations

import pytest

from MCP.mcp02 import features as FE
from MCP.mcp03.agent import CodeOracleView, make_code_control, query_item, row_item, run_code_episode
from MCP.mcp03.env import CodeEnv
from MCP.mcp03.utility import packet_utility_targets
from MCP.test_mcp03 import repo as ledger_repo


def test_read_target_is_packet_marginal_and_expand_has_horizon(ledger_repo):
    env = CodeEnv(ledger_repo["snap"], {}, ledger_repo["gold"], "search")
    uri = "pkg/core.py::Ledger.post"
    cands = [query_item("Ledger post"), row_item(env.row("s:" + uri), 1)]
    view = CodeOracleView(env)
    labels = [view.label(c) for c in cands]
    assert labels == [1, 1]
    obs = {"candidates": cands}
    first = packet_utility_targets(env, obs, labels, [], {}, packet_budget=1024, remaining_calls=8)
    assert first[0] == 0.30 and first[1] > first[0]
    final = packet_utility_targets(env, obs, labels, [], {}, packet_budget=1024, remaining_calls=1)
    assert final[0] == 0.0 and final[1] > 0
    seen = packet_utility_targets(env, obs, labels, [(1.0, uri)], {},
                                  packet_budget=1024, remaining_calls=8)
    assert seen == [0.0, 0.0]
    tiny = packet_utility_targets(env, obs, labels, [], {}, packet_budget=1, remaining_calls=1)
    assert tiny == [0.0, 0.0]
    with pytest.raises(ValueError):
        packet_utility_targets(env, obs, labels[:1], [], {}, packet_budget=1024, remaining_calls=8)


def test_private_targets_only_in_recording_and_horizon_is_public(ledger_repo):
    env = CodeEnv(ledger_repo["snap"], {}, ledger_repo["gold"], "search")
    events = []
    issue = "Ledger.post should round the amount"
    ep = run_code_episode(env, issue, policy=make_code_control("oracle_upper_bound"),
                          max_tool_calls=8, budgets=(), record=events.append,
                          utility_packet_budget=1024)
    decisions = [e for e in events if e["kind"] == "decide"]
    assert decisions and ep["tool_calls"] <= 8
    stats = ep["compute"]
    assert stats["read_symbol_actions"] + stats["expand_actions"] == stats["decisions"]
    assert 0 <= stats["read_candidate_steps"] <= stats["decisions"]
    assert stats["read_symbol_actions"] >= len(ep["reads"])
    for event in decisions:
        obs = event["obs"]
        assert "remaining_calls" in obs
        assert len(event["utility_values"]) == len(obs["candidates"])
        assert "utility_values" not in obs
        assert "gold" not in obs and "oracle" not in obs
    assert FE.utility_needs(decisions[-1]["obs"])[0].startswith("budget")
    with pytest.raises(ValueError):
        FE.utility_needs({"needs": ["request"]})


def test_dual_head_forward_and_fit_cpu_if_torch_available():
    torch = pytest.importorskip("torch")
    from MCP.mcp02.models import Router
    from MCP.mcp02.train import _utility_listwise_loss, fit

    config = {"arch": "looped", "d": 16, "heads": 4, "passes": 1,
              "encoder": "match", "utility_heads": "expand_read"}
    model = Router(config)
    assert model.expand_head is not None and model.read_head is not None
    obs = {"needs": ["Ledger post"], "remaining_calls": 8, "candidates": [
        {"text": "module Ledger post", "struct": [0, 0, 1, 0, 0, 0, 0, 0, 0]},
        {"text": "symbol Ledger post", "struct": [0, 0, 0, 0, 1, 0, 0, 0, 0]},
    ]}
    ids, segs, structs = FE.encode_observation(
        dict(obs, needs=FE.utility_needs(obs)), "match")
    with torch.no_grad():
        model.expand_head.weight.zero_()
        model.expand_head.bias.fill_(-2.0)
        model.read_head.weight.zero_()
        model.read_head.bias.fill_(2.0)
        scores, _ = model(torch.tensor(ids), torch.tensor(segs), torch.tensor(structs, dtype=torch.float32))
        assert scores.tolist() == pytest.approx([-2, 2], abs=1e-5)
    fake = {"decide": [{
        "need": FE.encode_text("budgetearly Ledger post", FE.NEED_TOKENS),
        "cands": [FE.encode_text(c["text"], FE.CAND_TOKENS) for c in obs["candidates"]],
        "structs": [c["struct"] for c in obs["candidates"]],
        "labels": [0, 1], "utilities": [0.30, 0.90],
    }], "rows": [], "meta": {"split": "train", "synthetic": True}}
    model, receipt = fit(config, fake, epochs=1, lr=0.002, batch_states=1,
                         device="cpu", seed=7, threads=1, max_steps=1, dev=fake,
                         log=lambda *_: None)
    assert receipt["objective"].startswith("packet-marginal")
    assert receipt["steps"] == 1
    assert receipt["dev_ranking"]["utility_states"] == 1
    assert torch.isfinite(_utility_listwise_loss(
        torch.tensor([1.0, -1.0]), torch.tensor([0, 0]),
        torch.tensor([0.0, 1.0]), 1))
