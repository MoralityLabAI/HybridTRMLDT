"""PocketGoblin Bench: search space, tool surface, test-split isolation, budget enforcement, scoring, transports."""
from __future__ import annotations

import copy
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from MCP.mcp02.world import make_cases
from MCP.pocketgoblin_bench import space
from MCP.pocketgoblin_bench.episode import Episode, aggregate, normalized

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "MCP/pocketgoblin_bench/configs/smoke.json"

TINY = {
    "name": "pgb-test",
    "world": {"catalog_size": 64, "arm": "layered", "layout": "SCB"},
    "worlds": {"train": 6, "dev": 3, "test": 4},
    "seeds": {"train": 1101, "dev": 2202, "test": 3303},
    "dataset": {"epsilons": [0.0], "max_candidates": 8},
    "limits": {"max_tool_calls": 24, "prefilter_k": 16, "handoff_budget_bytes": 2048},
    "budget": {"units": 120, "max_fits": 3, "max_evals": 3},
    "reference": {"random_search_seed": 1, "random_search_max_draws": 10},
}
CAND = {"family": "dense", "core": {"d": 16}, "train": {"worlds": 6, "epochs": 1, "batch_states": 16}}


def _torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


needs_torch = pytest.mark.skipif(not _torch(), reason="torch not installed")


def tiny(**budget) -> dict:
    cfg = copy.deepcopy(TINY)
    cfg["budget"].update(budget)
    return cfg


# -- search space -----------------------------------------------------------------------------
def test_normalize_fills_defaults_and_rejects_bad_candidates():
    cand = space.normalize({"family": "trm", "core": {"cycles": 3}}, 24, 40)
    assert cand["core"] == {"d": 24, "heads": 2, "net_layers": 1, "inner": 1, "cycles": 3}
    assert cand["train"]["worlds"] == 24 and cand["filters"]["max_tool_calls"] == 40
    for bad, fragment in [({"family": "mlp"}, "family"), ({"family": "dense", "core": {"heads": 2}}, "unknown keys"),
                          ({"family": "trm", "core": {"cycles": 9}}, "core.cycles"),
                          ({"family": "dense", "train": {"worlds": 99}}, "train.worlds"),
                          ({"family": "dense", "extra": 1}, "unknown keys"),
                          ({"family": "transformer", "core": {"d": 24, "heads": 4, "layers": 1}}, None),
                          ({"family": "transformer", "core": {"d": 16, "heads": 4}}, None)]:
        if fragment is None:
            space.normalize(bad, 24, 40)
            continue
        with pytest.raises(ValueError, match=fragment):
            space.normalize(bad, 24, 40)
    with pytest.raises(ValueError):
        space.normalize({"family": "trm", "core": {"d": 24, "heads": 4}, "encoder": "bogus"}, 24, 40)


def test_sampling_is_seeded_and_always_valid():
    a = [space.sample(random.Random(3), 24, 40) for _ in range(2)]
    assert a[0] == a[1]
    draws = [space.sample(random.Random(i), 24, 40) for i in range(200)]
    assert {d["family"] for d in draws} == set(space.FAMILIES)
    for d in draws:
        assert space.normalize(d, 24, 40) == d


# -- tool surface and isolation ---------------------------------------------------------------
def test_tool_surface_hides_test_worlds():
    ep = Episode(tiny())
    out = [ep.call("describe_mcp"), ep.call("list_mesh_families"), ep.call("sample_train_queries", {"n": 6}),
           ep.call("status")]
    text = json.dumps(out)
    test_cases = make_cases("test", ep.seed["test"], ep.n["test"], ep.catalog_size)
    dev_cases = make_cases("dev", ep.seed["dev"], ep.n["dev"], ep.catalog_size)
    for case in test_cases + dev_cases:
        assert case["id"] not in text
        assert case["task"]["request"] not in text
        assert not any(doc["uri"] in text for doc in case["docs"][:20])
    assert [q["request"] for q in out[2]["queries"]] == [c["task"]["request"] for c in ep.train_cases]
    assert all(q["evidence"] for q in out[2]["queries"] if not q["abstain"] or q["evidence"] is not None)
    assert {t["name"] for t in ep.tools()} == {"describe_mcp", "sample_train_queries", "list_mesh_families",
                                               "estimate_cost", "fit", "evaluate", "submit", "status"}


def test_errors_are_coded_and_free():
    ep = Episode(tiny())
    assert ep.call("fit", {"candidate": {"family": "nope"}})["code"] == "invalid_candidate"
    assert ep.call("evaluate", {"candidate_id": "c9"})["code"] == "unknown_candidate"
    assert ep.call("submit", {"candidate_id": "c9"})["code"] == "unknown_candidate"
    assert ep.call("bogus")["code"] == "unknown_tool"
    assert ep.spent == 0.0 and ep.receipt()["status"] == "no_submission" and ep.receipt()["score"] == 0.0


def test_controls_are_deterministic_and_bracket_the_scale():
    first, second = Episode(tiny()), Episode(tiny())
    a, b = first._controls_for("test"), second._controls_for("test")
    assert a == b
    assert a["oracle_upper_bound"]["packet_recall"] == 1.0
    assert a["sham_random"]["packet_recall"] < a["oracle_upper_bound"]["packet_recall"]
    assert a["lexical_bm25"]["packet_recall"] <= 1.0


def test_normalization():
    assert normalized(0.5, 0.5, 1.0) == 0.0 and normalized(1.0, 0.5, 1.0) == 1.0
    assert normalized(0.75, 0.5, 1.0) == pytest.approx(0.5)
    assert normalized(0.2, 0.5, 1.0) < 0
    assert normalized(1.0, 1.0, 1.0) == 1.0 and normalized(0.9, 1.0, 1.0) == 0.0


def test_packet_recall_is_zero_when_the_packet_exceeds_the_budget():
    ok = {"handoff_budget_exceeded": False, "grade": {"evidence_recall": 1.0}, "accepted": True,
          "handoff_utf8_bytes": 400, "tool_calls": 10, "compute": {"flops": 0}}
    big = dict(ok, handoff_budget_exceeded=True, accepted=False)
    m = aggregate([ok, big])
    assert m["packet_recall"] == 0.5 and m["accepted"] == 0.5 and m["packet_tokens"] == 100.0


def _rpc(proc, method, params=None, ident=1):
    message = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}}
    proc.stdin.write((json.dumps(message) + "\n").encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def test_stdio_server_framing_and_hidden_tools():
    import subprocess
    proc = subprocess.Popen([sys.executable, "-u", "-m", "MCP.pocketgoblin_bench.server", "--config", str(SMOKE)],
                            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        init = _rpc(proc, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                         "clientInfo": {"name": "t", "version": "0"}})
        assert init["result"]["protocolVersion"] == "2025-06-18"
        proc.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        proc.stdin.flush()
        tools = _rpc(proc, "tools/list", ident=2)["result"]["tools"]
        assert [t["name"] for t in tools] == ["describe_mcp", "sample_train_queries", "list_mesh_families",
                                              "estimate_cost", "fit", "evaluate", "submit", "status"]
        reply = _rpc(proc, "tools/call", {"name": "sample_train_queries", "arguments": {"n": 2}}, ident=3)
        assert reply["result"]["isError"] is False and len(reply["result"]["structuredContent"]["queries"]) == 2
        bad = _rpc(proc, "tools/call", {"name": "evaluate", "arguments": {"candidate_id": "c1"}}, ident=4)
        assert bad["result"]["isError"] is True and bad["result"]["structuredContent"]["code"] == "unknown_candidate"
        assert "error" in _rpc(proc, "tools/call", {"name": "fit", "arguments": {}}, ident=5)
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


# -- training, budget and scoring (torch) -------------------------------------------------------
@needs_torch
def test_fit_charges_within_the_estimate_and_budget_is_enforced():
    ep = Episode(tiny(units=60))
    estimate = ep.call("estimate_cost", {"candidate": CAND})
    fitted = ep.call("fit", {"candidate": CAND})
    assert "error" not in fitted and fitted["charged_units"] <= estimate["upper_bound_units"] + 1e-6
    assert ep.spent == pytest.approx(fitted["charged_units"], abs=1e-2)
    heavy = {"family": "trm", "core": {"d": 48, "heads": 4, "cycles": 4, "inner": 3, "net_layers": 2},
             "train": {"worlds": 6, "epochs": 6, "batch_states": 8}}
    before = ep.spent
    refused = ep.call("fit", {"candidate": heavy})
    assert refused["code"] == "budget_exceeded" and ep.spent == before and ep.fits == 1
    ep.call("fit", {"candidate": CAND})
    ep.call("fit", {"candidate": CAND})
    assert ep.call("fit", {"candidate": CAND})["code"] in ("fit_limit", "budget_exceeded")
    assert ep.spent <= ep.total + 1e-9


@needs_torch
def test_evaluate_submit_scores_and_ends_the_episode():
    ep = Episode(tiny())
    cid = ep.call("fit", {"candidate": CAND})["candidate_id"]
    dev = ep.call("evaluate", {"candidate_id": cid})
    assert dev["split"] == "dev" and 0.0 <= dev["score"] <= 1.0 and dev["cached"] is False
    spent = ep.spent
    again = ep.call("evaluate", {"candidate_id": cid})
    assert again["cached"] is True and ep.spent == spent
    final = ep.call("submit", {"candidate_id": cid})
    assert final["split"] == "test" and set(final["metrics"]) >= {"packet_recall", "packet_tokens", "tool_calls"}
    assert final["controls"]["oracle_upper_bound"]["packet_recall"] == 1.0
    assert ep.call("fit", {"candidate": CAND})["code"] == "episode_over"
    receipt = ep.receipt()
    assert receipt["status"] == "submitted" and receipt["score"] == final["score"]
    assert receipt["budget"]["spent"] == pytest.approx(ep.spent, abs=1e-2)
    assert "request" not in json.dumps(receipt)


@needs_torch
def test_scoring_is_deterministic_given_seeds():
    results = []
    for _ in range(2):
        ep = Episode(tiny())
        cid = ep.call("fit", {"candidate": CAND})["candidate_id"]
        dev = ep.call("evaluate", {"candidate_id": cid})
        final = ep.call("submit", {"candidate_id": cid})
        results.append((dev["metrics"], dev["score"], final["metrics"], final["score"], ep.spent))
    assert results[0] == results[1]


@needs_torch
def test_reference_trainers_inprocess_and_stdio_agree(tmp_path):
    from MCP.pocketgoblin_bench import reference
    cfg_path = tmp_path / "tiny.json"
    cfg_path.write_text(json.dumps(tiny()), encoding="utf-8")
    inproc = reference.InProcessClient(tiny(), 0)
    reference.run_default(inproc)
    a = inproc.receipt()
    stdio = reference.StdioToolClient(cfg_path, 0, 4, tmp_path / "r.json")
    reference.run_default(stdio)
    b = stdio.receipt()
    assert a["test_metrics"] == b["test_metrics"] and a["score"] == b["score"]
    assert a["budget"]["spent"] == pytest.approx(b["budget"]["spent"], abs=1e-2)
    rnd = reference.InProcessClient(tiny(units=200, max_fits=4, max_evals=4), 0)
    reference.run_random(rnd, seed=1, max_draws=12)
    r = rnd.receipt()
    assert r["status"] == "submitted" and r["budget"]["spent"] <= r["budget"]["total"] + 1e-9


@needs_torch
def test_llm_runner_drives_the_tools_over_a_loopback_endpoint(tmp_path):
    cfg_path = tmp_path / "tiny.json"
    cfg_path.write_text(json.dumps(tiny()), encoding="utf-8")
    script = [("list_mesh_families", {}),
              ("fit", {"candidate": json.dumps(CAND)}),
              ("evaluate", {"candidate_id": "c1"}),
              ("submit", {"candidate_id": "c1"})]
    state = {"turn": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            name, arguments = script[state["turn"]]
            state["turn"] += 1
            call = {"id": f"call{state['turn']}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "receipts.jsonl"
    try:
        from MCP.pocketgoblin_bench.__main__ import main
        code = main(["llm", "--config", str(cfg_path), "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                     "--model", "scripted", "--out", str(out), "--max-turns", "8"])
    finally:
        server.shutdown()
    assert code == 0
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["stop_reason"] == "submitted" and record["episode"]["status"] == "submitted"
    assert record["usage"] == {"prompt_tokens": 40, "completion_tokens": 20}
    assert [m["role"] for m in record["transcript"]].count("tool") == 4


def test_llm_runner_rejects_non_loopback_endpoints():
    from MCP.pocketgoblin_bench.llm_runner import _loopback
    with pytest.raises(ValueError):
        _loopback("https://api.example.com/v1")
    _loopback("http://127.0.0.1:8091/v1")
