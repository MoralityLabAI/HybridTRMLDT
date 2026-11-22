"""MCP-02 contracts: hidden ground truth, grouped splits, certified exclusions, judge, replay."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from MCP.mcp02 import features as FE
from MCP.mcp02.agent import (CONTROL_POLICIES, Composer, OraclePolicy, ShamPolicy, build_observation,
                             make_control, run_episode, symbolic_exclusions)
from MCP.mcp02.economics import paired_difference, summarize_group, wilson
from MCP.mcp02.env import ARMS, Env02, plan_needs, violates
from MCP.mcp02.textutil import canonical_json, digest
from MCP.mcp02.world import FAMILIES, SPLITS, TEMPLATE_SPLIT, _word_pool, make_cases, make_world

ARM_LAYOUTS = (("flat", ""), ("search", ""), ("layered", "SCB"), ("layered_ldt", "PSCB"), ("layered", "KBE"))


@pytest.fixture(scope="module")
def cases():
    return make_cases("test", 3303, 8, 128)


def test_world_is_reproducible_and_sized():
    a, b = make_world("dev", 5, 3, 256), make_world("dev", 5, 3, 256)
    assert digest(a) == digest(b)
    assert len(a["docs"]) == 256 and len({d["uri"] for d in a["docs"]}) == 256
    assert digest(make_world("dev", 5, 4, 256)) != digest(a)
    with pytest.raises(ValueError):
        make_world("dev", 5, 0, 32)


def test_every_family_has_reachable_required_documents(cases):
    assert {c["family"] for c in cases} == set(FAMILIES)
    for case in cases:
        uris = {d["uri"] for d in case["docs"]}
        for chain in case["oracle"]["required_uris"]:
            assert set(chain) <= uris
        if case["family"] == "missing":
            assert case["oracle"]["answer"] is None and case["oracle"]["abstain"]


def test_splits_are_grouped_by_names_and_templates():
    pools = {s: set(_word_pool(s)) for s in SPLITS}
    assert not (pools["train"] & pools["test"]) and not (pools["train"] & pools["dev"])
    assert not (pools["dev"] & pools["test"])
    for split in SPLITS:
        for case in make_cases(split, 9, 8, 96):
            assert TEMPLATE_SPLIT[case["task"]["template"]] == split
            words = {w.lower() for d in case["docs"] for w in d["name"].split()}
            assert words <= pools[split]
            assert case["group_id"].startswith(f"mcp02-world-v1/{split}/")


def test_public_task_hides_oracle(cases):
    for case in cases:
        env = Env02(case, "search")
        text = canonical_json(env.task) + canonical_json(env.tools())
        for chain in case["oracle"]["required_uris"]:
            for uri in chain:
                assert uri not in text
        assert "oracle" not in env.task and "hidden" not in env.task
        if case["oracle"]["answer"] is not None:
            assert str(case["oracle"]["answer"]) not in canonical_json(env.task["plan"])


def test_observations_carry_no_private_state(cases):
    seen = []

    class Spy(ShamPolicy):
        def score(self, obs):
            seen.append(canonical_json({k: v for k, v in obs.items() if k != "task"}))
            return super().score(obs)

    case = cases[1]
    run_episode(case, arm="layered", layout="SCB", policy=Spy(), max_tool_calls=12)
    blob = "".join(seen)
    assert "required_uris" not in blob and "oracle" not in blob and "chain_values" not in blob


@pytest.mark.parametrize("arm,layout", ARM_LAYOUTS)
def test_oracle_bound_accepts_every_family(cases, arm, layout):
    for case in cases:
        ep = run_episode(case, arm=arm, layout=layout, policy=make_control("oracle_upper_bound"))
        assert ep["accepted"], (case["family"], ep["failure"], ep["grade"]["reasons"])
        assert ep["private_checks"]["certified_exclusions_containing_required"] == 0


def test_controls_run_and_failures_are_counted(cases):
    for name in CONTROL_POLICIES:
        ep = run_episode(cases[2], arm="layered", layout="SCB", policy=make_control(name), max_tool_calls=6)
        assert ep["tool_calls"] <= 6
        if not ep["accepted"]:
            assert ep["failure"] or ep["grade"]["reasons"]
    group = [run_episode(c, arm="layered", layout="SCB", policy=make_control("sham_random"), max_tool_calls=4)
             for c in cases[:3]]
    prices = {"bytes_per_token_estimate": 4.0, "frontier_input_usd_per_mtok": 3.0, "local_cpu_usd_per_hour": 0.05,
              "local_gpu_usd_per_hour": 0.3, "tool_call_usd": 1e-5, "verification_check_usd": 1e-9}
    row = summarize_group(group, prices, "cpu")
    assert row["accepted"] == 0 and row["cost_per_success_usd"] is None
    assert row["cost_total_usd"]["total_usd"] > 0 and sum(row["failures"].values()) == 3


def test_sham_is_deterministic(cases):
    a = run_episode(cases[0], arm="search", layout="", policy=make_control("sham_random"))
    b = run_episode(cases[0], arm="search", layout="", policy=make_control("sham_random"))
    assert a["results_sha256"] == b["results_sha256"] and a["decisions"] == b["decisions"]


def test_tape_replay_reproduces_outcomes(cases):
    from MCP.mcp02.run import TapeOnly
    for arm, layout in ARM_LAYOUTS:
        ep = run_episode(cases[4], arm=arm, layout=layout, policy=make_control("deterministic_rules"))
        out = run_episode(cases[4], arm=arm, layout=layout, policy=TapeOnly("tape"), tape=ep["decisions"])
        for field in ("accepted", "tool_calls", "results_sha256", "handoff_utf8_bytes", "environment_stats"):
            assert out[field] == ep[field]
    from MCP.mcp02.run import compact_tape, expand_tape
    ldt = run_episode(cases[4], arm="layered_ldt", layout="PSCB", policy=make_control("deterministic_rules"))
    compact = expand_tape(json.loads(json.dumps(compact_tape(ldt["decisions"]))))
    out = run_episode(cases[4], arm="layered_ldt", layout="PSCB", policy=TapeOnly("tape"), tape=compact)
    assert out["results_sha256"] == ldt["results_sha256"] and out["compute"]["flops"] == ldt["compute"]["flops"]
    bad = [dict(ep["decisions"][0], obs_sha="0" * 16)] + ep["decisions"][1:]
    out = run_episode(cases[4], arm="search", layout="", policy=TapeOnly("tape"), tape=bad)
    assert out["failure"] == "transport_or_environment_error"


def _root_children(env):
    return env.call("index_children", {"node_id": "root", "limit": 16})["structuredContent"]["children"]


def test_certified_exclusion_rejects_unsound_and_keeps_unknown(cases):
    case = cases[1]
    env = Env02(case, "layered_ldt", "KSB")
    rows = _root_children(env)
    needs = plan_needs(env.task)
    kinds = {n["kind"] for n in needs}
    needed = [r for r in rows if any(env._docs[u]["kind"] in kinds for u in env._nodes[r["node_id"]]["uris"])]
    result = env.call("candidate_refine", {"node_id": "root", "exclude": [needed[0]["node_id"]],
                                           "predicate": "kind"})["structuredContent"]
    assert result["excluded"] == [] and result["candidates_after"] == result["candidates_before"]
    unneeded = [r for r in rows if r not in needed]
    if unneeded:
        result = env.call("candidate_refine", {"node_id": "root", "exclude": [unneeded[0]["node_id"]],
                                               "predicate": "kind"})["structuredContent"]
        assert result["excluded"] == [unneeded[0]["node_id"]]
        assert result["candidates_after"] == result["candidates_before"] - 1
        again = env.call("candidate_refine", {"node_id": "root", "exclude": [unneeded[0]["node_id"]],
                                              "predicate": "kind"})["structuredContent"]
        assert again["excluded"] == []
    # Unknown registered metadata never proves a violation.
    assert not violates({"status": None, "scope": None, "unit": None}, "service", needs, "status")
    assert not violates({"status": None, "scope": None, "unit": None}, "service", needs, "scope")
    assert violates({"status": "archived", "scope": "archive", "unit": None}, "service", needs, "status")


def test_ldt_pagination_survives_exclusions(cases):
    case = cases[0]
    env = Env02(case, "layered_ldt", "PSCB")
    first = env.call("index_children", {"node_id": "root", "limit": 2})["structuredContent"]
    full = Env02(case, "layered", "PSCB").call("index_children", {"node_id": "root", "limit": 16})
    all_ids = [r["node_id"] for r in full["structuredContent"]["children"]]
    excluded = []
    for row, predicate in symbolic_exclusions(env.task, first["children"]):
        excluded += env.call("candidate_refine", {"node_id": "root", "exclude": [row], "predicate": predicate}
                             )["structuredContent"]["excluded"]
    second = env.call("index_children", {"node_id": "root", "limit": 2, "cursor": first["next_cursor"]})
    shown = [r["node_id"] for r in second["structuredContent"]["children"]]
    assert all(i in all_ids[2:4] for i in shown)


def test_epoch_roll_resets_state_and_invalidates_old_nodes():
    case = next(c for c in make_cases("test", 3303, 8, 128) if c["family"] == "epoch_roll")
    env = Env02(case, "layered_ldt", "SCB")
    after = case["hidden"]["epoch_roll"]["after_calls"]
    old_root = env.call("index_children", {"node_id": "root"})["structuredContent"]
    child = old_root["children"][0]["node_id"]
    for _ in range(after):
        env.call("index_children", {"node_id": "root"})
    assert env.rolled and env.catalog_epoch == case["hidden"]["epoch_roll"]["new_epoch"]
    stale = env.call("index_children", {"node_id": child})
    assert stale["isError"] and "previous catalog epoch" in stale["structuredContent"]["error"]
    assert env.stats["ldt_state_resets"] == 1


def test_judge_rejects_stale_forged_extra_and_wrong_abstention():
    cases = make_cases("test", 3303, 8, 128)
    roll = next(c for c in cases if c["family"] == "epoch_roll")
    env = Env02(roll, "search")
    refs = [env.call("resource_read", {"uri": u})["structuredContent"]["ref"] for u in roll["oracle"]["required_uris"][0]]
    while not env.rolled:
        env.call("catalog_search", {"query": "x"})
    stale = env.grade({"answer": roll["oracle"]["answer"], "evidence_refs": refs, "abstain": False})
    assert not stale["success"] and "invalid_stale_or_extra_evidence" in stale["reasons"]
    fresh = list(refs[:-1]) + [env.call("resource_read", {"uri": env.required_uris()[0][-1]})["structuredContent"]["ref"]]
    assert env.grade({"answer": env.expected_answer(), "evidence_refs": fresh, "abstain": False})["success"]
    assert not env.grade({"answer": env.expected_answer(), "evidence_refs": fresh + ["ev-forged"], "abstain": False})["success"]
    missing = next(c for c in cases if c["family"] == "missing")
    env = Env02(missing, "search")
    refs = [env.call("resource_read", {"uri": u})["structuredContent"]["ref"] for u in missing["oracle"]["required_uris"][0]]
    assert env.grade({"answer": None, "evidence_refs": refs, "abstain": True})["success"]
    assert not env.grade({"answer": 5, "evidence_refs": refs, "abstain": False})["success"]
    assert not env.grade({"answer": None, "evidence_refs": refs})["success"]


def test_malformed_tool_calls_are_errors(cases):
    env = Env02(cases[0], "layered_ldt", "SCB")
    for name, args in (("nope", {}), ("index_children", {}), ("index_children", {"node_id": "root", "limit": 99}),
                       ("resource_read", {"uri": 7}), ("candidate_refine", {"node_id": "root", "exclude": [],
                                                                              "predicate": "kind"})):
        assert env.call(name, args)["isError"]
    assert env.stats["tool_errors"] == 5


def test_composer_uses_public_reads_only(cases):
    composer = Composer(cases[1]["task"])
    needs = composer.needs()
    assert needs[0]["kind"] == "service" and needs[0]["name"] == cases[1]["task"]["plan"]["chains"][0]["anchor"]["name"]
    assert not composer.done() and composer.partial_submission()["abstain"]


def test_stdio_transport_matches_replay(cases):
    ep = run_episode(cases[0], arm="layered_ldt", layout="PSCB", policy=make_control("deterministic_rules"),
                     transport="stdio", stdio_options={"seed": 3303, "index": 0})
    assert ep["transport_replay_verified"] is True and ep["compute"]["replay_mismatches"] == 0


def test_encoder_has_fixed_shape_and_no_lexical_features(cases):
    env = Env02(cases[0], "layered", "SCB")
    composer = Composer(env.task)
    from MCP.mcp02.agent import branch_item
    rows = _root_children(env)
    obs = build_observation(env.task, composer, [dict(branch_item(r, 1), order=i) for i, r in enumerate(rows)],
                            0, 40, "layered")
    ids, segs, structs = FE.encode_observation(obs)
    assert all(len(r) == FE.SEQ_LEN for r in ids) and all(len(s) == FE.STRUCT_DIM for s in structs)
    assert {v for row in segs for v in row} <= {0, 1}
    ids, segs, _ = FE.encode_observation(obs, "match")
    assert {v for row in segs for v in row} <= {0, 1, 2, 3}
    need = FE.encode_text("need find team entity tm-3f2a", 32)
    cand = FE.encode_text("resource team entity tm-3f2a scope people", 40)
    seg = FE.segments(need, cand, "match")
    # Shared identity tokens are flagged; template glue ("need", "entity", "scope") never is.
    flagged = {t for t, s in zip([1] + need + [2] + cand, seg) if s in (2, 3)}
    assert FE.token_id("3f2a") in flagged and FE.token_id("team") in flagged
    assert FE.token_id("entity") not in flagged and FE.token_id("scope") not in flagged


def test_economics_helpers():
    assert wilson(0, 0) is None and wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]
    diff = paired_difference([1, 1, 0], [0, 1, 0])
    assert diff["mean"] == pytest.approx(1 / 3) and diff["wins"] == 1 and diff["losses"] == 0


class _ScriptedChat(BaseHTTPRequestHandler):
    script: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        text = self.script.pop(0)
        body = json.dumps({"model": "scripted", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_llm_controller_through_local_adapter():
    from MCP.local_model import LocalModel
    from MCP.mcp02.llm import ScriptedModel, run_llm_episode
    case = make_cases("test", 3303, 1, 96)[0]
    env = Env02(case, "search")
    refs = [env.call("resource_read", {"uri": u})["structuredContent"]["ref"] for u in case["oracle"]["required_uris"][0]]
    script = ["not json"] + [json.dumps({"tool": "resource_read", "arguments": {"uri": u}})
                             for u in case["oracle"]["required_uris"][0]]
    script.append(json.dumps({"tool": "finish", "arguments": {"answer": case["oracle"]["answer"],
                                                              "evidence_refs": refs, "abstain": False}}))
    _ScriptedChat.script = list(script)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = LocalModel(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", model="scripted", timeout=10)
        ep = run_llm_episode(case, arm="search", layout="", model=model, parameter_count=1000)
    finally:
        server.shutdown()
    assert ep["accepted"] and ep["llm"]["parse_errors"] == 1 and ep["llm"]["prompt_tokens"] == 100 * len(script)
    assert ep["compute"]["flops"] == 2 * 1000 * 110 * len(script)
    again = run_llm_episode(case, arm="search", layout="", model=ScriptedModel(ep["responses"]))
    assert again["results_sha256"] == ep["results_sha256"] and again["accepted"]


def _torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch(), reason="torch not installed")
def test_architectures_share_interface_and_measure_flops(cases):
    from MCP.mcp02.models import RouterPolicy, build, flops_per_candidate, parameter_counts
    configs = {"dense": {"arch": "dense", "d": 16},
               "transformer": {"arch": "transformer", "d": 16, "heads": 2, "layers": 2},
               "looped": {"arch": "looped", "d": 16, "heads": 2, "passes": 2},
               "trm": {"arch": "trm", "d": 16, "heads": 2, "inner": 1, "cycles": 1},
               "trm_ldt": {"arch": "trm_ldt", "d": 16, "heads": 2, "inner": 1, "cycles": 1},
               "hrm": {"arch": "hrm", "d": 16, "heads": 2, "low_steps": 1, "high_cycles": 1}}
    for name, config in configs.items():
        model = build(config)
        policy = RouterPolicy(name, model)
        ep = run_episode(cases[3], arm="layered_ldt", layout="PSCB", policy=policy, max_tool_calls=10)
        assert ep["compute"]["flops"] > 0 or name == "dense"
        assert ep["private_checks"]["certified_exclusions_containing_required"] == 0
        assert parameter_counts(model)["total"] > 0
    looped = build(configs["looped"])
    two = flops_per_candidate(looped)["flops"]
    looped.depth_override = 4
    assert flops_per_candidate(looped)["flops"] > 1.8 * two - 1
