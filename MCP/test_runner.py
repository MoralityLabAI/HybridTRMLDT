"""Episode-level tests for policy errors, measured boundaries and protocol replay."""
import copy

import pytest

from MCP.environment import make_cases
from MCP.local_model import LocalModelError
from MCP.run import ARCHITECTURES, model_observation, run_episode


def case():
    return make_cases(split="test", seed=17, n=1, catalog_size=8, depth=2)[0]


def run_reference(architecture="search", **kwargs):
    return run_episode(case(), architecture=architecture, controller="reference",
                       catalog_size=8, depth=2, **kwargs)


class RecordedActions:
    def __init__(self, actions):
        self.actions = iter(actions)

    def step(self, messages):
        action = next(self.actions)
        return {"action": action, "response_text": "malformed" if action is None else __import__("json").dumps(action),
                "parse_error": "test parse error" if action is None else None,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0},
                "latency_seconds": 0.01}


def recorded_success():
    reference = run_reference()
    return [{"tool": e["tool"], "arguments": e["arguments"]} for e in reference["history"]] + [
        {"tool": "finish", "arguments": reference["submission"]}]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_stdio_episode_has_same_evidence_as_inprocess(architecture):
    direct = run_reference(architecture)
    stdio = run_reference(architecture, transport="stdio", stdio_options={"split": "test", "seed": 17, "n": 1})
    assert direct["accepted"] and stdio["accepted"]
    assert direct["handoff"] == stdio["handoff"]
    assert stdio["transport_replay_verified"]


def test_parser_repair_attempt_stays_in_usage():
    actions = recorded_success()
    result = run_episode(case(), architecture="search", controller="local", catalog_size=8, depth=2,
                         local_model=RecordedActions([None] + actions))
    assert result["accepted"]
    assert len(result["model_calls"]) == len(actions) + 1
    assert result["model_input_utf8_bytes"] > 0


def test_unknown_tool_is_counted_and_recoverable_over_stdio():
    actions = recorded_success()
    result = run_episode(case(), architecture="search", controller="local", catalog_size=8, depth=2,
                         local_model=RecordedActions([{"tool": "unknown", "arguments": {}}] + actions),
                         transport="stdio", stdio_options={"split": "test", "seed": 17, "n": 1})
    assert result["accepted"]
    assert result["history"][0]["protocol_error"]["code"] == -32602
    assert result["tool_calls"] == len(actions)


def test_small_packet_budget_fails_without_truncating_evidence():
    result = run_reference(handoff_budget_bytes=1)
    assert result["grade"]["success"]
    assert not result["accepted"]
    assert result["handoff_budget_exceeded"]
    assert result["handoff"]["evidence"]


def test_exhausted_turn_budget_remains_failure():
    result = run_reference(max_steps=1)
    assert not result["accepted"]
    assert result["failure"] == "step_budget_exhausted"
    assert result["tool_calls"] == 1


def test_failed_local_request_preserves_unknown_usage():
    class Unreachable:
        def step(self, messages):
            raise LocalModelError("unreachable local test endpoint", {"usage": {}, "latency_seconds": 0.1})
    result = run_episode(case(), architecture="search", controller="local", catalog_size=8, depth=2,
                         local_model=Unreachable())
    assert not result["accepted"]
    assert result["failure"] == "local_model_error"
    assert len(result["model_calls"]) == 1
    assert result["model_calls"][0]["usage"] == {}


def test_one_result_representation_reaches_model():
    payload = {"answer": 42}
    original = {"content": [{"type": "text", "text": '{"answer":42}'}], "structuredContent": payload, "isError": False}
    copy_before = copy.deepcopy(original)
    assert model_observation(original) == {"isError": False, "result": payload}
    assert original == copy_before


def test_forged_final_answer_does_not_bypass_evidence():
    forged = recorded_success()[-1]
    forged["arguments"]["evidence_refs"] = []
    result = run_episode(case(), architecture="search", controller="local", catalog_size=8, depth=2,
                         local_model=RecordedActions([forged]))
    assert not result["accepted"]
    assert not result["grade"]["evidence_valid"]


def test_later_transport_mismatch_clears_verified_flag(monkeypatch):
    from MCP.environment import RetrievalEnv
    from MCP import transport

    class CorruptClient:
        def __init__(self, **kwargs):
            self.env = RetrievalEnv(case(), "search", catalog_size=8, depth=2)
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def initialize(self):
            return {"_meta": {"publicTask": self.env.task}}

        def tools(self):
            return self.env.tools()

        def call(self, name, arguments):
            result = self.env.call(name, arguments)
            self.calls += 1
            if self.calls == 2:
                result["structuredContent"]["injected"] = True
            return result

    monkeypatch.setattr(transport, "StdioClient", CorruptClient)
    result = run_reference(transport="stdio")
    assert result["failure"] == "transport_replay_mismatch"
    assert not result["transport_replay_verified"]
    assert not result["accepted"]
