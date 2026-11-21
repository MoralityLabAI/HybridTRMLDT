from MCP.metrics import aggregate_usage, byte_size, summarize


def test_missing_usage_stays_unknown_and_keeps_known_spend():
    usage = aggregate_usage([
        {"usage": {"prompt_tokens": 120, "completion_tokens": 30, "cached_tokens": 0}},
        {"usage": {}},
    ])
    assert usage["prompt_tokens"] is None
    assert usage["prompt_tokens_known_sum"] == 120
    assert usage["prompt_tokens_missing_calls"] == 1
    assert not usage["usage_complete"]


def test_unicode_bytes_are_exact_and_not_tokens():
    assert byte_size({"x": "λ"}) == len('{"x":"λ"}'.encode("utf-8"))


def test_failed_episode_is_charged_to_success_denominator():
    common = dict(controller="local", architecture="search", catalog_size=32, depth=3,
                  tool_calls=1, catalog_utf8_bytes=10, model_input_utf8_bytes=100,
                  tool_result_utf8_bytes=50, handoff_utf8_bytes=20,
                  handoff_budget_exceeded=False, elapsed_seconds=1,
                  local_inference_seconds=0.5, transport_replay_verified=True)
    good = dict(common, accepted=True, grade={"success": True},
                model_calls=[{"usage": {"prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 0}}])
    bad = dict(common, accepted=False, grade={"success": False},
               model_calls=[{"usage": {"prompt_tokens": 200, "completion_tokens": 80, "cached_tokens": 50}}])
    result = summarize([good, bad])["arms"][0]
    assert result["prompt_tokens_per_accepted_task"] == 300
    assert result["completion_tokens_per_accepted_task"] == 100
    assert result["success_rate"] == 0.5
    assert result["failures"] == 1


def test_reference_runs_do_not_invent_token_measurements():
    usage = aggregate_usage([])
    assert usage["model_calls"] == 0
    assert usage["prompt_tokens"] is None
    assert not usage["usage_complete"]
