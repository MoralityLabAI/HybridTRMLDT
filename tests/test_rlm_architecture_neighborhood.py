from __future__ import annotations

import json
from pathlib import Path

from research_gym.integrity import canonical_file_sha256
from research_gym.benchmarks.rlm_architecture_neighborhood import (
    answer_matches,
    architecture_manifest,
    materialize_tasks,
    pareto_frontier,
    split_prompt,
    trajectory_metrics,
    usage_totals,
)


def test_task_suite_is_deterministic_and_contains_four_distinct_families() -> None:
    left = materialize_tasks()
    right = materialize_tasks()
    assert left == right
    assert left["task_count"] == 4
    assert len({task["family"] for task in left["tasks"]}) == 4
    for task in left["tasks"]:
        assert task["expected_answer"]
        assert task["prompt_lines"] > 600


def test_architecture_manifest_spans_local_rlm_neighborhood() -> None:
    rows = {row["architecture_id"]: row for row in architecture_manifest()}
    assert set(rows) == {
        "direct_full_context",
        "map_reduce_4",
        "rlm_depth_1",
        "rlm_depth_2",
    }
    assert rows["rlm_depth_1"]["distance_from_rlm_depth_1"] == 0
    assert rows["rlm_depth_2"]["distance_from_rlm_depth_1"] == 1
    assert len({row["architecture_hash"] for row in rows.values()}) == 4


def test_answer_match_is_token_bounded() -> None:
    assert answer_matches("KX-731-OMEGA", "KX-731-OMEGA")
    assert answer_matches("Answer: KX-731-OMEGA.", "KX-731-OMEGA")
    assert not answer_matches("KX-731-OMEGAX", "KX-731-OMEGA")


def test_map_chunks_cover_context_once() -> None:
    prompt = materialize_tasks(filler_count=12)["tasks"][0]["prompt"]
    chunks = split_prompt(prompt, 4)
    assert len(chunks) == 4
    combined = "\n".join(chunk.split("CONTEXT_START\n", 1)[1] for chunk in chunks)
    for line in prompt.split("CONTEXT_START\n", 1)[1].splitlines()[:-1]:
        assert combined.count(line) == 1


def test_usage_and_nested_trajectory_metrics() -> None:
    usage = {
        "model_usage_summaries": {
            "root": {"total_calls": 2, "total_input_tokens": 100, "total_output_tokens": 20},
            "leaf": {"total_calls": 1, "total_input_tokens": 30, "total_output_tokens": 5},
        }
    }
    assert usage_totals(usage)["total_tokens"] == 155
    metadata = {
        "iterations": [
            {
                "code_blocks": [
                    {
                        "result": {
                            "stderr": "",
                            "rlm_calls": [
                                {"metadata": {"iterations": [{"code_blocks": []}]}},
                                {"metadata": None},
                            ],
                        }
                    }
                ]
            }
        ]
    }
    metrics = trajectory_metrics(metadata)
    assert metrics["iterations"] == 2
    assert metrics["code_blocks"] == 1
    assert metrics["subcalls"] == 2
    assert metrics["recursive_subcalls"] == 1
    assert metrics["maximum_observed_depth"] == 1


def test_pareto_frontier_does_not_scalarize_tradeoffs() -> None:
    rows = [
        {"architecture_id": "accurate", "accuracy": 1.0, "total_tokens": 100, "execution_time": 10.0},
        {"architecture_id": "cheap", "accuracy": 0.5, "total_tokens": 10, "execution_time": 1.0},
        {"architecture_id": "dominated", "accuracy": 0.5, "total_tokens": 120, "execution_time": 12.0},
    ]
    assert pareto_frontier(rows) == ["accurate", "cheap"]


def test_sealed_task_bundle_and_registration_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    tasks_path = root / "data" / "benchmarks" / "rlm_architecture_neighborhood_v0_tasks.json"
    registration_path = root / "configs" / "rlm_architecture_neighborhood_v0_registration.json"
    if not tasks_path.exists() or not registration_path.exists():
        return
    assert json.loads(tasks_path.read_text(encoding="utf-8")) == materialize_tasks()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    config_path = root / registration["config_path"]
    assert canonical_file_sha256(config_path) == registration["config_sha256"]
    assert registration["provider_outcomes_observed"] is False


def test_resource_wrapper_accepts_explicit_python_interpreter() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts" / "run_lsa_kappa_surface_phase.ps1").read_text(
        encoding="utf-8"
    )
    assert '[string]$PythonExe = "python"' in wrapper
    assert 'Start-Process -FilePath $PythonPath' in wrapper
