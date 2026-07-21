from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_gym.benchmarks.rlm_architecture_neighborhood import architecture_manifest
from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import bench_rlm_architecture_neighborhood_v0_1 as runner


class _ProviderError(Exception):
    def __init__(self, status_code=None, code=None):
        self.status_code = status_code
        self.body = {"error": {"code": code, "type": "test_error"}}


@pytest.mark.parametrize(
    ("status_code", "code"),
    [(401, None), (403, None), (400, "invalid_api_key"), (404, "model_not_found")],
)
def test_global_provider_errors_are_fail_fast(status_code, code) -> None:
    assert runner._is_global_provider_error(_ProviderError(status_code, code))


def test_transient_provider_error_is_not_global() -> None:
    assert not runner._is_global_provider_error(_ProviderError(429, "rate_limit"))


def test_successor_preserves_v0_tasks_architectures_and_claim_endpoints() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "rlm_architecture_neighborhood_v0_1.json").read_text(
            encoding="utf-8"
        )
    )
    v0 = json.loads(
        (root / "configs" / "rlm_architecture_neighborhood_v0.json").read_text(
            encoding="utf-8"
        )
    )
    hashes = {row["architecture_id"]: row["architecture_hash"] for row in architecture_manifest()}
    assert config["task_suite"]["sha256"] == v0["task_suite"]["sha256"]
    assert config["official_rlm_source"] == v0["official_rlm_source"]
    assert config["architecture_order"] == v0["architecture_order"]
    assert config["restricted_repl"] == v0["restricted_repl"]
    assert config["registered_endpoints"] == v0["registered_endpoints"]
    assert config["preserved_from_v0"]["architecture_hashes"] == hashes
    assert config["runtime"]["model"] == "gpt-4.1-mini"


def test_registration_hash_matches_successor_config_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "rlm_architecture_neighborhood_v0_1.json"
    registration_path = root / "configs" / "rlm_architecture_neighborhood_v0_1_registration.json"
    if not registration_path.exists():
        return
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    assert registration["config_sha256"] == canonical_file_sha256(config_path)
    assert registration["task_suite_sha256"] == json.loads(
        config_path.read_text(encoding="utf-8")
    )["task_suite"]["sha256"]


def test_capability_gate_precedes_any_task_cell(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "rlm_architecture_neighborhood_v0_1.json").read_text(
            encoding="utf-8"
        )
    )
    task_calls = []

    def fail_gate(*_args, **_kwargs):
        raise RuntimeError("capability failed")

    monkeypatch.setattr(runner, "_provider_capability_gate", fail_gate)
    monkeypatch.setattr(runner, "run_architecture", lambda *_args: task_calls.append(True))
    with pytest.raises(RuntimeError, match="capability failed"):
        runner.run(config, "config-hash", tmp_path)
    assert task_calls == []
    assert not (tmp_path / "trajectories").exists()
