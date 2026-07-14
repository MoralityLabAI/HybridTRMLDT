from pathlib import Path

from research_gym.adapters.verifiers_v1 import VerifiersV1Integration, _version_tuple


def test_version_tuple_handles_dev_releases():
    assert _version_tuple("0.1.14") == (0, 1, 14)
    assert _version_tuple("0.1.15.dev3") == (0, 1, 15)


def test_v1_integration_commands_and_package_contract():
    integration = VerifiersV1Integration()
    payload = integration.to_jsonable()

    assert integration.package_complete or not (integration.env_path / "data/replay_tasks.jsonl").exists()
    assert payload["target_version"] == "0.1.14"
    assert payload["minimum_uv_version"] == "0.11.1"
    assert isinstance(payload["installed_uv_compatible"], bool)
    assert integration.prime_eval_command()[:4] == ["prime", "eval", "run", "hybrid-sequencer-v1"]
    assert "verifiers==0.1.14" in integration.exact_release_smoke_command()


def test_environment_source_uses_v1_taskset_harness_shape():
    source = Path("environments/hybrid_sequencer_v1/hybrid_sequencer_v1.py").read_text(encoding="utf-8")
    metadata = Path("environments/hybrid_sequencer_v1/pyproject.toml").read_text(encoding="utf-8")

    assert "import verifiers.v1 as vf" in source
    assert "def load_taskset(" in source
    assert "def load_harness(" in source
    assert "def load_environment(config: vf.EnvConfig)" in source
    assert '"verifiers>=0.1.14,<0.2"' in metadata
    assert '"airis_das"' in source
    assert '"data/airis_rules.json"' in metadata
