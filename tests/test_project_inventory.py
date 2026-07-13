from pathlib import Path

from research_gym.discovery.project_inventory import inventory_markdown, scan_projects
from research_gym.discovery.application_map import application_map_markdown


def test_project_inventory_finds_game_environment_and_skips_dependency_noise(tmp_path: Path):
    game = tmp_path / "TinyGame"
    game.mkdir()
    (game / "README.md").write_text(
        "A storyworld game environment with an action policy and control harness.",
        encoding="utf-8",
    )
    dependencies = game / "node_modules"
    dependencies.mkdir()
    (dependencies / "README.md").write_text("diplomacy arena verifier", encoding="utf-8")
    unrelated = tmp_path / "TaxForms"
    unrelated.mkdir()
    (unrelated / "README.md").write_text("Deterministic accounting forms.", encoding="utf-8")

    payload = scan_projects(tmp_path, minimum_score=5)

    assert payload["scanned_project_count"] == 2
    assert [candidate["name"] for candidate in payload["candidates"]] == ["TinyGame"]
    assert "TinyGame" in inventory_markdown(payload)


def test_application_map_keeps_native_and_proxy_status_explicit():
    report = application_map_markdown()

    assert "TheySing" in report
    assert "SmallControlHarness" in report
    assert "proxy benchmarked" in report
    assert "native adapter next" in report
