from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.benchmarks.control_harness_bench import (
    ARCHITECTURE_NOTES,
    run_control_harness_benchmark,
    summary_markdown,
    write_experiment_bundle,
)
from research_gym.discovery.project_inventory import scan_projects, write_inventory
from research_gym.discovery.application_map import application_map_markdown


def training_notes(payload: dict[str, object], inventory: dict[str, object]) -> str:
    summary = payload["summary"]
    routes = payload["skill_routes"]
    assert isinstance(summary, dict)
    assert isinstance(routes, dict)
    lines = [
        "# Hybrid Control Harness Experiment Notes",
        "",
        "## Scope",
        "",
        "The project scan is real and local. The benchmark cases are deterministic source-inspired proxies",
        "built from the control interfaces found during the scan; they do not execute or score the neighboring",
        "projects themselves. This keeps the experiment dependency-free and prevents proxy results from being",
        "misreported as native game performance.",
        "",
        f"- projects scanned: `{inventory['scanned_project_count']}`",
        f"- relevant candidates retained: `{inventory['candidate_count']}`",
        f"- train cases per application: `{payload['n_train_per_application']}`",
        f"- held-out cases per application: `{payload['n_eval_per_application']}`",
        f"- random seed: `{payload['seed']}`",
        f"- selected confidence margin: `{payload['tuned']['gamma']:.2f}`",
        f"- selected counterfactual beam width: `{payload['tuned']['beam_width']}`",
        "",
        "## Source-inspired applications",
        "",
        "- `GPTStoryworld`: exact secret-route reachability versus soft morality and opponent-model scores.",
        "- `AI_Diplomacy`: exact order legality versus conditional coalition/betrayal forecasts.",
        "- `SmallControlHarness`: attested provenance versus confident but semantically re-anchored evidence.",
        "- `BitVMArena`: exact protocol compatibility versus replay-derived control-profile rankings.",
        "- `StoryForge`: exact choice gates versus model-sound prisoner-dilemma best responses.",
        "- `TheySing`: exact channel/campaign permissions versus modeled persuasion and treaty response.",
        "- `Blighted Galaxy`: exact strategic-tick legality versus replay-derived action value.",
        "- `CodexGameStudio`: exact agent scope versus learned task-to-specialist routing.",
        "",
        "## Structures tested",
        "",
    ]
    for policy, description in ARCHITECTURE_NOTES.items():
        lines.append(f"- `{policy}`: {description}")

    lines.extend(["", "## Held-out results", ""])
    for policy, metrics in summary.items():
        assert isinstance(metrics, dict)
        lines.append(
            f"- `{policy}` accuracy={float(metrics['accuracy']):.3f} "
            f"utility={float(metrics['mean_utility']):.3f} "
            f"unsafe_rate={float(metrics['unsafe_rate']):.3f} "
            f"cost={float(metrics['mean_cost']):.2f}"
        )

    lines.extend(["", "## Learned skill routing", ""])
    for skill, policy in sorted(routes.items()):
        lines.append(f"- `{skill}` -> `{policy}`")

    lines.extend(
        [
            "",
            "## Design decisions",
            "",
            "- Environment-derived candidate exclusions are the only hard constraints.",
            "- Model- and experience-derived candidates are visible to hard-gate ablations but not promoted by typed policies.",
            "- Hyperparameters and skill routes are selected on train cases and reported on a disjoint deterministic eval split.",
            "- The selection objective penalizes unsafe actions and deliberation cost in addition to rewarding utility and exact choice.",
            "- Oracle utility is used only for calibration/evaluation, never as an input to an architecture at decision time.",
            "",
            "## Native follow-ups",
            "",
            "- Wrap `GPTStoryworld/storyworld/env/diplomacy_env.py` action normalization and turn traces directly.",
            "- Add a controller adapter around `SmallControlHarness` attestation decisions and shared registry memory.",
            "- Route `BitVMArena` settlement profiles by trajectory phase, then certify protocol compatibility before funding.",
            "- Use `StoryworldTRM` SWMD PICK traces to train the skill router instead of selecting from synthetic score cards.",
            "- Apply typed delegation to `CodexGameStudio` skill scopes, then optimize consultation cost on real task traces.",
            "- Insert the membrane into `TheySing` bridge policies and compare hard/soft/graduated treaty scenarios.",
            "- Wrap `Blighted Galaxy` replay ticks to test whether beam certification survives longer horizons.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan local game environments and benchmark hybrid control structures.")
    parser.add_argument("--projects-root", type=Path, default=Path(r"C:\projects"))
    parser.add_argument("--minimum-score", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=48)
    parser.add_argument("--n-eval", type=int, default=48)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--inventory-out",
        type=Path,
        default=Path("data/benchmarks/local_game_env_inventory.json"),
    )
    parser.add_argument(
        "--inventory-report",
        type=Path,
        default=Path("reports/local_game_env_inventory.md"),
    )
    parser.add_argument(
        "--application-map-report",
        type=Path,
        default=Path("reports/hybrid_application_map.md"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/control_harness_results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/control_harness_bench.md"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/control_harness_structures"),
    )
    args = parser.parse_args()

    inventory = scan_projects(args.projects_root, minimum_score=args.minimum_score)
    write_inventory(inventory, args.inventory_out, args.inventory_report)
    args.application_map_report.parent.mkdir(parents=True, exist_ok=True)
    args.application_map_report.write_text(application_map_markdown(), encoding="utf-8")

    payload = run_control_harness_benchmark(
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = summary_markdown(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_experiment_bundle(payload, args.experiment_dir, notes=training_notes(payload, inventory))
    print(report)


if __name__ == "__main__":
    main()
