from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.adapters.verifiers_v1 import VerifiersV1Integration
from research_gym.benchmarks.sequencer_control_bench import (
    SequencerBenchmarkConfig,
    run_sequencer_control_benchmark,
    summary_markdown,
    v1_replay_rows,
)
from research_gym.envs.routing import (
    DEFAULT_ROUTING_ENVS,
    DEFAULT_TESSERACT_DATA_ROOT,
    load_tesseract_routing_examples,
)


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def training_notes(payload: dict[str, object], registration: dict[str, object]) -> str:
    fit = payload["fit_receipt"]
    summary = payload["summary"]
    comparisons = payload["comparisons"]
    integration = payload["verifiers_v1_integration"]
    primary = next(item for item in comparisons if item["control"] == "global_signed")
    lineage = next(item for item in comparisons if item["control"] == "lineage_only")
    return "\n".join(
        [
            "# Sequencer Control-Math Benchmark Notes",
            "",
            "## Registration",
            "",
            f"- registered on: `{registration['registered_on']}`",
            f"- primary endpoint: `{registration['primary_endpoint']}`",
            f"- protocol SHA-256: `{payload['protocol_sha256']}`",
            f"- calibration SHA-256: `{fit['calibration_sha256']}`",
            f"- evaluation SHA-256: `{payload['eval_sha256']}`",
            "",
            "## Design",
            "",
            "- Candidate skill sequences are proposal-only TRM, deduction-only LDT, and typed propose/certify hybrid.",
            "- Sequence scores are fitted on calibration episodes only and frozen before evaluation is passed in.",
            "- Two calibration folds define independent transport paths; loop disagreement and orientation feed the",
            "  conservative RSITopology control bound.",
            "- Full control uses global signed transfer only inside the frozen bound; otherwise it invokes a local",
            "  calibrated section. Lineage-only deliberately omits the loop/orientation term.",
            "- Families are macro-weighted so routing volume cannot dominate Sudoku, ARC, or storyworld objectives.",
            "- Confidence intervals use paired family-stratified hierarchical context-cluster bootstrap resampling.",
            "  P-values use context-cluster sign flips and Holm correction across three registered controls.",
            "",
            "## Primary result",
            "",
            f"- global signed macro utility: `{summary['global_signed']['macro_utility']:.6f}`",
            f"- full control-math macro utility: `{summary['control_math']['macro_utility']:.6f}`",
            f"- paired delta: `{primary['macro_utility_delta']:+.6f}`",
            f"- 95% bootstrap CI: `[{primary['utility_delta_ci_95'][0]:+.6f}, "
            f"{primary['utility_delta_ci_95'][1]:+.6f}]`",
            f"- paired sign-flip p: `{primary['paired_sign_flip_p_value']:.6g}`",
            f"- Holm-adjusted p: `{primary['holm_adjusted_p_value']:.6g}`",
            f"- registered context section rate: `{payload['section_rate']:.6f}`",
            "",
            "## Negative result",
            "",
            f"- control math versus lineage-only delta: `{lineage['macro_utility_delta']:+.6f}`",
            f"- clustered sign-flip p: `{lineage['paired_sign_flip_p_value']:.6g}`",
            "- This run does not establish incremental utility from holonomy/orientation beyond lineage-only control.",
            "",
            "## Scope",
            "",
            "- Sudoku uses rule-preserving automorphisms of the local 4x4 puzzles.",
            "- ARC-1/ARC-2 are balanced procedural tasks over the local primitive rule library; these are ARC-style,",
            "  not official ARC leaderboard tasks.",
            "- ARC tasks are retained only when all candidate sequences solve them, isolating efficiency rather",
            "  than solver coverage.",
            "- Routing uses real normalized Tesseract trajectories with model-train, sequencer-calibration, and eval",
            "  partitions.",
            "- Storyworld uses disjoint finite-state starts under secret-ending and moral-optimization objectives.",
            "- No neural model was trained, infused, promoted, or edited. The measured object is the sequencer.",
            "- Error-budget sweeps are post-registration sensitivity analysis, not alternate primary endpoints.",
            "",
            "## Verifiers v1",
            "",
            "The saved eval rows are exported into the local Verifiers 0.1.14 v1 Taskset/Harness package as an",
            "LLM-free replay environment. Package contract tests validate its shape; exact runtime status is",
            "recorded separately and is not claimed as a native model evaluation.",
            "",
            f"- target Verifiers: `{integration['target_version']}`",
            f"- installed Verifiers: `{integration['installed_version']}`",
            f"- installed Verifiers compatible: `{integration['installed_version_compatible']}`",
            f"- minimum uv: `{integration['minimum_uv_version']}`",
            f"- installed uv: `{integration['installed_uv_version']}`",
            f"- installed uv compatible: `{integration['installed_uv_compatible']}`",
            f"- native Windows Prime blocked: `{integration['native_windows_prime_blocked']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RSITopology control math on hybrid skill sequencers.")
    parser.add_argument("--config", type=Path, default=Path("configs/sequencer_control_v1.json"))
    parser.add_argument("--routing-root", type=Path, default=DEFAULT_TESSERACT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/sequencer_control_results.json"))
    parser.add_argument(
        "--episodes-out",
        type=Path,
        default=Path("data/benchmarks/sequencer_control_episodes.jsonl"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/sequencer_control_bench.md"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/sequencer_control_math"),
    )
    parser.add_argument(
        "--v1-taskset-out",
        type=Path,
        default=Path("environments/hybrid_sequencer_v1/data/replay_tasks.jsonl"),
    )
    args = parser.parse_args()

    registration = json.loads(args.config.read_text(encoding="utf-8"))
    config = SequencerBenchmarkConfig(**registration["benchmark"])
    routing_examples = load_tesseract_routing_examples(
        args.routing_root,
        DEFAULT_ROUTING_ENVS,
        max_per_env=config.routing_max_per_env,
    )
    payload, rows = run_sequencer_control_benchmark(routing_examples, config)
    payload["registration"] = registration
    payload["routing_source"] = {
        "root": str(args.routing_root),
        "environments": list(DEFAULT_ROUTING_ENVS),
        "loaded_examples": len(routing_examples),
    }
    integration = VerifiersV1Integration()
    payload["verifiers_v1_integration"] = integration.to_jsonable()

    _write_json(payload, args.out)
    _write_jsonl(rows, args.episodes_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(summary_markdown(payload), encoding="utf-8")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(payload, args.experiment_dir / "results.json")
    _write_jsonl(rows, args.experiment_dir / "episodes.jsonl")
    (args.experiment_dir / "training_notes.md").write_text(
        training_notes(payload, registration), encoding="utf-8"
    )
    _write_jsonl(v1_replay_rows(rows), args.v1_taskset_out)
    print(summary_markdown(payload))


if __name__ == "__main__":
    main()
