from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.benchmarks.routing_bench import run_routing_benchmark, summary_markdown, write_experiment_bundle
from research_gym.envs.routing import DEFAULT_ROUTING_ENVS, DEFAULT_TESSERACT_DATA_ROOT, load_tesseract_routing_examples


def training_notes(payload: dict[str, object], data_root: Path, max_per_env: int) -> str:
    lines = [
        "# Routing Training Notes",
        "",
        "Task: environment pointer routing from prompt text to local environment ID.",
        "",
        "Models:",
        "",
        "- `ldt`: explicit token lattice router. Tokens refine candidate environment sets.",
        "- `trm`: dependency-light lexical TRM analogue, mirroring Tesseract's TF-IDF router objective.",
        "- `hybrid`: soft LDT candidate telemetry plus TRM scoring.",
        "- `hybrid_hard_filter`: ablation that forces LDT candidates as hard filters before TRM scoring.",
        "",
        "Data:",
        "",
        f"- data root: `{data_root}`",
        f"- max records per env: `{max_per_env}`",
        f"- train examples: `{payload['train_size']}`",
        f"- test examples: `{payload['test_size']}`",
        f"- envs: `{', '.join(payload['envs'])}`",
        "",
        "This run does not train QLoRA adapters or use VPD. It isolates router behavior.",
        "",
    ]
    for result in payload["results"]:
        lines.append(
            f"- `{result['router']}` accuracy={float(result['accuracy']):.3f} "
            f"correct={result['correct']}/{result['total']} abstained={result['abstained']} "
            f"avg_candidates={float(result['avg_candidates']):.2f}"
        )
    for result in payload.get("ablations", []):
        lines.append(
            f"- `{result['router']}` accuracy={float(result['accuracy']):.3f} "
            f"correct={result['correct']}/{result['total']} abstained={result['abstained']} "
            f"avg_candidates={float(result['avg_candidates']):.2f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate LDT, TRM, and hybrid env-pointer routers.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TESSERACT_DATA_ROOT)
    parser.add_argument("--max-per-env", type=int, default=80)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/routing_results.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/routing_bench.md"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/routing_env_pointer"))
    args = parser.parse_args()

    examples = load_tesseract_routing_examples(args.data_root, DEFAULT_ROUTING_ENVS, max_per_env=args.max_per_env)
    if not examples:
        raise RuntimeError(f"No routing examples found under {args.data_root}")

    payload = run_routing_benchmark(examples, train_ratio=args.train_ratio, seed=args.seed)
    payload["data_root"] = str(args.data_root)
    payload["max_per_env"] = args.max_per_env
    payload["seed"] = args.seed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = summary_markdown(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_experiment_bundle(payload, args.experiment_dir, notes=training_notes(payload, args.data_root, args.max_per_env))
    print(report)


if __name__ == "__main__":
    main()
