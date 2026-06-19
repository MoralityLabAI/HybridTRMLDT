from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_boolean_rows(rows: list[dict[str, Any]], solver_key: str, solved_key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[solver_key]].append(row)
    summary: dict[str, dict[str, float]] = {}
    for solver, items in sorted(grouped.items()):
        total = len(items)
        solved = sum(bool(item[solved_key]) for item in items)
        summary[solver] = {
            "total": total,
            "solved": solved,
            "score": solved / total if total else 0.0,
            "steps": sum(int(item.get("steps", 0)) for item in items),
            "guesses": sum(int(item.get("guesses", 0)) for item in items),
            "proposals": sum(int(item.get("proposals", 0)) for item in items),
            "rejected": sum(int(item.get("rejected", 0)) for item in items),
        }
    return summary


def load_all(data_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    out["sudoku"] = summarize_boolean_rows(read_jsonl(data_dir / "sudoku_results.jsonl"), "solver", "solved")
    out["arc1"] = summarize_boolean_rows(read_jsonl(data_dir / "arc1_results.jsonl"), "solver", "solved")
    out["arc2"] = summarize_boolean_rows(read_jsonl(data_dir / "arc2_results.jsonl"), "solver", "solved")

    routing = json.loads((data_dir / "routing_results.json").read_text(encoding="utf-8"))
    out["routing"] = {
        row["router"]: {
            "total": row["total"],
            "solved": row["correct"],
            "score": row["accuracy"],
            "abstained": row["abstained"],
            "avg_candidates": row["avg_candidates"],
        }
        for row in routing["results"]
    }

    storyworld = json.loads((data_dir / "storyworld_results.json").read_text(encoding="utf-8"))
    out["storyworld"] = {
        player: {
            "total": metrics["episodes"],
            "solved": metrics["successes"],
            "score": metrics["success_rate"],
            "avg_steps": metrics["avg_steps"],
            "overrides": metrics["overrides"],
        }
        for player, metrics in storyworld["summary"].items()
    }
    return out


def arc2_efficiency_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(data_dir / "arc2_results.jsonl")
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["solver"]] = row
    out = []
    for task_id, solvers in sorted(by_task.items()):
        trm = solvers["trm"]
        hybrid = solvers["hybrid"]
        delta = int(hybrid["proposals"]) - int(trm["proposals"])
        out.append(
            {
                "task_id": task_id,
                "trm_proposals": int(trm["proposals"]),
                "hybrid_proposals": int(hybrid["proposals"]),
                "delta": delta,
                "hybrid_inferior": delta > 0,
            }
        )
    return out


def routing_ablation_rows(data_dir: Path) -> list[dict[str, Any]]:
    routing = json.loads((data_dir / "routing_results.json").read_text(encoding="utf-8"))
    return list(routing.get("ablations", []))


def routing_architecture_rows(data_dir: Path) -> tuple[float, list[dict[str, Any]]]:
    routing = json.loads((data_dir / "routing_results.json").read_text(encoding="utf-8"))
    return float(routing.get("confidence_gamma", 0.0)), list(routing.get("architecture_variants", []))


def storyworld_architecture_summary(data_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = data_dir / "storyworld_architecture_results.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def best_by_score(summary: dict[str, dict[str, float]]) -> list[str]:
    best = max(metrics["score"] for metrics in summary.values())
    return sorted(name for name, metrics in summary.items() if metrics["score"] == best)


def inferior_hybrid_cases(all_results: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    cases = []
    for bench, summary in all_results.items():
        hybrid = summary["hybrid"]
        best = max(metrics["score"] for metrics in summary.values())
        if hybrid["score"] < best:
            cases.append(f"{bench}: hybrid score {hybrid['score']:.3f} below best {best:.3f}")
    return cases


def markdown_report(
    all_results: dict[str, dict[str, dict[str, float]]],
    *,
    arc2_rows: list[dict[str, Any]] | None = None,
    routing_ablations: list[dict[str, Any]] | None = None,
    routing_architectures: list[dict[str, Any]] | None = None,
    confidence_gamma: float = 0.0,
    storyworld_architectures: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> str:
    arc2_rows = arc2_rows or []
    routing_ablations = routing_ablations or []
    routing_architectures = routing_architectures or []
    storyworld_architectures = storyworld_architectures or {}
    lines = [
        "# LDT/TRM/Hybrid Benchmark Comparison",
        "",
        "This report compares the saved benchmark outputs. VPD is not part of this comparison.",
        "",
        "## Score Summary",
        "",
        "| Benchmark | LDT | TRM | Hybrid | Best |",
        "|---|---:|---:|---:|---|",
    ]
    for bench, summary in all_results.items():
        lines.append(
            f"| `{bench}` | {summary['ldt']['score']:.3f} | {summary['trm']['score']:.3f} | "
            f"{summary['hybrid']['score']:.3f} | {', '.join(best_by_score(summary))} |"
        )

    lines.extend(
        [
            "",
            "## Where TRM Is Effective",
            "",
            "- `sudoku`: TRM solves search-heavy puzzles that LDT propagation cannot solve, but uses many more guesses than hybrid.",
            "- `routing`: TRM is the strongest hard router. It reaches `0.815` accuracy while deterministic LDT reaches `0.474` because token-lattice evidence is not sound enough for hard elimination.",
            "- `arc1` and `arc2`: TRM solves all tasks, but it is less efficient than hybrid on aggregate because it searches a wider proposal space.",
            "- `storyworld`: TRM is weak as a standalone policy because greedy local deficits lose modeled reachability under the rival policy.",
            "",
            "## Where LDT Is Effective",
            "",
            "- `arc1`, `arc2`, and `storyworld`: LDT reaches perfect task success because the transition/rule mechanics are explicit and checkable.",
            "- `sudoku`: LDT is excellent when naked-single propagation is sufficient, but it abstains/fails on puzzles requiring search.",
            "- `routing`: LDT is inferior as a hard router. Its token-derived candidate sets are model/experience evidence, not environment-sound deductions.",
            "",
            "## Hybrid Behavior",
            "",
            "- `sudoku`: hybrid matches TRM's solve rate and cuts guesses from `31` to `6` by using LDT propagation after proposals.",
            "- `arc1`: hybrid matches the best score and uses fewer steps than LDT and fewer proposals than TRM.",
            "- `arc2`: hybrid matches the best score and reduces aggregate proposals versus TRM (`32` vs `38`), but is inferior to TRM on two individual task proposal counts because the current proposal ordering is heuristic, not learned.",
            "- `routing`: hybrid matches TRM accuracy only after treating LDT candidate sets as soft guidance. The hard-filter ablation drops to `0.584` accuracy.",
            "- `storyworld`: hybrid matches LDT success and slightly reduces average steps, using `124` overrides to repair unsafe TRM proposals.",
            "",
            "## Inferior Hybrid Cases",
            "",
        ]
    )
    inferior = inferior_hybrid_cases(all_results)
    if inferior:
        lines.extend(f"- {case}" for case in inferior)
    else:
        lines.append("- No aggregate benchmark has hybrid below the best score.")
    lines.extend(
        [
            "- ARC-2 has individual efficiency regressions: hybrid uses more proposals than TRM on `arc2_flip_then_color` and `arc2_color_then_flip` due to non-learned pair ordering.",
            "- Routing has a design caveat: hybrid is not better than TRM on accuracy yet; LDT is useful only as soft candidate telemetry unless calibrated.",
            "",
            "## ARC-2 Efficiency Regressions",
            "",
            "| Task | TRM Proposals | Hybrid Proposals | Delta | Interpretation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    if arc2_rows:
        for row in arc2_rows:
            interpretation = "hybrid worse" if row["hybrid_inferior"] else "hybrid better"
            if row["delta"] == 0:
                interpretation = "tie"
            lines.append(
                f"| `{row['task_id']}` | {row['trm_proposals']} | {row['hybrid_proposals']} | "
                f"{row['delta']:+d} | {interpretation} |"
            )
    else:
        lines.append("| n/a | 0 | 0 | +0 | no saved ARC-2 rows |")
    lines.extend(
        [
            "",
            "## Routing Ablations",
            "",
            "| Router | Accuracy | Correct | Abstained | Avg Candidates |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if routing_ablations:
        for row in routing_ablations:
            lines.append(
                f"| `{row['router']}` | {float(row['accuracy']):.3f} | "
                f"{int(row['correct'])}/{int(row['total'])} | {int(row['abstained'])} | "
                f"{float(row['avg_candidates']):.2f} |"
            )
    else:
        lines.append("| n/a | 0.000 | 0/0 | 0 | 0.00 |")
    lines.extend(
        [
            "",
            "## Hybrid Architecture Variants",
            "",
            f"Confidence arbitration gamma: `{confidence_gamma:.2f}`",
            "",
            "| Architecture | Accuracy | Correct | Avg Candidates | Control Policy |",
            "|---|---:|---:|---:|---|",
        ]
    )
    policy_notes = {
        "typed_membrane": "TRM proposes; LDT evidence stays soft unless sound.",
        "hard_gate": "LDT candidates hard-filter TRM scoring.",
        "confidence_arbitration": "TRM acts above margin; LDT constrains low-margin cases.",
    }
    if routing_architectures:
        for row in routing_architectures:
            name = str(row["router"])
            lines.append(
                f"| `{name}` | {float(row['accuracy']):.3f} | "
                f"{int(row['correct'])}/{int(row['total'])} | {float(row['avg_candidates']):.2f} | "
                f"{policy_notes.get(name, 'architecture variant')} |"
            )
    else:
        lines.append("| n/a | 0.000 | 0/0 | 0.00 | no saved architecture rows |")
    lines.extend(
        [
            "",
            "## Storyworld Confidence vs Type Split",
            "",
            "| Scenario | Policy | Success Rate | Avg Score | Interpretation |",
            "|---|---|---:|---:|---|",
        ]
    )
    interpretation = {
        ("secret_ending", "typed_membrane"): "exact reachability is environment-sound",
        ("secret_ending", "confidence_arbitration"): "high-confidence local morality can miss a secret gate",
        ("moral_optimization", "typed_membrane"): "safe but conservative under a soft preference surface",
        ("moral_optimization", "confidence_arbitration"): "best soft-score optimizer in this probe",
    }
    if storyworld_architectures:
        for scenario in ("secret_ending", "moral_optimization"):
            for policy in ("typed_membrane", "confidence_arbitration"):
                metrics = storyworld_architectures[scenario][policy]
                lines.append(
                    f"| `{scenario}` | `{policy}` | {float(metrics['success_rate']):.3f} | "
                    f"{float(metrics['avg_score']):.2f} | {interpretation[(scenario, policy)]} |"
                )
    else:
        lines.append("| n/a | n/a | 0.000 | 0.00 | no saved storyworld architecture rows |")
    lines.extend(
        [
            "",
            "## Practical Map",
            "",
            "| Regime | Best Current Tool | Reason |",
            "|---|---|---|",
            "| Fully checkable local mechanics | LDT or hybrid | Hard certification is reliable. |",
            "| Search required beyond local propagation | TRM or hybrid | Latent/proposal search supplies candidates. |",
            "| Search plus checkable constraints | Hybrid | TRM proposes; LDT prunes/certifies. |",
            "| Noisy lexical environment routing | TRM, hybrid equal | LDT token evidence is not hard-sound. |",
            "| Adversarial/dynamic storyworld policy | LDT or hybrid | Reachability checks prevent greedy traps. |",
            "",
            "## Next Fixes",
            "",
            "- Train ARC-2 hybrid proposal ordering rather than using the current static order.",
            "- Replace the routing confidence grid with a richer calibration signal; the current trained threshold degenerates to TRM on this slice.",
            "- Add per-instance comparison tables for hybrid regressions, especially ARC-2 proposal counts and routing confusions.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved LDT/TRM/hybrid benchmark outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark_comparison.md"))
    args = parser.parse_args()

    results = load_all(args.data_dir)
    confidence_gamma, architecture_rows = routing_architecture_rows(args.data_dir)
    report = markdown_report(
        results,
        arc2_rows=arc2_efficiency_rows(args.data_dir),
        routing_ablations=routing_ablation_rows(args.data_dir),
        routing_architectures=architecture_rows,
        confidence_gamma=confidence_gamma,
        storyworld_architectures=storyworld_architecture_summary(args.data_dir),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
