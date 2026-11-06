"""Exact ARC pass@2 scoring and independently sourced logic-verdict aggregation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from Kimilings.index import audit_index, manifest_for_task, read_jsonl, sha256_file


def valid_grid(grid: object) -> bool:
    if not isinstance(grid, list) or not (1 <= len(grid) <= 30):
        return False
    width = len(grid[0]) if isinstance(grid[0], list) else 0
    if not (1 <= width <= 30):
        return False
    return all(isinstance(row, list) and len(row) == width and all(
        type(cell) is int and 0 <= cell <= 9 for cell in row
    ) for row in grid)


def _exact_attempts(predictions: list[dict], valid_ids: set[str], max_attempts: int = 2):
    attempts: dict[tuple, list] = defaultdict(list)
    participants: set[tuple[str, int]] = set()
    for row in predictions:
        for field in ("system", "task_id", "seed", "test_index", "attempt", "grid"):
            if field not in row:
                raise ValueError(f"Missing prediction field {field}")
        system, task_id, seed = row["system"], row["task_id"], row["seed"]
        idx, attempt = row["test_index"], row["attempt"]
        if not isinstance(system, str) or not system or task_id not in valid_ids:
            raise ValueError("Prediction uses invalid system or task ID")
        if type(seed) is not int or type(idx) is not int or idx < 0:
            raise ValueError("Seed and test index must be nonnegative integers")
        if type(attempt) is not int or not 1 <= attempt <= max_attempts:
            raise ValueError("Invalid ARC attempt number")
        if not valid_grid(row["grid"]):
            raise ValueError("ARC prediction must be a 1..30 rectangular 0..9 grid")
        key = (system, seed, task_id, idx)
        if any(p["attempt"] == attempt for p in attempts[key]):
            raise ValueError("Duplicate prediction attempt")
        attempts[key].append(row)
        participants.add((system, seed))
    return attempts, participants


def score_arc(
    manifest: dict,
    task_dir: Path,
    predictions_path: Path,
    *,
    split: str = "validation",
    unlock_final: bool = False,
    ledger_path: Path | None = None,
    strict_final_count: bool = True,
) -> dict:
    """Grade task exactness, passing only if ALL test inputs pass in <=2 attempts.

    Final grading locks a one-shot ledger before reading public evaluation JSON.
    Answers and prompts are never included in the returned report.
    """
    audit_index(manifest)
    if split == "final":
        if not unlock_final or ledger_path is None:
            raise PermissionError("Final ARC scoring needs --unlock-final and a ledger")
        if strict_final_count and manifest["sealed_final_count"] != 120:
            raise ValueError("Official ARC-AGI-2 public final must have 120 tasks")
    elif unlock_final:
        raise ValueError("--unlock-final applies only to final ARC evaluation")
    elif task_dir.name.lower() == "evaluation":
        raise PermissionError("Evaluation directory requires final unlock")
    selected = [r for r in manifest["records"]
                if r["family"] == "arc_agi_2" and r["split"] == split]
    if not selected:
        raise ValueError(f"No ARC tasks registered for split {split}")
    pred_rows = read_jsonl(predictions_path)
    attempts, participants = _exact_attempts(
        pred_rows, {r["task_id"] for r in selected}
    )
    if not participants:
        raise ValueError("No valid predictions were supplied")
    if split == "final":
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create is the operational one-shot guard, including failed
        # downstream attempts. Preserve this artifact in durable storage.
        with ledger_path.open("x", encoding="utf-8") as ledger:
            json.dump({
                "status": "final_grading_attempt_consumed",
                "arc_revision": manifest["arc_revision"],
                "manifest_sha256": hashlib.sha256(
                    json.dumps(manifest, sort_keys=True).encode()
                ).hexdigest(),
                "predictions_sha256": sha256_file(predictions_path),
            }, ledger, indent=2)
            ledger.write("\n")
    correct: dict[tuple[str, int], int] = defaultdict(int)
    covered: dict[tuple[str, int], int] = defaultdict(int)
    for row in selected:
        path = task_dir / (row["task_id"].split("/", 1)[1] + ".json")
        if sha256_file(path) != row["content_sha256"]:
            raise ValueError(f"Source task checksum mismatch: {row['task_id']}")
        task = json.loads(path.read_text(encoding="utf-8"))
        test_cases = task.get("test")
        if not isinstance(test_cases, list) or not test_cases:
            raise ValueError("Task has no test cases")
        for case in test_cases:
            if not valid_grid(case.get("output")) or not valid_grid(case.get("input")):
                raise ValueError("ARC source has invalid grid")
        n = len(test_cases)
        for system, seed in participants:
            task_attempts = [
                (i, attempts.get((system, seed, row["task_id"], i), []))
                for i in range(n)
            ]
            if all(p for _, p in task_attempts):
                covered[(system, seed)] += 1
            if all(any(p["grid"] == test_cases[i]["output"] for p in ps)
                   for i, ps in task_attempts):
                correct[(system, seed)] += 1
    # No extra test indices for the submitted task set.
    expected_case_counts = {
        row["task_id"]: len(json.loads(
            (task_dir / (row["task_id"].split("/", 1)[1] + ".json")).read_text()
        )["test"]) for row in selected
    }
    for _, _, task_id, idx in attempts:
        if idx >= expected_case_counts[task_id]:
            raise ValueError("Prediction references an impossible test index")
    n_tasks = len(selected)
    return {
        "status": "graded_external_arc_tasks",
        "benchmark": "official_arc_agi_2" if split == "final" else "official_arc_agi_2_training_partition",
        "split": split,
        "source_revision": manifest["arc_revision"],
        "task_count": n_tasks,
        "max_attempts_per_test_input": 2,
        "systems": [{
            "system": system, "seed": seed,
            "solved": correct[(system, seed)],
            "solve_rate": correct[(system, seed)] / n_tasks,
            "complete_task_coverage": covered[(system, seed)] / n_tasks,
        } for system, seed in sorted(participants)],
    }


def score_logic_verdicts(manifest: dict, verdicts: list[dict], *, split: str) -> dict:
    """Aggregate external task-verifier receipts; never score free-form text ourselves."""
    task_index = manifest_for_task(manifest)
    valid_ids = {tid for tid, r in task_index.items()
                 if r["family"] != "arc_agi_2" and r["split"] == split}
    if not valid_ids:
        raise ValueError("No registered logic tasks in split")
    outcomes = {}
    cost_units = set()
    for row in verdicts:
        fields = ("task_id", "system", "seed", "correct",
                  "verifier_id", "verifier_revision", "receipt_sha256", "cost", "cost_unit")
        if any(field not in row for field in fields):
            raise ValueError("Logic result needs independent verifier metadata")
        if row["task_id"] not in valid_ids or not row["verifier_id"] or not row["verifier_revision"]:
            raise ValueError("Unknown logic task or missing verifier identity")
        if type(row["seed"]) is not int or type(row["correct"]) is not bool:
            raise ValueError("Invalid logic seed / correct type")
        if not isinstance(row["cost_unit"], str) or not row["cost_unit"]:
            raise ValueError("Measured cost_unit is required")
        cost_units.add(row["cost_unit"])
        if type(row["cost"]) not in (int, float) or row["cost"] < 0:
            raise ValueError("Cost must be nonnegative measured units")
        digest = row["receipt_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError("Verifier receipt must be a SHA-256 digest")
        key = (row["system"], row["seed"], row["task_id"])
        if key in outcomes:
            raise ValueError("Duplicate logic verdict")
        outcomes[key] = row
    if len(cost_units) != 1:
        raise ValueError("Cost units must match across all arms")
    participants = sorted({(s, seed) for s, seed, _ in outcomes})
    if not participants:
        raise ValueError("No logic receipts")
    report = []
    for system, seed in participants:
        covered = [outcomes[(system, seed, task_id)] for task_id in valid_ids
                   if (system, seed, task_id) in outcomes]
        report.append({
            "system": system, "seed": seed,
            "task_count": len(valid_ids), "verdict_count": len(covered),
            "accuracy_all_tasks": sum(r["correct"] for r in covered) / len(valid_ids),
            "coverage": len(covered) / len(valid_ids),
            "mean_cost_covered": sum(r["cost"] for r in covered) / len(covered),
            "cost_unit": next(iter(cost_units)),
        })
    paired = []
    for i, (system_a, seed_a) in enumerate(participants):
        for system_b, seed_b in participants[i + 1:]:
            if seed_a != seed_b:
                continue
            matched = [
                task_id for task_id in valid_ids
                if (system_a, seed_a, task_id) in outcomes
                and (system_b, seed_b, task_id) in outcomes
            ]
            if matched:
                paired.append({
                    "system_a": system_a, "system_b": system_b, "seed": seed_a,
                    "paired_n": len(matched),
                    "accuracy_delta_a_minus_b": sum(
                        outcomes[(system_a, seed_a, tid)]["correct"]
                        - outcomes[(system_b, seed_b, tid)]["correct"]
                        for tid in matched
                    ) / len(matched),
                    "mean_cost_delta_a_minus_b": sum(
                        outcomes[(system_a, seed_a, tid)]["cost"]
                        - outcomes[(system_b, seed_b, tid)]["cost"]
                        for tid in matched
                    ) / len(matched),
                })
    return {
        "status": "reported_external_verifier_receipts_unverified_signature",
        "split": split, "systems": report, "paired": paired,
        "scope": "normalized per-task external verifier results",
    }
