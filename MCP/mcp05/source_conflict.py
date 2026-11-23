"""Stage and score the public Aleph source-conflict pairs through MCP-05."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "aleph-source-conflict-development-v1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_action(observation: dict) -> str:
    authorized = observation["authorized_revision"]
    found = [report["route"] for report in observation["reports"] if authorized == report["revision"]]
    if len(found) != 1 or found[0] not in observation["legal_actions"]:
        raise ValueError("the bulletin must identify exactly one legal route")
    return found[0]


def verified_pairs(tasks: Path, receipt: Path) -> list[dict]:
    control = load(receipt)
    source = tasks.read_bytes()
    if hashlib.sha256(source).hexdigest() != control["public_tasks_sha256"]:
        raise ValueError("task file does not match its source receipt")
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    if (control["schema"] != SCHEMA or control["heldout_sealed"] is not False
            or len(rows) != 2 * control["development_roots"] or not rows):
        raise ValueError("expected paired public development cases")
    ids = [row["task_id"] for row in rows]
    if any(not (len(id_) == 24 and all(ch in "0123456789abcdef" for ch in id_)) for id_ in ids):
        raise ValueError("invalid source task identifier")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source task identifier")
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        if first["schema"] != SCHEMA or second["schema"] != SCHEMA:
            raise ValueError("task schema differs from its source receipt")
        left, right = first["observation"], second["observation"]
        changed = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
        if changed != {"authorized_revision"} or expected_action(left) == expected_action(right):
            raise ValueError("paired cases did not preserve the advertised invariant")
    return rows


def cases(rows: list[dict], workspace: Path) -> list[dict]:
    return [{"id": row["task_id"], "suite": SCHEMA, "repo": str(workspace), "task": row["prompt"]}
            for row in rows]


def parse_action(answer: str, choices: list[str]) -> str | None:
    try:
        value = json.loads(answer, object_pairs_hook=tuple)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, tuple) or len(value) != 1:
        return None
    key, action = value[0]
    return action if key == "action" and isinstance(action, str) and action in choices else None


def score_case(row: dict, payload: dict) -> dict:
    observation = row["observation"]
    target = expected_action(observation)
    preflight = payload.get("preflight", [])
    valid_response = (not payload.get("error") and payload.get("inference_calls") == 1
                      and len(preflight) == 1 and preflight[0].get("provider_status") == 200
                      and preflight[0].get("provider_preflight_parity") is True)
    action = parse_action(payload.get("answer", ""), observation["legal_actions"]) if valid_response else None
    status = "infrastructure-error" if not valid_response else "malformed" if action is None else "scored"
    return {"id": row["task_id"], "status": status, "action": action, "expected": target,
            "correct": action == target, "prompt_tokens": payload.get("provider_input_tokens", 0),
            "output_tokens": payload.get("provider_output_tokens", 0)}


def score(rows: list[dict], output: Path) -> dict:
    results = []
    for row in rows:
        path = output / row["task_id"] / "A/gdk.json"
        if path.is_file():
            results.append(score_case(row, load(path)))
        else:
            results.append({"id": row["task_id"], "status": "pending", "correct": False})
    pairs = list(zip(results[::2], results[1::2], strict=True))
    ready = {"malformed", "scored"}
    return {
        "schema": SCHEMA,
        "evidence_class": "model responses on public synthetic development roots",
        "assigned": len(rows), "scored": sum(row["status"] in ready for row in results),
        "pending": sum(row["status"] == "pending" for row in results),
        "malformed": sum(row["status"] == "malformed" for row in results),
        "infrastructure_errors": sum(row["status"] == "infrastructure-error" for row in results),
        "correct": sum(row["correct"] for row in results),
        "pairs_assigned": len(pairs),
        "pairs_scored": sum(all(case["status"] in ready for case in pair) for pair in pairs),
        "pairs_correct": sum(all(case["correct"] for case in pair) for pair in pairs),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--workspace", type=Path, help="Stage the model cases from the public task file")
    group.add_argument("--campaign", type=Path, help="Score the resulting A-arm Goose receipts")
    args = parser.parse_args()
    rows = verified_pairs(args.tasks, args.controls)
    result = cases(rows, args.workspace.resolve(strict=True)) if args.workspace else score(rows, args.campaign)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
