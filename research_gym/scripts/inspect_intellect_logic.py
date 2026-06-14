from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.adapters.intellect_logic import DEFAULT_LOGIC_ENV_PATH, inspect_logic_env


def markdown_report(data: dict[str, object]) -> str:
    tasks = data["verifier_tasks"]
    assert isinstance(tasks, list)
    lines = [
        "# Intellect-3 Logic Environment",
        "",
        f"Environment path: `{data['env_path']}`",
        f"Environment exists: `{data['exists']}`",
        f"Environment ID: `{data['env_id']}`",
        f"Dataset: `{data['dataset_name']}` subset `{data['dataset_subset']}`",
        "",
        "Smoke command:",
        "",
        "```bash",
        " ".join(str(part) for part in data["smoke_command"]),
        "```",
        "",
        f"Verifier tasks discovered: {len(tasks)}",
        "",
    ]
    for task in tasks[:30]:
        lines.append(f"- `{task}`")
    if len(tasks) > 30:
        lines.append(f"- ... {len(tasks) - 30} more")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local INTELLECT-3 logic environment fork.")
    parser.add_argument("--env-path", type=Path, default=DEFAULT_LOGIC_ENV_PATH)
    parser.add_argument("--out", type=Path, default=Path("reports/intellect_logic_env.md"))
    parser.add_argument("--json-out", type=Path, default=Path("data/benchmarks/intellect_logic_env.json"))
    args = parser.parse_args()

    integration = inspect_logic_env(args.env_path)
    data = integration.to_jsonable()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = markdown_report(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
