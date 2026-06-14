from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_LOGIC_ENV_PATH = Path("C:/projects/prime_intellect_research_environments/environments/logic_env")


@dataclass(frozen=True)
class LogicEnvIntegration:
    """Local descriptor for the Prime Intellect INTELLECT-3 logic environment."""

    env_path: Path = DEFAULT_LOGIC_ENV_PATH
    env_id: str = "logic-env"
    dataset_name: str = "PrimeIntellect/INTELLECT-3-RL"
    dataset_subset: str = "logic"
    default_skip_tasks: tuple[str, ...] = ("arc_agi", "arc_agi_2", "buggy_tables")
    verifier_tasks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def exists(self) -> bool:
        return (self.env_path / "logic_env" / "logic_env.py").exists()

    def vf_eval_command(self, *, n: int = 1, rollouts: int = 1, debug: bool = True, verbose: bool = True) -> list[str]:
        command = ["uv", "run", "vf-eval", "--env", self.env_id, f"-n{n}", f"-r{rollouts}"]
        if debug:
            command.append("-d")
        if verbose:
            command.append("-v")
        return command

    def to_jsonable(self) -> dict[str, object]:
        return {
            "env_path": str(self.env_path),
            "env_id": self.env_id,
            "dataset_name": self.dataset_name,
            "dataset_subset": self.dataset_subset,
            "default_skip_tasks": list(self.default_skip_tasks),
            "verifier_tasks": list(self.verifier_tasks),
            "exists": self.exists,
            "smoke_command": self.vf_eval_command(),
        }


def _extract_string_keys_from_dict_assignment(source: str, assignment_name: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
    return tuple(sorted(set(keys)))


def inspect_logic_env(env_path: Path = DEFAULT_LOGIC_ENV_PATH) -> LogicEnvIntegration:
    task_map = env_path / "logic_env" / "task2verifier.py"
    verifier_tasks: tuple[str, ...] = ()
    if task_map.exists():
        verifier_tasks = _extract_string_keys_from_dict_assignment(task_map.read_text(encoding="utf-8"), "verifier_classes")
    return LogicEnvIntegration(env_path=env_path, verifier_tasks=verifier_tasks)
