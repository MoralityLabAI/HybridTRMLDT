from pathlib import Path

from research_gym.adapters.intellect_logic import (
    DEFAULT_LOGIC_ENV_PATH,
    LogicEnvIntegration,
    _extract_string_keys_from_dict_assignment,
    inspect_logic_env,
)


def test_extract_string_keys_from_dict_assignment():
    source = '''
other = {"x": 1}
verifier_classes = {
    "sudoku": SudokuVerifier,
    "minesweeper": MinesweeperVerifier,
    **bbh_classes,
}
'''

    assert _extract_string_keys_from_dict_assignment(source, "verifier_classes") == ("minesweeper", "sudoku")


def test_logic_env_integration_command_shape():
    integration = LogicEnvIntegration(env_path=Path("C:/tmp/missing"), verifier_tasks=("sudoku",))

    assert integration.vf_eval_command(n=2, rollouts=3) == [
        "uv",
        "run",
        "vf-eval",
        "--env",
        "logic-env",
        "-n2",
        "-r3",
        "-d",
        "-v",
    ]
    assert integration.to_jsonable()["dataset_subset"] == "logic"


def test_inspect_logic_env_local_fork_if_present():
    integration = inspect_logic_env(DEFAULT_LOGIC_ENV_PATH)

    assert integration.env_id == "logic-env"
    if integration.exists:
        assert "sudoku" in integration.verifier_tasks
        assert "minesweeper" in integration.verifier_tasks
