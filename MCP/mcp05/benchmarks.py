"""Stage offline Storyworld, SWE-bench Lite and research-planning pilot cases."""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from MCP.mcp03.repo import gold_symbols, snapshot_from_dir
from .gym import GUARD, ROOT, dump


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def freeze(root: Path) -> None:
    git(root, "init", "-q")
    git(root, "-c", "core.autocrlf=false", "add", ".")
    git(root, "-c", "user.name=MCP fixture", "-c", "user.email=mcp@example.invalid", "commit", "-qm",
        "frozen pilot workspace")


def storyworld(source: Path, out: Path) -> dict:
    name = "storyworld-tiny-diplomacy-accord"
    root = out / name
    (root / "world").mkdir(parents=True)
    (root / "reference").mkdir()
    shutil.copyfile(source / "storyworld/examples/diplomacy_min.json", root / "world/diplomacy_min.json")
    shutil.copyfile(source / "benchmarks/storyworld_builder_small_model/README.md",
                    root / "reference/STORYWORLD_BUILDER.md")
    (out / "storyworld-native").mkdir()
    validator = out / "storyworld-native/validate_storyworld.py"
    shutil.copyfile(source / "storyworld/validators/validate_storyworld.py", validator)
    check = out / "storyworld-native/verify_accord.py"
    check.write_text(
        "import json, runpy\nfrom pathlib import Path\n"
        "validator = runpy.run_path(__file__.replace('verify_accord.py', 'validate_storyworld.py'))\n"
        "data = json.loads(Path('world/diplomacy_min.json').read_text(encoding='utf-8'))\n"
        "errors = validator['manual_validate'](data)\n"
        "assert not errors, errors\n"
        "nodes = {node['id']: node for node in data['nodes']}\n"
        "assert {'node_start', 'node_coalition', 'node_betrayal'} <= nodes.keys(), 'Preserve existing nodes'\n"
        "accord = nodes.get('node_council_accord', {})\n"
        "assert accord.get('terminal') is True and 'Accord' in accord.get('label',''), 'Missing terminal accord'\n"
        "outcomes = data['rules']['outcomes']\n"
        "assert {'betrayal','coalition_formed','stalemate','maneuver'} <= outcomes.keys(), 'Preserve existing outcomes'\n"
        "assert outcomes.get('council_accord', {}).get('next_node') == 'node_council_accord', 'Missing outcome route'\n"
        "print('Native Storyworld structure and new ending verified')\n",
        encoding="utf-8")
    freeze(root)
    return {"id": name, "suite": "gptstoryworld-structure", "repo": str(root),
            "source_revision": git(source, "rev-parse", "HEAD"),
            "task": ("Build a negotiated council accord ending for Tiny Diplomacy in world/diplomacy_min.json. "
                     "Add a terminal node with id node_council_accord and an evocative label containing 'Accord'. "
                     "Add the council_accord outcome under rules.outcomes pointing at that node. "
                     "Preserve all existing agents, nodes, and outcomes. Verify the actual storyworld file."),
            "verifier": {"argv": [sys.executable, "-B", str(GUARD), "--allow-read", str(validator),
                                  "--script", str(check)],
                         "editable": ["world/diplomacy_min.json"]}}


SWE_CHECKS = {
    "pallets__flask-4045": (
        "from flask import Blueprint\n"
        "Blueprint('ordinary', __name__)\n"
        "try:\n    Blueprint('nested.name', __name__)\n"
        "except ValueError:\n    pass\n"
        "else:\n    raise AssertionError('Blueprint names containing a dot must be rejected')\n"),
    "pallets__flask-4992": (
        "import tomllib\nfrom pathlib import Path\nfrom flask.config import Config\n"
        "source = Path('mcp05-verifier-fixture.toml')\n"
        "source.write_text('FLASK_ITEM = 17\\n', encoding='utf-8')\n"
        "try:\n    cfg = Config('.')\n"
        "    assert cfg.from_file(source.name, load=tomllib.load, text=False)\n"
        "    assert cfg['FLASK_ITEM'] == 17\n"
        "finally:\n    source.unlink()\n"),
}


def swebench(data: Path, bare_roots: Path, out: Path, python: str, ids: list[str]) -> list[dict]:
    wanted = set(ids)
    cases = []
    for line in data.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        name = item["instance_id"]
        if name not in wanted:
            continue
        repo = bare_roots / item["repo"].replace("/", "__")
        root = out / name
        root.mkdir()
        archive = subprocess.run(["git", "--git-dir", str(repo), "archive", item["base_commit"]],
                                 capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(root, filter="data")
        freeze(root)
        check = out / f"{name}-native.py"
        check.write_text("import sys\nfrom pathlib import Path\n"
                         "sys.path.insert(0, str(Path.cwd() / 'src'))\n"
                         + SWE_CHECKS[name], encoding="utf-8")
        check_data = gold_symbols(snapshot_from_dir(root), item["patch"])
        cases.append({"id": name, "suite": "swe-bench-lite", "repo": str(root),
                      "base_commit": item["base_commit"], "task": item["problem_statement"],
                      "gold_symbols": check_data["symbols"],
                      "verifier": {"argv": [python, "-B", str(GUARD), "--preload-click", "--script", str(check)],
                                   "editable": ["src/flask/"]}})
    if len(cases) != len(wanted):
        raise ValueError("one or more SWE-bench Lite instance IDs were unavailable")
    return cases


def research(out: Path, datagoblin: Path | None) -> list[dict]:
    root = out / "research-evidence"
    root.mkdir()
    shutil.copyfile(ROOT / "MCP/mcp03/RU02_MATCHED_RESULTS.md", root / "RU02_MATCHED_RESULTS.md")
    shutil.copyfile(ROOT / "MCP/mcp02/models.py", root / "models.py")
    cases = [
        {"id": "research-ram-trm", "suite": "research-planning-rubric", "repo": str(root),
         "task": ("Inspect the RU02 results in this repository. Write a short preregistration for a four-thread "
                  "CPU SetTRM experiment with a 3 GiB peak RSS limit. Say how to falsify a recurrent-state "
                  "benefit, keep optimizer and data comparisons fair, and distinguish localization from repair. "
                  "Cite the source file, give one numeric result and identify a result that would change your mind."),
         "rubric": [r"RU02_MATCHED_RESULTS\.md", r"shuffl", r"\breset\b", r"slates?", r"\bRSS\b",
                    r"(\b3\s*GiB\b|\b3\s*GB\b)", r"verifi\w*|regression", r"\b0\.49(?:38|53)\b"]},
    ]
    if datagoblin:
        report = datagoblin / "docs/inkling_variable_loop_local_report.md"
        shutil.copyfile(report, root / report.name)
        cases.append(
            {"id": "research-cuda-interpretability", "suite": "research-planning-rubric", "repo": str(root),
             "task": ("Read the existing Inkling variable-loop local report. Propose a small RTX 5080 CUDA "
                      "mechanistic-interpretability experiment to distinguish loop-state causal reuse from "
                      "a mere compute-depth effect. Specify instrumentation, intervention, at least two "
                      "negative controls, a held-out criterion, and resource stop conditions. Cite the report. "
                      "Design the experiment only, with no need to run GPU kernels."),
             "rubric": [r"inkling_variable_loop_local_report\.md", r"loop[- ](?:state|pass)|residual",
                        r"patch|interven", r"shuffl|random", r"norm[- ]match|orthogonal|zero|sham",
                        r"held[- ]out", r"VRAM|memory", r"seed|paired"]})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--storyworld", type=Path, required=True)
    parser.add_argument("--datagoblin", type=Path)
    parser.add_argument("--swe-data", type=Path)
    parser.add_argument("--swe-repos", type=Path)
    parser.add_argument("--swe-python", default=sys.executable)
    parser.add_argument("--swe-ids", nargs="+", choices=sorted(SWE_CHECKS), default=sorted(SWE_CHECKS))
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = [storyworld(args.storyworld, root), *research(root, args.datagoblin)]
    if args.swe_data and args.swe_repos:
        cases.extend(swebench(args.swe_data, args.swe_repos, root, args.swe_python, args.swe_ids))
    dump(root / "cases.json", cases)
    print(json.dumps({"staged": [case["id"] for case in cases]}))


if __name__ == "__main__":
    main()
