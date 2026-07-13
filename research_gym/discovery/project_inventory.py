from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path


EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "site-packages",
}

SIGNALS = {
    "game": 3,
    "storyworld": 4,
    "diplomacy": 4,
    "arena": 3,
    "environment": 2,
    "env": 1,
    "harness": 3,
    "control": 2,
    "agent": 1,
    "policy": 1,
    "action": 1,
    "reward": 1,
    "skill": 1,
    "router": 1,
    "verifier": 2,
    "simulation": 2,
}


@dataclass(frozen=True)
class ProjectCandidate:
    name: str
    path: str
    relevance_score: int
    matched_signals: tuple[str, ...]
    application: str
    evidence_files: tuple[str, ...]

    def to_jsonable(self) -> dict[str, object]:
        record = asdict(self)
        record["matched_signals"] = list(self.matched_signals)
        record["evidence_files"] = list(self.evidence_files)
        return record


def _application_hint(text: str) -> str:
    if "diplomacy" in text or "coalition" in text or "betrayal" in text:
        return "coalition planning and model-sound opponent reasoning"
    if "storyworld" in text or "encounter" in text or "secret ending" in text:
        return "choice routing, reachability certification, and moral optimization"
    if "controlarena" in text or "control harness" in text or "oracle" in text:
        return "trusted/untrusted control flow and provenance-aware intervention"
    if "arena" in text or "game" in text:
        return "game action selection and safety-constrained search"
    if "router" in text or "skill" in text:
        return "skill routing and architecture selection"
    if "environment" in text or " env" in text:
        return "environment action validation and typed transition control"
    return "agent control and policy arbitration"


def _evidence_text(
    project: Path,
    *,
    max_depth: int = 2,
    max_files: int = 8,
    max_directories: int = 48,
    max_entries_per_directory: int = 256,
) -> tuple[str, tuple[str, ...]]:
    chunks = [project.name]
    evidence: list[str] = []
    queue: deque[tuple[Path, int]] = deque([(project, 0)])
    visited = 0
    while queue and visited < max_directories:
        current_path, depth = queue.popleft()
        visited += 1
        try:
            with os.scandir(current_path) as iterator:
                entries = []
                for entry_index, entry in enumerate(iterator):
                    if entry_index >= max_entries_per_directory:
                        break
                    entries.append(entry)
        except OSError:
            continue
        entries.sort(key=lambda entry: entry.name.lower())
        for entry in entries:
            name = entry.name
            lower = name.lower()
            if entry.is_dir(follow_symlinks=False):
                if depth < max_depth and name not in EXCLUDED_DIRS:
                    chunks.append(name)
                    queue.append((Path(entry.path), depth + 1))
                continue
            if not (
                lower.startswith("readme")
                or lower in {"agents.md", "pyproject.toml", "package.json"}
                or any(signal in lower for signal in SIGNALS)
            ):
                continue
            path = Path(entry.path)
            relative = path.relative_to(project).as_posix()
            evidence.append(relative)
            chunks.append(relative)
            if lower.startswith("readme") or lower in {"agents.md", "pyproject.toml", "package.json"}:
                try:
                    chunks.append(path.read_text(encoding="utf-8", errors="ignore")[:32_000])
                except OSError:
                    pass
            if len(evidence) >= max_files:
                return "\n".join(chunks).lower(), tuple(evidence)
    return "\n".join(chunks).lower(), tuple(evidence)


def inspect_project(project: Path) -> ProjectCandidate:
    text, evidence = _evidence_text(project)
    matched = tuple(sorted(signal for signal in SIGNALS if signal in text))
    score = sum(SIGNALS[signal] for signal in matched)
    return ProjectCandidate(
        name=project.name,
        path=str(project.resolve()),
        relevance_score=score,
        matched_signals=matched,
        application=_application_hint(text),
        evidence_files=evidence,
    )


def scan_projects(root: Path, *, minimum_score: int = 5) -> dict[str, object]:
    root = root.resolve()
    projects = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower())
    inspected = [inspect_project(project) for project in projects]
    candidates = sorted(
        (candidate for candidate in inspected if candidate.relevance_score >= minimum_score),
        key=lambda candidate: (-candidate.relevance_score, candidate.name.lower()),
    )
    return {
        "root": str(root),
        "minimum_score": minimum_score,
        "scanned_project_count": len(projects),
        "candidate_count": len(candidates),
        "candidates": [candidate.to_jsonable() for candidate in candidates],
    }


def inventory_markdown(payload: dict[str, object]) -> str:
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    lines = [
        "# Local AI/Game Environment Inventory",
        "",
        f"Scanned `{payload['scanned_project_count']}` project directories under `{payload['root']}`.",
        f"Retained `{payload['candidate_count']}` candidates at relevance score >= `{payload['minimum_score']}`.",
        "",
        "The score is a discovery heuristic, not a quality ranking. It uses repository names, manifests,",
        "near-root documentation, and interface-like filenames while excluding generated dependency trees.",
        "",
        "| Project | Score | Application surface | Signals |",
        "|---|---:|---|---|",
    ]
    for candidate in candidates:
        assert isinstance(candidate, dict)
        signals = ", ".join(f"`{signal}`" for signal in candidate["matched_signals"])
        lines.append(
            f"| `{candidate['name']}` | {candidate['relevance_score']} | "
            f"{candidate['application']} | {signals} |"
        )
    return "\n".join(lines) + "\n"


def write_inventory(payload: dict[str, object], json_path: Path, report_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(inventory_markdown(payload), encoding="utf-8")
