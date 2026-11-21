"""Repository snapshots at a base commit, symbol tables and gold edit targets.

Snapshots are read from bare clones with ``git ls-tree`` and ``git cat-file
--batch``; nothing is checked out. Python files are parsed with ``ast`` and
cached per blob SHA, so unchanged files are parsed once across many commits.
Gold targets come from the instance's patch: changed files and the innermost
function, method or class enclosing each changed line of the old file.
"""
from __future__ import annotations

import ast
import gzip
import json
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TEST_PARTS = ("tests", "test", "testing")
DOC_PARTS = ("doc", "docs", "examples", "example", "benchmarks", "asv_benchmarks", "tutorials", "galleries")


def git(repo: Path, *args: str, input: bytes | None = None) -> bytes:
    return subprocess.run(["git", "--git-dir", str(repo), *args], input=input, capture_output=True, check=True).stdout


def path_flags(path: str) -> dict:
    parts = path.lower().split("/")
    name = parts[-1]
    is_test = any(p in TEST_PARTS for p in parts[:-1]) or name.startswith("test_") or name.endswith("_test.py") \
        or name == "conftest.py"
    is_doc = any(p in DOC_PARTS for p in parts[:-1])
    return {"is_test": is_test, "is_doc": is_doc, "is_setup": name in ("setup.py", "conf.py") and len(parts) <= 2}


@dataclass
class Symbol:
    path: str
    qualname: str
    kind: str            # module, class, function, method
    start: int
    end: int
    signature: str
    doc: str

    @property
    def uri(self) -> str:
        return f"{self.path}::{self.qualname}" if self.kind != "module" else self.path


def _signature(node: ast.AST, lines: list[str]) -> str:
    line = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
    return line[:160]


def parse_symbols(path: str, source: str) -> list[Symbol]:
    lines = source.splitlines()
    module_doc = ""
    out: list[Symbol] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
        module_doc = (ast.get_docstring(tree) or "").strip().split("\n")[0][:160]
    except (SyntaxError, ValueError, RecursionError):
        return [Symbol(path, "<module>", "module", 1, max(1, len(lines)), "", "")]
    out.append(Symbol(path, "<module>", "module", 1, max(1, len(lines)), "", module_doc))

    def visit(node: ast.AST, prefix: str, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(child, ast.ClassDef) else ("method" if in_class else "function")
                qual = f"{prefix}{child.name}"
                doc = (ast.get_docstring(child) or "").strip().split("\n")[0][:160]
                end = getattr(child, "end_lineno", child.lineno) or child.lineno
                start = min([child.lineno] + [d.lineno for d in getattr(child, "decorator_list", [])])
                out.append(Symbol(path, qual, kind, start, end, _signature(child, lines), doc))
                visit(child, qual + ".", isinstance(child, ast.ClassDef))
    visit(tree, "", False)
    return out


@dataclass
class Snapshot:
    repo: str
    commit: str
    files: dict[str, str]                    # path -> blob sha
    symbols: dict[str, list[Symbol]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def all_symbols(self) -> Iterable[Symbol]:
        for path in sorted(self.symbols):
            yield from self.symbols[path]


class SnapshotStore:
    """Read snapshots from bare clones with a per-blob parse cache on disk."""

    def __init__(self, repos_dir: Path, cache_dir: Path):
        self.repos_dir, self.cache_dir = repos_dir, cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._blob_cache: dict[str, tuple[str, list[Symbol]]] = {}

    def bare(self, repo: str) -> Path:
        return self.repos_dir / repo.replace("/", "__")

    def snapshot(self, repo: str, commit: str, with_sources: bool = True) -> Snapshot:
        bare = self.bare(repo)
        listing = git(bare, "ls-tree", "-r", commit).decode("utf-8", "replace").splitlines()
        files = {}
        for line in listing:
            meta, path = line.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind == "blob" and path.endswith(".py"):
                files[path] = sha
        snap = Snapshot(repo, commit, files)
        missing = [sha for sha in set(files.values()) if sha not in self._blob_cache]
        if missing:
            self._load_blobs(bare, missing)
        by_sha = {}
        for path, sha in files.items():
            source, symbols = self._blob_cache[sha]
            if symbols and symbols[0].path != path:
                symbols = [Symbol(path, s.qualname, s.kind, s.start, s.end, s.signature, s.doc) for s in symbols]
            snap.symbols[path] = symbols
            if with_sources:
                snap.sources[path] = source
            by_sha[sha] = True
        return snap

    def _load_blobs(self, bare: Path, shas: list[str]) -> None:
        payload = ("\n".join(shas) + "\n").encode()
        proc = subprocess.run(["git", "--git-dir", str(bare), "cat-file", "--batch"], input=payload,
                              capture_output=True, check=True)
        data, pos = proc.stdout, 0
        for sha in shas:
            header_end = data.index(b"\n", pos)
            _sha, _kind, size = data[pos:header_end].split()
            size = int(size)
            body = data[header_end + 1:header_end + 1 + size]
            pos = header_end + 1 + size + 1
            source = body.decode("utf-8", "replace")
            self._blob_cache[sha] = (source, parse_symbols("?", source))


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def patch_targets(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Old-file line spans touched by each file in a unified diff (non-new files)."""
    out: dict[str, list[tuple[int, int]]] = {}
    current = None
    old_line = 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            name = line[4:].strip()
            current = name[2:] if name.startswith("a/") else None
            if current is not None:
                out.setdefault(current, [])
            continue
        if line.startswith("+++ "):
            continue
        m = _HUNK.match(line)
        if m:
            old_line = int(m.group(1))
            continue
        if current is None:
            continue
        if line.startswith("-"):
            out[current].append((old_line, old_line))
            old_line += 1
        elif line.startswith("+"):
            out[current].append((max(1, old_line - 1), old_line))
        elif line.startswith(" "):
            old_line += 1
    return out


def gold_symbols(snapshot: Snapshot, patch: str) -> dict:
    """Gold files and innermost enclosing symbols; module-level edits map to the module."""
    targets = patch_targets(patch)
    files, symbols = [], set()
    for path, spans in targets.items():
        if path not in snapshot.symbols:
            continue
        files.append(path)
        table = snapshot.symbols[path]
        for lo, hi in spans:
            for line in range(lo, hi + 1):
                inner = None
                for s in table:
                    if s.kind != "module" and s.start <= line <= s.end:
                        if inner is None or (s.end - s.start) <= (inner.end - inner.start):
                            inner = s
                symbols.add((inner or table[0]).uri)
    return {"files": sorted(files), "symbols": sorted(symbols)}


def load_instances(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", "build", "dist", ".tox",
             ".mypy_cache", ".pytest_cache", "site-packages"}


def snapshot_from_dir(root: Path, max_files: int = 20000) -> Snapshot:
    """Snapshot of a working tree's Python files (for live use by the Goose integration)."""
    files: dict[str, str] = {}
    sources: dict[str, str] = {}
    symbols: dict[str, list[Symbol]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files[rel] = f"worktree:{len(text)}"
        sources[rel] = text
        symbols[rel] = parse_symbols(rel, text)
        if len(files) >= max_files:
            break
    return Snapshot(repo=root.name, commit="worktree", files=files, symbols=symbols, sources=sources)
