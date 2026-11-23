"""Revision-bound working set shared by the MCP service and the repair gym."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from MCP.mcp03.env import CodeEnv, PREDICATES, code_tokens
from MCP.mcp03.repo import SKIP_DIRS, Snapshot, Symbol, parse_symbols
from .json_sections import sections

TEXT_SUFFIXES = {".py", ".rs", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".json", ".md", ".toml", ".lean", ".metta",
                 ".yaml", ".yml", ".txt", ".cu", ".cpp", ".c", ".h", ".html", ".css"}
MAX_SOURCE_BYTES = 512 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path) -> dict[str, tuple[int, ...]]:
    files = {}
    for base, dirs, names in os.walk(root):
        directory = Path(base)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and d != "target"
                         and not d.startswith(".") and not (directory / d).is_symlink()
                         and not getattr(directory / d, "is_junction", lambda: False)())
        for name in sorted(names):
            path = directory / name
            if path.suffix.lower() not in TEXT_SUFFIXES or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if path.is_file() and 0 < stat.st_size <= MAX_SOURCE_BYTES:
                files[path.relative_to(root).as_posix()] = (
                    stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            if len(files) >= 20_000:
                return files
    return files


class WorkingSet:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.identity: dict = {}
        self.env: CodeEnv | None = None
        self.revision = ""
        self.bodies: dict[str, dict] = {}
        self.latest: dict[tuple[str, int], str] = {}
        self.reads: list[str] = []
        self.repeated_reads = 0
        self.reused = 0
        self.invalidated = 0
        self.postings = 0
        self.exclusions: list[dict] = []

    def refresh(self) -> CodeEnv:
        identity = inventory(self.root)
        if self.env is not None and identity == self.identity:
            return self.env
        snapshot = Snapshot(str(self.root), "", {})
        for rel, stat in identity.items():
            if self.env is not None and self.identity.get(rel) == stat:
                snapshot.files[rel] = self.env.snapshot.files[rel]
                snapshot.sources[rel] = self.env.snapshot.sources[rel]
                snapshot.symbols[rel] = self.env.snapshot.symbols[rel]
                continue
            path = self.root / rel
            if path.resolve() != path:
                raise ValueError("source path changed during scan")
            raw = path.read_bytes()
            source = raw.decode("utf-8", errors="replace")
            snapshot.files[rel] = sha(raw)
            snapshot.sources[rel] = source
            suffix = path.suffix.lower()
            if suffix == ".py":
                snapshot.symbols[rel] = parse_symbols(rel, source)
            else:
                module = Symbol(rel, "<module>", "module", 1, max(1, len(source.splitlines())), "", "")
                snapshot.symbols[rel] = [module]
                if suffix == ".json":
                    snapshot.symbols[rel].extend(sections(rel, source))
        if identity != inventory(self.root):
            raise ValueError("working tree changed during indexing; retry")
        snapshot.commit = sha(repr(sorted(snapshot.files.items())).encode())
        for handle, body in list(self.bodies.items()):
            if snapshot.files.get(body["path"]) != body["source_digest"]:
                self.invalidated += 1
                self.bodies.pop(handle)
                self.latest.pop((body["uri"], body["start"]), None)
        self.revision, self.identity = snapshot.commit, identity
        self.env = CodeEnv(snapshot, {}, {"files": [], "symbols": []}, "layered_ldt")
        self.exclusions = []
        return self.env

    def query(self, query: str, limit: int = 8, cursor: int = 0) -> dict:
        env = self.refresh()
        before = env.index.operations
        rows = env.index.search(query)
        self.postings += env.index.operations - before
        # Whole-file identifiers remain searchable alongside structural sections.
        if any(Path(path).suffix.lower() != ".py" for path in env.snapshot.files):
            before = env.file_index.operations
            other = [(score, uri) for score, uri in env.file_index.search(query)
                     if Path(uri).suffix.lower() != ".py"]
            self.postings += env.file_index.operations - before
            scores = dict((uri, score) for score, uri in rows)
            for score, uri in other:
                scores[uri] = max(score, scores.get(uri, 0))
            rows = sorted(((score, uri) for uri, score in scores.items()),
                          key=lambda pair: (-pair[0], pair[1]))
        return {"revision": self.revision,
                "candidates": [dict(env.row("s:" + uri), score=round(score, 6))
                               for score, uri in rows[cursor:cursor + limit]],
                "next_cursor": cursor + limit if cursor + limit < len(rows) else None}

    def expand(self, node_id: str, limit: int = 8, cursor: int = 0,
               predicate: str | None = None, exclude: list[str] | None = None) -> dict:
        env = self.refresh()
        if predicate is not None:
            if predicate not in PREDICATES or not exclude:
                raise ValueError("an exclusion requires a registered predicate and explicit children")
            result = env.call("candidate_refine", {"node_id": node_id, "predicate": predicate,
                                                   "exclude": exclude})
            if result["isError"]:
                raise ValueError(result["structuredContent"]["error"])
            certificate = {"revision": self.revision, "predicate": predicate,
                           **result["structuredContent"]}
            self.exclusions.append(certificate)
        result = env.call("index_children", {"node_id": node_id, "limit": limit, "cursor": cursor})
        if result["isError"]:
            raise ValueError(result["structuredContent"]["error"])
        return {"revision": self.revision, **result["structuredContent"]}

    def read(self, *, uri: str | None = None, handle: str | None = None,
             start: int | None = None, record: bool = True) -> dict:
        env = self.refresh()
        if (uri is None) == (handle is None) or (handle and start is not None):
            raise ValueError("supply a symbol URI or a cached source handle")
        if handle is not None:
            body = self.bodies.get(handle)
            if body is None:
                raise ValueError("unknown or stale source handle; query the current revision")
            if not self.matches_snapshot(body["path"], body["source_digest"]):
                self.identity.pop(body["path"], None)
                self.refresh()
                raise ValueError("source changed since this handle was issued")
            if record:
                self.reused += 1
        else:
            symbol = env.symbols.get(uri)
            if symbol is None:
                raise ValueError("unknown symbol URI")
            if not self.matches_snapshot(symbol.path, env.snapshot.files[symbol.path]):
                self.identity.pop(symbol.path, None)
                env = self.refresh()
                symbol = env.symbols.get(uri)
                if symbol is None:
                    raise ValueError("symbol disappeared after source changed")
            start = start if start is not None else symbol.start
            if not symbol.start <= start <= symbol.end:
                raise ValueError("line is outside the symbol")
            handle = self.latest.get((uri, start))
            if handle is not None:
                self.repeated_reads += 1
                body = self.bodies[handle]
            else:
                lines = env.lines[symbol.path]
                end = min(symbol.end, start + 79)
                digest = env.snapshot.files[symbol.path]
                handle = "source:" + sha(f"{uri}\0{digest}\0{start}".encode())
                body = {"uri": uri, "path": symbol.path, "start": start, "end": end,
                        "source_digest": digest, "handle": handle,
                        "next_line": end + 1 if end < symbol.end else None,
                        "source": "\n".join(lines[start - 1:end])}
                self.bodies[handle] = body
                self.latest[(uri, start)] = handle
        if record:
            self.reads.append(uri)
        return {**body, "revision": self.revision}

    def matches_snapshot(self, rel: str, revision: str) -> bool:
        path = self.root / rel
        try:
            return path.resolve(strict=True) == path and sha(path.read_bytes()) == revision
        except FileNotFoundError:
            return False

    def state(self) -> dict:
        self.refresh()
        return {"revision": self.revision,
                "read_history": self.reads, "repeated_reads": self.repeated_reads,
                "invalidated_handles": self.invalidated, "reused_handles": self.reused,
                "search_postings_ops": self.postings, "exclusions": self.exclusions,
                "evidence": [{key: value for key, value in body.items() if key != "source"}
                             for body in self.bodies.values()]}


def candidate_features(query: str, row: dict) -> tuple[str, list[float]]:
    text = " ".join(str(row.get(field, "")) for field in ("uri", "path", "signature", "doc"))
    q, c = set(code_tokens(query)), set(code_tokens(text))
    return text, [min(float(row.get("score", 0)) / 20, 1), len(q & c) / max(1, len(q)),
                  min(len(c) / 40, 1), float(str(row.get("uri", "")).count("::") > 0)]
