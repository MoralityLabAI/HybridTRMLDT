"""Immutable repository indexes and a bounded, proof-carrying cross-index search."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path, PurePosixPath

from research_gym.core.hybrid import (
    CandidateState,
    HybridMode,
    LatticeProposal,
    certify_and_apply_verified,
)
from research_gym.core.typed_soundness import SoundnessType

from .context import MAX_SOURCE_BYTES, TEXT_SUFFIXES

SHARD_ID = re.compile(r"[a-z][a-z0-9-]{0,39}\Z")
PROTECTED = re.compile(
    r"(?:^|[^a-z0-9])(?:sealed|private|hidden|holdout|heldout|held[-_]out|gold|secret|secrets|credentials|lock)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
IGNORED = {"node_modules", "target", "checkpoints", "vendor", "training_data", "transcripts", "__pycache__"}
STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "find", "for", "from", "how",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "use", "what", "which", "with"}
KNOWN_EXTENSIONS = TEXT_SUFFIXES | {".lean", ".metta", ".tex", ".typ", ".rst", ".ipynb", ".sh", ".ps1", ".sv", ".v"}


def eligible(filename: str) -> bool:
    path = PurePosixPath(filename)
    return (path.suffix.lower() in KNOWN_EXTENSIONS
            and not any(part.startswith(".") or part.lower() in IGNORED or PROTECTED.search(part)
                        for part in path.parts))


def terms(query: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    useful = [token for token in tokens if len(token) > 1 and token not in STOP]
    return tuple(dict.fromkeys((useful or tokens)[:24]))


def git_entries(repo: Path, commit: str):
    tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-rlz", commit])
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, oid, size = metadata.decode("ascii").split()
        path = name.decode("utf-8", errors="replace")
        if kind == "blob" and mode in {"100644", "100755"} and eligible(path):
            weight = int(size)
            if 0 < weight <= MAX_SOURCE_BYTES:
                yield path, oid, weight


def sections(text: str):
    lines = text.splitlines()
    for offset in range(0, len(lines), 64):
        yield offset + 1, "\n".join(lines[offset:offset + 64])


def build_shard(repo: Path, commit: str, output: Path) -> dict:
    counters = {"files": 0, "chunks": 0, "bytes": 0, "binary": 0}
    with closing(sqlite3.connect(output)) as database, subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    ) as git:
        database.execute("PRAGMA cache_size=-16384")
        database.executescript("""
            CREATE TABLE files (path TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE VIRTUAL TABLE chunks USING fts5(path UNINDEXED, start UNINDEXED, body, names);
            CREATE VIRTUAL TABLE vocabulary USING fts5vocab(chunks, row);
        """)
        for path, oid, size in git_entries(repo, commit):
            git.stdin.write(oid.encode("ascii") + b"\n")
            git.stdin.flush()
            expected = f"{oid} blob {size}\n".encode("ascii")
            if git.stdout.readline() != expected:
                raise ValueError(f"Git returned an unexpected blob for {path}")
            raw = git.stdout.read(size)
            if len(raw) != size or git.stdout.read(1) != b"\n":
                raise ValueError(f"Git returned an incomplete blob for {path}")
            if b"\x00" in raw or raw.startswith(b"version https://git-lfs.github.com/spec/"):
                counters["binary"] += 1
                continue
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError:
                counters["binary"] += 1
                continue
            database.execute("INSERT INTO files VALUES (?, ?)", (path, hashlib.sha256(raw).hexdigest()))
            for start, segment in sections(body):
                database.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (path, start, segment, path))
                counters["chunks"] += 1
            counters["files"] += 1
            counters["bytes"] += size
        git.stdin.close()
        if git.wait() != 0:
            raise ValueError("Git object reader failed")
        database.commit()
    return counters


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)


def build(spec: Path, output: Path) -> dict:
    output = output.resolve()
    repositories = json.loads(spec.read_text(encoding="utf-8-sig"))["repositories"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="federation-build-", dir=output.parent) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        repos = {}
        with closing(sqlite3.connect(stage / "catalog.sqlite")) as catalog:
            catalog.execute("PRAGMA cache_size=-16384")
            catalog.execute("CREATE TABLE postings (term TEXT, shard TEXT, documents INTEGER, "
                            "PRIMARY KEY (term, shard)) WITHOUT ROWID")
            for item in repositories:
                name = item["id"]
                if not SHARD_ID.fullmatch(name) or name in repos:
                    raise ValueError(f"invalid or duplicate index id: {name}")
                repo = Path(item["path"]).resolve(strict=True)
                commit = subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "--verify", item.get("commit", "HEAD") + "^{commit}"],
                    text=True,
                ).strip()
                path = stage / f"{name}.sqlite"
                counts = build_shard(repo, commit, path)
                if not counts["chunks"]:
                    raise ValueError(f"no eligible source in {name}")
                with closing(readonly(path)) as shard:
                    catalog.executemany("INSERT INTO postings VALUES (?, ?, ?)", (
                        (term, name, documents) for term, documents in shard.execute("SELECT term, doc FROM vocabulary")
                    ))
                catalog.commit()
                repos[name] = {"commit": commit, "label": item.get("label", name), **counts}
                print(json.dumps({"index": name, **repos[name]}), flush=True)
        if not repos:
            raise ValueError("no repository indexes configured")
        revision = hashlib.sha256(json.dumps(repos, sort_keys=True).encode()).hexdigest()
        manifest = {"version": 1, "revision": revision, "repositories": repos}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return manifest


class Federation:
    def __init__(self, root: Path, router=None):
        self.root = root.resolve(strict=True)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.repos = self.manifest["repositories"]
        if (self.manifest["version"] != 1 or not self.repos
                or not all(SHARD_ID.fullmatch(name) for name in self.repos)):
            raise ValueError("unsupported federation manifest")
        self.revision = self.manifest["revision"]
        self.router = router
        self.handles = {}
        self.reads = []
        self.routes = []
        self.previous_search = None

    def matches(self, tokens: tuple[str, ...]) -> dict:
        placeholders = ",".join("?" for _ in tokens)
        hits = defaultdict(dict)
        with closing(readonly(self.root / "catalog.sqlite")) as catalog:
            for term, shard, documents in catalog.execute(
                f"SELECT term, shard, documents FROM postings WHERE term IN ({placeholders})", tokens
            ):
                hits[shard][term] = documents
        return hits

    def shortlist(self, query: str, tokens: tuple[str, ...], *, scope=None) -> tuple[list[dict], list[str]]:
        hits = self.matches(tokens)
        scope = frozenset(self.repos if scope is None else scope)
        current = CandidateState({"indexes": scope})
        eliminated = scope - hits.keys()
        if eliminated:
            def verify(_proposal, revision):
                # Recheck the complete vocabulary of this revision for the exact OR query.
                if revision == self.revision and eliminated.isdisjoint(self.matches(tokens)):
                    return SoundnessType.ENV_SOUND_DEAD
                return SoundnessType.UNKNOWN

            decision = certify_and_apply_verified(
                current, LatticeProposal(current.eliminate("indexes", eliminated), SoundnessType.MODEL_SOUND_DEAD,
                                         mode=HybridMode.ABSTAIN if eliminated == scope else HybridMode.DEDUCE),
                provenance_verifier=verify, verifier_context=self.revision, require_progress=True,
            )
            if decision.accepted:
                current = decision.after
            else:
                eliminated = frozenset()
        rows = []
        for name in sorted(current.domains["indexes"]):
            documents = self.repos[name]["chunks"]
            frequencies = hits.get(name, {})
            score = sum(1 / (1 + 12 * frequencies[term] / documents) for term in frequencies)
            rows.append({"id": name, "path": self.repos[name]["label"], "uri": f"index:{name}",
                         "signature": " ".join(frequencies), "score": score})
        rows.sort(key=lambda row: (-row["score"], row["id"]))
        if self.router and len(rows) > 1:
            rows = self.router.rank(query, rows)
        return rows, sorted(eliminated)

    def query(self, query: str, limit: int = 8, cursor: int = 0, *, fanout: int = 2) -> dict:
        if type(limit) is not int or not 1 <= limit <= 16 or type(cursor) is not int or not 0 <= cursor <= 1024:
            raise ValueError("invalid index pagination")
        scope = None
        if query.startswith("@"):
            selector, separator, query = query.partition(" ")
            name = selector[1:]
            if not separator or name not in self.repos:
                raise ValueError("use @index-id followed by search terms")
            scope = (name,)
        tokens = terms(query)
        if not tokens:
            raise ValueError("search requires at least one word")
        key = (query, scope, fanout)
        if cursor and self.previous_search and self.previous_search[0] == key:
            _, ranked, excluded = self.previous_search
        else:
            ranked, excluded = self.shortlist(query, tokens, scope=scope)
            self.previous_search = (key, ranked, excluded)
        searched = [row["id"] for row in ranked[:fanout]]
        expression = " OR ".join(f'"{token}"' for token in tokens)
        rows = []
        for priority, name in enumerate(searched):
            with closing(readonly(self.root / f"{name}.sqlite")) as shard:
                matches = shard.execute("""
                    SELECT rowid, path, start, bm25(chunks, 0, 0, 1, 2)
                    FROM chunks WHERE chunks MATCH ? ORDER BY 4, rowid LIMIT ?
                """, (expression, cursor + limit + 1)).fetchall()
            for position, (rowid, path, start, rank) in enumerate(matches):
                candidate = {"uri": f"index:{name}:{rowid}", "path": f"{name}/{path}", "start": start,
                             "rank": rank, "index": name, "commit": self.repos[name]["commit"]}
                rows.append(((position, priority), candidate))
        # FTS ranks are local to a shard. Interleave the lists in router order.
        rows.sort(key=lambda row: row[0])
        route = {"searched": searched, "deferred": [row["id"] for row in ranked[fanout:]],
                 "excluded": excluded, "proof": "no exact OR-term postings", "terms": tokens,
                 "revision": self.revision, "policy": "settrm" if self.router else "lexical"}
        self.routes.append(route)
        del self.routes[:-10]
        end = cursor + limit
        return {"candidates": [candidate for _, candidate in rows[cursor:end]], "routing": route,
                "next_cursor": end if end <= 1024 and len(rows) > end else None}

    def read(self, *, uri: str | None = None, handle: str | None = None, start: int | None = None) -> dict:
        if (uri is None) == (handle is None) or (handle is not None and start is not None):
            raise ValueError("supply an indexed source URI or a previously returned handle")
        if handle is not None:
            if handle not in self.handles:
                raise ValueError("unknown indexed source handle")
            uri, start = self.handles[handle]
        prefix, name, rowid = uri.split(":")
        if prefix != "index" or name not in self.repos or not rowid.isdecimal() or len(rowid) > 12:
            raise ValueError("invalid indexed source URI")
        with closing(readonly(self.root / f"{name}.sqlite")) as shard:
            item = shard.execute("""
                SELECT c.path, c.start, c.body, f.digest FROM chunks c
                JOIN files f ON f.path = c.path WHERE c.rowid = ?
            """, (int(rowid),)).fetchone()
        if item is None:
            raise ValueError("unknown indexed source URI")
        path, first, source, digest = item
        lines = source.split("\n")
        if start is None:
            start = first
        if type(start) is not int or not first <= start < first + len(lines):
            raise ValueError("start is outside the indexed source section")
        fragment = []
        remaining = 10_000
        for line in lines[start - first:]:
            if len(line) > remaining:
                if not fragment:
                    fragment.append(line[:remaining])
                break
            fragment.append(line)
            remaining -= len(line) + 1
        end = start + len(fragment) - 1
        handle = "indexed:" + hashlib.sha256(f"{self.revision}:{uri}:{start}".encode()).hexdigest()
        self.handles[handle] = (uri, start)
        self.reads.append(uri)
        del self.reads[:-40]
        return {"uri": uri, "path": f"{name}/{path}", "start": start, "end": end,
                "source": "\n".join(fragment), "handle": handle, "source_digest": digest,
                "revision": self.revision, "commit": self.repos[name]["commit"],
                "next_line": end + 1 if end + 1 < first + len(lines) else None}

    def state(self) -> dict:
        return {"revision": self.revision, "indexes": self.repos,
                "recent_routes": self.routes[-10:], "read_history": self.reads[-40:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.spec, args.out)


if __name__ == "__main__":
    main()
