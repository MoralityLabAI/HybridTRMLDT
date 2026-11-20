"""Code-repository MCP surface over one snapshot, with a private localisation judge.

Arms:

* ``search``: ``code_search`` (BM25 over symbol documents) plus reads.
* ``layered``: ``index_children`` over directories, modules and symbols plus reads.
* ``hybrid``: both discovery tools.
* ``layered_ldt``: ``hybrid`` plus ``candidate_refine``. Exclusions use the
  registered path predicates ``test_path`` and ``doc_path``. The public task
  constraint says edits belong in library source, so a certified exclusion
  needs every file below the excluded node to carry the flag. Each exclusion
  passes through ``certify_and_apply_verified``.

Reads: ``read_outline(path)`` lists a module's symbols; ``read_symbol(uri)``
returns up to ``MAX_READ_LINES`` source lines. All results are JSON.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from research_gym.core.hybrid import CandidateState, HybridMode, LatticeProposal, certify_and_apply_verified
from research_gym.core.typed_soundness import SoundnessType

from MCP.mcp02.textutil import canonical_json, digest
from .repo import Snapshot, Symbol, path_flags

ARMS = ("search", "layered", "hybrid", "layered_ldt")
PREDICATES = ("test_path", "doc_path")
MAX_READ_LINES = 80
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")


def code_tokens(text: str) -> list[str]:
    """Identifiers plus their snake/camel parts, lower-cased."""
    out = []
    for ident in _IDENT.findall(text):
        low = ident.lower()
        out.append(low)
        parts = [p.lower() for chunk in ident.split("_") for p in _SPLIT.findall(chunk)]
        if len(parts) > 1:
            out.extend(parts)
    return out


class SymbolIndex:
    """BM25 over symbol documents: path, qualname, signature, docstring, body identifiers."""

    def __init__(self, docs: list[tuple[str, list[str]]], k1: float = 1.2, b: float = 0.75):
        self.ids = [d for d, _ in docs]
        self.tf = [Counter(toks) for _, toks in docs]
        self.len = [len(toks) for _, toks in docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tf):
            for t in tf:
                self.postings.setdefault(t, []).append(i)
        self.k1, self.b = k1, b
        self.operations = 0

    def search(self, query: str) -> list[tuple[float, str]]:
        scores: dict[int, float] = {}
        for term in set(code_tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                self.operations += 1
                tf = self.tf[i][term]
                norm = tf + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return sorted(((s, self.ids[i]) for i, s in scores.items()), key=lambda x: (-x[0], x[1]))


def symbol_document(sym: Symbol, source_lines: list[str]) -> list[str]:
    body = " ".join(source_lines[sym.start - 1:min(sym.end, sym.start + 59)]) if sym.kind != "module" else ""
    return code_tokens(f"{sym.path} {sym.qualname} {sym.signature} {sym.doc}") + code_tokens(body)[:200]


class CodeEnv:
    """One task episode over a repository snapshot."""

    def __init__(self, snapshot: Snapshot, task: dict, oracle: dict, arm: str, page_limit: int = 8,
                 index: SymbolIndex | None = None, file_index: SymbolIndex | None = None):
        if arm not in ARMS:
            raise ValueError("unknown arm")
        self.snapshot, self.task, self.arm, self.page_limit = snapshot, task, arm, page_limit
        self._oracle = oracle
        self.symbols: dict[str, Symbol] = {s.uri: s for s in snapshot.all_symbols()}
        self.lines = {p: src.splitlines() for p, src in snapshot.sources.items()}
        self.flags = {p: path_flags(p) for p in snapshot.files}
        self._index = index
        self._file_index = file_index
        self._children: dict[str, list[tuple[str, str]]] = {}
        self._build_tree()
        self._states: dict[str, CandidateState] = {}
        self.read_log: list[str] = []
        self.stats = {"tool_calls": 0, "tool_errors": 0, "search_calls": 0, "index_calls": 0, "outline_calls": 0,
                      "read_calls": 0, "refine_calls": 0, "ldt_checks": 0, "ldt_metadata_checks": 0,
                      "ldt_excluded": 0, "ldt_rejected": 0, "search_postings_ops": 0}
        self.private = {"certified_exclusions_containing_gold": 0}

    @property
    def index(self) -> SymbolIndex:
        if self._index is None:
            docs = [(s.uri, symbol_document(s, self.lines.get(s.path, []))) for s in self.symbols.values()]
            self._index = SymbolIndex(docs)
        return self._index

    @property
    def file_index(self) -> SymbolIndex:
        if self._file_index is None:
            self._file_index = SymbolIndex([(p, code_tokens(src)) for p, src in sorted(self.snapshot.sources.items())])
        return self._file_index

    def _build_tree(self) -> None:
        dirs: dict[str, set[str]] = {"": set()}
        for path in self.snapshot.files:
            parts = path.split("/")
            for i in range(len(parts) - 1):
                parent, child = "/".join(parts[:i]), "/".join(parts[:i + 1])
                dirs.setdefault(parent, set()).add("d:" + child)
                dirs.setdefault(child, set())
            dirs["/".join(parts[:-1])].add("f:" + path)
        self._dir_files: dict[str, list[str]] = {}
        for d, kids in dirs.items():
            self._children["d:" + d] = [(k[:1], k) for k in sorted(kids, key=lambda k: (k[:1] != "d", k))]
        for d in dirs:
            prefix = d + "/" if d else ""
            self._dir_files[d] = [p for p in self.snapshot.files if p.startswith(prefix)]
        for path, table in self.snapshot.symbols.items():
            top = [s for s in table if s.kind in ("class", "function")]
            self._children["f:" + path] = [("s", "s:" + s.uri) for s in top]
            for s in table:
                if s.kind == "class":
                    self._children["s:" + s.uri] = [("s", "s:" + m.uri) for m in table
                                                    if m.kind == "method" and m.qualname.startswith(s.qualname + ".")
                                                    and m.qualname.count(".") == s.qualname.count(".") + 1]

    def files_under(self, node: str) -> list[str]:
        if node.startswith("d:"):
            return self._dir_files.get(node[2:], [])
        if node.startswith("f:"):
            return [node[2:]]
        return [node[2:].split("::")[0]]

    def row(self, node: str) -> dict:
        if node.startswith("d:"):
            files = self.files_under(node)
            sketch = sorted({f.split("/")[-1][:-3] for f in files})[:6]
            return {"kind": "dir", "node_id": node, "path": node[2:] or ".", "files": len(files), "sketch": sketch,
                    "flags": {k: all(self.flags[f][k] for f in files) if files else False for k in ("is_test", "is_doc")}}
        if node.startswith("f:"):
            path = node[2:]
            table = self.snapshot.symbols.get(path, [])
            return {"kind": "module", "node_id": node, "path": path, "symbols": max(0, len(table) - 1),
                    "doc": table[0].doc if table else "", "flags": dict(self.flags[path])}
        s = self.symbols[node[2:]]
        return {"kind": "symbol", "node_id": node, "uri": s.uri, "path": s.path, "qualname": s.qualname,
                "type": s.kind, "signature": s.signature, "doc": s.doc, "lines": s.end - s.start + 1}

    def tools(self) -> list[dict]:
        def tool(name, desc, props, req):
            return {"name": name, "description": desc,
                    "inputSchema": {"type": "object", "properties": props, "required": req,
                                    "additionalProperties": False}}
        page = {"limit": {"type": "integer", "minimum": 1, "maximum": 16}, "cursor": {"type": "integer", "minimum": 0}}
        reads = [tool("read_outline", "List a module's classes, functions and methods with signatures.",
                      {"path": {"type": "string"}}, ["path"]),
                 tool("read_symbol", f"Read up to {MAX_READ_LINES} source lines of a symbol (path::qualname).",
                      {"uri": {"type": "string"}}, ["uri"])]
        out = []
        if self.arm in ("search", "hybrid", "layered_ldt"):
            out.append(tool("code_search", "BM25 search over symbols (path, name, signature, docstring, body "
                            "identifiers). Paginated.", {"query": {"type": "string"}, **page}, ["query"]))
        if self.arm in ("layered", "hybrid", "layered_ldt"):
            out.append(tool("index_children", "List children of an index node: directories, modules, classes, "
                            "methods. Root is d:.", {"node_id": {"type": "string"}, **page}, ["node_id"]))
        if self.arm == "layered_ldt":
            out.append(tool("candidate_refine", "Exclude children whose every file carries a registered path "
                            "flag (test_path, doc_path). Each exclusion is independently verified.",
                            {"node_id": {"type": "string"}, "exclude": {"type": "array", "items": {"type": "string"},
                                                                         "minItems": 1, "maxItems": 64},
                             "predicate": {"type": "string", "enum": list(PREDICATES)}},
                            ["node_id", "exclude", "predicate"]))
        out.append(tool("file_search", "BM25 over whole files; returns ranked file paths only (no content).",
                        {"query": {"type": "string"}, **page}, ["query"]))
        return out + reads

    def _page(self, args: dict) -> tuple[int, int]:
        limit, cursor = args.get("limit", self.page_limit), args.get("cursor", 0)
        if type(limit) is not int or not 1 <= limit <= 16 or type(cursor) is not int or cursor < 0:
            raise ValueError("bad pagination")
        return limit, cursor

    def _state(self, node: str) -> CandidateState:
        if node not in self._states:
            self._states[node] = CandidateState({"children": frozenset(k for _, k in self._children.get(node, []))})
        return self._states[node]

    def _refine(self, args: dict) -> dict:
        node, exclude, predicate = args["node_id"], args["exclude"], args["predicate"]
        if node not in self._children or predicate not in PREDICATES or not isinstance(exclude, list):
            raise ValueError("bad refine request")
        self.stats["refine_calls"] += 1
        flag = {"test_path": "is_test", "doc_path": "is_doc"}[predicate]
        state, accepted, rejected = self._state(node), [], []
        gold = set(self._oracle["files"])
        for child in exclude:
            if child not in state.domains["children"]:
                rejected.append(child)
                continue
            proposal = LatticeProposal(state.eliminate("children", {child}), SoundnessType.MODEL_SOUND_DEAD,
                                       mode=HybridMode.DEDUCE)

            def verify(p: LatticeProposal, ctx: Any, child=child) -> SoundnessType:
                self.stats["ldt_checks"] += 1
                if ctx != self.snapshot.commit:
                    return SoundnessType.UNKNOWN
                files = self.files_under(child)
                self.stats["ldt_metadata_checks"] += len(files)
                return SoundnessType.ENV_SOUND_DEAD if files and all(self.flags[f][flag] for f in files) \
                    else SoundnessType.UNKNOWN

            decision = certify_and_apply_verified(state, proposal, provenance_verifier=verify,
                                                  verifier_context=self.snapshot.commit, require_progress=True)
            if decision.accepted:
                state = decision.after
                accepted.append(child)
                if gold & set(self.files_under(child)):
                    self.private["certified_exclusions_containing_gold"] += 1
            else:
                rejected.append(child)
        self._states[node] = state
        self.stats["ldt_excluded"] += len(accepted)
        self.stats["ldt_rejected"] += len(rejected)
        return {"node_id": node, "excluded": accepted, "rejected": rejected,
                "candidates_after": len(state.domains["children"])}

    def call(self, name: str, args: dict) -> dict:
        self.stats["tool_calls"] += 1
        try:
            names = {t["name"] for t in self.tools()}
            if name not in names or not isinstance(args, dict):
                raise ValueError("unknown tool or bad arguments")
            if name == "code_search":
                self.stats["search_calls"] += 1
                limit, cursor = self._page(args)
                before = self.index.operations
                hits = self.index.search(str(args["query"]))
                self.stats["search_postings_ops"] += self.index.operations - before
                rows = [self.row("s:" + uri) for _s, uri in hits[cursor:cursor + limit]]
                payload = {"query": args["query"], "results": rows, "total": len(hits),
                           "next_cursor": cursor + limit if cursor + limit < len(hits) else None}
            elif name == "index_children":
                self.stats["index_calls"] += 1
                node = args["node_id"]
                if node == "root":
                    node = "d:"
                if node not in self._children:
                    raise ValueError("unknown node_id")
                limit, cursor = self._page(args)
                kids = self._children[node]
                window = kids[cursor:cursor + limit]
                if self.arm == "layered_ldt":
                    kept = self._state(node).domains["children"]
                    window = [k for k in window if k[1] in kept]
                payload = {"node_id": node, "children": [self.row(k) for _t, k in window], "total": len(kids),
                           "next_cursor": cursor + limit if cursor + limit < len(kids) else None}
            elif name == "file_search":
                self.stats["search_calls"] += 1
                limit, cursor = self._page({"limit": args.get("limit", 16), "cursor": args.get("cursor", 0)})
                before = self.file_index.operations
                hits = self.file_index.search(str(args["query"]))
                self.stats["search_postings_ops"] += self.file_index.operations - before
                payload = {"query": args["query"], "paths": [p for _s, p in hits[cursor:cursor + limit]],
                           "total": len(hits)}
            elif name == "read_outline":
                self.stats["outline_calls"] += 1
                path = args["path"]
                if path not in self.snapshot.symbols:
                    raise ValueError("unknown path")
                payload = {"path": path, "symbols": [{"uri": s.uri, "type": s.kind, "signature": s.signature}
                                                     for s in self.snapshot.symbols[path] if s.kind != "module"][:60]}
            elif name == "read_symbol":
                self.stats["read_calls"] += 1
                uri = args["uri"]
                if uri not in self.symbols:
                    raise ValueError("unknown uri")
                s = self.symbols[uri]
                lines = self.lines.get(s.path, [])[s.start - 1:s.end]
                payload = {"uri": uri, "path": s.path, "start": s.start, "end": s.end,
                           "truncated": len(lines) > MAX_READ_LINES, "source": "\n".join(lines[:MAX_READ_LINES])}
                self.read_log.append(uri)
            else:
                payload = self._refine(args)
            return {"content": [{"type": "text", "text": canonical_json(payload)}], "structuredContent": payload,
                    "isError": False}
        except (ValueError, KeyError, TypeError) as error:
            self.stats["tool_errors"] += 1
            payload = {"error": str(error)}
            return {"content": [{"type": "text", "text": canonical_json(payload)}], "structuredContent": payload,
                    "isError": True}

    # -- private judge ----------------------------------------------------------------
    def grade(self, packet: dict) -> dict:
        """Localisation of the hidden gold edit targets within the submitted packet."""
        gold_files, gold_syms = set(self._oracle["files"]), set(self._oracle["symbols"])
        snippet_uris = [s["uri"] for s in packet.get("snippets", []) if s.get("uri") in self.symbols]
        mapped = set(packet.get("map", []))
        files_hit = {f for f in gold_files if f in mapped or any(u.split("::")[0] == f for u in snippet_uris)}
        sym_hit = set()
        for g in gold_syms:
            for u in snippet_uris:
                if u == g:
                    sym_hit.add(g)
                elif "::" in g and "::" in u and g.split("::")[0] == u.split("::")[0]:
                    s_u, s_g = self.symbols[u], self.symbols.get(g)
                    if s_g is not None and s_u.start <= s_g.start and s_g.end <= s_u.end:
                        sym_hit.add(g)
                elif "::" not in g and u.split("::")[0] == g:
                    sym_hit.add(g)
        n_f, n_s = len(gold_files), len(gold_syms)
        return {"file_recall": len(files_hit) / n_f if n_f else 1.0, "all_files": files_hit == gold_files,
                "symbol_recall": len(sym_hit) / n_s if n_s else 1.0, "all_symbols": sym_hit == gold_syms,
                "gold_files": n_f, "gold_symbols": n_s,
                "first_file_rank": min((i for i, f in enumerate(packet.get("ranked_files", [])) if f in gold_files),
                                       default=None)}

    def gold_digest(self) -> str:
        return digest(self._oracle)
