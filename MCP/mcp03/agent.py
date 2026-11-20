"""Routing loop, packet builder and baselines for code-repository retrieval.

The observation reuses MCP-02's boundary: ``needs`` (salient issue terms) plus a
frontier of candidates with ``text`` and the nine-value ``struct`` vector, so
MCP-02's routers and encoders run unchanged. Item types map onto MCP-02's
slots: search query = ``query``, directory or module = ``branch``, symbol =
``doc``, result page = ``page``.

The packet is the compact prompt handed to a frontier agent: a ranked file map
followed by source snippets of the symbols the router read, cut to a token
budget with a named tokenizer.
"""
from __future__ import annotations

import math
import random
import re
import time
from typing import Any

from MCP.mcp02.agent import ITEM_TYPES, Frontier, Policy
from MCP.mcp02.textutil import BM25, byte_size, digest
from .env import CodeEnv, code_tokens
from .tokens import count_tokens

_BACKTICK = re.compile(r"`([^`\n]{1,80})`")
_PATH = re.compile(r"[\w./-]+\.py")
_TRACE_FN = re.compile(r"\bin ([A-Za-z_][A-Za-z0-9_]*)\b")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_STOP = set("the a an and or of to in is it this that for with on as be are was by not when if from".split())


def salient_terms(issue: str, limit: int = 24) -> list[str]:
    """Public, deterministic extraction of code-like terms from an issue, in order of appearance."""
    found: list[tuple[int, str]] = []
    for rx in (_BACKTICK, _PATH, _TRACE_FN, _DOTTED, _CAMEL, _SNAKE):
        for m in rx.finditer(issue):
            text = m.group(1) if rx.groups else m.group(0)
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
                if len(tok) > 2 and tok.lower() not in _STOP:
                    found.append((m.start(), tok))
    title = issue.strip().split("\n")[0]
    for i, w in enumerate(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", title)):
        if w.lower() not in _STOP:
            found.append((10_000_000 + i, w))
    out, seen = [], set()
    for _pos, tok in sorted(found, key=lambda x: x[0]):
        if tok.lower() not in seen:
            seen.add(tok.lower())
            out.append(tok)
    return out[:limit]


def _struct(item_type: str, depth: float = 0, cursor: int = 0, count: int = 0) -> list[float]:
    return [1.0 if item_type == t else 0.0 for t in ITEM_TYPES] + [depth / 6.0, min(cursor, 64) / 32.0,
                                                                   math.log1p(count) / 8.0]


def row_item(row: dict, depth: int) -> dict:
    kind = row["kind"]
    if kind == "symbol":
        text = f"symbol {row['type']} {row['qualname']} in {row['path']} {row['signature']} {row['doc']}"
        return {"id": row["node_id"], "type": "doc", "tool": "read_symbol", "arguments": {"uri": row["uri"]},
                "text": text, "meta": row, "struct": _struct("doc", depth, 0, row["lines"])}
    if kind == "module":
        flags = " ".join(k for k, v in row["flags"].items() if v)
        text = f"module {row['path']} {row['doc']} symbols {row['symbols']} {flags}"
        return {"id": row["node_id"], "type": "branch", "tool": "index_children",
                "arguments": {"node_id": row["node_id"]}, "text": text, "meta": row,
                "struct": _struct("branch", depth, 0, row["symbols"])}
    flags = " ".join(k for k, v in row["flags"].items() if v)
    text = f"directory {row['path']} modules {' '.join(row['sketch'])} files {row['files']} {flags}"
    return {"id": row["node_id"], "type": "branch", "tool": "index_children", "arguments": {"node_id": row["node_id"]},
            "text": text, "meta": row, "struct": _struct("branch", depth, 0, row["files"])}


def query_item(query: str) -> dict:
    return {"id": f"query:{query}", "type": "query", "tool": "code_search", "arguments": {"query": query},
            "text": f"search {query}", "meta": {"query": query}, "struct": _struct("query")}


def page_item(tool: str, key: str, value: str, cursor: int, remaining: int, depth: int) -> dict:
    return {"id": f"page:{tool}:{value}:{cursor}", "type": "page", "tool": tool,
            "arguments": {key: value, "cursor": cursor}, "text": f"more results {value} remaining {remaining}",
            "meta": {"key": value, "cursor": cursor}, "struct": _struct("page", depth, cursor, remaining)}


class CodeOracleView:
    """Private labels: a candidate is positive when it leads to a gold edit target."""

    def __init__(self, env: CodeEnv):
        self.env = env
        self.files = set(env._oracle["files"])
        self.syms = set(env._oracle["symbols"])

    def _symbol_positive(self, uri: str) -> bool:
        if uri in self.syms:
            return True
        s = self.env.symbols.get(uri)
        if s is None:
            return False
        for g in self.syms:
            if "::" in g and g.split("::")[0] == s.path:
                t = self.env.symbols.get(g)
                if t is not None and s.start <= t.start and t.end <= s.end and s.kind == "class":
                    return True
        return False

    def label(self, item: dict) -> int:
        env = self.env
        if item["type"] == "doc":
            return int(self._symbol_positive(item["meta"]["uri"]))
        if item["type"] == "branch":
            return int(bool(self.files & set(env.files_under(item["id"]))))
        if item["type"] == "query":
            hits = env.index.search(item["meta"]["query"])[:16]
            return int(any(self._symbol_positive(u) or u.split("::")[0] in self.files and u in self.syms
                           for _s, u in hits))
        if item["type"] == "page":
            key, cursor = item["meta"]["key"], item["meta"]["cursor"]
            if item["tool"] == "code_search":
                hits = env.index.search(key)[cursor:cursor + 16]
                return int(any(self._symbol_positive(u) for _s, u in hits))
            kids = env._children.get(key, [])[cursor:]
            return int(any(self.files & set(env.files_under(k)) for _t, k in kids))
        return 0

    def exclusion_label(self, row: dict) -> str:
        files = self.env.files_under(row["node_id"]) if row["kind"] != "symbol" else []
        if files and all(self.env.flags[f]["is_test"] for f in files):
            return "test_path"
        if files and all(self.env.flags[f]["is_doc"] for f in files):
            return "doc_path"
        return "none"


def symbolic_code_exclusions(rows: list[dict]) -> list[tuple[str, str]]:
    out = []
    for row in rows:
        if row["kind"] in ("dir", "module"):
            if row["flags"].get("is_test"):
                out.append((row["node_id"], "test_path"))
            elif row["flags"].get("is_doc"):
                out.append((row["node_id"], "doc_path"))
    return out


def _fixed_priority(cand: dict, score: float) -> float:
    """Queries carry every issue term and pages repeat a query, so the lexical controls rank them by rule:
    issue each search query first, in frontier order; take another page only when nothing else is left."""
    if cand["type"] == "query":
        return 1e6 - cand["order"]
    if cand["type"] == "page":
        return -1e6 + score * 1e-3
    return score


class CodeLexicalPolicy(Policy):
    name, family = "lexical_bm25", "lexical"

    def score(self, obs: dict) -> list[float]:
        index = BM25([" ".join(code_tokens(c["text"])) for c in obs["candidates"]])
        query = " ".join(code_tokens(obs["task"]["request"]))
        scores = index.scores(query)
        self._ops = index.operations
        return [_fixed_priority(c, s) for s, c in zip(scores, obs["candidates"])]

    def last_cost(self) -> dict:
        return {"flops": 0, "router_tokens": 0, "lexical_ops": getattr(self, "_ops", 0)}


class CodeIdentifierPolicy(Policy):
    """Deterministic control: count salient issue identifiers matched exactly in the candidate."""

    name, family = "identifier_rules", "deterministic"

    def score(self, obs: dict) -> list[float]:
        terms = [t.lower() for t in obs["need_terms"]]
        weights = {t: 1.0 / (1 + i * 0.1) for i, t in enumerate(terms)}
        out = []
        for c in obs["candidates"]:
            toks = set(code_tokens(c["text"]))
            s = sum(w for t, w in weights.items() if t in toks)
            flags = c["meta"].get("flags", {}) if isinstance(c["meta"], dict) else {}
            s -= 2.0 * bool(flags.get("is_test")) + 2.0 * bool(flags.get("is_doc"))
            s += {"doc": 0.5, "branch": 0.2}.get(c["type"], 0)
            out.append(_fixed_priority(c, s))
        return out


class CodeShamPolicy(Policy):
    name, family = "sham_random", "sham"

    def score(self, obs: dict) -> list[float]:
        rng = random.Random(digest([self.episode_key, obs["step"], [c["id"] for c in obs["candidates"]]]))
        return [rng.random() for _ in obs["candidates"]]


class CodeOraclePolicy(Policy):
    name, family, uses_private_oracle = "oracle_upper_bound", "oracle", True
    view: CodeOracleView | None = None

    def score(self, obs: dict) -> list[float]:
        return [float(self.view.label(c)) + 0.01 * (c["type"] == "doc") for c in obs["candidates"]]


def make_code_control(name: str) -> Policy:
    return {"lexical_bm25": CodeLexicalPolicy, "identifier_rules": CodeIdentifierPolicy,
            "sham_random": CodeShamPolicy, "oracle_upper_bound": CodeOraclePolicy}[name]()


CODE_CONTROLS = ("identifier_rules", "lexical_bm25", "sham_random", "oracle_upper_bound")


def _prefilter(need: str, items: list[dict], k: int) -> tuple[list[dict], int]:
    keep = [it for it in items if it["type"] == "query"]
    rest = [it for it in items if it["type"] != "query"]
    if len(rest) <= k:
        return keep + rest, 0
    index = BM25([" ".join(code_tokens(it["text"])) for it in rest])
    scores = index.scores(" ".join(code_tokens(need)))
    ranked = sorted(range(len(rest)), key=lambda i: (-scores[i], rest[i]["order"]))[:k]
    return keep + sorted((rest[i] for i in ranked), key=lambda it: it["order"]), index.operations


def build_packet(env: CodeEnv, reads: list[tuple[float, str]], seen: dict[str, float], budget: int,
                 snippet_lines: int = 40, file_map: list[str] | None = None) -> dict:
    """Ranked file map plus snippets of read symbols, cut to ``budget`` tokens.

    ``file_map`` replaces the router-derived file ranking (the composed packet uses the ranked
    paths returned by one ``file_search`` call)."""
    file_score: dict[str, float] = {}
    for uri, score in seen.items():
        path = uri.split("::")[0][2:] if uri.startswith(("s:", "f:")) else None
        if path:
            file_score[path] = max(file_score.get(path, -1e9), score)
    for score, uri in reads:
        path = uri.split("::")[0]
        file_score[path] = max(file_score.get(path, -1e9), score + 1.0)
    ranked_files = [p for p, _s in sorted(file_score.items(), key=lambda x: (-x[1], x[0]))]
    if file_map is not None:
        ranked_files = list(file_map) + [p for p in ranked_files if p not in file_map]
    parts, used = [], 0
    header = "Relevant files (ranked):\n"
    map_files = []
    for path in ranked_files[:12]:
        line = f"- {path}\n"
        cost = count_tokens(line)
        if used + cost + count_tokens(header) > budget:
            break
        map_files.append(path)
        used += cost
    if map_files:
        used += count_tokens(header)
        parts.append(header + "".join(f"- {p}\n" for p in map_files))
    snippets = []
    for score, uri in sorted(reads, key=lambda x: -x[0]):
        s = env.symbols[uri]
        lines = env.lines.get(s.path, [])[s.start - 1:min(s.end, s.start - 1 + snippet_lines)]
        text = f"### {uri} (lines {s.start}-{s.end})\n" + "\n".join(lines) + "\n"
        cost = count_tokens(text)
        if used + cost > budget:
            continue
        snippets.append({"uri": uri, "tokens": cost})
        parts.append(text)
        used += cost
    body = "\n".join(parts)
    return {"text": body, "tokens": count_tokens(body), "bytes": len(body.encode("utf-8")), "snippets": snippets,
            "map": map_files, "ranked_files": ranked_files, "budget": budget}


def run_code_episode(env: CodeEnv, issue: str, *, policy: Policy, max_tool_calls: int = 24,
                     budgets: tuple[int, ...] = (1024, 2048, 4096), prefilter_k: int = 32,
                     record=None, tape: list | None = None, return_text: bool = False,
                     compose: bool = False) -> dict:
    started = time.perf_counter()
    terms = salient_terms(issue)
    need = " ".join(terms) or issue.strip().split("\n")[0]
    view = CodeOracleView(env)
    if isinstance(policy, CodeOraclePolicy):
        policy.view = view
    policy.reset(f"{env.snapshot.repo}/{env.snapshot.commit}/{env.arm}")
    frontier = Frontier()
    if env.arm in ("search", "hybrid", "layered_ldt"):
        frontier.add(query_item(need))
        for t in terms[:3]:
            frontier.add(query_item(t))
    if env.arm in ("layered", "hybrid", "layered_ldt"):
        frontier.add(row_item(env.row("d:"), 0))
    reads: list[tuple[float, str]] = []
    seen: dict[str, float] = {}
    tape_out: list = []
    tape_iter = iter(tape) if tape is not None else None
    totals = {"flops": 0, "router_tokens": 0, "lexical_ops": 0, "prefilter_ops": 0, "router_seconds": 0.0,
              "decisions": 0, "candidates_scored": 0, "result_bytes": 0, "result_tokens": 0}
    result_digests = []
    task = {"request": issue}
    while env.stats["tool_calls"] < max_tool_calls:
        eligible = frontier.eligible()
        if not eligible:
            break
        cands, ops = _prefilter(need, eligible, prefilter_k)
        totals["prefilter_ops"] += ops
        obs = {"task": task, "needs": [need], "need_terms": terms, "candidates": cands,
               "step": env.stats["tool_calls"], "evidence": [u for _s, u in reads]}
        if record is not None:
            record({"kind": "decide", "obs": obs, "labels": [view.label(c) for c in cands]})
        t0 = time.perf_counter()
        if tape_iter is not None:
            entry = next(tape_iter)
            if entry[0] != "d" or len(entry[2]) != len(cands):
                raise RuntimeError("tape observation mismatch")
            best, norm = entry[1], entry[2]
            cost = {"flops": entry[3], "router_tokens": entry[4], "lexical_ops": entry[5]}
        else:
            scores = policy.score(obs)
            cost = policy.last_cost()
            if len(scores) != len(cands):
                raise RuntimeError("policy returned the wrong number of scores")
            best = max(range(len(cands)), key=lambda i: (scores[i], -cands[i]["order"]))
            lo, hi = min(scores), max(scores)
            norm = [round((x - lo) / (hi - lo), 4) if hi > lo else 0.5 for x in scores]
        totals["router_seconds"] += time.perf_counter() - t0
        for k in ("flops", "router_tokens", "lexical_ops"):
            totals[k] += int(cost.get(k, 0))
        totals["decisions"] += 1
        totals["candidates_scored"] += len(cands)
        item = cands[best]
        tape_out.append(["d", best, norm, int(cost.get("flops", 0)), int(cost.get("router_tokens", 0)),
                         int(cost.get("lexical_ops", 0))])
        # Normalised scores within this decision order the packet's snippets and file map.
        rel = norm[best]
        for c, x in zip(cands, norm):
            seen[c["id"]] = max(seen.get(c["id"], -1e9), x)
        frontier.visited.add(item["id"])
        frontier.remove(item["id"])
        args = dict(item["arguments"])
        if item["tool"] in ("code_search", "index_children"):
            args["limit"] = env.page_limit
        result = env.call(item["tool"], args)
        payload = result["structuredContent"]
        rtext = result["content"][0]["text"]
        totals["result_bytes"] += len(rtext.encode("utf-8"))
        totals["result_tokens"] += count_tokens(rtext)
        result_digests.append(digest(payload))
        if result["isError"]:
            continue
        depth = item["struct"][len(ITEM_TYPES)] * 6 + 1
        if item["tool"] == "code_search":
            for row in payload["results"]:
                frontier.add(row_item(row, 1))
            if payload["next_cursor"] is not None:
                frontier.add(page_item("code_search", "query", payload["query"], payload["next_cursor"],
                                       payload["total"] - payload["next_cursor"], 1))
        elif item["tool"] == "index_children":
            rows = payload["children"]
            for row in rows:
                frontier.add(row_item(row, int(depth)))
            if payload["next_cursor"] is not None:
                frontier.add(page_item("index_children", "node_id", payload["node_id"], payload["next_cursor"],
                                       payload["total"] - payload["next_cursor"], int(depth)))
            if env.arm == "layered_ldt" and rows and env.stats["tool_calls"] < max_tool_calls:
                if record is not None:
                    record({"kind": "rows", "obs": obs, "rows": rows,
                            "labels": [view.exclusion_label(r) for r in rows]})
                if tape_iter is not None:
                    proposals = [tuple(p) for p in next(tape_iter)[1]]
                else:
                    proposals = (policy.code_exclusions(obs, rows) if hasattr(policy, "code_exclusions")
                                 else symbolic_code_exclusions(rows))
                tape_out.append(["p", [list(p) for p in proposals]])
                by_pred: dict[str, list[str]] = {}
                for child, pred in proposals:
                    by_pred.setdefault(pred, []).append(child)
                for pred, children in sorted(by_pred.items()):
                    if env.stats["tool_calls"] >= max_tool_calls:
                        break
                    refine = env.call("candidate_refine", {"node_id": payload["node_id"], "exclude": children[:64],
                                                           "predicate": pred})
                    for child in refine["structuredContent"].get("excluded", []):
                        frontier.remove(child)
                        frontier.visited.add(child)
        elif item["tool"] == "read_symbol":
            reads.append((rel, payload["uri"]))
    packets = {}
    composed_map = None
    if compose and budgets:
        # One counted MCP call after exploration: whole-file BM25 ranking used as the packet's file map.
        fs = env.call("file_search", {"query": issue, "limit": 16})
        totals["result_bytes"] += len(fs["content"][0]["text"].encode("utf-8"))
        totals["result_tokens"] += count_tokens(fs["content"][0]["text"])
        result_digests.append(digest(fs["structuredContent"]))
        composed_map = fs["structuredContent"].get("paths", [])[:12]
        for b in budgets:
            packet = build_packet(env, reads, seen, b, file_map=composed_map)
            packets[f"c{b}"] = {"tokens": packet["tokens"], "bytes": packet["bytes"], "grade": env.grade(packet),
                                "snippets": len(packet["snippets"]), "map": len(packet["map"])}
            if return_text:
                packets[f"c{b}"]["text"] = packet["text"]
    for b in budgets:
        packet = build_packet(env, reads, seen, b)
        packets[str(b)] = {"tokens": packet["tokens"], "bytes": packet["bytes"], "grade": env.grade(packet),
                           "snippets": len(packet["snippets"]), "map": len(packet["map"])}
        if return_text:
            packets[str(b)]["text"] = packet["text"]
    return {"repo": env.snapshot.repo, "commit": env.snapshot.commit, "arm": env.arm,
            "policy": policy.describe(), "tool_calls": env.stats["tool_calls"], "environment_stats": dict(env.stats),
            "private_checks": dict(env.private), "reads": [u for _s, u in reads], "packets": packets,
            "compute": totals, "results_sha256": digest(result_digests), "tape": tape_out,
            "need_terms": terms, "elapsed_seconds": time.perf_counter() - started}


def bm25_file_baseline(env: CodeEnv, issue: str, budgets: tuple[int, ...]) -> dict:
    """SWE-bench-style retrieval: BM25 of the issue over whole files, packed whole until the budget."""
    paths = sorted(env.snapshot.sources)
    index = BM25([" ".join(code_tokens(env.snapshot.sources[p])) for p in paths])
    scores = index.scores(" ".join(code_tokens(issue)))
    ranked = [p for _s, p in sorted(zip(scores, paths), key=lambda x: (-x[0], x[1]))]
    out = {}
    for b in budgets:
        used, chosen = 0, []
        for p in ranked[:50]:
            cost = count_tokens(env.snapshot.sources[p])
            if used + cost > b:
                continue
            chosen.append(p)
            used += cost
        packet = {"snippets": [], "map": chosen, "ranked_files": ranked[:50]}
        out[str(b)] = {"tokens": used, "grade": env.grade(packet), "files": len(chosen)}
    maps = {}
    for k in (5, 12, 25):
        text = "Relevant files (ranked):" + chr(10) + "".join(f"- {p}" + chr(10) for p in ranked[:k])
        maps[str(k)] = {"tokens": count_tokens(text), "grade": env.grade({"snippets": [], "map": ranked[:k],
                                                                          "ranked_files": ranked[:k]})}
    return {"packets": out, "path_maps": maps, "lexical_ops": index.operations}


def oracle_context_tokens(env: CodeEnv) -> dict:
    """Token size of the gold files in full (SWE-bench 'oracle' retrieval) and of the whole snapshot."""
    gold = env._oracle["files"]
    return {"gold_files_tokens": sum(count_tokens(env.snapshot.sources[f]) for f in gold if f in env.snapshot.sources),
            "repo_python_bytes": sum(len(s.encode("utf-8")) for s in env.snapshot.sources.values()),
            "repo_python_files": len(env.snapshot.sources)}
