"""Shared routing loop: one observation format and one action space for every policy.

A policy sees the public task, the current evidence needs and a frontier of
candidate expansions (index branches, result pages, resources, links, search
queries). It returns one score per candidate. The loop expands the best
unvisited candidate with exactly one MCP tool call. In the ``layered_ldt`` arm a
policy may also propose exclusions of index children; the environment certifies
each one before it changes the candidate set.

The plan tracker (``Composer``) uses only the public task and read results. It
is shared by all policies, so differences between policies come from routing.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable

from .env import ARMS, Env02, plan_needs, violates
from .textutil import BM25, byte_size, canonical_json, digest, tokenize
from .world import CONCEPTS, KIND_HOME, KIND_WORDS, ROLE_WORDS, SCOPE, TARGET_KIND

SURFACE_TO_CONCEPT = {surface: concept for concept, names in CONCEPTS.items() for surface in names}
RELATION_KIND = {"owner_team": "team", "cost_center": "cost_center", "vendor": "vendor"}
ITEM_TYPES = ("list", "query", "branch", "page", "doc", "pointer")
KINDWORD_TO_KIND = {word: kind for kind, words in KIND_WORDS.items() for word in words}


def payload_of(result: dict) -> dict:
    data = result.get("structuredContent")
    return data if isinstance(data, dict) else {}


# -- public plan tracker ------------------------------------------------------------------
class _Chain:
    def __init__(self, spec: dict):
        self.spec = spec
        kinds = ["service"] + [RELATION_KIND[h] for h in spec["hops"]]
        target_kind = TARGET_KIND[spec["target"]]
        self.steps: list[dict] = [{"kind": k} for k in kinds]
        if target_kind != kinds[-1]:
            self.steps.append({"kind": target_kind, "same_entity": target_kind == "service_metrics"})
        self.evidence: list[dict] = []        # read results accepted per step
        self.pointer: str | None = None       # URI that the next step must read
        self.rejected: set[str] = set()
        self.revalidate: list[int] = []        # evidence indices to re-read after an epoch change

    @property
    def complete(self) -> bool:
        return len(self.evidence) == len(self.steps) and not self.revalidate

    def need(self) -> dict | None:
        if self.revalidate:
            ev = self.evidence[self.revalidate[0]]
            return {"mode": "revalidate", "uri": ev["uri"], "kind": ev["type"], "name": ev["name"],
                    "entity": ev["entity"]}
        if self.complete:
            return None
        i = len(self.evidence)
        step = self.steps[i]
        kind = step["kind"]
        need = {"mode": "find", "kind": kind, "scope": SCOPE[KIND_HOME[kind]], "step": i,
                "unit": self.spec["unit"] if kind == TARGET_KIND[self.spec["target"]] else None,
                "field": self.spec["target"] if kind == TARGET_KIND[self.spec["target"]] else None}
        if i == 0:
            need.update(name=self.spec["anchor"]["name"], region=self.spec["anchor"]["region"], entity=None)
        else:
            prev = self.evidence[-1]
            if step.get("same_entity"):
                need.update(entity=prev["entity"], name=prev["name"])
            else:
                hop = self.spec["hops"][i - 1] if i - 1 < len(self.spec["hops"]) else None
                link = _link(prev, hop) if hop else None
                need.update(entity=(link or {}).get("id"), name=(link or {}).get("name"))
        if self.pointer:
            need.update(mode="pointer", uri=self.pointer)
        return need

    def value(self) -> Any:
        target = self.evidence[-1] if self.evidence else None
        if target is None:
            return None
        for surface, value in target["fields"].items():
            if SURFACE_TO_CONCEPT.get(surface) == self.spec["target"]:
                return value
        return None


def _link(read: dict, concept: str) -> dict | None:
    for surface, link in read.get("links", {}).items():
        if SURFACE_TO_CONCEPT.get(surface) == concept:
            return link
    return None


class Composer:
    """Tracks which public evidence need is open; no access to the private judge."""

    def __init__(self, task: dict):
        self.task = task
        self.chains = [_Chain(spec) for spec in task["plan"]["chains"]]
        self.epoch = task["catalog_epoch"]
        self.epoch_changes = 0
        self.rejections = 0
        self.revalidations = 0
        self.cache: dict[str, dict] = {}     # latest read per URI

    def observe_epoch(self, epoch: Any) -> bool:
        if isinstance(epoch, str) and epoch != self.epoch:
            self.epoch = epoch
            self.epoch_changes += 1
            for chain in self.chains:
                chain.revalidate = list(range(len(chain.evidence)))
            return True
        return False

    def needs(self) -> list[dict]:
        out = []
        for index, chain in enumerate(self.chains):
            need = chain.need()
            if need is not None:
                out.append(dict(need, chain=index))
        return out

    def done(self) -> bool:
        return all(chain.complete for chain in self.chains)

    def ingest(self, read: dict) -> list[str]:
        """Return newly discovered link URIs (sections, replacements)."""
        links: list[str] = []
        self.cache[read["uri"]] = read
        self._ingest_one(read, links)
        self._advance_from_cache(links)
        return links

    def _accept(self, chain: _Chain, read: dict, links: list[str]) -> None:
        chain.evidence.append(read)
        chain.pointer = None
        if read["type"] == "cost_center":
            section = _link(read, "monthly_budget")
            if section and "uri" in section:
                chain.pointer = section["uri"]
                links.append(section["uri"])

    def _advance_from_cache(self, links: list[str]) -> None:
        """Reuse reads from the current epoch instead of re-reading a satisfied need."""
        for chain in self.chains:
            while True:
                need = chain.need()
                if need is None or need["mode"] == "revalidate":
                    break
                cited = {ev["uri"] for ev in chain.evidence}
                match = next((r for u, r in sorted(self.cache.items()) if u not in cited
                              and r.get("catalog_epoch") == self.epoch and self._matches(need, r)), None)
                if match is None:
                    break
                self._accept(chain, match, links)

    def _ingest_one(self, read: dict, links: list[str]) -> None:
        for chain in self.chains:
            if chain.revalidate:
                index = chain.revalidate[0]
                if chain.evidence[index]["uri"] == read["uri"]:
                    self.revalidations += 1
                    if read["provenance"]["status"] == "active" \
                            and read["provenance"]["revision"] == chain.evidence[index]["provenance"]["revision"]:
                        chain.revalidate.pop(0)
                    else:
                        del chain.evidence[index:]
                        chain.revalidate = []
                        if read.get("superseded_by"):
                            chain.pointer = read["superseded_by"]
                            links.append(read["superseded_by"])
                continue
            need = chain.need()
            if need is None:
                continue
            if self._matches(need, read):
                self._accept(chain, read, links)
            elif read["type"] == need["kind"] and self._identifies(need, read):
                if read.get("superseded_by"):
                    chain.pointer = read["superseded_by"]
                    links.append(read["superseded_by"])
                chain.rejected.add(read["uri"])
                self.rejections += 1

    @staticmethod
    def _identifies(need: dict, read: dict) -> bool:
        if need.get("uri"):
            return read["uri"] == need["uri"]
        if need.get("entity"):
            return read["entity"] == need["entity"]
        return read["name"] == need.get("name")

    def _matches(self, need: dict, read: dict) -> bool:
        prov = read["provenance"]
        if read["type"] != need["kind"] or not self._identifies(need, read):
            return False
        if prov["status"] != "active" or prov["scope"] != need["scope"]:
            return False
        if need.get("unit") and prov["unit"] != need["unit"]:
            return False
        if need.get("region") and read.get("region") != need["region"]:
            return False
        return True

    def submission(self) -> dict:
        values = [chain.value() for chain in self.chains]
        combine = self.task["plan"]["combine"]
        missing = any(v is None for v in values)
        answer = None if missing else (values[0] if combine is None else
                                       values[0] + values[1] if combine == "sum" else values[0] - values[1])
        refs = [ev["ref"] for chain in self.chains for ev in chain.evidence]
        return {"answer": answer, "abstain": missing, "evidence_refs": list(dict.fromkeys(refs))}

    def partial_submission(self) -> dict:
        refs = [ev["ref"] for chain in self.chains for ev in chain.evidence]
        return {"answer": None, "abstain": True, "evidence_refs": list(dict.fromkeys(refs))}


def need_text(need: dict) -> str:
    parts = [f"need {need['mode']} {need['kind']}"]
    for key in ("name", "entity", "region", "field", "unit", "scope", "uri"):
        if need.get(key):
            parts.append(f"{key} {need[key]}")
    return " ".join(parts)


def query_for(need: dict) -> str:
    if need.get("entity"):
        return f"{need['entity']} {need.get('name') or ''}".strip()
    return need.get("name") or need["kind"]


# -- frontier items -------------------------------------------------------------------------
def _doc_text(row: dict) -> str:
    reg = row.get("registry", {})
    return (f"resource {row['type']} {row['title']} entity {row['entity']} server {row['server']} "
            f"collection {row['collection']} fields {', '.join(row['fields'])} scope {reg.get('scope') or 'unknown'} "
            f"status {reg.get('status') or 'unknown'} unit {reg.get('unit') or 'unknown'}")


def _struct(item_type: str, depth: int = 0, cursor: int = 0, count: int = 0) -> list[float]:
    import math
    onehot = [1.0 if item_type == t else 0.0 for t in ITEM_TYPES]
    return onehot + [depth / 6.0, min(cursor, 64) / 32.0, math.log1p(count) / 8.0]


class Frontier:
    def __init__(self):
        self.items: dict[str, dict] = {}
        self.visited: set[str] = set()
        self.order = 0

    def add(self, item: dict) -> None:
        key = item["id"]
        if key in self.visited:
            return
        if key in self.items:
            if item["type"] == "pointer" and self.items[key]["type"] != "pointer":
                item["order"] = self.items[key]["order"]
                self.items[key] = item
            return
        self.order += 1
        item["order"] = self.order
        self.items[key] = item

    def remove(self, key: str) -> None:
        self.items.pop(key, None)

    def eligible(self) -> list[dict]:
        return sorted((it for k, it in self.items.items() if k not in self.visited), key=lambda it: it["order"])


def doc_item(row: dict, origin: str, depth: int) -> dict:
    return {"id": row["uri"], "type": "doc", "tool": "resource_read", "arguments": {"uri": row["uri"]},
            "text": _doc_text(row), "meta": {"registry": row.get("registry", {}), "type": row["type"],
                                             "entity": row["entity"], "uri": row["uri"], "title": row["title"]},
            "struct": _struct("doc", depth), "origin": origin}


def branch_item(row: dict, depth: int) -> dict:
    return {"id": row["node_id"], "type": "branch", "tool": "index_children",
            "arguments": {"node_id": row["node_id"]},
            "text": f"branch {row['facet']} {row['label']} sketch {', '.join(row['sketch'])} count {row['count']}",
            "meta": {"facet": row["facet"], "label": row["label"], "sketch": row["sketch"], "count": row["count"],
                     "node_id": row["node_id"]},
            "struct": _struct("branch", depth, 0, row["count"]), "origin": row["node_id"].rsplit("/", 1)[0]}


def page_item(tool: str, key_name: str, key: str, cursor: int, parent_text: str, remaining: int, depth: int) -> dict:
    return {"id": f"page:{tool}:{key}:{cursor}", "type": "page", "tool": tool,
            "arguments": {key_name: key, "cursor": cursor},
            "text": f"more results {parent_text} page {cursor // 8 + 1} remaining {remaining}",
            "meta": {"key": key, "cursor": cursor, "remaining": remaining},
            "struct": _struct("page", depth, cursor, remaining), "origin": key}


def pointer_item(uri: str, reason: str, need: dict | None) -> dict:
    text = f"link {reason} {uri.split('/', 3)[-1].replace('/', ' ')}"
    if need:
        text += " for " + need_text(need)
    return {"id": f"ptr:{uri}:{reason}", "type": "pointer", "tool": "resource_read", "arguments": {"uri": uri},
            "text": text, "meta": {"uri": uri, "reason": reason}, "struct": _struct("pointer"), "origin": uri}


def query_item(query: str, epoch_index: int = 0) -> dict:
    return {"id": f"query:e{epoch_index}:{query}", "type": "query", "tool": "catalog_search", "arguments": {"query": query},
            "text": f"search {query}", "meta": {"query": query}, "struct": _struct("query"), "origin": "search"}


# -- policies ---------------------------------------------------------------------------
class Policy:
    """Interface shared by every router. ``score`` must not see private state."""

    name = "base"
    family = "control"
    proposer = "symbolic"     # exclusion proposals in the layered_ldt arm
    uses_private_oracle = False

    def reset(self, episode_key: str) -> None:
        self.episode_key = episode_key

    def score(self, obs: dict) -> list[float]:
        raise NotImplementedError

    def propose_exclusions(self, obs: dict, rows: list[dict]) -> list[tuple[str, str]]:
        if self.proposer == "symbolic":
            return symbolic_exclusions(obs["task"], rows)
        return []

    def last_cost(self) -> dict:
        return {"flops": 0, "router_tokens": 0, "lexical_ops": 0}

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family, "parameters": 0, "recurrent_depth": 0,
                "proposer": self.proposer, "uses_private_oracle": self.uses_private_oracle}


def symbolic_exclusions(task: dict, rows: list[dict]) -> list[tuple[str, str]]:
    """Propose exclusions visible from public registry metadata and facet labels."""
    needs = plan_needs(task)
    kinds = {need["kind"] for need in needs}
    scopes = {need["scope"] for need in needs}
    out = []
    for row in rows:
        if row.get("kind") == "resource":
            for predicate in ("kind", "status", "scope", "unit"):
                if violates(row.get("registry", {}), row["type"], needs, predicate):
                    out.append((row["uri"], predicate))
                    break
        elif row.get("kind") == "branch":
            label = row["label"]
            if row["facet"] == "provenance":
                scope = label.split(",")[0].replace("scope ", "").strip()
                status = label.split("status ")[-1].strip()
                if status not in ("active", "unregistered status") :
                    out.append((row["node_id"], "status"))
                elif scope not in scopes and scope != "unregistered scope":
                    out.append((row["node_id"], "scope"))
            elif row["facet"] == "kind":
                kind = KINDWORD_TO_KIND.get(label.replace(" records", ""))
                if kind is not None and kind not in kinds:
                    out.append((row["node_id"], "kind"))
            elif row["facet"] == "server":
                role_words = label.split(": ", 1)[-1]
                role = next((r for r, words in ROLE_WORDS.items() if role_words in words), None)
                if role is not None and SCOPE[role] not in scopes:
                    out.append((row["node_id"], "scope"))
    return out


class LexicalPolicy(Policy):
    """BM25 between the open needs plus request and each candidate's text."""

    name, family = "lexical_bm25", "lexical"

    def __init__(self):
        self._ops = 0

    def score(self, obs: dict) -> list[float]:
        index = BM25([c["text"] for c in obs["candidates"]])
        scores = index.scores(" ".join(obs["needs"]) + " " + obs["task"]["request"])
        self._ops = index.operations
        # Links and queries carry no catalog vocabulary; keep them reachable.
        return [s + (5.0 if c["type"] in ("pointer", "list") else 0.0) for s, c in zip(scores, obs["candidates"])]

    def last_cost(self) -> dict:
        return {"flops": 0, "router_tokens": 0, "lexical_ops": self._ops}


class ShamPolicy(Policy):
    """Random scores with the same loop, budget and candidate pipeline."""

    name, family = "sham_random", "sham"

    def score(self, obs: dict) -> list[float]:
        rng = random.Random(digest([self.episode_key, obs["step"], [c["id"] for c in obs["candidates"]]]))
        return [rng.random() for _ in obs["candidates"]]


class DeterministicPolicy(Policy):
    """Hand-written public rules: exact identity match, record kind, provenance, role labels."""

    name, family = "deterministic_rules", "deterministic"

    def score(self, obs: dict) -> list[float]:
        needs = obs["need_records"]
        kinds = {n["kind"] for n in needs}
        scopes = {n.get("scope") for n in needs if n.get("scope")}
        idents = {str(v).lower() for n in needs for v in (n.get("entity"), n.get("name"), n.get("uri")) if v}
        scores = []
        for c in obs["candidates"]:
            meta, s = c["meta"], 0.0
            if c["type"] == "pointer":
                s = 100.0
            elif c["type"] in ("list", "query"):
                s = 80.0
            elif c["type"] == "doc":
                reg = meta.get("registry", {})
                ident = {meta.get("entity", "").lower(), meta.get("uri", "").lower(),
                         meta.get("title", "").split(" - ")[0].lower()}
                s = 10.0 + 40.0 * bool(ident & idents) + 10.0 * (meta.get("type") in kinds)
                s += 5.0 * (reg.get("scope") in scopes) - 30.0 * (reg.get("status") not in (None, "active"))
                s -= 30.0 * (reg.get("scope") is not None and reg.get("scope") not in scopes)
            elif c["type"] == "branch":
                text = c["text"].lower()
                sketch = {name.lower() for name in meta.get("sketch", [])}
                s = 5.0 + 30.0 * bool(sketch & idents) + 2.0 * meta.get("count", 0) ** 0.25
                for kind in kinds:
                    role = KIND_HOME[kind]
                    if any(word in text for word in ROLE_WORDS[role] + KIND_WORDS[kind]):
                        s += 12.0
                if any(word in text for word in ROLE_WORDS["mirror"] + ROLE_WORDS["archive"] + ("replica", "archived")):
                    s -= 20.0
            elif c["type"] == "page":
                s = 8.0
            scores.append(s)
        return scores


class OraclePolicy(Policy):
    """Upper bound only: reads private required documents through ``OracleView``."""

    name, family, uses_private_oracle = "oracle_upper_bound", "oracle", True

    def __init__(self):
        self.view: OracleView | None = None

    def score(self, obs: dict) -> list[float]:
        assert self.view is not None
        labels = self.view.labels(obs)
        return [float(label) for label in labels]


class OracleView:
    """Private labels for training data and the oracle bound. Never part of an observation."""

    def __init__(self, env: Env02, composer: Composer):
        self.env, self.composer = env, composer
        self.query_results: dict[str, list[str]] = {}

    def targets(self) -> set[str]:
        required = self.env.required_uris()
        out = set()
        for chain_index, chain in enumerate(self.composer.chains):
            if chain.revalidate:
                out.add(chain.evidence[chain.revalidate[0]]["uri"])
            elif not chain.complete:
                step = len(chain.evidence)
                if step < len(required[chain_index]):
                    out.add(required[chain_index][step])
        return out

    def _contains(self, item: dict, targets: set[str]) -> bool:
        env = self.env
        if item["type"] in ("doc", "pointer"):
            uri = item["meta"]["uri"]
            if uri in targets:
                return True
            doc = env._docs.get(uri)
            return bool(doc and doc["superseded_by"] in targets)
        if item["type"] == "branch":
            node = env._nodes.get(item["meta"]["node_id"])
            return bool(node and targets & set(node["uris"]))
        if item["type"] == "list":
            return True
        if item["type"] == "query":
            scores = env._search.scores(item["meta"]["query"])
            hits = {u for s, u in zip(scores, env._search_order) if s > 0}
            return bool(hits & targets)
        if item["type"] == "page":
            key, cursor = item["meta"]["key"], item["meta"]["cursor"]
            if item["tool"] == "index_children":
                node = env._nodes.get(key)
                if not node:
                    return False
                for _kind, child in node["children"][cursor:]:
                    if targets & set(env._subtree(child)):
                        return True
                return False
            scores = env._search.scores(key)
            ranked = [u for _s, u in sorted((-s, u) for s, u in zip(scores, env._search_order) if s > 0)]
            return bool(targets & set(ranked[cursor:]))
        return False

    def labels(self, obs: dict) -> list[int]:
        targets = self.targets()
        return [int(self._contains(c, targets)) for c in obs["candidates"]]

    def exclusion_labels(self, rows: list[dict]) -> list[str]:
        """Strongest certifiable predicate per child, from registered metadata (verifier semantics)."""
        env = self.env
        out = []
        for row in rows:
            key = row.get("node_id") or row.get("uri")
            uris = env._subtree(key)
            label = "none"
            for predicate in ("kind", "status", "scope", "unit"):
                if all(violates(env._docs[u]["registry"], env._docs[u]["kind"], env._needs, predicate) for u in uris):
                    label = predicate
                    break
            out.append(label)
        return out


# -- the episode loop -------------------------------------------------------------------
def build_observation(task: dict, composer: Composer, candidates: list[dict], step: int, budget: int,
                      arm: str) -> dict:
    needs = composer.needs()
    evidence = [f"{ev['type']} {ev['name']} {ev['entity']}" for ch in composer.chains for ev in ch.evidence]
    return {"task": task, "needs": [need_text(n) for n in needs], "need_records": needs,
            "evidence": evidence, "candidates": candidates, "step": step, "budget_left": budget - step,
            "arm": arm}


def observation_digest(obs: dict) -> str:
    return digest({"needs": obs["needs"], "evidence": obs["evidence"], "step": obs["step"],
                   "candidates": [c["id"] for c in obs["candidates"]]})


def prefilter(obs_needs: list[str], request: str, items: list[dict], k: int) -> tuple[list[dict], int]:
    """Keep links and queries plus the top-k other candidates by BM25; returns lexical ops."""
    keep = [it for it in items if it["type"] in ("pointer", "query", "list")]
    rest = [it for it in items if it["type"] not in ("pointer", "query", "list")]
    if len(rest) <= k:
        return keep + rest, 0
    index = BM25([it["text"] for it in rest])
    scores = index.scores(" ".join(obs_needs) + " " + request)
    ranked = sorted(range(len(rest)), key=lambda i: (-scores[i], rest[i]["order"]))[:k]
    chosen = sorted((rest[i] for i in ranked), key=lambda it: it["order"])
    return keep + chosen, index.operations


def run_episode(case: dict, *, arm: str, layout: str, policy: Policy, max_tool_calls: int = 40,
                prefilter_k: int = 32, handoff_budget_bytes: int = 2048, transport: str = "inprocess",
                stdio_options: dict | None = None, record: Callable[[dict], None] | None = None,
                tape: list[dict] | None = None, debug: bool = False) -> dict:
    """Run one episode. ``record`` receives private-labelled events for training-data builds only."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    started = time.perf_counter()
    env = Env02(case, arm, layout if arm.startswith("layered") else "")
    task = env.task
    composer = Composer(task)
    view = OracleView(env, composer)
    if isinstance(policy, OraclePolicy):
        policy.view = view
    policy.reset(f"{case['id']}/{arm}/{layout}")
    frontier = Frontier()
    decisions: list[dict] = []
    history: list[dict] = []
    totals = {"flops": 0, "router_tokens": 0, "lexical_ops": 0, "prefilter_ops": 0,
              "router_seconds": 0.0, "candidates_scored": 0, "observation_bytes": 0, "result_bytes": 0,
              "refine_calls": 0, "decisions": 0, "replay_mismatches": 0}
    failure, error_detail = None, None
    client = None
    issued_queries: set[str] = set()
    epoch_index = 0

    def seed_frontier():
        if arm == "flat":
            frontier.add({"id": "list", "type": "list", "tool": "catalog_list", "arguments": {},
                          "text": "list entire catalog", "meta": {}, "struct": _struct("list"), "origin": "root"})
        elif arm.startswith("layered"):
            frontier.add(branch_item({"node_id": f"e{epoch_index}:root", "facet": "root", "label": "catalog root",
                                      "sketch": [], "count": len(env._docs)}, 0))

    if transport == "stdio":
        from MCP.transport import StdioClient
        options = dict(stdio_options or {})
        command = [__import__("sys").executable, "-u", "-m", "MCP.mcp02.serve", "--arm", arm,
                   "--layout", layout if arm.startswith("layered") else "", "--split", case["split"],
                   "--seed", str(options["seed"]), "--index", str(options["index"]),
                   "--catalog-size", str(len(case["docs"]))]
        client = StdioClient(command=command, timeout=30.0)
        client.__enter__()
        client.initialize()
        if client.tools() != env.tools():
            raise RuntimeError("stdio tool catalogue differs from replay")

    result_digests: list[str] = []

    def call(name: str, arguments: dict) -> dict:
        replay = env.call(name, arguments)
        if client is not None:
            result = client.call(name, arguments)
            if result != replay:
                totals["replay_mismatches"] += 1
        totals["result_bytes"] += byte_size(replay)
        result_digests.append(digest(replay))
        history.append([name, arguments, bool(replay.get("isError", False))])
        return replay

    seed_frontier()
    submission = None
    tape_iter = iter(tape) if tape is not None else None
    try:
        while True:
            if composer.done():
                submission = composer.submission()
                break
            if env.stats["tool_calls"] >= max_tool_calls:
                failure = "tool_budget_exhausted"
                break
            needs = composer.needs()
            for need in needs:
                if need["mode"] in ("pointer", "revalidate"):
                    frontier.add(pointer_item(need["uri"], need["mode"], need))
                if arm == "search" and need["mode"] == "find":
                    q = query_for(need)
                    if q not in issued_queries:
                        frontier.add(query_item(q, epoch_index))
            eligible = frontier.eligible()
            if not eligible:
                failure = "frontier_exhausted"
                break
            candidates, ops = prefilter([need_text(n) for n in needs], task["request"], eligible, prefilter_k)
            totals["prefilter_ops"] += ops
            step = env.stats["tool_calls"]
            obs = build_observation(task, composer, candidates, step, max_tool_calls, arm)
            totals["observation_bytes"] += byte_size({"needs": obs["needs"], "evidence": obs["evidence"],
                                                      "candidates": [c["text"] for c in obs["candidates"]]})
            labels = None
            if record is not None:
                labels = view.labels(obs)
                record({"kind": "decide", "obs": obs, "labels": labels})
            t0 = time.perf_counter()
            if tape_iter is not None:
                entry = next(tape_iter)
                if entry["kind"] != "decide" or entry["obs_sha"] != observation_digest(obs)[:16]                         or entry["n"] != len(candidates):
                    raise RuntimeError("tape observation mismatch")
                scores = [1.0 if i == entry["choice_index"] else 0.0 for i in range(len(candidates))]
                cost = entry["cost"]
            else:
                scores = policy.score(obs)
                cost = policy.last_cost()
            totals["router_seconds"] += time.perf_counter() - t0
            if len(scores) != len(candidates):
                raise RuntimeError("policy returned the wrong number of scores")
            for key in ("flops", "router_tokens", "lexical_ops"):
                totals[key] += int(cost.get(key, 0))
            totals["candidates_scored"] += len(candidates)
            totals["decisions"] += 1
            best = max(range(len(candidates)), key=lambda i: (scores[i], -candidates[i]["order"]))
            item = candidates[best]
            top = sorted(range(len(candidates)), key=lambda i: -scores[i])[:5]
            decisions.append({"kind": "decide", "obs_sha": observation_digest(obs)[:16], "n": len(candidates),
                              "choice_index": best, "choice_type": item["type"],
                              "cost": {k: int(cost.get(k, 0)) for k in ("flops", "router_tokens", "lexical_ops")}})
            if debug:
                decisions[-1].update(choice=item["id"], top=[[candidates[i]["id"], round(float(scores[i]), 6)]
                                                             for i in top])
            frontier.visited.add(item["id"])
            frontier.remove(item["id"])
            if item["type"] == "query":
                issued_queries.add(item["meta"]["query"])
            arguments = dict(item["arguments"])
            if item["tool"] in ("index_children", "catalog_search"):
                arguments["limit"] = env.page_limit
            result = call(item["tool"], arguments)
            payload = payload_of(result)
            if item["tool"] == "resource_read":
                for key in [k for k, it in frontier.items.items() if it["arguments"].get("uri") == arguments["uri"]
                            and it["meta"].get("reason") != "revalidate"]:
                    frontier.remove(key)
                    frontier.visited.add(key)
            if composer.observe_epoch(payload.get("catalog_epoch")):
                epoch_index = env.epoch_index
                for key in [k for k, it in frontier.items.items() if it["type"] in ("branch", "page")]:
                    frontier.remove(key)
                issued_queries.clear()
                seed_frontier()
            if result.get("isError"):
                continue
            depth = item["struct"][len(ITEM_TYPES)] * 6.0 + 1
            if item["tool"] == "catalog_list":
                for row in payload["resources"]:
                    frontier.add(doc_item(row, "list", 1))
            elif item["tool"] == "catalog_search":
                for row in payload["resources"]:
                    frontier.add(doc_item(row, "search", 1))
                if payload.get("next_cursor") is not None:
                    frontier.add(page_item("catalog_search", "query", payload["query"], payload["next_cursor"],
                                           f"search {payload['query']}", max(0, payload["total"] - payload["next_cursor"]), 1))
            elif item["tool"] == "index_children":
                rows = payload["children"]
                for row in rows:
                    frontier.add(branch_item(row, int(depth)) if row["kind"] == "branch" else doc_item(row, payload["node_id"], int(depth)))
                if payload.get("next_cursor") is not None:
                    frontier.add(page_item("index_children", "node_id", payload["node_id"], payload["next_cursor"],
                                           payload["label"], max(0, payload["total"] - payload["next_cursor"]), int(depth)))
                if record is not None and arm == "layered_ldt":
                    record({"kind": "rows", "obs": obs, "rows": rows, "labels": view.exclusion_labels(rows)})
                if arm == "layered_ldt" and rows and env.stats["tool_calls"] < max_tool_calls:
                    if tape_iter is not None:
                        entry = next(tape_iter)
                        if entry["kind"] != "propose":
                            raise RuntimeError("tape proposal mismatch")
                        proposals = [tuple(p) for p in entry["proposals"]]
                        extra = entry.get("cost", {})
                    else:
                        proposals = policy.propose_exclusions(obs, rows)
                        extra = policy.last_cost()
                    for key in ("flops", "router_tokens", "lexical_ops"):
                        totals[key] += int(extra.get(key, 0))
                    decisions.append({"kind": "propose", "node": payload["node_id"],
                                      "proposals": [list(p) for p in proposals],
                                      "cost": {k: int(extra.get(k, 0)) for k in ("flops", "router_tokens", "lexical_ops")}})
                    by_predicate: dict[str, list[str]] = {}
                    for child, predicate in proposals:
                        by_predicate.setdefault(predicate, []).append(child)
                    for predicate, children in sorted(by_predicate.items()):
                        if env.stats["tool_calls"] >= max_tool_calls:
                            break
                        refine = call("candidate_refine", {"node_id": payload["node_id"],
                                                           "exclude": children[:64], "predicate": predicate})
                        totals["refine_calls"] += 1
                        for child in payload_of(refine).get("excluded", []):
                            frontier.remove(child)
                            frontier.visited.add(child)
            else:  # resource_read
                for uri in composer.ingest(payload):
                    reason = "replacement" if payload.get("superseded_by") == uri else "section"
                    frontier.add(pointer_item(uri, reason, None))
                if payload.get("superseded_by") and payload["superseded_by"] not in frontier.visited:
                    frontier.add(pointer_item(payload["superseded_by"], "replacement", None))
    except Exception as exc:  # infrastructure failures stay visible in the record
        failure = "transport_or_environment_error"
        error_detail = f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None:
            client.close()
    abstained_on_failure = submission is None
    if submission is None:
        submission = composer.partial_submission()
    grade = env.grade(submission)
    handoff = env.handoff(submission)
    handoff_bytes = byte_size(handoff)
    over_budget = handoff_bytes > handoff_budget_bytes
    accepted = bool(grade["success"] and not over_budget and failure is None)
    return {
        "case_id": case["id"], "case_sha256": digest(case), "family": case["family"], "split": case["split"],
        "group_id": case["group_id"], "template_group": case["template_group"],
        "catalog_size": len(case["docs"]), "arm": arm, "layout": layout if arm.startswith("layered") else "",
        "policy": policy.describe(), "transport": transport,
        "submission": submission, "grade": grade, "accepted": accepted,
        "failure": failure, "error_detail": error_detail, "abstained_on_failure": abstained_on_failure,
        "handoff_utf8_bytes": handoff_bytes, "handoff_budget_bytes": handoff_budget_bytes,
        "handoff_budget_exceeded": over_budget, "handoff": handoff,
        "tool_calls": env.stats["tool_calls"], "environment_stats": dict(env.stats),
        "private_checks": dict(env.private), "rolled": env.rolled,
        "composer": {"epoch_changes": composer.epoch_changes, "rejections": composer.rejections,
                     "revalidations": composer.revalidations},
        "compute": totals, "decisions": decisions, "history": history,
        "results_sha256": digest(result_digests),
        "max_tool_calls": max_tool_calls, "prefilter_k": prefilter_k,
        "transport_replay_verified": (None if transport == "inprocess" else
                                      totals["replay_mismatches"] == 0 and failure != "transport_or_environment_error"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def make_control(name: str) -> Policy:
    return {"lexical_bm25": LexicalPolicy, "sham_random": ShamPolicy, "deterministic_rules": DeterministicPolicy,
            "oracle_upper_bound": OraclePolicy}[name]()


CONTROL_POLICIES = ("deterministic_rules", "lexical_bm25", "sham_random", "oracle_upper_bound")
