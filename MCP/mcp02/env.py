"""MCP-02 environment: one episode over a multi-server world with a private judge.

Arms share the same world and judge and differ only in discovery interface:

* ``flat``: one ``catalog_list`` call returns every resource descriptor.
* ``search``: ``catalog_search`` runs BM25 over descriptors, with pagination.
* ``layered``: ``index_children`` walks a configurable facet hierarchy.
* ``layered_ldt``: the layered tools plus ``candidate_refine``. The environment
  keeps an explicit candidate set per expanded node, bound to the catalog epoch.
  Every proposed exclusion passes through ``certify_and_apply_verified`` with an
  environment-owned check over registered metadata. Unknown metadata keeps a
  candidate available, and an epoch change starts a new candidate state.

Index layouts are strings over facets: S server, C collection, K record kind
(capability), P registered provenance (scope and status), B name bucket with a
partial name sketch, E entity (all records about one entity). The empty layout
pages every resource under the root.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from research_gym.core.hybrid import (
    CandidateState, HybridMode, LatticeProposal, certify_and_apply_verified,
)
from research_gym.core.typed_soundness import SoundnessType

from .textutil import BM25, canonical_json, digest, stable_int
from .world import CONCEPTS, KIND_HOME, ROLE_WORDS, SCOPE, TARGET_KIND

ARMS = ("flat", "search", "layered", "layered_ldt")
FACETS = {"S": "server", "C": "collection", "K": "kind", "P": "provenance", "B": "bucket", "E": "entity"}
PREDICATES = ("status", "kind", "scope", "unit", "any")
MAX_LIMIT = 16


def _tool(name: str, description: str, properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": list(required), "additionalProperties": False},
            "annotations": {"readOnlyHint": name != "candidate_refine", "destructiveHint": False}}


_PAGE = {"limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": 8},
         "cursor": {"type": "integer", "minimum": 0, "default": 0}}


def plan_needs(task: dict) -> list[dict]:
    """Record kinds a valid citation can have, with their home scope and unit."""
    needs = []
    for chain in task["plan"]["chains"]:
        kinds = ["service"] + [{"owner_team": "team", "cost_center": "cost_center", "vendor": "vendor"}[h]
                               for h in chain["hops"]]
        target_kind = TARGET_KIND[chain["target"]]
        for kind in kinds:
            needs.append({"kind": kind, "scope": SCOPE[KIND_HOME[kind]],
                          "unit": chain["unit"] if kind == target_kind else None})
        if target_kind not in kinds:
            needs.append({"kind": target_kind, "scope": SCOPE[KIND_HOME[target_kind]], "unit": chain["unit"]})
    return needs


def violates(registry: dict, kind: str, needs: list[dict], predicate: str) -> bool:
    """True only when registered metadata proves the record serves no need."""
    if predicate == "any":
        return any(violates(registry, kind, needs, p) for p in ("status", "kind", "scope", "unit"))
    if predicate == "status":
        return registry.get("status") is not None and registry["status"] != "active"
    if predicate == "kind":
        return all(need["kind"] != kind for need in needs)
    if predicate == "scope":
        scope = registry.get("scope")
        return scope is not None and all(need["scope"] != scope for need in needs)
    if predicate == "unit":
        unit = registry.get("unit")
        matching = [need for need in needs if need["kind"] == kind]
        return (unit is not None and bool(matching)
                and all(need["unit"] is not None and need["unit"] != unit for need in matching))
    raise ValueError("unknown predicate")


class Env02:
    """One episode; ``task``, ``tools`` and ``call`` form the actor-visible surface."""

    def __init__(self, case: dict, arm: str, layout: str = "SCB", page_limit: int = 8):
        if arm not in ARMS:
            raise ValueError("unknown arm")
        if any(ch not in FACETS for ch in layout) or len(set(layout)) != len(layout):
            raise ValueError("layout must use distinct facet letters from SCKPBE")
        frozen = copy.deepcopy(case)
        self.case_id = frozen["id"]
        self.group = frozen["group_id"]
        self.task = frozen["task"]
        self.arm, self.layout, self.page_limit = arm, layout, page_limit
        self.task["discovery"] = {"arm": arm, "layout": layout if arm.startswith("layered") else None,
                                  "index_root": "root" if arm.startswith("layered") else None}
        self._oracle = frozen["oracle"]
        self._hidden = frozen.get("hidden", {})
        self._servers = frozen["servers"]
        self._docs = {d["uri"]: d for d in frozen["docs"]}
        self._needs = plan_needs(self.task)
        self._sketch_exclude = {self._docs[u]["name"] for u in self._hidden.get("sketch_exclude", [])}
        self.epoch_index = 0
        self.catalog_epoch = self.task["catalog_epoch"]
        self._observed: dict[str, dict] = {}   # ref -> read result
        self._ref_uri: dict[str, str] = {}
        self.rolled = False
        self.stats = {"tool_calls": 0, "tool_errors": 0, "read_calls": 0, "search_calls": 0,
                      "index_calls": 0, "list_calls": 0, "refine_calls": 0, "ldt_checks": 0,
                      "ldt_metadata_checks": 0, "ldt_excluded": 0, "ldt_rejected": 0,
                      "ldt_state_resets": 0, "search_term_ops": 0}
        self.private = {"certified_exclusions_containing_required": 0}
        self._rebuild()

    # -- catalog structures -----------------------------------------------------------
    def _rebuild(self) -> None:
        self._registry_digest = digest({u: d["registry"] for u, d in sorted(self._docs.items())})
        self._descriptors = {u: self._descriptor(d) for u, d in self._docs.items()}
        order = sorted(self._docs)
        self._search = BM25([canonical_json(self._descriptors[u]) for u in order])
        self._search_order = order
        self._nodes: dict[str, dict] = {}
        self._build(f"e{self.epoch_index}:root", order, 0)
        self._states: dict[str, CandidateState] = {}

    def _descriptor(self, doc: dict) -> dict:
        surface = self._servers[doc["role"]]["surface"]
        names = sorted(surface[c] for c in list(doc["fields"]) + list(doc["links"]))
        return {"kind": "resource", "uri": doc["uri"], "server": doc["server"],
                "collection": doc["collection"], "type": doc["kind"], "entity": doc["entity_id"],
                "title": f"{doc['name']} - {doc['kind_word']}", "fields": names,
                "registry": dict(doc["registry"])}

    def _facet_key(self, facet: str, doc: dict, n_docs: int) -> tuple[str, str]:
        if facet == "S":
            return doc["server"], f"{doc['server']}: {self._servers[doc['role']]['description']}"
        if facet == "C":
            words = sorted({d["kind_word"] for d in self._docs.values()
                            if d["server"] == doc["server"] and d["collection"] == doc["collection"]})[:2]
            return (f"{doc['server']}/{doc['collection']}",
                    f"{doc['collection']} on {doc['server']}: {', '.join(words)}")
        if facet == "K":
            return doc["kind"], f"{doc['kind_word']} records"
        if facet == "P":
            reg = doc["registry"]
            scope = reg.get("scope") or "unregistered scope"
            status = reg.get("status") or "unregistered status"
            return f"{scope}|{status}", f"scope {scope}, status {status}"
        if facet == "B":
            buckets = max(2, math.ceil(n_docs / 12))
            k = stable_int(self.group, "bucket", doc["name"]) % buckets
            return f"b{k:02d}", f"name bucket {k + 1} of {buckets}"
        return doc["entity_id"], f"{doc['name']} ({doc['entity_id']})"

    def _build(self, node_id: str, uris: list[str], level: int) -> None:
        names = sorted({self._docs[u]["name"] for u in uris} - self._sketch_exclude,
                       key=lambda name: stable_int(self.group, "sketch", name))
        node = {"uris": uris, "children": [], "sketch": names[:6], "label": "catalog root", "facet": "root"}
        self._nodes[node_id] = node
        if level == len(self.layout):
            node["children"] = [("doc", u) for u in sorted(uris)]
            return
        facet = self.layout[level]
        groups: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        for u in uris:
            key, label = self._facet_key(facet, self._docs[u], len(uris))
            groups.setdefault(key, []).append(u)
            labels.setdefault(key, label)
        for i, key in enumerate(sorted(groups, key=lambda k: (labels[k], k))):
            child = f"{node_id}/{i}"
            self._build(child, groups[key], level + 1)
            self._nodes[child].update(label=labels[key], facet=FACETS[facet])
            node["children"].append(("branch", child))

    def _branch_row(self, node_id: str) -> dict:
        node = self._nodes[node_id]
        return {"kind": "branch", "node_id": node_id, "facet": node["facet"], "label": node["label"],
                "sketch": list(node["sketch"]), "count": len(node["uris"])}

    def required_uris(self) -> list[list[str]]:
        roll = self._hidden.get("epoch_roll")
        required = copy.deepcopy(self._oracle["required_uris"])
        if self.rolled and roll:
            required = [[roll["new_uri"] if u == roll["uri"] else u for u in chain] for chain in required]
        return required

    # -- epoch change -----------------------------------------------------------------
    def _maybe_roll(self) -> None:
        roll = self._hidden.get("epoch_roll")
        if not roll or self.rolled or self.stats["tool_calls"] <= roll["after_calls"]:
            return
        old = self._docs[roll["uri"]]
        slug = digest([self.group, "rolled", roll["uri"]])[:10]
        new_uri = f"mcp://{old['server']}/{old['collection']}/{slug}"
        new = copy.deepcopy(old)
        new.update(uri=new_uri, epoch=roll["new_epoch"], revision=f"r{int(old['revision'][1:]) + 1}")
        new["fields"]["contract_value"] = roll["new_value"]
        new["registry"] = {"status": "active", "scope": new["scope"], "unit": new["unit"], "epoch": roll["new_epoch"]}
        old.update(status="superseded", superseded_by=new_uri)
        old["registry"]["status"] = "superseded"
        self._docs[new_uri] = new
        roll["new_uri"] = new_uri
        self.rolled = True
        self.epoch_index += 1
        self.catalog_epoch = roll["new_epoch"]
        if self._states:
            self.stats["ldt_state_resets"] += 1
        self._rebuild()

    # -- tools ------------------------------------------------------------------------
    def tools(self) -> list[dict]:
        read = _tool("resource_read", "Read one resource by URI. Returns a citation ref, provenance "
                     "(server, scope, status, unit, epoch, revision), field values and links to related "
                     "records. A superseded record names its replacement.",
                     {"uri": {"type": "string"}}, ("uri",))
        if self.arm == "flat":
            return [_tool("catalog_list", "List every resource descriptor on every server in one result.", {}), read]
        if self.arm == "search":
            return [_tool("catalog_search", "BM25 search over resource descriptors (title, entity, fields, "
                          "server, collection, registered metadata). Results are paginated.",
                          {"query": {"type": "string"}, **_PAGE}, ("query",)), read]
        tools = [_tool("index_children", "List immediate children of an index node, starting at root. "
                       "Branches carry a facet label, a partial name sketch and a record count. Node IDs "
                       "belong to one catalog epoch.", {"node_id": {"type": "string"}, **_PAGE}, ("node_id",)), read]
        if self.arm == "layered_ldt":
            tools.append(_tool(
                "candidate_refine", "Propose excluding children of an index node from its candidate set. "
                "Each exclusion is applied only when the environment verifies, from registered metadata, "
                "that every record below it violates the named predicate for every record kind the task "
                "can cite. Unknown metadata keeps a candidate. Rejected exclusions change nothing.",
                {"node_id": {"type": "string"},
                 "exclude": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64},
                 "predicate": {"type": "string", "enum": list(PREDICATES)}},
                ("node_id", "exclude", "predicate")))
        return tools

    def _page(self, arguments: dict) -> tuple[int, int]:
        limit, cursor = arguments.get("limit", self.page_limit), arguments.get("cursor", 0)
        if type(limit) is not int or not 1 <= limit <= MAX_LIMIT or type(cursor) is not int or cursor < 0:
            raise ValueError(f"limit must be 1..{MAX_LIMIT} and cursor a nonnegative integer")
        return limit, cursor

    def _node(self, node_id: Any) -> dict:
        if not isinstance(node_id, str):
            raise ValueError("node_id must be a string")
        if node_id == "root":
            node_id = f"e{self.epoch_index}:root"
        if node_id not in self._nodes:
            if not node_id.startswith(f"e{self.epoch_index}:"):
                raise ValueError("node_id belongs to a previous catalog epoch; restart from root")
            raise ValueError("unknown node_id")
        return {"id": node_id, **self._nodes[node_id]}

    def _child_ids(self, node: dict) -> list[str]:
        return [key for _kind, key in node["children"]]

    def _state(self, node: dict) -> CandidateState:
        if node["id"] not in self._states:
            self._states[node["id"]] = CandidateState({"children": frozenset(self._child_ids(node))})
        return self._states[node["id"]]

    def _subtree(self, child: str) -> list[str]:
        return self._nodes[child]["uris"] if child in self._nodes else [child]

    def _read(self, uri: Any) -> dict:
        if not isinstance(uri, str) or uri not in self._docs:
            raise ValueError("unknown uri")
        doc = self._docs[uri]
        surface = self._servers[doc["role"]]["surface"]
        ref = "ev-" + digest([self.group, uri, doc["revision"]])[:20]
        result = {
            "ref": ref, "uri": uri, "type": doc["kind"], "entity": doc["entity_id"], "name": doc["name"],
            "title": f"{doc['name']} - {doc['kind_word']}",
            "provenance": {"server": doc["server"], "collection": doc["collection"], "scope": doc["scope"],
                           "status": doc["status"], "unit": doc["unit"], "epoch": doc["epoch"],
                           "revision": doc["revision"]},
            "fields": {surface[c]: v for c, v in doc["fields"].items()},
            "links": {surface[c]: dict(v) for c, v in doc["links"].items()},
            "superseded_by": doc["superseded_by"], "catalog_epoch": self.catalog_epoch,
        }
        if doc["region"] is not None:
            result["region"] = doc["region"]
        self._observed[ref] = copy.deepcopy(result)
        self._ref_uri[ref] = uri
        self.stats["read_calls"] += 1
        return result

    def _refine(self, arguments: dict) -> dict:
        node = self._node(arguments["node_id"])
        exclude, predicate = arguments["exclude"], arguments["predicate"]
        if (not isinstance(exclude, list) or not 1 <= len(exclude) <= 64
                or any(not isinstance(x, str) for x in exclude) or len(set(exclude)) != len(exclude)):
            raise ValueError("exclude must list 1..64 distinct child IDs")
        if predicate not in PREDICATES:
            raise ValueError("unknown predicate")
        self.stats["refine_calls"] += 1
        state = self._state(node)
        before = len(state.domains["children"])
        accepted, rejected = [], []
        context = (self.catalog_epoch, self._registry_digest)
        for child in exclude:
            if child not in state.domains["children"]:
                rejected.append({"id": child, "reason": "not a current candidate"})
                continue
            proposal = LatticeProposal(state.eliminate("children", {child}), SoundnessType.MODEL_SOUND_DEAD,
                                       mode=HybridMode.DEDUCE, metadata={"predicate": predicate})

            def verify(candidate: LatticeProposal, ctx: Any, child=child) -> SoundnessType:
                self.stats["ldt_checks"] += 1
                if ctx != (self.catalog_epoch, self._registry_digest):
                    return SoundnessType.UNKNOWN
                removed = state.domains["children"] - candidate.proposed_state.domains["children"]
                for item in removed:
                    for uri in self._subtree(item):
                        self.stats["ldt_metadata_checks"] += 1
                        doc = self._docs[uri]
                        if not violates(doc["registry"], doc["kind"], self._needs, predicate):
                            return SoundnessType.UNKNOWN
                return SoundnessType.ENV_SOUND_DEAD

            decision = certify_and_apply_verified(state, proposal, provenance_verifier=verify,
                                                  verifier_context=context, require_progress=True)
            if decision.accepted:
                state = decision.after
                accepted.append(child)
                required = {u for chain in self.required_uris() for u in chain}
                if required & set(self._subtree(child)):
                    self.private["certified_exclusions_containing_required"] += 1
            else:
                rejected.append({"id": child, "reason": "verification did not establish the exclusion"})
        self._states[node["id"]] = state
        self.stats["ldt_excluded"] += len(accepted)
        self.stats["ldt_rejected"] += len(rejected)
        return {"node_id": node["id"], "predicate": predicate, "excluded": accepted, "rejected": rejected,
                "candidates_before": before, "candidates_after": len(state.domains["children"]),
                "independently_checked": True, "catalog_epoch": self.catalog_epoch}

    def call(self, name: str, arguments: dict) -> dict:
        """Return an MCP CallToolResult; malformed calls return ``isError``."""
        self.stats["tool_calls"] += 1
        self._maybe_roll()
        try:
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            schemas = {tool["name"]: tool["inputSchema"] for tool in self.tools()}
            if name not in schemas:
                raise ValueError("unknown tool")
            schema = schemas[name]
            if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
                raise ValueError("unexpected or missing argument")
            if name == "resource_read":
                payload = self._read(arguments["uri"])
            elif name == "catalog_list":
                self.stats["list_calls"] += 1
                payload = {"resources": [self._descriptors[u] for u in sorted(self._docs)],
                           "total": len(self._docs), "catalog_epoch": self.catalog_epoch}
            elif name == "catalog_search":
                self.stats["search_calls"] += 1
                if not isinstance(arguments["query"], str):
                    raise ValueError("query must be a string")
                limit, cursor = self._page(arguments)
                before = self._search.operations
                scores = self._search.scores(arguments["query"])
                self.stats["search_term_ops"] += self._search.operations - before
                ranked = sorted((-s, u) for s, u in zip(scores, self._search_order) if s > 0)
                rows = [self._descriptors[u] for _s, u in ranked]
                payload = {"query": arguments["query"], "resources": rows[cursor:cursor + limit],
                           "total": len(rows), "next_cursor": cursor + limit if cursor + limit < len(rows) else None,
                           "catalog_epoch": self.catalog_epoch}
            elif name == "index_children":
                self.stats["index_calls"] += 1
                node = self._node(arguments["node_id"])
                limit, cursor = self._page(arguments)
                children = node["children"]
                window = children[cursor:cursor + limit]
                annotation = None
                total = len(children)
                if self.arm == "layered_ldt":
                    # Cursors index the unfiltered child order, so exclusions never hide unseen pages.
                    kept = self._state(node).domains["children"]
                    annotation = {"candidates": len(kept), "excluded": len(children) - len(kept)}
                    window = [c for c in window if c[1] in kept]
                    total = len(kept)
                rows = [self._branch_row(key) if kind == "branch" else self._descriptors[key]
                        for kind, key in window]
                payload = {"node_id": node["id"], "label": node["label"], "children": rows,
                           "total": total,
                           "next_cursor": cursor + limit if cursor + limit < len(children) else None,
                           "catalog_epoch": self.catalog_epoch}
                if annotation is not None:
                    payload["candidate_state"] = annotation
            else:
                payload = self._refine(arguments)
            result = {"content": [{"type": "text", "text": canonical_json(payload)}],
                      "structuredContent": payload, "isError": False}
        except (ValueError, TypeError, KeyError) as error:
            self.stats["tool_errors"] += 1
            payload = {"error": str(error), "catalog_epoch": self.catalog_epoch}
            result = {"content": [{"type": "text", "text": canonical_json(payload)}],
                      "structuredContent": payload, "isError": True}
        return result

    # -- private judge ----------------------------------------------------------------
    def expected_answer(self) -> Any:
        roll = self._hidden.get("epoch_roll")
        if self.rolled and roll:
            return roll["new_value"]
        return self._oracle["answer"]

    def grade(self, submission: Any) -> dict:
        """Private terminal judge; expected answers are never returned to the actor."""
        ok = (isinstance(submission, dict) and set(submission) == {"answer", "evidence_refs", "abstain"}
              and type(submission["abstain"]) is bool and isinstance(submission["evidence_refs"], list)
              and all(isinstance(r, str) for r in submission["evidence_refs"]))
        if not ok:
            return {"success": False, "answer_correct": False, "evidence_valid": False, "evidence_recall": 0.0,
                    "evidence_precision": 0.0, "abstention_correct": False, "reasons": ["malformed_submission"]}
        required = {u for chain in self.required_uris() for u in chain}
        refs = submission["evidence_refs"]
        valid = set()
        for ref in refs:
            uri = self._ref_uri.get(ref)
            if uri in required and self._docs[uri]["status"] == "active" \
                    and self._observed[ref]["provenance"]["revision"] == self._docs[uri]["revision"]:
                valid.add(uri)
        unique = len(refs) == len(set(refs))
        evidence_valid = bool(refs) and unique and len(valid) == len(refs)
        recall = len(valid) / len(required) if required else 1.0
        precision = len(valid) / len(refs) if refs else 0.0
        expected = self.expected_answer()
        answer = submission["answer"]
        numeric = type(answer) is int or (type(answer) is float and math.isfinite(answer))
        answer_correct = answer is None if expected is None else bool(numeric and answer == expected)
        abstain_expected = expected is None
        abstention_correct = submission["abstain"] == abstain_expected
        coherent = (answer is None) == submission["abstain"]
        success = answer_correct and evidence_valid and recall == 1.0 and abstention_correct and coherent
        reasons = [name for name, passed in (
            ("wrong_answer", answer_correct), ("invalid_stale_or_extra_evidence", evidence_valid),
            ("incomplete_evidence", recall == 1.0), ("wrong_abstention", abstention_correct),
            ("inconsistent_answer_abstention", coherent)) if not passed]
        return {"success": bool(success), "answer_correct": answer_correct, "evidence_valid": evidence_valid,
                "evidence_recall": recall, "evidence_precision": precision,
                "abstention_correct": abstention_correct, "reasons": reasons}

    def handoff(self, submission: Any) -> dict:
        """Compact packet: answer plus cited evidence the controller actually read."""
        submission = submission if isinstance(submission, dict) else {}
        refs = submission.get("evidence_refs", [])
        refs = list(dict.fromkeys(r for r in refs if isinstance(r, str))) if isinstance(refs, list) else []
        answer = submission.get("answer")
        if type(answer) not in (int, float) or (type(answer) is float and not math.isfinite(answer)):
            answer = None
        evidence = []
        for ref in refs:
            if ref in self._observed:
                obs = self._observed[ref]
                evidence.append({"ref": ref, "uri": obs["uri"], "type": obs["type"], "name": obs["name"],
                                 "fields": obs["fields"], "links": obs["links"],
                                 "provenance": {k: obs["provenance"][k] for k in ("scope", "status", "epoch", "revision")}})
        return {"task_id": self.task["id"], "answer": answer, "abstain": submission.get("abstain"),
                "evidence": evidence}


ROLE_DESCRIPTIONS = ROLE_WORDS
CONCEPT_SURFACES = CONCEPTS
