"""Synthetic retrieval fixtures with a public MCP surface and a private judge.

The exact LDT arm checks registered metadata. Semantic relevance stays a search
ordering choice. Fixtures exercise explicit keys and arithmetic; they provide
no evidence about natural-language routing or trained neural performance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any

from research_gym.core.hybrid import (
    CandidateState, HybridMode, LatticeProposal, certify_and_apply_verified,
)
from research_gym.core.typed_soundness import SoundnessType

ARCHITECTURES = ("flat", "search", "layered", "layered_ldt")
FAMILIES = ("lookup", "difference", "cross_branch_sum", "stale_version", "missing_evidence")
_PROVENANCE = ("entity", "field", "revision", "scope", "epoch", "unit")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def make_cases(split: str = "test", seed: int = 7, n: int = 24,
               catalog_size: int = 32, depth: int = 3) -> list[dict]:
    """Generate disjoint, reproducible synthetic source groups for each split.

    Answer labels and required evidence IDs live only in ``oracle``. Every arm
    receives exactly the same case. The parameter ``depth`` affects index shape
    at environment construction and never changes fixture values or labels.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    if type(n) is not int or n < 0 or type(catalog_size) is not int or catalog_size < 8:
        raise ValueError("n must be nonnegative and catalog_size at least eight")
    if type(depth) is not int or not 1 <= depth <= 12:
        raise ValueError("depth must be an integer from one through twelve")
    cases = []
    for index in range(n):
        group = f"mcp-fixture-v1/{split}/{seed}/{index:06d}"
        rng = random.Random(int(_digest(group), 16))
        stem = _digest(group)[:16]
        epoch, scope = f"snapshot-{stem}", f"workspace-{stem}"
        family = FAMILIES[index % len(FAMILIES)]
        entity_count = catalog_size - 3
        # A hidden five-object core fixes the question as distractors are added.
        # Opaque, independently derived IDs reveal no core/distractor marker.
        entities = [f"site-{i % 2}-{stem}-asset-{_digest([group, 'entity', i])[:10]}"
                    for i in range(entity_count)]
        task_rng = random.Random(int(_digest([group, "task"]), 16))
        first, second = task_rng.sample(range(5), 2)
        if family == "cross_branch_sum":
            first, second = task_rng.choice((0, 2, 4)), task_rng.choice((1, 3))
        wanted = [entities[first]]
        operation = "lookup"
        if family in ("difference", "cross_branch_sum", "missing_evidence"):
            wanted.append(entities[second])
            operation = "difference" if family == "difference" else "sum"
        records = []
        for i, entity in enumerate(entities):
            source = _digest([group, "record", i])[:24]
            value_rng = random.Random(int(_digest([group, "value", i]), 16))
            records.append({
                "record_id": f"record-{source}",
                "ref": f"evidence-{_digest([group, 'citation', i])[:24]}",
                "entity": entity, "field": "reading", "revision": "v2",
                "scope": scope, "epoch": epoch, "unit": "points",
                "value": value_rng.randint(-200, 1000),
            })
        if family == "missing_evidence":
            records[second]["value"] = None
        required = [records[entities.index(entity)]["ref"] for entity in wanted]
        # Plausible near-matches have different registered provenance and values.
        for j, changed in enumerate((
            {"revision": "v1"}, {"scope": scope + "-foreign"},
            {"epoch": epoch + "-old"},
        )):
            row = dict(records[first])
            row.update(changed)
            row.update(record_id=f"record-{_digest([group, 'trap', j])[:24]}",
                       ref=f"evidence-{_digest([group, 'trap-citation', j])[:24]}",
                       value=int(records[first]["value"]) + (j + 1) * 1001)
            records.append(row)
        for row in records:
            row["registry_metadata"] = {k: row[k] for k in _PROVENANCE}
        rng.shuffle(records)
        values = [next(row["value"] for row in records if row["ref"] == ref) for ref in required]
        answer = None if any(value is None for value in values) else (
            values[0] - values[1] if operation == "difference" else sum(values)
        )
        wording = {"lookup": "Read", "difference": "Subtract the second reading from the first for",
                   "sum": "Add the readings for"}[operation]
        task = {
            "id": stem,
            "request": f"{wording} {', '.join(wanted)}. Use revision v2, scope {scope}, "
                       f"epoch {epoch}, and unit points. Cite the records you read. "
                       "When a required reading is missing, abstain and cite its missing-evidence record.",
            "operation": operation,
            "required_records": [{"entity": entity, "field": "reading"} for entity in wanted],
            "required_revision": "v2", "scope": scope, "epoch": epoch, "unit": "points",
        }
        cases.append({
            "id": stem, "family": family, "split": split, "group_id": group,
            "provenance": {"kind": "synthetic", "generator": "mcp-fixture-v1", "seed": seed},
            "task": task, "corpus": records,
            "oracle": {"answer": answer, "required_refs": required, "abstain": answer is None},
        })
    return cases


def _tool(name: str, description: str, properties: dict, required: list[str] = ()) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": list(required), "additionalProperties": False},
            "annotations": {"readOnlyHint": True, "destructiveHint": False}}


_FIELDS = {"type": "array", "items": {"type": "string", "enum": ["value"]},
           "description": "Projected data fields; default is [value].", "maxItems": 1}
_PAGE = {"limit": {"type": "integer", "minimum": 1, "maximum": 64, "default": 8},
         "cursor": {"type": "integer", "minimum": 0, "default": 0}}


class RetrievalEnv:
    """One episode; only ``task``, ``tools`` and ``call`` belong to the actor."""

    def __init__(self, case: dict, architecture: str, catalog_size: int = 32, depth: int = 3):
        if architecture not in ARCHITECTURES:
            raise ValueError("unknown architecture")
        if type(depth) is not int or not 1 <= depth <= 12:
            raise ValueError("depth must be an integer from one through twelve")
        if len(case["corpus"]) != catalog_size:
            raise ValueError("catalog_size must match the fixture corpus")
        frozen = copy.deepcopy(case)
        self.task = frozen["task"]
        self.task.update(index_root="root", index_depth=depth)
        self.architecture = architecture
        self._oracle = frozen["oracle"]
        self._records = {row["record_id"]: row for row in frozen["corpus"]}
        if len(self._records) != catalog_size:
            raise ValueError("record IDs must be unique")
        self._by_ref = {row["ref"]: row for row in self._records.values()}
        if len(self._by_ref) != catalog_size:
            raise ValueError("evidence refs must be unique")
        self._leaves = {f"leaf_{i:04d}": key for i, key in enumerate(self._records)}
        self._observed: dict[str, dict] = {}
        self._nodes: dict[str, dict] = {}
        self.stats = {"tool_calls": 0, "read_calls": 0, "index_calls": 0,
                      "catalog_calls": 0, "ldt_checks": 0, "ldt_eliminations": 0,
                      "metadata_checks": 0}
        self._build_tree("root", list(self._records), 1, depth)
        self._registry_digest = _digest({key: row["registry_metadata"]
                                         for key, row in self._records.items()})

    def _build_tree(self, node_id: str, ids: list[str], level: int, depth: int) -> None:
        entities = sorted({self._records[key]["entity"] for key in ids})
        node = {"ids": ids, "children": [], "entity_min": entities[0], "entity_max": entities[-1]}
        self._nodes[node_id] = node
        if level == depth:
            node["children"] = [("record", key) for key in sorted(ids)]
            return
        # Stable public site shards ensure cross-site queries span both root
        # branches for every catalogue width. Deeper levels use entity ranges.
        left = ({entity for entity in entities if entity.startswith("site-0-")}
                if level == 1 and all(entity.startswith(("site-0-", "site-1-")) for entity in entities)
                else set(entities[:max(1, len(entities) // 2)]))
        groups = ([key for key in ids if self._records[key]["entity"] in left],
                  [key for key in ids if self._records[key]["entity"] not in left])
        for i, group in enumerate(groups):
            if group:
                child = f"{node_id}/{i}"
                node["children"].append(("branch", child))
                self._build_tree(child, group, level + 1, depth)

    def _descriptor(self, record_id: str) -> dict:
        row = self._records[record_id]
        return {"kind": "record", "record_id": record_id,
                "metadata": copy.deepcopy(row["registry_metadata"])}

    def tools(self) -> list[dict]:
        if self.architecture == "flat":
            return [_tool(name, "Read registered evidence " + _json(self._descriptor(key)),
                          {"fields": _FIELDS}) for name, key in self._leaves.items()]
        read = _tool("evidence_read", "Read an evidence record by ID. Returns a citation handle, "
                     "source provenance and projected data. Discovery hits contain metadata only.",
                     {"record_id": {"type": "string"}, "fields": _FIELDS}, ["record_id"])
        if self.architecture == "search":
            return [_tool("catalog_search", "Search registered entity and field metadata. "
                          "An exact entity query returns its source versions. Empty query lists the catalogue.",
                          {"query": {"type": "string"}, **_PAGE}, ["query"]), read]
        return [_tool("index_children", "List immediate children of an index node, starting at root. "
                      "Branches give inclusive entity ranges. Siblings remain available. "
                      "The layered_ldt arm checks revision, scope, epoch and unit metadata.",
                      {"node_id": {"type": "string", "default": "root"}, **_PAGE}), read]

    def _possible(self, record_id: str) -> bool:
        self.stats["metadata_checks"] += 1
        metadata = self._records[record_id]["registry_metadata"]
        expected = {"revision": self.task["required_revision"], "scope": self.task["scope"],
                    "epoch": self.task["epoch"], "unit": self.task["unit"]}
        # Unknown metadata and missing data preserve the candidate.
        return all(metadata.get(key) is None or metadata[key] == value for key, value in expected.items())

    def _checked_children(self, children: list[tuple[str, str]]) -> tuple[list, dict]:
        mapping = {key: (self._nodes[key]["ids"] if kind == "branch" else [key])
                   for kind, key in children}
        kept = {key for key, ids in mapping.items() if any(self._possible(item) for item in ids)}
        current = CandidateState({"children": frozenset(mapping)})
        proposed = CandidateState({"children": frozenset(kept)})
        proposal = LatticeProposal(proposed, SoundnessType.ENV_SOUND_DEAD,
                                  mode=HybridMode.DEDUCE if kept else HybridMode.ABSTAIN)

        def verify(candidate: LatticeProposal, context: Any) -> SoundnessType:
            self.stats["ldt_checks"] += 1
            if context != (self.task["epoch"], self._registry_digest):
                return SoundnessType.UNKNOWN
            removed = set(mapping) - set(candidate.proposed_state.domains["children"])
            sound = all(not self._possible(item) for key in removed for item in mapping[key])
            return SoundnessType.ENV_SOUND_DEAD if sound else SoundnessType.UNKNOWN

        decision = certify_and_apply_verified(
            current, proposal, provenance_verifier=verify,
            verifier_context=(self.task["epoch"], self._registry_digest), require_progress=False)
        if not decision.accepted:
            raise RuntimeError("registered metadata refinement failed verification")
        excluded = len(children) - len(kept)
        self.stats["ldt_eliminations"] += excluded
        return [item for item in children if item[1] in kept], {
            "independently_checked": True, "before": len(children), "after": len(kept),
            "excluded": excluded, "registry_digest": self._registry_digest,
        }

    @staticmethod
    def _page(arguments: dict) -> tuple[int, int]:
        limit, cursor = arguments.get("limit", 8), arguments.get("cursor", 0)
        if type(limit) is not int or not 1 <= limit <= 64 or type(cursor) is not int or cursor < 0:
            raise ValueError("limit must be 1..64 and cursor a nonnegative integer")
        return limit, cursor

    def _read(self, record_id: str, fields: Any) -> dict:
        if record_id not in self._records:
            raise ValueError("unknown record_id")
        if fields is None:
            fields = ["value"]
        if not isinstance(fields, list) or len(fields) > 1 or any(field != "value" for field in fields):
            raise ValueError("fields must be [] or [value]")
        row = self._records[record_id]
        result = {"ref": row["ref"], "record_id": record_id, "available": row["value"] is not None,
                  "provenance": {key: row[key] for key in _PROVENANCE},
                  "data": {key: row[key] for key in fields}}
        observed = copy.deepcopy(self._observed.get(row["ref"], result))
        observed["data"].update(result["data"])
        self._observed[row["ref"]] = observed
        self.stats["read_calls"] += 1
        return result

    def call(self, name: str, arguments: dict) -> dict:
        """Return an MCP CallToolResult, including handled malformed calls."""
        self.stats["tool_calls"] += 1
        try:
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            schemas = {tool["name"]: tool["inputSchema"] for tool in self.tools()}
            if name not in schemas:
                raise ValueError("unknown tool")
            schema = schemas[name]
            if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
                raise ValueError("unexpected or missing argument")
            if "fields" in arguments and arguments["fields"] is None:
                raise ValueError("fields must be an array")
            if name in self._leaves:
                payload = self._read(self._leaves[name], arguments.get("fields"))
            elif name == "evidence_read":
                if not isinstance(arguments["record_id"], str):
                    raise ValueError("record_id must be a string")
                payload = self._read(arguments["record_id"], arguments.get("fields"))
            elif name == "catalog_search":
                self.stats["catalog_calls"] += 1
                if not isinstance(arguments["query"], str):
                    raise ValueError("query must be a string")
                limit, cursor = self._page(arguments)
                tokens = arguments["query"].lower().split()
                rows = [self._descriptor(key) for key in self._records]
                rows = [row for row in rows if all(token in _json(row["metadata"]).lower() for token in tokens)]
                rows.sort(key=lambda row: row["record_id"])
                payload = {"records": rows[cursor:cursor + limit], "total": len(rows),
                           "next_cursor": cursor + limit if cursor + limit < len(rows) else None,
                           "epoch": self.task["epoch"]}
            else:
                self.stats["index_calls"] += 1
                node_id = arguments.get("node_id", "root")
                if not isinstance(node_id, str) or node_id not in self._nodes:
                    raise ValueError("unknown node_id")
                limit, cursor = self._page(arguments)
                children = self._nodes[node_id]["children"]
                verification = None
                if self.architecture == "layered_ldt":
                    children, verification = self._checked_children(children)
                rows = []
                for kind, key in children[cursor:cursor + limit]:
                    if kind == "record":
                        rows.append(self._descriptor(key))
                    else:
                        node = self._nodes[key]
                        rows.append({"kind": "branch", "node_id": key,
                                     "entity_min": node["entity_min"], "entity_max": node["entity_max"]})
                payload = {"node_id": node_id, "children": rows, "total": len(children),
                           "next_cursor": cursor + limit if cursor + limit < len(children) else None,
                           "epoch": self.task["epoch"]}
                if verification is not None:
                    payload["verification"] = verification
            return {"content": [{"type": "text", "text": _json(payload)}],
                    "structuredContent": payload, "isError": False}
        except (ValueError, TypeError, KeyError) as error:
            payload = {"error": str(error)}
            return {"content": [{"type": "text", "text": _json(payload)}],
                    "structuredContent": payload, "isError": True}

    def grade(self, submission: dict) -> dict:
        """Private terminal judge; expected answers are never returned."""
        schema_valid = (isinstance(submission, dict)
                        and set(submission) == {"answer", "evidence_refs", "abstain"}
                        and type(submission["abstain"]) is bool
                        and isinstance(submission["evidence_refs"], list)
                        and all(isinstance(ref, str) for ref in submission["evidence_refs"]))
        if not schema_valid:
            return {"success": False, "answer_correct": False, "evidence_valid": False,
                    "evidence_recall": 0.0, "abstention_correct": False,
                    "reasons": ["malformed_submission"]}
        expected_refs = set(self._oracle["required_refs"])
        refs = submission["evidence_refs"]
        valid = set()
        for ref in refs:
            observed = self._observed.get(ref)
            if (ref in expected_refs and observed is not None and "value" in observed["data"]
                    and _source_matches(observed["provenance"], self.task)):
                valid.add(ref)
        evidence_valid = bool(refs) and len(refs) == len(set(refs)) and len(valid) == len(refs)
        recall = len(valid) / len(expected_refs) if expected_refs else 1.0
        answer = submission["answer"]
        expected = self._oracle["answer"]
        numeric = type(answer) is int or (type(answer) is float and math.isfinite(answer))
        answer_correct = answer is None if expected is None else numeric and answer == expected
        abstention_correct = submission["abstain"] == self._oracle["abstain"]
        coherent = (answer is None) == submission["abstain"]
        success = answer_correct and evidence_valid and recall == 1.0 and abstention_correct and coherent
        reasons = [name for name, passed in (
            ("wrong_answer", answer_correct), ("invalid_or_unread_evidence", evidence_valid),
            ("incomplete_evidence", recall == 1.0), ("wrong_abstention", abstention_correct),
            ("inconsistent_answer_abstention", coherent),
        ) if not passed]
        return {"success": bool(success), "answer_correct": bool(answer_correct),
                "evidence_valid": evidence_valid, "evidence_recall": recall,
                "abstention_correct": abstention_correct, "reasons": reasons}

    def handoff(self, submission: dict) -> dict:
        """Expose only cited evidence already obtained through reads."""
        submission = submission if isinstance(submission, dict) else {}
        refs = submission.get("evidence_refs", [])
        refs = refs if isinstance(refs, list) else []
        refs = list(dict.fromkeys(ref for ref in refs if isinstance(ref, str)))
        answer = submission.get("answer")
        if type(answer) not in (int, float) or (type(answer) is float and not math.isfinite(answer)):
            answer = None
        return {"task_id": self.task["id"], "answer": answer,
                "abstain": submission.get("abstain"),
                "evidence": [copy.deepcopy(self._observed[ref]) for ref in refs if ref in self._observed]}


def _source_matches(metadata: dict, task: dict) -> bool:
    return (any(metadata.get("entity") == item["entity"] and metadata.get("field") == item["field"]
                for item in task["required_records"])
            and metadata.get("revision") == task["required_revision"]
            and all(metadata.get(key) == task[key] for key in ("scope", "epoch", "unit")))


def _payload(result: dict) -> dict:
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    for item in result.get("content", []):
        if item.get("type") == "text":
            try:
                return json.loads(item["text"])
            except (TypeError, ValueError):
                pass
    return {}


def choose_reference_action(env_task: dict, tools: list[dict], history: list[dict]) -> dict:
    """Exact public-key controller, independent of the fixture oracle and env.

    It consumes only the same schemas, task and tool outputs available to a
    local model. The reference tests transport and accounting, not neural skill.
    """
    names = {tool["name"] for tool in tools}
    records, observed, pages, branches = {}, {}, set(), {}
    for tool in tools:
        prefix = "Read registered evidence "
        if tool["description"].startswith(prefix):
            descriptor = json.loads(tool["description"][len(prefix):])
            descriptor["tool"] = tool["name"]
            records[descriptor["record_id"]] = descriptor
    for event in history:
        result = event.get("result", {})
        if result.get("isError"):
            continue
        payload = _payload(result)
        arguments = event.get("arguments", {})
        if event["tool"] in ("catalog_search", "index_children"):
            key = arguments.get("query", "") if event["tool"] == "catalog_search" else arguments.get("node_id", "root")
            pages.add((event["tool"], key, arguments.get("cursor", 0)))
            if payload.get("next_cursor") is not None:
                branches[(event["tool"], key, payload["next_cursor"])] = None
        for item in payload.get("records", []) + payload.get("children", []):
            if item.get("kind") == "record":
                records[item["record_id"]] = item
            elif item.get("kind") == "branch":
                branches[("index_children", item["node_id"], 0)] = item
        if "ref" in payload and "value" in payload.get("data", {}):
            if _source_matches(payload["provenance"], env_task):
                observed[payload["provenance"]["entity"]] = payload
    required = [item["entity"] for item in env_task["required_records"]]
    if all(entity in observed for entity in required):
        entries = [observed[entity] for entity in required]
        values = [entry["data"]["value"] for entry in entries]
        missing = any(value is None for value in values)
        answer = None if missing else (values[0] - values[1] if env_task["operation"] == "difference" else sum(values))
        return {"tool": "finish", "arguments": {"answer": answer, "abstain": missing,
                "evidence_refs": [entry["ref"] for entry in entries]}}
    for descriptor in records.values():
        meta = descriptor["metadata"]
        # Unknown registry metadata remains eligible for inspection.
        matches = any(all(meta.get(key) is None or meta.get(key) == target[key]
                          for key in ("entity", "field")) for target in env_task["required_records"])
        constraints = {"revision": env_task["required_revision"], "scope": env_task["scope"],
                       "epoch": env_task["epoch"], "unit": env_task["unit"]}
        if (matches and meta.get("entity") not in observed
                and all(meta.get(key) is None or meta.get(key) == value for key, value in constraints.items())):
            name = descriptor.get("tool", "evidence_read")
            arguments = {"fields": ["value"]}
            if name == "evidence_read":
                arguments["record_id"] = descriptor["record_id"]
            if not any(event["tool"] == name and event.get("arguments") == arguments for event in history):
                return {"tool": name, "arguments": arguments}
    if "catalog_search" in names:
        for entity in required:
            if entity not in observed and ("catalog_search", entity, 0) not in pages:
                return {"tool": "catalog_search", "arguments": {"query": entity, "limit": 8}}
    if "index_children" in names:
        if ("index_children", "root", 0) not in pages:
            return {"tool": "index_children", "arguments": {"node_id": "root", "limit": 8}}
    for (name, key, cursor), branch in branches.items():
        if (name, key, cursor) in pages:
            continue
        if branch and not any(entity not in observed and branch["entity_min"] <= entity <= branch["entity_max"]
                              for entity in required):
            continue
        arguments = {"query" if name == "catalog_search" else "node_id": key,
                     "cursor": cursor, "limit": 8}
        return {"tool": name, "arguments": arguments}
    return {"tool": "finish", "arguments": {"answer": None, "abstain": True,
            "evidence_refs": [entry["ref"] for entry in observed.values()]}}
