"""MCP-04: typed working-set compaction plus exact llama-server token preflight.

Designed for a pinned llama-server fork that supports:
  POST /apply-template {"messages":[...],"tools":[...],"add_generation_prompt":true}
      -> {"prompt": "<fully formatted chat prompt>"}
  POST /tokenize {"content":"...","add_special":false}
      -> {"tokens":[...]}
No approximate count is ever accepted as context-safety evidence. Missing or
incompatible upstream preflight endpoints fail closed.

Existing raw/goblin compaction remains unchanged. The packet preserves live
edit anchors, user constraints, source handles and original recent tool pairs.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

from .packet import _content_text, _split_regions, estimate_tokens

class PreflightError(RuntimeError):
    """Refuse to forward when the real applied template cannot be measured."""


class WorkingSetError(RuntimeError):
    """Refuse to compact away unpaired or required mutable state."""


def _json_object(payload: bytes, description: str) -> dict:
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise PreflightError(f"{description} returned malformed JSON") from exc
    if not isinstance(data, dict):
        raise PreflightError(f"{description} must return a JSON object")
    return data


def exact_preflight(
    forward: Callable[[str, str, bytes], tuple[int, bytes]],
    request: dict, *, context_window: int = 32768, completion_reserve: int = 4096,
    safety_margin: int = 1024,
) -> dict:
    """Count the complete chat template with the upstream's exact token IDs."""
    if min(context_window, completion_reserve, safety_margin) < 0 or context_window <= 0:
        raise ValueError("invalid context preflight budget")
    limit = context_window - completion_reserve - safety_margin
    if limit <= 0:
        raise ValueError("no available prompt window")
    template_request = {
        "messages": request.get("messages", []),
        "tools": request.get("tools", []),
        "add_generation_prompt": True,
    }
    status, raw = forward("POST", "/apply-template",
                          json.dumps(template_request, ensure_ascii=False).encode())
    if status != 200:
        raise PreflightError(f"/apply-template failed with HTTP {status}")
    formatted = _json_object(raw, "/apply-template")
    prompt = formatted.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise PreflightError("/apply-template response lacks full 'prompt' string")
    status, raw = forward("POST", "/tokenize",
                          json.dumps({"content": prompt, "add_special": False}).encode())
    if status != 200:
        raise PreflightError(f"/tokenize failed with HTTP {status}")
    tokens = _json_object(raw, "/tokenize").get("tokens")
    if not isinstance(tokens, list) or not all(type(t) is int and t >= 0 for t in tokens):
        raise PreflightError("/tokenize must return a list of token IDs")
    n = len(tokens)
    return {
        "method": "upstream:/apply-template+/tokenize",
        "prompt_tokens": n,
        "prompt_limit": limit,
        "fits": n <= limit,
        "context_window": context_window,
        "completion_reserve": completion_reserve,
        "safety_margin": safety_margin,
        "formatted_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "tools_counted": len(template_request["tools"]),
    }


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _args(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments") or {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        x = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return x if isinstance(x, dict) else {}


def _path(args: dict) -> str:
    for key in ("path", "file_path", "filename", "uri", "relative_path", "target"):
        val = args.get(key)
        if isinstance(val, str) and val:
            return val[:240]
    return "(unknown path)"


def _site(args: dict) -> str:
    fields = ("start_line", "end_line", "line", "old_str", "new_str", "old_string",
              "new_string", "search", "replace", "position")
    bits = []
    for field in fields:
        val = args.get(field)
        if isinstance(val, (str, int, float)) and val != "":
            text = str(val)
            bits.append(f"{field}={text[:180]}")
    return " | ".join(bits)[:600]


_READ = re.compile(r"(read|open|cat|fetch|inspect|search|query|list|expand)", re.I)
_EDIT = re.compile(r"(edit|write|patch|replace|insert|delete|apply)", re.I)
_EMPTY = re.compile(r"\b(?:no matches?|no results?|0 results?|not found)\b", re.I)


def typed_packet(messages: list[dict], *, budget_tokens: int = 1800, keep_recent: int = 2,
                 store: dict[str, str] | None = None) -> tuple[list[dict], dict]:
    """Build an episode working-set packet from complete history.

    Every older assistant tool call must be followed by its result before it
    can be summarized. Editing calls leave an immutable digest and precise
    relevant anchor fields. Handles are hash-addressed to full tool results.
    """
    if budget_tokens < 200 or keep_recent < 1:
        raise ValueError("bad packet budget or recent-turn count")
    if store is None:
        store = {}
    head_end, tail_start = _split_regions(messages, keep_recent)
    if tail_start <= head_end:
        raise WorkingSetError("history has no closed middle turns available for compaction")
    head, middle, tail = messages[:head_end], messages[head_end:tail_start], messages[tail_start:]
    calls: dict[str, dict] = {}
    edits: list[str] = []
    user_constraints: list[str] = []
    reads: dict[str, str] = {}
    empty: list[str] = []
    n_tools = 0
    for m in middle:
        role = m.get("role")
        if role == "user":
            user_constraints.append(_content_text(m))
        elif role == "assistant":
            for call in m.get("tool_calls") or []:
                cid = call.get("id")
                if not isinstance(cid, str) or not cid or cid in calls:
                    raise WorkingSetError("duplicate or missing tool call ID in archived history")
                calls[cid] = {"call": call, "result": None}
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid not in calls or calls[cid]["result"] is not None:
                raise WorkingSetError("orphan or duplicate tool response in archived history")
            calls[cid]["result"] = m
    if any(entry["result"] is None for entry in calls.values()):
        raise WorkingSetError("cannot compact an outstanding old tool call")
    for cid, entry in calls.items():
        call, result = entry["call"], entry["result"]
        fn = call.get("function") or {}
        name = str(fn.get("name", "?"))
        args = _args(call)
        path = _path(args)
        site = _site(args)
        body = _content_text(result)
        sha = _hash(body)
        handle = "mcp04:" + sha
        store[handle] = body
        n_tools += 1
        status = "error" if result.get("isError") else "returned"
        if _EDIT.search(name):
            edits.append(f"{name}({path}) result={status} args_sha={_hash(str(fn.get('arguments', '')))[:16]} "
                         f"result_handle={handle} anchor={site or '(unavailable)'}")
        elif _READ.search(name):
            reads[path] = f"{name}({path}) {status} handle={handle}"
            if _EMPTY.search(body):
                empty.append(f"observed-empty {name}({path}) handle={handle}")
    # A later user instruction is essential mutable state, retained in full.
    task_constraints = "\n".join(user_constraints)
    essential = [
        "[MCP-04 typed working set]",
        "Older tool output is available by immutable result handle; source is not re-read until needed.",
        "User constraints added since original task:",
        task_constraints if task_constraints else "(none)",
        "Applied edit calls and exact available anchors (chronological):",
        "\n".join(edits) if edits else "(none)",
    ]
    base = "\n".join(essential)
    if estimate_tokens(base) > budget_tokens:
        raise WorkingSetError("user constraints or edit anchors exceed bounded packet; preserve raw history")
    optional = ["Observed empty queries (not certified irrelevant):"] + (empty or ["(none)"])
    optional += ["Recent read handles (full text recoverable):"]
    optional += list(reads.values())[::-1] or ["(none)"]
    optional += [f"Archived paired tool calls: {n_tools}",
                 "Keep original task and recent turns authoritative; validate edit anchors against current source."]
    packet = base
    for line in optional:
        candidate = packet + "\n" + line
        if estimate_tokens(candidate) <= budget_tokens:
            packet = candidate
    new_messages = [*head, {"role": "user", "content": packet}, *tail]
    return new_messages, {
        "compacted": True,
        "method": "typed-working-set",
        "archive_messages": len(middle),
        "recent_turns_kept": keep_recent,
        "archive_paired_calls": n_tools,
        "archive_edit_calls": len(edits),
        "read_handles": len(reads),
        "observed_empty_queries": len(empty),
        "durable_user_constraints": len(user_constraints),
        "packet_tokens_est": estimate_tokens(packet),
        "packet_sha256": _hash(packet),
    }


def adaptive_preflight(
    request: dict, forward: Callable[[str, str, bytes], tuple[int, bytes]],
    *, context_window: int = 32768, completion_reserve: int = 4096,
    safety_margin: int = 1024, keep_recent: int = 2,
    packet_budget: int = 1800, store: dict[str, str] | None = None
) -> tuple[dict, dict]:
    """Raw-first preflight; compress closed history only when the exact prompt exceeds limit."""
    request = dict(request)
    baseline = exact_preflight(forward, request, context_window=context_window,
                               completion_reserve=completion_reserve, safety_margin=safety_margin)
    if baseline["fits"]:
        return request, {"compacted": False, "preflight": baseline, "trials": 1}
    errors = []
    trials = 1
    for recent, budget in ((keep_recent, packet_budget), (1, min(900, packet_budget)),
                           (1, min(500, packet_budget))):
        try:
            messages, receipt = typed_packet(request["messages"], budget_tokens=budget,
                                             keep_recent=recent, store=store)
            candidate = dict(request, messages=messages)
            measured = exact_preflight(forward, candidate, context_window=context_window,
                                       completion_reserve=completion_reserve, safety_margin=safety_margin)
            trials += 1
            if measured["fits"]:
                return candidate, dict(receipt, preflight=measured, raw_prompt_tokens=baseline["prompt_tokens"],
                                       trials=trials)
            errors.append(f"compact still {measured['prompt_tokens']}>{measured['prompt_limit']}")
        except WorkingSetError as exc:
            errors.append(str(exc))
    raise PreflightError("MCP-04 context safety rejection: " + "; ".join(errors))
