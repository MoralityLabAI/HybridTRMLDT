"""Goblin packet compaction for long agent histories.

A multi-turn agent episode grows one tool result at a time. Past a token
threshold, the packet arm replaces the middle of the history with one compact
packet and keeps the task and the most recent turns verbatim:

* head: leading system messages and the first user message (the task);
* tail: the last ``keep_recent`` assistant turns, each with the tool results
  and user messages that follow it, so every tool message keeps its call;
* packet: an action ledger (every earlier tool call, arguments cut to a fixed
  width) and the earlier lines that score highest against the task and the
  tail, cut to ``budget_tokens``.

Line scoring is the MCP-03 packet scorer applied to conversation lines: BM25
over the lines (``MCP.mcp02.textutil.BM25``) plus an exact-match channel that
counts the task's salient terms (``MCP.mcp03.agent.salient_terms``) and
identifier-like tokens from the tail that occur verbatim in the line. Selected
lines keep their original order. The function is pure and deterministic.
"""
from __future__ import annotations

import json
import re
from typing import Any

from MCP.mcp02.textutil import BM25, digest

_IDENT = re.compile(r"\b(?=[A-Za-z0-9_.:/-]*\d)[A-Za-z0-9_][A-Za-z0-9_.:/-]{2,}\b|\"[^\"\n]{3,40}\"")
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
CHARS_PER_TOKEN = 3.2
PACKET_HEADER = "[goblin packet]"


def estimate_tokens(value: Any) -> int:
    """Character estimate used only for the compaction threshold and budget."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _content_text(message: dict) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _exact_terms(text: str, limit: int = 64) -> list[str]:
    from MCP.mcp03.agent import salient_terms

    found = list(salient_terms(text, limit=limit))
    for rx in (_IDENT, _SNAKE, _CAMEL):
        for match in rx.finditer(text):
            token = match.group(0).strip('"')
            if len(token) >= 3:
                found.append(token)
    out, seen = [], set()
    for token in found:
        if token.lower() not in seen:
            seen.add(token.lower())
            out.append(token)
    return out[:limit]


def _split_regions(messages: list[dict], keep_recent: int) -> tuple[int, int]:
    """Return (head_end, tail_start) indexes into ``messages``."""
    head_end = 0
    while head_end < len(messages) and messages[head_end].get("role") == "system":
        head_end += 1
    if head_end < len(messages) and messages[head_end].get("role") == "user":
        head_end += 1
    assistant_idx = [i for i in range(head_end, len(messages)) if messages[i].get("role") == "assistant"]
    if len(assistant_idx) <= keep_recent:
        return head_end, head_end
    return head_end, assistant_idx[-keep_recent]


def _call_line(step: int, call: dict) -> str:
    fn = call.get("function", {}) if isinstance(call, dict) else {}
    name = fn.get("name", "?")
    args = fn.get("arguments", "")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    args = " ".join(args.split())
    if len(args) > 160:
        args = args[:157] + "..."
    return f"step {step}: {name}({args})"


def _units(middle: list[dict], first_step: int) -> tuple[list[str], list[tuple[int, str, str]]]:
    """Action ledger lines and (order, label, line) evidence units of the middle region."""
    ledger: list[str] = []
    units: list[tuple[int, str, str]] = []
    step = first_step
    order = 0
    for message in middle:
        role = message.get("role")
        if role == "assistant":
            step += 1
            for call in message.get("tool_calls") or []:
                ledger.append(_call_line(step, call))
        label = f"step {step} {role}"
        if role == "tool" and message.get("name"):
            label = f"step {step} tool {message['name']}"
        for line in _content_text(message).splitlines():
            line = line.rstrip()
            if line.strip():
                units.append((order, label, line[:400]))
                order += 1
    return ledger, units


def compact_messages(messages: list[dict], *, threshold_tokens: int = 6000, budget_tokens: int = 1800,
                     keep_recent: int = 2, per_message_cap: int = 6000) -> tuple[list[dict], dict]:
    """Compact ``messages`` when their estimate exceeds ``threshold_tokens``.

    Returns the new message list and a receipt. Below the threshold, or when
    there is no middle region, the messages are returned unchanged apart from
    the per-message cap on tail messages.
    """
    before = estimate_tokens(messages)
    receipt: dict[str, Any] = {"tokens_before_est": before, "compacted": False, "capped_messages": 0}
    out = [dict(m) for m in messages]
    head_end, tail_start = _split_regions(out, keep_recent)
    query_text = "\n".join(_content_text(m) for m in out[:head_end] if m.get("role") == "user")
    tail_text = "\n".join(_content_text(m) for m in out[tail_start:])

    capped = 0
    for i in range(tail_start, len(out)):
        text = _content_text(out[i])
        if out[i].get("role") in ("tool", "user") and estimate_tokens(text) > per_message_cap:
            out[i]["content"] = _select_lines(text, query_text + "\n" + tail_text, per_message_cap,
                                              note="[goblin cap] long result cut to the lines most relevant to the task")
            capped += 1
    receipt["capped_messages"] = capped

    if before <= threshold_tokens or tail_start <= head_end:
        receipt["tokens_after_est"] = estimate_tokens(out)
        return out, receipt

    middle = out[head_end:tail_start]
    prior_steps = 0
    ledger, units = _units(middle, prior_steps)
    query = query_text + "\n" + tail_text
    exact = _exact_terms(query)
    exact_lower = [t.lower() for t in exact]
    bm25 = BM25([u[2] for u in units]) if units else None
    scores = bm25.scores(query) if bm25 else []
    ranked = []
    for (order, label, line), score in zip(units, scores):
        low = line.lower()
        hits = sum(1 for t in exact_lower if t in low)
        ranked.append((score + 2.0 * hits, order, label, line))
    ranked.sort(key=lambda r: (-r[0], r[1]))

    ledger_text = "\n".join(ledger) if ledger else "(no tool calls)"
    budget = budget_tokens - estimate_tokens(ledger_text) - 60
    chosen: list[tuple[int, str, str]] = []
    used = 0
    for score, order, label, line in ranked:
        if score <= 0:
            break
        cost = estimate_tokens(line) + 4
        if used + cost > budget:
            continue
        chosen.append((order, label, line))
        used += cost
    chosen.sort()
    lines, last_label = [], None
    for _order, label, line in chosen:
        if label != last_label:
            lines.append(f"[{label}]")
            last_label = label
        lines.append(line)
    packet = (f"{PACKET_HEADER} Earlier steps ({len(middle)} messages, about {estimate_tokens(middle)} tokens) "
              f"were compressed. The action ledger lists every earlier tool call; the evidence lines are the "
              f"earlier lines most relevant to the task, verbatim. Re-run a tool if you need an omitted detail.\n"
              f"Actions so far:\n{ledger_text}\nRetained evidence:\n" + ("\n".join(lines) if lines else "(none)"))
    new = out[:head_end] + [{"role": "user", "content": packet}] + out[tail_start:]
    receipt.update({
        "compacted": True,
        "middle_messages": len(middle),
        "middle_tokens_est": estimate_tokens(middle),
        "ledger_calls": len(ledger),
        "units": len(units),
        "units_kept": len(chosen),
        "exact_terms": len(exact),
        "packet_tokens_est": estimate_tokens(packet),
        "packet_digest": digest(packet),
        "tokens_after_est": estimate_tokens(new),
    })
    return new, receipt


def _select_lines(text: str, query: str, cap_tokens: int, note: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text[: int(cap_tokens * CHARS_PER_TOKEN)]
    exact_lower = [t.lower() for t in _exact_terms(query)]
    scores = BM25(lines).scores(query)
    ranked = sorted(((s + 2.0 * sum(1 for t in exact_lower if t in ln.lower()), i) for i, (ln, s) in
                     enumerate(zip(lines, scores))), key=lambda r: (-r[0], r[1]))
    keep, used = set(), estimate_tokens(note) + 8
    for i in range(min(5, len(lines))):
        keep.add(i)
        used += estimate_tokens(lines[i]) + 1
    for _score, i in ranked:
        cost = estimate_tokens(lines[i]) + 1
        if i in keep or used + cost > cap_tokens:
            continue
        keep.add(i)
        used += cost
    return note + "\n" + "\n".join(lines[i] for i in sorted(keep))
