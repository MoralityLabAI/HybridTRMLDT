"""Receipt Goblin MCP server: check numeric claims against run receipts over stdio.

Tools (all read-only):

* ``extract_claims(doc_path)``: every number in a Markdown document with its sentence or table row.
* ``claim_packet(text, receipts_root, target | start+end, ...)``: a token-budgeted packet of receipt
  leaves (``L01 file :: path = value``) that may record the claim's number.
* ``certify(packet_id, leaf)``: compare the claimed number with the leaf a model linked (``L07`` or
  ``NONE``) and return ``supported``, ``contradicted`` or ``unsupported``.

The flow is extract -> packet -> link (the model) -> certify. Used by the ``receipt-goblin`` GDK app
and as a Goose extension:

    goose session --with-extension "python -m MCP.receipt_goblin.server"
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

from MCP.mcp03.tokens import describe as tokenizer_name
from MCP.serve import serve

from .claims import MARK_CLOSE, MARK_OPEN, extract
from .numbers import Num, parse
from .packet import build, certify, number_matches
from .receipts import ReceiptIndex

MAX_PACKETS = 512


def _freshness(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.json")):
        try:
            st = path.stat()
        except OSError:
            continue
        h.update(f"{path.relative_to(root).as_posix()}:{st.st_size}:{st.st_mtime_ns}\n".encode("utf-8"))
    return h.hexdigest()


def _target(text: str, arguments: dict) -> Num:
    nums = parse(text)
    if "start" in arguments and "end" in arguments:
        for num in nums:
            if num.start == arguments["start"] and num.end == arguments["end"]:
                return num
        raise ValueError("start/end do not delimit a number in text")
    target = str(arguments.get("target", "")).strip()
    for num in nums:
        if num.surface == target:
            return num
    raise ValueError("give target (a number as written in text) or start and end offsets")


def _result(payload: dict, text: str | None = None, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text if text is not None else json.dumps(payload)}],
            "structuredContent": payload, "isError": error}


class ReceiptService:
    """Duck-typed environment for MCP.serve: ``task``, ``tools()``, ``call()``."""

    task = {"service": "receipt-goblin", "purpose": "check numeric claims against run receipts"}

    def __init__(self):
        self.indexes: dict[str, tuple[str, ReceiptIndex]] = {}
        self.packets: OrderedDict[str, tuple[ReceiptIndex, Num, dict[str, int]]] = OrderedDict()

    def tools(self) -> list[dict]:
        def tool(name, desc, props, req):
            return {"name": name, "description": desc, "annotations": {"readOnlyHint": True},
                    "inputSchema": {"type": "object", "properties": props, "required": req,
                                    "additionalProperties": False}}
        return [
            tool("extract_claims", "List every number in a Markdown document as a claim: its sentence or "
                 "table row (text), the number as written (target) and its character offsets.",
                 {"doc_path": {"type": "string"}}, ["doc_path"]),
            tool("claim_packet", "Build a compact packet of receipt leaves that may record a claim's number. "
                 "Each line is 'Lnn file :: path = value'. Pick the leaf that measures what the claim's "
                 "number describes, then call certify with its id.",
                 {"text": {"type": "string"}, "receipts_root": {"type": "string"},
                  "target": {"type": "string"}, "start": {"type": "integer", "minimum": 0},
                  "end": {"type": "integer", "minimum": 0},
                  "budget_tokens": {"type": "integer", "minimum": 128, "maximum": 12000},
                  "k_lexical": {"type": "integer", "minimum": 1, "maximum": 400},
                  "k_neighbours": {"type": "integer", "minimum": 0, "maximum": 64},
                  "blind": {"type": "boolean"}},
                 ["text", "receipts_root"]),
            tool("certify", "Compare the claim's number with the value of the linked packet leaf. leaf is an "
                 "id such as L07, or NONE when no leaf records the claimed quantity.",
                 {"packet_id": {"type": "string"}, "leaf": {"type": "string"}}, ["packet_id", "leaf"]),
        ]

    def _index(self, root_arg: str) -> ReceiptIndex:
        root = Path(root_arg).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("receipts_root is not a directory")
        fresh = _freshness(root)
        entry = self.indexes.get(str(root))
        if entry is None or entry[0] != fresh:
            entry = (fresh, ReceiptIndex(root))
            self.indexes[str(root)] = entry
        return entry[1]

    def call(self, name: str, arguments: dict) -> dict:
        try:
            if name == "extract_claims":
                path = Path(arguments["doc_path"]).expanduser().resolve()
                if not path.is_file():
                    raise ValueError("doc_path is not a file")
                claims = [{"cid": c.cid, "line": c.line, "text": c.text, "target": c.target.surface,
                           "start": c.target.start, "end": c.target.end} for c in extract(path, path.name)]
                return _result({"doc": str(path), "claims": claims})
            if name == "claim_packet":
                index = self._index(arguments["receipts_root"])
                text = arguments["text"]
                target = _target(text, arguments)
                query = text[:target.start] + " " + text[target.end:]
                packet = build(index, query, target, budget_tokens=int(arguments.get("budget_tokens", 3000)),
                               k_lexical=int(arguments.get("k_lexical", 40)),
                               k_neighbours=int(arguments.get("k_neighbours", 8)),
                               blind=bool(arguments.get("blind", False)))
                key = json.dumps([str(index.root), text, target.start, packet.text], ensure_ascii=False)
                packet_id = "p" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
                self.packets[packet_id] = (index, target, packet.ids)
                while len(self.packets) > MAX_PACKETS:
                    self.packets.popitem(last=False)
                marked = text[:target.start] + MARK_OPEN + target.surface + MARK_CLOSE + text[target.end:]
                payload = {"packet_id": packet_id, "claim": marked, "target": target.surface,
                           "packet": packet.text, "packet_tokens": packet.tokens, "tokenizer": tokenizer_name(),
                           "leaf_paths": [index.leaves[lid].path for lid in packet.ids.values()],
                           "number_matches": len(number_matches(index, target)),
                           "leaves_indexed": len(index.leaves), "files_indexed": len(index.files)}
                return _result(payload, f"packet_id: {packet_id}\nclaim: {marked}\n{packet.text}")
            if name == "certify":
                entry = self.packets.get(arguments["packet_id"])
                if entry is None:
                    raise ValueError("unknown packet_id; build the packet first")
                index, target, ids = entry
                leaf = arguments["leaf"].strip().upper()
                if leaf != "NONE" and leaf not in ids:
                    raise ValueError(f"leaf {leaf} is not in packet {arguments['packet_id']}")
                payload = certify(index, target, None if leaf == "NONE" else ids[leaf])
                payload.update({"packet_id": arguments["packet_id"], "leaf_id": leaf, "claimed": target.surface})
                return _result(payload)
            raise ValueError("unknown tool")
        except (ValueError, KeyError, OSError) as error:
            return _result({"error": str(error)}, error=True)


def main() -> None:
    serve(ReceiptService(), sys.stdin.buffer, sys.stdout.buffer, max_message_bytes=8 * 1024 * 1024)


if __name__ == "__main__":
    main()
