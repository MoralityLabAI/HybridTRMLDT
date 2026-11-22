"""Shared observation encoder for learned routers (standard library only).

Every learned architecture receives the same input for each candidate: a
hashed word-token sequence ``[CLS] needs [SEP] candidate`` of fixed length and a
small structural vector (item type one-hot, depth, page cursor, log count). No
BM25 score is supplied. Encoder ``tokens`` gives segment ids 0 (needs) and 1
(candidate). Encoder ``match`` additionally marks tokens that occur in both
segments (segment ids 2 and 3), an exact-match channel; template glue words
never count as matches.
"""
from __future__ import annotations

import zlib

from .textutil import tokenize

VOCAB_BUCKETS = 4096
SPECIALS = {"pad": 0, "cls": 1, "sep": 2}
OFFSET = 8
NEED_TOKENS = 32
CAND_TOKENS = 40
SEQ_LEN = 1 + NEED_TOKENS + 1 + CAND_TOKENS
STRUCT_DIM = 9
EXCLUSION_LABELS = ("none", "kind", "status", "scope", "unit")
ENCODER_VERSION = "mcp02-hash4096-seq74-v1"
ENCODERS = ("tokens", "match")
_GLUE = ("need", "find", "pointer", "revalidate", "name", "entity", "region", "field", "unit", "scope", "uri",
         "resource", "branch", "sketch", "count", "server", "collection", "fields", "status", "more", "results",
         "page", "remaining", "link", "for", "search", "of", "list", "unknown", "on", "records", "the", "and")


def token_id(word: str) -> int:
    return OFFSET + zlib.crc32(word.encode("utf-8")) % VOCAB_BUCKETS


def encode_text(text: str, limit: int) -> list[int]:
    return [token_id(w) for w in tokenize(text)[:limit]]


def segments(a: list[int], b: list[int], encoder: str = "tokens") -> list[int]:
    """Segment ids for ``[CLS] a [SEP] b`` under the named encoder."""
    if encoder == "match":
        glue = _glue_ids()
        sa, sb = set(a) - glue, set(b) - glue
        return [0] + [2 if t in sb else 0 for t in a] + [0] + [3 if t in sa else 1 for t in b]
    return [0] * (len(a) + 2) + [1] * len(b)


_GLUE_IDS: set[int] = set()


def _glue_ids() -> set[int]:
    if not _GLUE_IDS:
        _GLUE_IDS.update(token_id(w) for w in _GLUE)
    return _GLUE_IDS


def encode_pair(need: str, candidate: str, encoder: str = "tokens") -> tuple[list[int], list[int]]:
    """Return token ids and segment ids, padded to ``SEQ_LEN``."""
    a = encode_text(need, NEED_TOKENS)
    b = encode_text(candidate, CAND_TOKENS)
    ids = [SPECIALS["cls"]] + a + [SPECIALS["sep"]] + b
    seg = segments(a, b, encoder)
    pad = SEQ_LEN - len(ids)
    return ids + [SPECIALS["pad"]] * pad, seg + [0] * pad


def utility_needs(obs: dict) -> list[str]:
    """Horizon marker prepended only for MCP-03 dual-head checkpoints."""
    remaining = obs.get("remaining_calls")
    if not isinstance(remaining, int) or remaining < 0:
        raise ValueError("dual-utility router needs a public remaining_calls count")
    marker = "budgetlast" if remaining <= 2 else "budgetlate" if remaining <= 6 else "budgetearly"
    return [marker] + list(obs["needs"])


def encode_observation(obs: dict, encoder: str = "tokens") -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    need = " ; ".join(obs["needs"])
    ids, segs, structs = [], [], []
    for cand in obs["candidates"]:
        i, s = encode_pair(need, cand["text"], encoder)
        ids.append(i)
        segs.append(s)
        structs.append(list(cand["struct"]))
    return ids, segs, structs


def rows_as_candidates(rows: list[dict], depth: int) -> list[dict]:
    """Index children rendered with the same text/struct as frontier candidates."""
    from .agent import branch_item, doc_item
    return [branch_item(r, depth) if r.get("kind") == "branch" else doc_item(r, "rows", depth) for r in rows]


def router_tokens(ids: list[list[int]]) -> int:
    return sum(sum(1 for t in row if t != SPECIALS["pad"]) for row in ids)
