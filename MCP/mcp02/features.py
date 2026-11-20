"""Shared observation encoder for learned routers (standard library only).

Every learned architecture receives the same input for each candidate: a
hashed word-token sequence ``[CLS] needs [SEP] candidate`` of fixed length and a
small structural vector (item type one-hot, depth, page cursor, log count). No
lexical-match or BM25 features are supplied; matching must be learned.
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


def token_id(word: str) -> int:
    return OFFSET + zlib.crc32(word.encode("utf-8")) % VOCAB_BUCKETS


def encode_text(text: str, limit: int) -> list[int]:
    return [token_id(w) for w in tokenize(text)[:limit]]


def encode_pair(need: str, candidate: str) -> tuple[list[int], list[int]]:
    """Return token ids and segment ids, padded to ``SEQ_LEN``."""
    a = encode_text(need, NEED_TOKENS)
    b = encode_text(candidate, CAND_TOKENS)
    ids = [SPECIALS["cls"]] + a + [SPECIALS["sep"]] + b
    seg = [0] * (len(a) + 2) + [1] * len(b)
    pad = SEQ_LEN - len(ids)
    return ids + [SPECIALS["pad"]] * pad, seg + [0] * pad


def encode_observation(obs: dict) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    need = " ; ".join(obs["needs"])
    ids, segs, structs = [], [], []
    for cand in obs["candidates"]:
        i, s = encode_pair(need, cand["text"])
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
