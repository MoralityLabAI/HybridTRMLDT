"""Deterministic tokenization, hashing and BM25 shared by the environment and routers."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Iterable

_WORD = re.compile(r"[a-z0-9]+")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def byte_size(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_int(*parts: Any) -> int:
    return int(digest(list(parts))[:16], 16)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed document list; IDF comes from that list only."""

    def __init__(self, documents: Iterable[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [tokenize(text) for text in documents]
        self.k1, self.b = k1, b
        self.lengths = [len(doc) for doc in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        self.tf = [Counter(doc) for doc in self.docs]
        frequency: Counter[str] = Counter()
        for doc in self.docs:
            frequency.update(set(doc))
        n = len(self.docs)
        self.idf = {term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}
        self.operations = 0

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, length in zip(self.tf, self.lengths):
            score = 0.0
            for term in terms:
                count = tf.get(term)
                self.operations += 1
                if count:
                    norm = count + self.k1 * (1 - self.b + self.b * length / (self.avg or 1.0))
                    score += self.idf.get(term, 0.0) * count * (self.k1 + 1) / norm
            out.append(score)
        return out
