"""Named token counter for packets and tool results.

With the ``tokenizers`` package and ``MCP03_TOKENIZER`` pointing at a
``tokenizer.json``, counts are exact for that tokenizer. Otherwise the counter
is the labelled estimate ceil(UTF-8 bytes / 4). ``describe()`` says which ran.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

_TOKENIZER = None
_NAME = "estimate:utf8_bytes_div_4"


def _load():
    global _TOKENIZER, _NAME
    path = os.environ.get("MCP03_TOKENIZER")
    if _TOKENIZER is None and path and os.path.isfile(path):
        try:
            from tokenizers import Tokenizer
            _TOKENIZER = Tokenizer.from_file(path)
            _NAME = "tokenizers:" + os.environ.get("MCP03_TOKENIZER_NAME", path)
        except ImportError:
            pass
    return _TOKENIZER


@lru_cache(maxsize=200_000)
def count_tokens(text: str) -> int:
    tok = _load()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False).ids)
    return math.ceil(len(text.encode("utf-8")) / 4)


def describe() -> str:
    _load()
    return _NAME
