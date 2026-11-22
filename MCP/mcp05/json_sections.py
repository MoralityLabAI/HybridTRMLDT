"""Index bounded JSON sections using the original source lines."""
from bisect import bisect_right
from collections.abc import Iterator
import json
import re

from MCP.mcp03.repo import Symbol


WHITESPACE = re.compile(r"[ \t\r\n]*")
MAX_SECTIONS = 128


def members(source: str, start: int, decoder: json.JSONDecoder) -> Iterator[tuple[str, int, object, int, int]]:
    """Yield the key and value bounds from an already validated JSON object."""
    keys = set()
    cursor = WHITESPACE.match(source, start + 1).end()
    while source[cursor] != "}":
        key, key_end = decoder.raw_decode(source, cursor)
        if key in keys:
            raise ValueError("duplicate JSON section key")
        keys.add(key)
        colon = WHITESPACE.match(source, key_end).end()
        value_start = WHITESPACE.match(source, colon + 1).end()
        value, value_end = decoder.raw_decode(source, value_start)
        yield key, cursor, value, value_start, value_end
        cursor = WHITESPACE.match(source, value_end).end()
        if source[cursor] == ",":
            cursor = WHITESPACE.match(source, cursor + 1).end()


def sections(path: str, source: str) -> list[Symbol]:
    decoder = json.JSONDecoder()
    try:
        if not isinstance(decoder.decode(source), dict):
            return []
        line_offsets = [0]
        for line in source.splitlines(keepends=True):
            line_offsets.append(line_offsets[-1] + len(line))
        items = []

        def add_object(start: int, parent: str = "") -> None:
            for key, key_start, value, value_start, value_end in members(source, start, decoder):
                if len(items) == MAX_SECTIONS:
                    return
                if not isinstance(value, (dict, list)):
                    continue
                uri_part = key.replace("~", "~0").replace("/", "~1")
                pointer = f"{parent}/{uri_part}"
                first = bisect_right(line_offsets, key_start)
                last = bisect_right(line_offsets, value_end - 1)
                if first != last:
                    kind = "object" if isinstance(value, dict) else "array"
                    items.append(Symbol(path, pointer, "section", first, last, f"{pointer} ({kind})", ""))
                if not parent and isinstance(value, dict) and len(items) < MAX_SECTIONS:
                    add_object(value_start, pointer)

        add_object(WHITESPACE.match(source).end())
        return items
    except (ValueError, RecursionError):
        return []
