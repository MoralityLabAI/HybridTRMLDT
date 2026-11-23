"""Assemble a source-attributed long prompt from an immutable local federation."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from collections import deque
from contextlib import closing
from pathlib import Path

from .federation import Federation, readonly, terms


def sections(federation: Federation, lane: dict, *, required: bool = True) -> deque:
    name = lane["index"]
    queries = tuple(lane["prefixes"])
    if not queries:
        raise ValueError("each index lane needs operator-approved path prefixes")
    prefix_filter = " OR ".join("substr(path, 1, length(?)) = ?" for _ in queries)
    bindings = [value for prefix in queries for value in (prefix, prefix)]
    expression = " OR ".join(f'"{term}"' for term in terms(lane["query"]))
    rows = {}
    with closing(readonly(federation.root / f"{name}.sqlite")) as shard:
        for path in lane.get("anchors", []):
            for row in shard.execute(
                "SELECT rowid, path, start, body FROM chunks WHERE path = ? ORDER BY rowid", (path,)
            ):
                rows[row[0]] = row
        for row in shard.execute(
            "SELECT rowid, path, start, body FROM chunks WHERE chunks MATCH ? "
            f"AND ({prefix_filter}) ORDER BY bm25(chunks, 0, 0, 1, 2), rowid LIMIT 400",
            (expression, *bindings),
        ):
            rows.setdefault(row[0], row)
    result = deque()
    for rowid, path, start, body in rows.values():
        if 80 <= len(body) <= 9_000:
            result.append({
                "index": name, "uri": f"index:{name}:{rowid}", "path": path, "start": start,
                "commit": federation.repos[name]["commit"], "body": body,
            })
    if required and not result:
        raise ValueError(f"no evidence sections in {name}")
    return result


def order_sections(federation: Federation, lanes: list[dict]) -> list[dict]:
    pending = {lane["index"]: sections(federation, lane) for lane in lanes}
    weights = {lane["index"]: lane["weight"] for lane in lanes}
    if len(pending) != len(lanes) or not pending or min(weights.values()) <= 0:
        raise ValueError("lanes must have unique indexes and positive weights")
    allocated = dict.fromkeys(pending, 0)
    selected = []
    while pending:
        name = min(pending, key=lambda key: allocated[key] / weights[key])
        item = pending[name].popleft()
        allocated[name] += len(item["body"])
        selected.append(item)
        if not pending[name]:
            del pending[name]
    return selected


def payload(items: list[dict]) -> str:
    return "\n\n".join(
        f"[{item['index']}/{item['path']}:{item['start']} @ {item['commit'][:12]} {item['uri']}]"
        f"\n{item['body']}" for item in items
    ) + "\n"


def count(text: str, port: int) -> int:
    request = {"content": text.replace("<|", "<\u200b|"), "add_special": False, "parse_special": True}
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=30)) as connection:
        connection.request("POST", "/tokenize", body=json.dumps(request),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read())
        if response.status != 200 or not isinstance(data.get("tokens"), list):
            raise RuntimeError(f"local tokenizer returned HTTP {response.status}")
    return len(data["tokens"])


def pack(federation: Federation, lanes: list[dict], output: Path, target: int, port: int) -> dict:
    pool = order_sections(federation, lanes)
    return pack_ordered(federation, pool, output, target, port, required={lane["index"] for lane in lanes})


def pack_ordered(federation: Federation, pool: list[dict], output: Path, target: int, port: int,
                 *, lead: str = "", required: set[str] | None = None) -> dict:
    search = []
    nonspecial_bytes = 0
    for item in pool:
        search.append(item)
        nonspecial_bytes += len(item["body"].encode("utf-8"))
        if nonspecial_bytes > target * 6:
            break
    low, high = 0, len(search)
    while low < high:
        mid = (low + high + 1) // 2
        if count(lead + payload(search[:mid]), port) <= target:
            low = mid
        else:
            high = mid - 1
    picked = search[:low]
    text = lead + payload(picked)
    tokens = count(text, port)
    if not picked or tokens < target * 0.95:
        raise ValueError("eligible sections do not fill the requested prompt")
    if required is not None and {item["index"] for item in picked} != required:
        raise ValueError("eligible sections do not cover the required lanes")
    raw = text.encode("utf-8")
    manifest = {
        "index_revision": federation.revision, "target_evidence_tokens": target, "evidence_tokens": tokens,
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "sections": [{key: value for key, value in item.items() if key != "body"} for item in picked],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.md").write_bytes(raw)
    (output / "evidence.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "sections": len(picked)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=96_800)
    parser.add_argument("--port", type=int, default=18097)
    args = parser.parse_args()
    print(json.dumps(pack(Federation(args.indexes), json.loads(args.spec.read_text(encoding="utf-8-sig")),
                          args.out, args.target, args.port)))


if __name__ == "__main__":
    main()
