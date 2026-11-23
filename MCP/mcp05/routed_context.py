"""Prepare nested evidence budgets using the Pocket Goblin federation navigator."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from .evidence_packet import pack_ordered, sections
from .federation import Federation, terms


def candidate_stream(federation: Federation, lanes: list[dict]) -> tuple[list[dict], list[dict]]:
    by_index = {lane["index"]: lane for lane in lanes}
    queues, traces = [], []
    for lane in lanes:
        query = lane["query"]
        ranked, excluded = federation.shortlist(query, terms(query), scope=by_index)
        selected = []
        for row in ranked:
            name = row["id"]
            retrieved = sections(federation, {
                "index": name, "query": query, "prefixes": by_index[name]["prefixes"],
            }, required=False)
            if retrieved:
                queues.append(retrieved)
                selected.append(name)
            if len(selected) == 2:
                break
        traces.append({"query": query, "ranked": [row["id"] for row in ranked], "excluded": excluded,
                       "selected": selected, "revision": federation.revision})

    pool = []
    seen = set()
    per_path = Counter()
    while queues:
        for queue in queues:
            while queue:
                item = queue.popleft()
                path = (item["index"], item["path"])
                if item["uri"] not in seen and per_path[path] < 3:
                    seen.add(item["uri"])
                    per_path[path] += 1
                    pool.append(item)
                    break
        queues = [queue for queue in queues if queue]
    return pool, traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lead", type=Path, required=True)
    parser.add_argument("--router", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--port", type=int, default=18097)
    args = parser.parse_args()
    start = time.perf_counter()
    router = None
    if args.router:
        from .index_router import IndexRouter

        router = IndexRouter(args.router)
    federation = Federation(args.indexes, router)
    lanes = json.loads(args.spec.read_bytes())
    lead = args.lead.read_bytes().decode("utf-8")
    pool, traces = candidate_stream(federation, lanes)
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"index_revision": federation.revision, "lead_sha256": hashlib.sha256(lead.encode()).hexdigest(),
               "policy": "settrm" if router else "lexical", "queries": traces,
               "candidate_count": len(pool), "routing_and_retrieval_wall_seconds": time.perf_counter() - start,
               "packets": []}
    if args.router:
        receipt["router_sha256"] = hashlib.sha256(args.router.read_bytes()).hexdigest()
    for budget in sorted(set(args.budgets)):
        start = time.perf_counter()
        result = pack_ordered(federation, pool, args.out / str(budget), budget, args.port, lead=lead)
        receipt["packets"].append({"budget": budget, "evidence_tokens": result["evidence_tokens"],
                                   "sections": result["sections"], "evidence_sha256": result["evidence_sha256"],
                                   "packing_wall_seconds": time.perf_counter() - start})
    (args.out / "routing.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(receipt["packets"]), flush=True)


if __name__ == "__main__":
    main()
