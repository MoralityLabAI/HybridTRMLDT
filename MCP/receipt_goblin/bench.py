"""Receipt Goblin benchmark: annotated claims from this repository's result write-ups, with perturbations.

    python -m MCP.receipt_goblin.bench candidates [--per-doc 30]      # review sheet for annotation
    python -m MCP.receipt_goblin.bench build                          # gold.json -> bench.jsonl
    python -m MCP.receipt_goblin.bench baselines --out DIR            # rule baselines over bench.jsonl

Each annotated claim yields up to three items:

* ``clean``: the claim as written (label ``supported``);
* ``off_receipt``: the claimed number replaced by a nearby value that no receipt leaf reads as
  (label ``contradicted``); a plain number lookup catches these;
* ``decoy``: the claimed number replaced by the value of a sibling leaf, the same metric in a
  neighbouring cell or a neighbouring metric of the same cell (label ``contradicted``). The new
  number exists in the receipts, so a number lookup calls it supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .claims import Claim, extract
from .numbers import Num, matches, parse_one, render
from .packet import build as build_packet, certify, number_matches
from .receipts import ReceiptIndex

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TRIVIAL = {"95%", "5080", "80", "2", "27B"}


def load_config() -> list[dict]:
    return json.loads((HERE / "bench_config.json").read_text(encoding="utf-8"))["pairs"]


class Indexes(dict):
    def __missing__(self, receipts: str) -> ReceiptIndex:
        self[receipts] = ReceiptIndex(ROOT / receipts)
        return self[receipts]


def all_claims() -> list[tuple[dict, Claim]]:
    out = []
    for pair in load_config():
        out.extend((pair, claim) for claim in extract(ROOT / pair["doc"], pair["doc"]))
    return out


def cmd_candidates(args) -> None:
    indexes = Indexes()
    by_doc: dict[str, list] = {}
    for pair, claim in all_claims():
        if claim.target.surface in TRIVIAL:
            continue
        index = indexes[pair["receipts"]]
        hits = number_matches(index, claim.target)
        if 1 <= len(hits) <= args.max_matches:
            by_doc.setdefault(pair["doc"], []).append((claim, index, hits))
    sheet = {}
    for doc, rows in by_doc.items():
        rng = random.Random(int(hashlib.sha256(doc.encode()).hexdigest()[:8], 16))
        picked = sorted(rng.sample(rows, min(args.per_doc, len(rows))), key=lambda r: (r[0].line, r[0].target.start))
        for claim, index, hits in picked:
            scores = index.lexical_scores(claim.context(), hits)
            ranked = sorted(hits, key=lambda lid: (-scores.get(lid, 0.0), lid))[:args.show]
            sheet[claim.cid] = [index.leaves[lid].path for lid in ranked]
            text = claim.marked()
            at = text.index("⟦")
            print(f"## {claim.cid}  [{len(hits)} matches]")
            print("   " + text[max(0, at - 160):at + 120])
            for rank, lid in enumerate(ranked):
                print(f"   {rank}: {index.leaves[lid].path} = {index.leaves[lid].value:g}")
    if args.json:
        Path(args.json).write_text(json.dumps(sheet, indent=1), encoding="utf-8")


def cmd_gold(args) -> None:
    """Turn review decisions ({cid: [candidate ranks]}) into gold.json with leaf paths."""
    sheet = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    claims = {}
    for cid, ranks in decisions.items():
        claims[cid] = {"leaves": [sheet[cid][r] for r in ranks]}
    payload = {"annotator": args.annotator,
               "rule": "Gold leaves are the receipt values the sentence reports, judged from the sentence, the "
                       "table lead-in and the leaf path. Equivalent copies of one value are all gold. Claims "
                       "with an empty list were reviewed and have no single receipt leaf (derived, aggregate "
                       "or out-of-receipt numbers).",
               "claims": claims}
    (HERE / "gold.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"reviewed": len(claims), "with_gold": sum(1 for c in claims.values() if c["leaves"])}))


def _off_receipt(index: ReceiptIndex, num: Num, rng: random.Random) -> str | None:
    step = 10.0 ** (-num.decimals)
    base = num.mantissa
    offsets = [k for k in range(2, 40)] + [-k for k in range(2, 40)]
    rng.shuffle(offsets)
    for k in offsets:
        rel = abs(k * step) / max(abs(base), step)
        if rel < 0.04 or rel > 0.6:
            continue
        mant = base + k * step
        if mant <= 0 or (num.percent and mant > 100) or (num.decimals and base <= 1.0 and mant >= 1.0):
            continue
        surface = render(mant * num.scale / (100.0 if num.percent else 1.0), num)
        candidate = parse_one(surface)
        if candidate.surface == num.surface:
            continue
        if not any(matches(candidate, leaf.value) for leaf in index.leaves):
            return surface
    return None


def _decoy(index: ReceiptIndex, gold: list[int], num: Num, rng: random.Random) -> tuple[str, int] | None:
    options = []
    for lid in gold:
        for other, shared in index.siblings(lid):
            value = index.leaves[other].value
            if value <= 0 or value * 0.25 > num.value or value * 4 < num.value:
                continue
            if num.decimals == 0 and not float(value).is_integer() and not num.percent and num.scale == 1.0:
                continue
            surface = render(value, num)
            if surface == num.surface or any(matches(parse_one(surface), index.leaves[g].value) for g in gold):
                continue
            options.append((-shared, other, surface))
    if not options:
        return None
    options.sort()
    best = [o for o in options if o[0] == options[0][0]]
    _, other, surface = rng.choice(best)
    return surface, other


def _replace(claim: Claim, surface: str) -> tuple[str, int, int]:
    t = claim.target
    text = claim.text[:t.start] + surface + claim.text[t.end:]
    return text, t.start, t.start + len(surface)


def cmd_build(args) -> None:
    gold = json.loads((HERE / "gold.json").read_text(encoding="utf-8"))["claims"]
    claims = {claim.cid: (pair, claim) for pair, claim in all_claims()}
    indexes = Indexes()
    items = []
    for cid, entry in sorted(gold.items()):
        if not entry.get("leaves"):
            continue
        pair, claim = claims[cid]
        index = indexes[pair["receipts"]]
        by_path = {leaf.path: leaf.lid for leaf in index.leaves}
        gold_ids = [by_path[p] for p in entry["leaves"]]
        assert all(matches(claim.target, index.leaves[g].value) for g in gold_ids), cid
        rng = random.Random(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
        base = {"cid": cid, "doc": claim.doc, "line": claim.line, "receipts": pair["receipts"],
                "gold_leaves": entry["leaves"]}
        items.append({**base, "id": cid + "#clean", "variant": "clean", "label": "supported", "text": claim.text,
                      "target": claim.target.surface, "start": claim.target.start, "end": claim.target.end})
        off = _off_receipt(index, claim.target, rng)
        if off:
            text, s, e = _replace(claim, off)
            items.append({**base, "id": cid + "#off", "variant": "off_receipt", "label": "contradicted",
                          "text": text, "target": off, "start": s, "end": e, "original": claim.target.surface})
        decoy = _decoy(index, gold_ids, claim.target, rng)
        if decoy:
            text, s, e = _replace(claim, decoy[0])
            items.append({**base, "id": cid + "#decoy", "variant": "decoy", "label": "contradicted",
                          "text": text, "target": decoy[0], "start": s, "end": e, "original": claim.target.surface,
                          "decoy_leaf": index.leaves[decoy[1]].path})
    out = HERE / "bench.jsonl"
    out.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    counts = {}
    for item in items:
        counts[item["variant"]] = counts.get(item["variant"], 0) + 1
    print(json.dumps({"items": len(items), "by_variant": counts, "out": str(out.relative_to(ROOT))}))


def load_bench() -> list[dict]:
    return [json.loads(line) for line in (HERE / "bench.jsonl").read_text(encoding="utf-8").splitlines() if line]


def item_parts(item: dict) -> tuple[Num, str]:
    target = parse_one(item["target"])
    query = item["text"][:item["start"]] + " " + item["text"][item["end"]:]
    return target, query


def cmd_baselines(args) -> None:
    indexes = Indexes()
    rows = []
    for item in load_bench():
        index = indexes[item["receipts"]]
        target, query = item_parts(item)
        hits = number_matches(index, target)
        ranked = index.search(query, k=1)
        top1 = ranked[0][0] if ranked else None
        packet = build_packet(index, query, target, budget_tokens=args.budget, k_lexical=args.k_lexical,
                              k_neighbours=args.k_neighbours)
        in_packet = [lid for lid in packet.ids.values() if lid in set(hits)]
        by_path = {leaf.path: leaf.lid for leaf in index.leaves}
        gold_ids = {by_path[p] for p in item["gold_leaves"]}
        for arm, verdict, leaf in (
            ("number_lookup", "supported" if hits else "unsupported", None),
            ("lexical_top1", certify(index, target, top1)["verdict"], top1),
            ("packet_number_lookup", "supported" if in_packet else "unsupported", None),
        ):
            rows.append({"id": item["id"], "variant": item["variant"], "label": item["label"], "arm": arm,
                         "verdict": verdict, "linked_gold": (leaf in gold_ids) if leaf is not None else None,
                         "packet_gold_recall": bool(gold_ids & set(packet.ids.values())),
                         "packet_tokens": packet.tokens})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    from .score import summarize
    summary = summarize(rows)
    (out / "baselines_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--per-doc", type=int, default=30)
    c.add_argument("--max-matches", type=int, default=50)
    c.add_argument("--show", type=int, default=4)
    c.add_argument("--json", help="also write {cid: ranked leaf paths} for the gold command")
    g = sub.add_parser("gold")
    g.add_argument("--candidates", required=True)
    g.add_argument("--decisions", required=True)
    g.add_argument("--annotator", default="Claude Opus 5.5 (Claude Code), reviewing the candidate sheet")
    sub.add_parser("build")
    b = sub.add_parser("baselines")
    b.add_argument("--out", default=str(ROOT / "MCP/results/receipt_goblin/baselines"))
    b.add_argument("--budget", type=int, default=3000)
    b.add_argument("--k-lexical", type=int, default=40)
    b.add_argument("--k-neighbours", type=int, default=8)
    args = parser.parse_args()
    {"candidates": cmd_candidates, "gold": cmd_gold, "build": cmd_build,
     "baselines": cmd_baselines}[args.cmd](args)


if __name__ == "__main__":
    main()
