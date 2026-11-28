"""Render the MOLT-GOBLIN-03 campaign archive (hypotheses, architectures, Inky usage) as markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}"


def render(run: Path) -> str:
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    arch = st["archive"]
    name = {cid: a["origin"].get("name") or f"c{cid[:6]}" for cid, a in arch.items()}
    props = [h for h in st["history"] if h["kind"] == "proposal"]
    advs = [h for h in st["history"] if h["kind"] == "adversarial_task"]
    reqs = [json.loads(l) for l in (run / "inky_requests.jsonl").read_text(encoding="utf-8-sig").splitlines()] \
        if (run / "inky_requests.jsonl").exists() else []
    ok = [r for r in reqs if r.get("ok")]
    hours = (st.get("finished_at", 0) - st["started_at"]) / 3600
    out = [f"Campaign: {len(props)} proposal iterations ({sum(1 for h in props if h.get('candidate_id'))} trained, "
           f"{sum(1 for h in props if h.get('error'))} recorded failures), {len(advs)} adversarial task requests "
           f"({len(st['adversarial'])} verified and used), {len(st['cache'])} development training runs, {hours:.2f} h, "
           f"resumed {st.get('resumed', 0)}x. Inky Mini: {len(reqs)} requests, {len(ok)} ok, "
           f"{sum(int((r.get('usage') or {}).get('prompt_tokens') or 0) for r in ok):,} prompt / "
           f"{sum(int((r.get('usage') or {}).get('completion_tokens') or 0) for r in ok):,} completion tokens, "
           f"{sum(r.get('finish_reason') == 'length' for r in ok)} truncated; paid spend $0. "
           f"Peak process-tree RSS {(st.get('peak_tree_rss_bytes') or 0) / 2 ** 20:.0f} MiB.", "",
           "### Hypothesis archive (development roots, rung 0 = seed 11)", "",
           "| it | mechanism | base | candidate change | quality | score | visits | outcome |", "|---:|---|---|---|---:|---:|---:|---|"]
    for h in props:
        o = h.get("observed") or {}
        cand = json.dumps(h.get("candidate"), separators=(",", ":")) if h.get("candidate") is not None else "—"
        out.append(f"| {h['iteration']} | {h.get('mechanism') or '—'} | {h.get('base') or '—'} | `{cand[:80]}` | "
                   f"{_f(o.get('quality'))} | {_f(o.get('score'))} | {o.get('visits') or '—'} | "
                   f"{(h.get('error') or '')[:70] or h.get('candidate_id', '')} |")
    out += ["", "Hypotheses as stated by Inky:", ""]
    for h in props:
        if h.get("hypothesis"):
            out.append(f"* it{h['iteration']}: {str(h['hypothesis'])[:300]}")
    out += ["", "### Architecture archive (best rung)", "", "| id | origin | canonical diff | rung | quality | score | seeds |",
            "|---|---|---|---:|---:|---:|---|"]
    rows = []
    for cid, a in arch.items():
        if a["scores"]:
            s = a["scores"][str(a["rung"])]
            seeds = next((v.get("seeds") for v in s["families"].values() if v.get("seeds")), None)
            rows.append((a["rung"], s.get("score") if s.get("score") is not None else -9, cid, a, s, seeds))
    for rung, _, cid, a, s, seeds in sorted(rows, key=lambda r: (r[0], r[1]), reverse=True):
        out.append(f"| {name[cid]} | {a['origin']['kind']} | `{json.dumps(a['canonical_diff'], separators=(',', ':'))[:90]}` | "
                   f"{rung} | {_f(s.get('quality'))} | {_f(s.get('score'))} | {seeds} |")
    out += ["", "### Adversarial task variants", "", "| it | spec | accepted |", "|---:|---|---|"]
    for h in advs:
        out.append(f"| {h['iteration']} | `{json.dumps(h.get('spec'), separators=(',', ':'))}` | {h.get('accepted') or ('no: ' + str(h.get('error'))[:60])} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    md = render(a.run)
    if a.out:
        a.out.write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
