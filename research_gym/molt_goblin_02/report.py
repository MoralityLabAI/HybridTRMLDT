"""Render campaign archives (hypotheses, architectures, promotions, Inky usage) as markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x, nd=3):
    return "—" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def inky_usage(log: Path) -> dict:
    u = {"requests": 0, "ok": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0, "truncated": 0,
         "by_purpose": {}}
    if not log.exists():
        return u
    for line in log.read_text(encoding="utf-8-sig").splitlines():
        r = json.loads(line)
        u["requests"] += 1
        u["by_purpose"][r["purpose"]] = u["by_purpose"].get(r["purpose"], 0) + 1
        if r.get("ok"):
            u["ok"] += 1
            us = r.get("usage") or {}
            u["prompt_tokens"] += int(us.get("prompt_tokens") or 0)
            u["completion_tokens"] += int(us.get("completion_tokens") or 0)
            u["truncated"] += r.get("finish_reason") == "length"
        else:
            u["failed"] += 1
    return u


def render(run: Path) -> str:
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    arch = st["archive"]
    name = {}
    for cid, a in arch.items():
        name[cid] = a["origin"].get("name") or f"c{cid[:6]}"
    hist = st["history"]
    props = [h for h in hist if h["kind"] == "proposal"]
    advs = [h for h in hist if h["kind"] == "adversarial_task"]
    u = inky_usage(run / "inky_requests.jsonl")
    hours = ((st.get("finished_at") or max(h["t"] for h in hist)) - st["started_at"]) / 3600
    out = [f"Campaign `{run.name}`: {len(props)} proposal iterations, {sum(1 for h in props if h.get('candidate_id'))} trained, "
           f"{sum(1 for h in props if h.get('error'))} recorded failures (parse, duplicate, invalid or training errors), "
           f"{len(advs)} adversarial task requests ({len(st['adversarial'])} verified and used, {len(st['rejected_adversarial'])} rejected), "
           f"{len(st['cache'])} CPU training runs, {hours:.2f} h wall clock, proposer `{st['proposer']}`, resumed {st.get('resumed', 0)}×.",
           f"Inky Mini: {u['requests']} requests ({u['ok']} ok, {u['failed']} failed after retries, {u['truncated']} truncated by the "
           f"completion cap), {u['prompt_tokens']:,} prompt and {u['completion_tokens']:,} completion tokens, paid spend $0.", "",
           "### Hypothesis archive", "",
           "| it | mech | candidate change (from base) | base | score | Δ vs prefix by family (pc/rev/rep/pd) | Δ vs control | predicted→observed matches | outcome |",
           "|---:|---|---|---|---:|---|---|---|---|"]
    for h in props:
        obs = h.get("observed") or {}
        fams = ("pointer_chase", "revision", "repair", "proof_deps")
        dp = "/".join(_fmt((obs.get(f) or {}).get("cand_minus_prefix"), 2) for f in fams) if obs else "—"
        dc = "/".join(_fmt((obs.get(f) or {}).get("cand_minus_control"), 2) for f in fams) if obs else "—"
        ver = h.get("verdict") or {}
        vm = f"{sum(v['match'] for v in ver.values())}/{len(ver)}" if ver else "—"
        cand = json.dumps(h.get("candidate"), separators=(",", ":")) if h.get("candidate") is not None else "—"
        outcome = (h.get("error") or "")[:60] or h.get("candidate_id", "")
        out.append(f"| {h['iteration']} | {h.get('mechanism') or '—'} | `{cand[:70]}` | {h.get('base') or '—'} | {_fmt(obs.get('score'))} | {dp} | {dc} | {vm} | {outcome} |")
    out += ["", "### Architecture archive (best rung reached)", "",
            "| id | origin | diff from BASE | rung | score at rung | mean exact | seeds |", "|---|---|---|---:|---:|---:|---|"]
    rows = []
    for cid, a in arch.items():
        if not a["scores"]:
            continue
        s = a["scores"][str(a["rung"])]
        seeds = next((v.get("seeds") for v in s["families"].values() if v.get("seeds")), None)
        rows.append((a["rung"], s.get("score") if s.get("score") is not None else -9, cid, a, s, seeds))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    for rung, _, cid, a, s, seeds in rows:
        o = a["origin"]
        org = o["kind"] + (f" it{o['iteration']}" if "iteration" in o else "")
        out.append(f"| {name[cid]} | {org} | `{json.dumps(a['diff'], separators=(',', ':'))[:90]}` | {rung} | {_fmt(s.get('score'))} | "
                   f"{_fmt(s.get('mean_exact'))} | {seeds} |")
    out += ["", "### Adversarial task variants", "", "| it | spec | accepted | explanations it targets |", "|---:|---|---|---|"]
    for h in advs:
        ex = " vs ".join(str(e)[:80] for e in (h.get("explanations") or []))
        out.append(f"| {h['iteration']} | `{json.dumps(h.get('spec'), separators=(',', ':'))}` | {h.get('accepted') or ('no: ' + str(h.get('error'))[:60])} | {ex} |")
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
