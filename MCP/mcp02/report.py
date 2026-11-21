"""Render comparative Markdown tables from one or more summarized MCP-02 runs.

    python -m MCP.mcp02.report --run MCP/results/mcp02/pilot_cpu [--run ...] --out TABLES.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _usd(value):
    return "n/a" if value is None else f"{1e3 * value:.3f}"


def _receipts(run: Path) -> dict:
    path = run / "model_receipts.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _mem(run: Path, name: str) -> dict:
    path = run / "memory" / f"{name}.mem.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def model_table(run: Path, summary: dict) -> list[str]:
    receipts = _receipts(run)
    lines = ["| Policy | Family | Params (total / non-emb.) | Recurrent depth | FLOPs per candidate | "
             "Router ms per decision | Forward peak RSS delta MiB | Dev top-1 | Train s |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    ms = {}
    for row in summary["groups"]:
        if row["experiment"] == "main" and row["router_ms_per_decision"] is not None:
            ms.setdefault(row["policy"], []).append(row["router_ms_per_decision"])
    for name, rec in sorted(receipts.items()):
        params = rec.get("parameters", {})
        if isinstance(params, dict) and "non_embedding" in params:
            ptxt = f"{params['total']:,} / {params['non_embedding']:,}"
        else:
            ptxt = "0" if rec.get("family") not in ("llm",) else _fmt((params or {}).get("total"))
        flops = (rec.get("flops_per_candidate") or {}).get("flops")
        mem = _mem(run, name)
        delta = mem.get("forward_peak_rss_delta")
        training = rec.get("training") or {}
        dev = (training.get("dev_ranking") or {}).get("top1_positive")
        lines.append(f"| `{name}` | {rec.get('family')} | {ptxt} | {rec.get('recurrent_depth', 0)} | "
                     f"{_fmt(flops)} | {_fmt(sum(ms.get(name, [0])) / max(1, len(ms.get(name, [1]))), 2)} | "
                     f"{_fmt(delta / 2**20 if delta is not None else None, 1)} | {_fmt(dev)} | "
                     f"{_fmt(training.get('train_seconds'), 0)} |")
    return lines


def main_table(summary: dict, size: int) -> list[str]:
    rows = [r for r in summary["groups"] if r["experiment"] == "main" and r["catalog_size"] == size]
    arms = sorted({(r["arm"], r["layout"]) for r in rows})
    policies = sorted({r["policy"] for r in rows})
    index = {(r["arm"], r["layout"], r["policy"]): r for r in rows}
    head = "| Policy | " + " | ".join(f"{a}{('/' + l) if l else ''}" for a, l in arms) + " |"
    lines = [head, "| --- |" + " ---: |" * len(arms)]
    for p in policies:
        cells = []
        for a, l in arms:
            r = index.get((a, l, p))
            cells.append("" if r is None else f"{r['accepted']}/{r['episodes']} · {r['tool_calls_mean']:.1f} calls · "
                                              f"{_usd(r['cost_per_success_usd'])}")
        lines.append(f"| `{p}` | " + " | ".join(cells) + " |")
    return lines


def sweep_table(summary: dict, experiment: str, key: str) -> list[str]:
    rows = [r for r in summary["groups"] if r["experiment"] == experiment]
    if not rows:
        return []
    lines = [f"| Policy | Arm | {key} | Accepted | Calls per episode | Router GFLOPs per success | "
             "Cost per success (mUSD) |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: (r["policy"], r["arm"], r["layout"], str(r[key]))):
        gflops = r["router_flops_per_success"] / 1e9 if r["router_flops_per_success"] else None
        lines.append(f"| `{r['policy']}` | {r['arm']}{('/' + r['layout']) if r['layout'] else ''} | {r[key]} | "
                     f"{r['accepted']}/{r['episodes']} | {r['tool_calls_mean']:.1f} | {_fmt(gflops, 2)} | "
                     f"{_usd(r['cost_per_success_usd'])} |")
    return lines


def layout_table(summary: dict) -> list[str]:
    rows = [r for r in summary["groups"] if r["experiment"] == "layout"]
    if not rows:
        return []
    lines = ["| Policy | Arm | Layout (depth) | Accepted | Calls per episode | Tool-result KB per episode | "
             "Refine calls | Excluded | Metadata checks |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: (r["policy"], r["arm"], len(r["layout"]), r["layout"])):
        lines.append(f"| `{r['policy']}` | {r['arm']} | `{r['layout'] or '-'}` ({len(r['layout']) + 1}) | "
                     f"{r['accepted']}/{r['episodes']} | {r['tool_calls_mean']:.1f} | "
                     f"{r['tool_result_bytes_mean'] / 1024:.1f} | {r['refine_calls']} | {r['ldt_excluded']} | "
                     f"{r['ldt_metadata_checks']} |")
    return lines


def paired_table(summary: dict, reference: str, size: int) -> list[str]:
    rows = [p for p in summary["paired"] if p["reference"] == reference and p["catalog_size"] == size]
    if not rows:
        return []
    lines = [f"| Policy | Arm | Success diff vs `{reference}` (95% CI) | W/L | Cost diff mUSD (95% CI) |",
             "| --- | --- | --- | --- | --- |"]
    for p in sorted(rows, key=lambda p: (p["arm"], p["layout"], p["policy"])):
        s, c = p["success_difference"], p["cost_difference_usd"]
        ci = s["ci95"] or [None, None]
        cci = c["ci95"] or [None, None]
        lines.append(f"| `{p['policy']}` | {p['arm']}{('/' + p['layout']) if p['layout'] else ''} | "
                     f"{_fmt(s['mean'])} [{_fmt(ci[0])}, {_fmt(ci[1])}] | {s.get('wins')}/{s.get('losses')} | "
                     f"{_usd(c['mean'])} [{_usd(cci[0])}, {_usd(cci[1])}] |")
    return lines


def family_table(summary: dict) -> list[str]:
    data = summary["success_by_family_main"]
    families = sorted({f for fams in data.values() for f in fams})
    lines = ["| Policy | " + " | ".join(families) + " |", "| --- |" + " ---: |" * len(families)]
    for p, fams in sorted(data.items()):
        lines.append(f"| `{p}` | " + " | ".join(
            f"{fams[f][0]}/{fams[f][1]}" if f in fams else "" for f in families) + " |")
    return lines


def render(runs: list[Path]) -> str:
    out = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        out += [f"## Run `{run.name}`", "", f"Episodes: {summary['episodes']}. Device: {summary['device']}.", ""]
        out += ["### Router receipts", ""] + model_table(run, summary) + [""]
        sizes = sorted({r["catalog_size"] for r in summary["groups"] if r["experiment"] == "main"})
        for size in sizes:
            out += [f"### Main grid, catalog size {size}", "",
                    "Cells: accepted/episodes · mean tool calls · cost per accepted task in milli-USD (default prices).", ""]
            out += main_table(summary, size) + [""]
            for ref in ("deterministic_rules", "lexical_bm25"):
                table = paired_table(summary, ref, size)
                if table:
                    out += [f"#### Paired against `{ref}`, size {size}", ""] + table + [""]
        out += ["### Success by task family (main grid, all arms and sizes)", ""] + family_table(summary) + [""]
        for title, exp, key in (("Recurrent depth at test time", "depth", "depth"),
                                ("Local deliberation budget", "deliberation", "max_tool_calls")):
            table = sweep_table(summary, exp, key)
            if table:
                out += [f"### {title}", ""] + table + [""]
        table = layout_table(summary)
        if table:
            out += ["### Index layout and depth", ""] + table + [""]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.write_text(render(args.run), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
