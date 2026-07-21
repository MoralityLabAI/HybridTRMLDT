"""Render the sealed RLM/TRM/LDT neighborhood result as Markdown and SVG figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping
from xml.sax.saxutils import escape

from research_gym.benchmarks.rlm_hybrid_neighborhood import FAMILIES
from research_gym.integrity import canonical_file_sha256, verify_file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = ROOT / "experiments" / "rlm_trm_ldt_hybrid_neighborhood_v1" / "campaign" / "result.json"
DEFAULT_RECEIPT = ROOT / "experiments" / "rlm_trm_ldt_hybrid_neighborhood_v1" / "campaign" / "result_receipt.json"
DEFAULT_REPORT = ROOT / "reports" / "rlm_trm_ldt_hybrid_neighborhood_v1.md"
DEFAULT_FIGURES = ROOT / "reports" / "figures" / "rlm_trm_ldt_hybrid_neighborhood_v1"

LABELS = {
    "rlm_repl_only": "RLM only",
    "ldt_only": "LDT only",
    "proxy_trm_only": "Proxy TRM",
    "trained_trm_only": "ControlTRM",
    "proxy_trm_ldt_fixed": "Proxy TRM -> LDT",
    "trained_trm_ldt_fixed": "ControlTRM -> LDT",
    "rlm_ldt_membrane": "RLM -> LDT membrane",
    "proxy_trm_rlm_critic_ldt": "Proxy TRM -> RLM -> LDT",
    "trained_trm_rlm_critic_ldt": "ControlTRM -> RLM -> LDT",
    "rlm_tool_conductor": "RLM tool conductor",
    "rlm_recursive_conductor": "RLM recursive conductor",
}

COLORS = {
    "control": "#5f6b73",
    "membrane": "#d46a3a",
    "critic": "#177e89",
    "conductor": "#c19a2b",
    "safe": "#276749",
    "unsafe": "#b53b2e",
    "paper": "#f5f0e6",
    "ink": "#172126",
}


def _kind(architecture: str) -> str:
    if architecture == "rlm_ldt_membrane":
        return "membrane"
    if "critic" in architecture:
        return "critic"
    if "conductor" in architecture:
        return "conductor"
    return "control"


def _svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f'<rect width="100%" height="100%" fill="{COLORS["paper"]}"/>'
        f'<style>text{{font-family:Georgia,serif;fill:{COLORS["ink"]}}}.small{{font-size:12px}}'
        '.label{font-size:14px;font-weight:700}.title{font-size:21px;font-weight:700}</style>'
        f"{body}</svg>\n"
    )
    path.write_text(value, encoding="utf-8", newline="\n")


def render_topology(path: Path) -> None:
    rows = [
        ("H1  Typed membrane", ["Transcript", "RLM", "LDT", "Action"]),
        ("H2  Critic sandwich", ["TRM beam", "RLM critic", "LDT", "Action"]),
        ("H3  Conductor mesh", ["RLM", "TRM tool", "LDT tool", "Typed commit"]),
    ]
    parts = ['<text class="title" x="36" y="38">Registered hybrid control flows</text>']
    for row_index, (title, nodes) in enumerate(rows):
        y = 78 + row_index * 112
        parts.append(f'<text class="label" x="36" y="{y + 31}">{escape(title)}</text>')
        for index, node in enumerate(nodes):
            x = 235 + index * 158
            fill = "#ffffff" if index in {0, 3} else "#e5dfcf"
            parts.append(
                f'<rect x="{x}" y="{y}" width="126" height="56" rx="9" fill="{fill}" '
                f'stroke="{COLORS["ink"]}" stroke-width="1.5"/>'
            )
            parts.append(f'<text class="small" x="{x + 63}" y="{y + 34}" text-anchor="middle">{escape(node)}</text>')
            if index < len(nodes) - 1:
                parts.append(
                    f'<path d="M {x + 126} {y + 28} L {x + 154} {y + 28}" stroke="{COLORS["ink"]}" '
                    'stroke-width="2" marker-end="url(#arrow)"/>'
                )
        if row_index == 2:
            parts.append('<text class="small" x="235" y="355">recursive variant: one soft child, zero authority tools</text>')
    defs = (
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{COLORS["ink"]}"/></marker></defs>'
    )
    _svg(path, 900, 385, defs + "".join(parts))


def render_utility(path: Path, summary: list[Mapping[str, Any]]) -> None:
    rows = sorted(summary, key=lambda row: float(row["macro_utility"]), reverse=True)
    values = [float(row["macro_utility"]) for row in rows]
    low, high = min(0.0, min(values)), max(0.01, max(values))
    span = high - low or 1.0
    width, left, top, row_height = 1060, 250, 58, 31
    plot_width = 740
    zero_x = left + (0.0 - low) / span * plot_width
    parts = ['<text class="title" x="28" y="32">Equal-family macro utility</text>']
    parts.append(f'<line x1="{zero_x:.2f}" y1="48" x2="{zero_x:.2f}" y2="{top + len(rows) * row_height}" stroke="#847c6d"/>')
    for index, row in enumerate(rows):
        architecture = str(row["architecture_id"])
        value = float(row["macro_utility"])
        y = top + index * row_height
        value_x = left + (value - low) / span * plot_width
        x = min(zero_x, value_x)
        bar_width = max(1.0, abs(value_x - zero_x))
        parts.append(f'<text class="small" x="20" y="{y + 17}">{escape(LABELS[architecture])}</text>')
        parts.append(f'<rect x="{x:.2f}" y="{y + 4}" width="{bar_width:.2f}" height="17" fill="{COLORS[_kind(architecture)]}"/>')
        safety = "safe" if int(row["unsafe_count"]) == 0 else "unsafe"
        parts.append(f'<text class="small" x="{left + plot_width + 10}" y="{y + 17}">{value:+.3f} | {safety}</text>')
    _svg(path, width, top + len(rows) * row_height + 30, "".join(parts))


def render_pareto(path: Path, summary: list[Mapping[str, Any]], frontier: list[str]) -> None:
    max_tokens = max(1, max(int(row["total_tokens"]) for row in summary))
    utility = [float(row["macro_utility"]) for row in summary]
    low, high = min(utility), max(utility)
    span = high - low or 1.0
    parts = [
        '<text class="title" x="28" y="32">Utility / provider-token neighborhood</text>',
        '<line x1="82" y1="450" x2="850" y2="450" stroke="#172126"/>',
        '<line x1="82" y1="60" x2="82" y2="450" stroke="#172126"/>',
        '<text class="small" x="410" y="488">provider tokens (log1p scale)</text>',
        '<text class="small" x="18" y="250" transform="rotate(-90 18 250)">macro utility</text>',
    ]
    import math

    for row in summary:
        architecture = str(row["architecture_id"])
        tokens = int(row["total_tokens"])
        value = float(row["macro_utility"])
        x = 82 + math.log1p(tokens) / math.log1p(max_tokens) * 768
        y = 450 - (value - low) / span * 390
        radius = 9 if architecture in frontier else 6
        stroke = COLORS["ink"] if architecture in frontier else "none"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{COLORS[_kind(architecture)]}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text class="small" x="{x + 11:.2f}" y="{y + 4:.2f}">{escape(LABELS[architecture])}</text>')
    _svg(path, 1000, 510, "".join(parts))


def _family_means(records: list[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        grouped[(str(row["architecture_id"]), str(row["task_family"]))].append(float(row["utility"]))
    return {key: mean(values) for key, values in grouped.items()}


def render_family_heatmap(path: Path, summary: list[Mapping[str, Any]], records: list[Mapping[str, Any]]) -> None:
    means = _family_means(records)
    architectures = [str(row["architecture_id"]) for row in summary]
    values = list(means.values())
    low, high = min(values), max(values)
    span = high - low or 1.0
    parts = ['<text class="title" x="26" y="32">Architecture x task-family utility</text>']
    left, top, cell_w, cell_h = 260, 75, 150, 34
    for index, family in enumerate(FAMILIES):
        parts.append(f'<text class="small" x="{left + index * cell_w + cell_w / 2}" y="60" text-anchor="middle">{escape(family)}</text>')
    for row_index, architecture in enumerate(architectures):
        y = top + row_index * cell_h
        parts.append(f'<text class="small" x="18" y="{y + 22}">{escape(LABELS[architecture])}</text>')
        for column, family in enumerate(FAMILIES):
            value = means[(architecture, family)]
            ratio = (value - low) / span
            red = int(219 - ratio * 145)
            green = int(104 + ratio * 78)
            blue = int(72 + ratio * 72)
            x = left + column * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" fill="rgb({red},{green},{blue})"/>')
            parts.append(f'<text class="small" x="{x + (cell_w - 3) / 2}" y="{y + 21}" text-anchor="middle" fill="#ffffff">{value:+.3f}</text>')
    _svg(path, 900, top + len(architectures) * cell_h + 25, "".join(parts))


def _markdown_table(summary: list[Mapping[str, Any]]) -> str:
    lines = [
        "| Architecture | Macro utility | Accuracy | Unsafe | Fallback | Tokens | Calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda value: float(value["macro_utility"]), reverse=True):
        lines.append(
            f'| {LABELS[str(row["architecture_id"])]} | {float(row["macro_utility"]):.4f} | '
            f'{float(row["accuracy"]):.4f} | {float(row["unsafe_rate"]):.4f} | '
            f'{float(row["fallback_rate"]):.4f} | {int(row["total_tokens"]):,} | {int(row["total_calls"]):,} |'
        )
    return "\n".join(lines)


def render_report(result_path: Path, receipt_path: Path, report_path: Path, figures: Path) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not verify_file_sha256(result_path, receipt["result_sha256"]):
        raise RuntimeError("sealed result hash failed verification")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records_path = ROOT / result["records_path"]
    if not verify_file_sha256(records_path, result["records_sha256"]):
        raise RuntimeError("sealed evaluation records failed verification")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    figures.mkdir(parents=True, exist_ok=True)
    render_topology(figures / "architecture_topology.svg")
    render_utility(figures / "macro_utility.svg", result["summary"])
    render_pareto(figures / "pareto.svg", result["summary"], result["pareto_frontier"])
    render_family_heatmap(figures / "family_utility.svg", result["summary"], records)
    relative_figures = figures.relative_to(report_path.parent).as_posix()
    comparisons = []
    for row in result["comparisons"]:
        comparisons.append(
            f'- `{row["treatment"]}` vs `{row["control"]}`: delta {float(row["macro_utility_delta"]):+.4f}, '
            f'95% cluster bootstrap [{float(row["utility_delta_ci_95"][0]):+.4f}, '
            f'{float(row["utility_delta_ci_95"][1]):+.4f}], Holm p={float(row["holm_adjusted_p"]):.4f}.'
        )
    body = f"""# RLM x TRM/LDT Hybrid Architecture Neighborhood v1

This registered campaign compares three hybrid control-flow families with RLM-, LDT-, proxy-TRM-, trained-ControlTRM-, and fixed-flow controls on four generated long-context control task families. The held-out design contains {result['record_count']} paired records: 24 tasks, three registered replicates, and eleven architectures.

![Registered topology]({relative_figures}/architecture_topology.svg)

## Registered Endpoints

The co-primary endpoints are equal-family macro utility and unsafe executed-action rate. Tokens and latency are costs rather than ingredients in a scalar winner. The Pareto set is `{', '.join(result['pareto_frontier'])}`.

![Macro utility]({relative_figures}/macro_utility.svg)

{_markdown_table(result['summary'])}

![Pareto neighborhood]({relative_figures}/pareto.svg)

![Family effects]({relative_figures}/family_utility.svg)

## Registered Comparisons

{chr(10).join(comparisons)}

## Interpretation Boundary

{result['claim_boundary']}

Deterministic local arms are repeated to preserve pairing across task/seed cells; those repeats are not independent model-training replicates. The trained proposer is the gym's 17,203-parameter tied recurrent `ControlTRMProposer`, not an evaluation of the official TinyRecursiveModels implementation. Typed LDT membranes retain host-side action authority, and the recursive conductor's single child has soft-analysis authority only.

## Integrity

- Result SHA-256: `{receipt['result_sha256']}`
- Evaluation records SHA-256: `{receipt['records_sha256']}`
- Evaluation records: {receipt['record_count']}
- Report source result: `{result_path.relative_to(ROOT).as_posix()}`
- Report generated deterministically from the re-verified sealed artifacts above.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    render_report(args.result.resolve(), args.receipt.resolve(), args.report.resolve(), args.figures.resolve())
    print(json.dumps({"report": str(args.report), "report_sha256": canonical_file_sha256(args.report)}))


if __name__ == "__main__":
    main()
