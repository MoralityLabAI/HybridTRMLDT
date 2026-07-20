"""Render the kappa-surface optimizer-schedule audit as deterministic SVG."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    ROOT / "experiments" / "loop_schedule_kappa_surface_v1_recovery2" / "recovery2_result.json"
)
DEFAULT_RECORDS = (
    ROOT
    / "experiments"
    / "loop_schedule_kappa_surface_v1_recovery2"
    / "combined_records.jsonl"
)
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "lsa_kappa_optimizer_schedule_audit.svg"


def _points(values: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in values)


def _tied_surface(records: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in records:
        if row.get("regime") == "tied" and "kappa" in row:
            values[(int(row["rounds"]), int(row["exposure"]))].append(float(row["kappa"]))
    return {
        key: math.exp(sum(math.log(value) for value in samples) / len(samples))
        for key, samples in values.items()
    }


def render_svg(result: dict[str, Any], records: list[dict[str, Any]]) -> str:
    width, height = 1120, 650
    exposures = (0, 512, 1024, 2048, 4096)
    rounds_values = (16, 32, 64)
    colors = {16: "#146B74", 32: "#D9772D", 64: "#A62E2E"}
    surface = _tied_surface(records)
    registered_surface = {
        (int(row["rounds"]), int(row["exposure"])): float(row["kappa"])
        for row in result["tied_geometric_surface"]
    }
    if any(abs(surface[key] - value) > 1e-12 for key, value in registered_surface.items()):
        raise ValueError("combined records do not reproduce the registered surface")
    x0, x1 = 105.0, 1030.0
    plot_top, plot_bottom = 195.0, 465.0

    def x_at(exposure: int) -> float:
        return x0 + exposures.index(exposure) * (x1 - x0) / (len(exposures) - 1)

    def y_at(kappa: float) -> float:
        return plot_bottom - kappa / 3.0 * (plot_bottom - plot_top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1120" height="650" fill="#F4F0E7"/>',
        '<rect x="30" y="28" width="1060" height="594" rx="18" fill="#FFFCF5" stroke="#2B302D" stroke-width="1.5"/>',
        '<text x="62" y="70" font-family="Georgia, serif" font-size="25" font-weight="bold" fill="#202723">Only R=64 forms a trough at AdamW step 4</text>',
        '<text x="62" y="96" font-family="Georgia, serif" font-size="14" fill="#59615C">Step-aligned backfills were already present; an R-amplified optimizer transient remains open</text>',
        '<rect x="105" y="120" width="925" height="38" rx="8" fill="#E5EFE9" stroke="#146B74"/>',
        '<line x1="130" y1="132" x2="1005" y2="132" stroke="#146B74" stroke-width="4"/>',
        '<text x="557" y="151" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#103F44">constant LR 0.001 across every optimizer step</text>',
    ]
    for tick in (0.0, 1.0, 2.0, 3.0):
        y = y_at(tick)
        parts.extend(
            [
                f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#D8D2C6" stroke-width="1"/>',
                f'<text x="91" y="{y + 5:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#59615C">{tick:.0f}</text>',
            ]
    )
    for rounds in rounds_values:
        values = [
            (x_at(exposure), y_at(surface[(rounds, exposure)]))
            for exposure in exposures
            if (rounds, exposure) in surface
        ]
        color = colors[rounds]
        parts.append(
            f'<polyline points="{_points(values)}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for x, y in values:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#FFFCF5" stroke="{color}" stroke-width="3"/>'
            )
        step_four_exposure = 4 * 8 * rounds
        step_four_x = x_at(step_four_exposure)
        step_four_y = y_at(surface[(rounds, step_four_exposure)])
        parts.append(
            f'<rect x="{step_four_x - 7:.2f}" y="{step_four_y - 7:.2f}" width="14" height="14" transform="rotate(45 {step_four_x:.2f} {step_four_y:.2f})" fill="#FFFCF5" stroke="{color}" stroke-width="3"/>'
        )
    onset_x = x_at(2048)
    parts.extend(
        [
            f'<line x1="{onset_x:.2f}" y1="{plot_top}" x2="{onset_x:.2f}" y2="{plot_bottom}" stroke="#A62E2E" stroke-width="1.5" stroke-dasharray="5 5"/>',
            f'<text x="{onset_x - 9:.2f}" y="214" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#A62E2E">R=64 trough</text>',
            '<text x="61" y="330" transform="rotate(-90 61 330)" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">geometric-mean kappa</text>',
        ]
    )
    for exposure in exposures:
        x = x_at(exposure)
        parts.append(
            f'<text x="{x:.2f}" y="490" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#202723">{exposure}</text>'
        )
    parts.append(
        '<text x="567" y="516" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">state-visit exposure</text>'
    )
    for row_index, rounds in enumerate(rounds_values):
        y = 548 + row_index * 23
        parts.append(
            f'<text x="91" y="{y}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="{colors[rounds]}">R={rounds} step</text>'
        )
        for exposure in exposures:
            step = exposure // (8 * rounds)
            parts.append(
                f'<text x="{x_at(exposure):.2f}" y="{y}" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#59615C">{step}</text>'
            )
    for index, rounds in enumerate(rounds_values):
        x = 746 + index * 93
        parts.extend(
            [
                f'<line x1="{x}" y1="177" x2="{x + 25}" y2="177" stroke="{colors[rounds]}" stroke-width="4"/>',
                f'<text x="{x + 31}" y="182" font-family="Consolas, monospace" font-size="12" fill="#202723">R={rounds}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(result, records), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
