"""Render the sealed kappa data-order intervention as deterministic SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    ROOT
    / "experiments"
    / "loop_schedule_kappa_data_order_v0_2_1_recovery1"
    / "recovery_result.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "lsa_kappa_data_order_v0_2_1.svg"


def _points(values: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in values)


def render_svg(result: dict[str, Any]) -> str:
    summary = result["summary"]
    traces = summary["per_order_trace"]
    cancellation = summary["e2048_cancellation"]
    exposures = (0, 512, 1024, 2048, 4096)
    orders = (103, 211, 223, 227)
    colors = {103: "#496A63", 211: "#D07A28", 223: "#B33A2B", 227: "#226C8A"}
    labels = {103: "D103 replay", 211: "D211", 223: "D223 pass", 227: "D227 pass"}
    left_x0, left_x1 = 92.0, 680.0
    right_x0, right_x1 = 790.0, 1092.0
    top, bottom = 175.0, 485.0

    def x_at(exposure: int) -> float:
        return left_x0 + exposures.index(exposure) * (left_x1 - left_x0) / 4.0

    def y_at(kappa: float) -> float:
        return bottom - (kappa - 0.3) / 1.2 * (bottom - top)

    def bar_y(value: float) -> float:
        return bottom - value / 1.0 * (bottom - top)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="650" viewBox="0 0 1160 650">',
        '<rect width="1160" height="650" fill="#F1EBDD"/>',
        '<rect x="24" y="24" width="1112" height="602" rx="18" fill="#FFFDF7" stroke="#26342F" stroke-width="1.5"/>',
        '<text x="54" y="68" font-family="Georgia, serif" font-size="26" font-weight="bold" fill="#1F2D28">E2048 cancellation recurs; exit timing varies by data order</text>',
        '<text x="54" y="96" font-family="Georgia, serif" font-size="14" fill="#56645F">Fixed construction seed 103; only the training-batch sequence changes</text>',
        '<text x="92" y="142" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#1F2D28">Kappa trajectories</text>',
        '<text x="790" y="142" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#1F2D28">Sensitivity interference at E2048</text>',
        f'<rect x="{x_at(2048)-30:.2f}" y="{top}" width="60" height="{bottom-top}" fill="#ECDDCB" opacity="0.72"/>',
    ]
    for tick in (0.3, 0.6, 0.9, 1.2, 1.5):
        y = y_at(tick)
        parts.extend(
            [
                f'<line x1="{left_x0}" y1="{y:.2f}" x2="{left_x1}" y2="{y:.2f}" stroke="#D8D0C1"/>',
                f'<text x="80" y="{y+4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#56645F">{tick:.1f}</text>',
            ]
        )
    for order in orders:
        trace = traces[str(order)]["kappa"]
        values = [(x_at(exposure), y_at(float(trace[str(exposure)]))) for exposure in exposures]
        parts.append(
            f'<polyline points="{_points(values)}" fill="none" stroke="{colors[order]}" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in values:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#FFFDF7" stroke="{colors[order]}" stroke-width="2.5"/>'
            )
    for exposure in exposures:
        x = x_at(exposure)
        parts.append(
            f'<text x="{x:.2f}" y="507" text-anchor="middle" font-family="Consolas, monospace" font-size="11" fill="#26342F">{exposure}</text>'
        )
    parts.extend(
        [
            '<text x="386" y="535" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#26342F">state-visit exposure</text>',
            '<text x="47" y="335" transform="rotate(-90 47 335)" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#26342F">kappa</text>',
        ]
    )
    bar_width = 47.0
    gap = (right_x1 - right_x0 - len(orders) * bar_width) / (len(orders) - 1)
    threshold_y = bar_y(1.0)
    parts.extend(
        [
            f'<line x1="{right_x0}" y1="{threshold_y:.2f}" x2="{right_x1}" y2="{threshold_y:.2f}" stroke="#26342F" stroke-width="1.5" stroke-dasharray="5 4"/>',
            f'<text x="{right_x1}" y="{threshold_y-7:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#26342F">I_U=1: zero aggregate cross-term</text>',
        ]
    )
    for index, order in enumerate(orders):
        x = right_x0 + index * (bar_width + gap)
        value = float(cancellation[str(order)]["sensitivity_interference_ratio"])
        y = bar_y(value)
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bottom-y:.2f}" rx="4" fill="{colors[order]}"/>',
                f'<text x="{x+bar_width/2:.2f}" y="{y-8:.2f}" text-anchor="middle" font-family="Consolas, monospace" font-size="11" font-weight="bold" fill="#26342F">{value:.3f}</text>',
                f'<text x="{x+bar_width/2:.2f}" y="507" text-anchor="middle" font-family="Consolas, monospace" font-size="11" fill="#26342F">D{order}</text>',
            ]
        )
    legend_x = 105.0
    for index, order in enumerate(orders):
        x = legend_x + index * 145.0
        parts.extend(
            [
                f'<line x1="{x}" y1="574" x2="{x+25}" y2="574" stroke="{colors[order]}" stroke-width="4"/>',
                f'<text x="{x+32}" y="579" font-family="Consolas, monospace" font-size="11" fill="#26342F">{labels[order]}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="790" y="548" width="302" height="51" rx="8" fill="#E6EFEA" stroke="#496A63"/>',
            '<text x="941" y="570" text-anchor="middle" font-family="Georgia, serif" font-size="13" font-weight="bold" fill="#1F2D28">Registered outcome: 2/3 fresh orders pass</text>',
            '<text x="941" y="588" text-anchor="middle" font-family="Georgia, serif" font-size="12" fill="#56645F">Destructive sensitivity cross-term: 4/4 orders</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(result), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
