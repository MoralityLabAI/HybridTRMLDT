"""Render the registered kappa-surface recovery result as deterministic SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    ROOT / "experiments" / "loop_schedule_kappa_surface_v1_recovery2" / "recovery2_result.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "lsa_kappa_surface_recovery2.svg"


def _line(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_svg(result: dict[str, Any]) -> str:
    width, height = 1200, 620
    colors = {16: "#146B74", 32: "#D9772D", 64: "#A62E2E"}
    surface_exposures = (0, 1024, 2048, 4096)
    stress_exposures = (512, 1024, 2048, 4096)
    left_x0, left_x1 = 92.0, 555.0
    right_x0, right_x1 = 680.0, 1143.0
    top, bottom = 160.0, 510.0

    def x_at(
        exposure: int, start: float, end: float, domain: tuple[int, ...]
    ) -> float:
        return start + domain.index(exposure) * (end - start) / (len(domain) - 1)

    def kappa_y(value: float) -> float:
        return bottom - value / 3.0 * (bottom - top)

    def stress_y(value: float) -> float:
        low, high = math.log10(1.0), math.log10(128.0)
        return bottom - (math.log10(value) - low) / (high - low) * (bottom - top)

    surface = {
        (int(row["rounds"]), int(row["exposure"])): float(row["kappa"])
        for row in result["tied_geometric_surface"]
    }
    stress = {
        tuple(map(int, key.removeprefix("R").replace("_E", " ").split())): float(value)
        for key, value in result["gradient_coupling"]["stress_ratios"].items()
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="620" fill="#F4F0E7"/>',
        '<rect x="34" y="32" width="1132" height="556" rx="18" fill="#FFFCF5" stroke="#2B302D" stroke-width="1.5"/>',
        '<text x="65" y="72" font-family="Georgia, serif" font-size="25" font-weight="bold" fill="#202723">High-loop alignment forms a transient trough, not a persistent change point</text>',
        '<text x="65" y="96" font-family="Georgia, serif" font-size="14" fill="#59615C">Post-partial registered recovery; geometric means across seeds 101, 103, 107</text>',
        '<text x="92" y="128" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#202723">A. Tied visit alignment</text>',
        '<text x="680" y="128" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#202723">B. Tied / untied gradient stress</text>',
    ]
    for tick in (0.0, 1.0, 2.0, 3.0):
        y = kappa_y(tick)
        parts.extend(
            [
                f'<line x1="{left_x0}" y1="{y:.2f}" x2="{left_x1}" y2="{y:.2f}" stroke="#D8D2C6" stroke-width="1"/>',
                f'<text x="78" y="{y + 5:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#59615C">{tick:.0f}</text>',
            ]
        )
    for tick in (1, 2, 4, 8, 16, 32, 64, 128):
        y = stress_y(float(tick))
        parts.extend(
            [
                f'<line x1="{right_x0}" y1="{y:.2f}" x2="{right_x1}" y2="{y:.2f}" stroke="#D8D2C6" stroke-width="1"/>',
                f'<text x="666" y="{y + 5:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#59615C">{tick}</text>',
            ]
        )
    threshold_y = stress_y(5.0)
    parts.extend(
        [
            f'<line x1="{right_x0}" y1="{threshold_y:.2f}" x2="{right_x1}" y2="{threshold_y:.2f}" stroke="#202723" stroke-width="1.5" stroke-dasharray="7 6"/>',
            f'<text x="{right_x0 + 6}" y="{threshold_y + 18:.2f}" text-anchor="start" font-family="Consolas, monospace" font-size="11" fill="#202723">registered support threshold = 5x</text>',
        ]
    )
    for left_exposure, right_exposure in zip(surface_exposures, stress_exposures):
        left_x = x_at(left_exposure, left_x0, left_x1, surface_exposures)
        right_x = x_at(right_exposure, right_x0, right_x1, stress_exposures)
        parts.extend(
            [
                f'<text x="{left_x:.2f}" y="535" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#59615C">{left_exposure}</text>',
                f'<text x="{right_x:.2f}" y="535" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#59615C">{right_exposure}</text>',
            ]
        )
    for rounds in (16, 32, 64):
        left_points = [
            (
                x_at(exposure, left_x0, left_x1, surface_exposures),
                kappa_y(surface[(rounds, exposure)]),
            )
            for exposure in surface_exposures
        ]
        right_points = [
            (
                x_at(exposure, right_x0, right_x1, stress_exposures),
                stress_y(stress[(rounds, exposure)]),
            )
            for exposure in stress_exposures
        ]
        color = colors[rounds]
        parts.extend(
            [
                f'<polyline points="{_line(left_points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
                f'<polyline points="{_line(right_points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
            ]
        )
        for x, y in left_points + right_points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#FFFCF5" stroke="{color}" stroke-width="3"/>')
    onset_x = x_at(2048, left_x0, left_x1, surface_exposures)
    parts.extend(
        [
            f'<line x1="{onset_x:.2f}" y1="{top}" x2="{onset_x:.2f}" y2="{bottom}" stroke="#A62E2E" stroke-width="1.5" stroke-dasharray="4 5"/>',
            f'<text x="{onset_x - 8:.2f}" y="178" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#A62E2E">curvature onset</text>',
            '<text x="323" y="567" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">state-visit exposure</text>',
            '<text x="911" y="567" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">interval end exposure</text>',
            '<text x="51" y="316" transform="rotate(-90 51 316)" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">kappa</text>',
            '<text x="621" y="316" transform="rotate(-90 621 316)" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">stress ratio (log scale)</text>',
        ]
    )
    for index, rounds in enumerate((16, 32, 64)):
        x = 803 + index * 105
        color = colors[rounds]
        parts.extend(
            [
                f'<line x1="{x}" y1="102" x2="{x + 27}" y2="102" stroke="{color}" stroke-width="4"/>',
                f'<text x="{x + 34}" y="107" font-family="Consolas, monospace" font-size="12" fill="#202723">R={rounds}</text>',
            ]
        )
    parts.append('</svg>')
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
