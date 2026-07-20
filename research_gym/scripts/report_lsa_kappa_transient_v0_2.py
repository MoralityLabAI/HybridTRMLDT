"""Render the recovered R=128 timing discriminator as deterministic SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    ROOT
    / "experiments"
    / "loop_schedule_kappa_transient_v0_2_recovery1"
    / "recovery_result.json"
)
DEFAULT_RECORDS = (
    ROOT
    / "experiments"
    / "loop_schedule_kappa_transient_v0_2_recovery1"
    / "combined_records.jsonl"
)
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "lsa_kappa_transient_v0_2.svg"


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_svg(result: dict[str, Any], records: list[dict[str, Any]]) -> str:
    width, height = 1200, 650
    exposures = (1024, 2048, 4096, 8192)
    seed_colors = {101: "#287A81", 103: "#D48432", 107: "#70766F"}
    kappa = {
        (int(row["seed"]), int(row["exposure"])): float(row["kappa"])
        for row in records
        if row["regime"] == "tied" and "kappa" in row
    }
    summary = result["summary"]
    geometric = {int(key): float(value) for key, value in summary["tied_geometric_kappa"].items()}
    stress = {int(key): float(value) for key, value in summary["step_four_stress_ratios"].items()}
    left_x0, left_x1 = 95.0, 565.0
    right_x0, right_x1 = 695.0, 1135.0
    top, bottom = 170.0, 515.0

    def x_at(value: int, domain: tuple[int, ...], start: float, end: float) -> float:
        return start + domain.index(value) * (end - start) / (len(domain) - 1)

    def kappa_y(value: float) -> float:
        low, high = math.log10(0.025), math.log10(1.5)
        return bottom - (math.log10(value) - low) / (high - low) * (bottom - top)

    def stress_y(value: float) -> float:
        low, high = math.log2(16.0), math.log2(512.0)
        return bottom - (math.log2(value) - low) / (high - low) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="650" fill="#F4F0E7"/>',
        '<rect x="30" y="28" width="1140" height="594" rx="18" fill="#FFFCF5" stroke="#2B302D" stroke-width="1.5"/>',
        '<text x="64" y="70" font-family="Georgia, serif" font-size="26" font-weight="bold" fill="#202723">R=128 favors exposure-pinned timing and survives the excursion</text>',
        '<text x="64" y="96" font-family="Georgia, serif" font-size="14" fill="#59615C">Post-timeout deterministic recovery; three registered seeds; matched untied gradient controls</text>',
        '<text x="95" y="137" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#202723">A. Tied visit alignment at R=128</text>',
        '<text x="695" y="137" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#202723">B. Step-4 tied / untied stress</text>',
    ]
    for tick in (0.03, 0.1, 0.3, 1.0):
        y = kappa_y(tick)
        parts.extend(
            [
                f'<line x1="{left_x0}" y1="{y:.2f}" x2="{left_x1}" y2="{y:.2f}" stroke="#D8D2C6" stroke-width="1"/>',
                f'<text x="80" y="{y + 5:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#59615C">{tick:g}</text>',
            ]
        )
    for tick in (16, 32, 64, 128, 256, 512):
        y = stress_y(float(tick))
        parts.extend(
            [
                f'<line x1="{right_x0}" y1="{y:.2f}" x2="{right_x1}" y2="{y:.2f}" stroke="#D8D2C6" stroke-width="1"/>',
                f'<text x="680" y="{y + 5:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#59615C">{tick}</text>',
            ]
        )
    for seed in (101, 103, 107):
        points = [
            (x_at(exposure, exposures, left_x0, left_x1), kappa_y(kappa[(seed, exposure)]))
            for exposure in exposures
        ]
        color = seed_colors[seed]
        parts.append(
            f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0.7" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for x, y in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
    geometric_points = [
        (x_at(exposure, exposures, left_x0, left_x1), kappa_y(geometric[exposure]))
        for exposure in exposures
    ]
    parts.append(
        f'<polyline points="{_polyline(geometric_points)}" fill="none" stroke="#A62E2E" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for x, y in geometric_points:
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#FFFCF5" stroke="#A62E2E" stroke-width="3"/>'
        )
    for exposure, label in ((2048, "step 2"), (4096, "step 4")):
        x = x_at(exposure, exposures, left_x0, left_x1)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#A62E2E" stroke-width="1.2" stroke-dasharray="5 5"/>',
                f'<text x="{x:.2f}" y="{top + 16}" text-anchor="middle" font-family="Consolas, monospace" font-size="11" fill="#A62E2E">{label}</text>',
            ]
        )
    depths = (32, 64, 128)
    stress_points = [
        (x_at(rounds, depths, right_x0, right_x1), stress_y(stress[rounds]))
        for rounds in depths
    ]
    parts.append(
        f'<polyline points="{_polyline(stress_points)}" fill="none" stroke="#146B74" stroke-width="5" stroke-linecap="round"/>'
    )
    for (x, y), rounds in zip(stress_points, depths):
        parts.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#FFFCF5" stroke="#146B74" stroke-width="3"/>',
                f'<text x="{x:.2f}" y="{y - 13:.2f}" text-anchor="middle" font-family="Consolas, monospace" font-size="11" fill="#103F44">{stress[rounds]:.1f}x</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="915" y="190" text-anchor="middle" font-family="Georgia, serif" font-size="14" fill="#202723">descriptive exponent = {float(summary["three_depth_stress_exponent"]):.3f}</text>',
            '<rect x="64" y="552" width="1072" height="48" rx="10" fill="#EFE8DA"/>',
            '<text x="600" y="573" text-anchor="middle" font-family="Georgia, serif" font-size="14" font-weight="bold" fill="#202723">Frozen result: exposure-pinned, survived recovered excursion</text>',
            '<text x="600" y="592" text-anchor="middle" font-family="Georgia, serif" font-size="12" fill="#59615C">Same exposure (2048), different optimizer step (R64: 4; R128: 2); step-pinned early-Adam timing is not supported.</text>',
            '<text x="330" y="542" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">state-visit exposure</text>',
            '<text x="915" y="542" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">loop depth R at optimizer step 4</text>',
            '<text x="50" y="345" transform="rotate(-90 50 345)" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">kappa (log scale)</text>',
            '<text x="642" y="345" transform="rotate(-90 642 345)" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#202723">stress ratio (log2 scale)</text>',
        ]
    )
    for exposure in exposures:
        x = x_at(exposure, exposures, left_x0, left_x1)
        parts.append(
            f'<text x="{x:.2f}" y="535" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#59615C">{exposure}</text>'
        )
    for rounds in depths:
        x = x_at(rounds, depths, right_x0, right_x1)
        parts.append(
            f'<text x="{x:.2f}" y="535" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#59615C">{rounds}</text>'
        )
    for index, (label, color, width_value) in enumerate(
        (("S101", seed_colors[101], 2.5), ("S103", seed_colors[103], 2.5), ("S107", seed_colors[107], 2.5), ("geomean", "#A62E2E", 5.0))
    ):
        x = 270 + index * 82
        parts.extend(
            [
                f'<line x1="{x}" y1="147" x2="{x + 22}" y2="147" stroke="{color}" stroke-width="{width_value}"/>',
                f'<text x="{x + 27}" y="151" font-family="Consolas, monospace" font-size="10" fill="#202723">{label}</text>',
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
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(result, records), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
