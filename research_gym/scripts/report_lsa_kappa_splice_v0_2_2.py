"""Render the sealed kappa checkpoint-splice result as deterministic SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = ROOT / "experiments" / "loop_schedule_kappa_splice_v0_2_2" / "splice_result.json"
DEFAULT_RECORDS = ROOT / "experiments" / "loop_schedule_kappa_splice_v0_2_2" / "splice_records.jsonl"
DEFAULT_SOURCE = (
    ROOT / "experiments" / "loop_schedule_kappa_data_order_v0_2_1" / "data_order_records.jsonl"
)
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "lsa_kappa_splice_v0_2_2.svg"


def _source_kappa(records: list[dict[str, Any]], seed: int, exposure: int) -> float:
    return next(
        float(row["kappa"])
        for row in records
        if int(row["seed_channels"]["data_order_seed"]) == seed
        and int(row["exposure"]) == exposure
    )


def render_svg(
    result: dict[str, Any],
    records: list[dict[str, Any]],
    source: list[dict[str, Any]],
) -> str:
    colors = {
        "S211->D211": "#4B6F66",
        "S211->D223": "#B43B2A",
        "S211->D227": "#226E8B",
        "S223->D211": "#D27A26",
    }
    traces: dict[str, list[tuple[int, float]]] = {
        "S211->D211": [
            (2048, _source_kappa(source, 211, 2048)),
            (4096, _source_kappa(source, 211, 4096)),
        ],
        "S211->D223": [(2048, _source_kappa(source, 211, 2048))],
        "S211->D227": [(2048, _source_kappa(source, 211, 2048))],
        "S223->D211": [(2048, _source_kappa(source, 223, 2048))],
    }
    cell_to_trace = {
        "extension_S211_D211": "S211->D211",
        "splice_S211_D223": "S211->D223",
        "splice_S211_D227": "S211->D227",
        "splice_S223_D211": "S223->D211",
    }
    for row in records:
        key = cell_to_trace.get(str(row["cell_id"]))
        if key:
            traces[key].append((int(row["exposure"]), float(row["kappa"])))
    traces = {key: sorted(dict(values).items()) for key, values in traces.items()}
    starts = {key: values[0][1] for key, values in traces.items()}
    left_x0, left_x1 = 94.0, 715.0
    right_x0, right_x1 = 815.0, 1090.0
    top, bottom = 180.0, 485.0

    def x_at(exposure: int) -> float:
        return left_x0 + (exposure - 2048) / (8192 - 2048) * (left_x1 - left_x0)

    def y_at(change: float) -> float:
        return bottom - (change + 0.7) / 1.4 * (bottom - top)

    def bar_y(value: float) -> float:
        return bottom - value * (bottom - top)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="650" viewBox="0 0 1160 650">',
        '<rect width="1160" height="650" fill="#EEE8DA"/>',
        '<rect x="24" y="24" width="1112" height="602" rx="18" fill="#FFFDF8" stroke="#26342F" stroke-width="1.5"/>',
        '<text x="54" y="68" font-family="Georgia, serif" font-size="25" font-weight="bold" fill="#1F2D28">Rebound is state-by-suffix mixed; sensitivity cancellation persists</text>',
        '<text x="54" y="96" font-family="Georgia, serif" font-size="14" fill="#56645F">Checkpoint splice at E2048; dashed line is the registered +0.1 log-kappa recovery threshold</text>',
        '<text x="94" y="146" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#1F2D28">Continuation trajectories</text>',
        '<text x="815" y="146" font-family="Georgia, serif" font-size="17" font-weight="bold" fill="#1F2D28">I_U at first recovery / endpoint</text>',
    ]
    for tick in (-0.6, -0.4, -0.2, 0.0, 0.1, 0.2, 0.4, 0.6):
        y = y_at(tick)
        dash = ' stroke-dasharray="6 4" stroke-width="2"' if tick == 0.1 else ""
        color = "#9B3D2F" if tick == 0.1 else "#D8D0C1"
        parts.extend(
            [
                f'<line x1="{left_x0}" y1="{y:.2f}" x2="{left_x1}" y2="{y:.2f}" stroke="{color}"{dash}/>',
                f'<text x="82" y="{y+4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#56645F">{tick:+.1f}</text>',
            ]
        )
    for key, values in traces.items():
        points = [
            (x_at(exposure), y_at(math.log(kappa / starts[key])))
            for exposure, kappa in values
        ]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{point_text}" fill="none" stroke="{colors[key]}" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#FFFDF8" stroke="{colors[key]}" stroke-width="2.5"/>'
            )
    for exposure in (2048, 3072, 4096, 5120, 6144, 8192):
        x = x_at(exposure)
        parts.append(
            f'<text x="{x:.2f}" y="507" text-anchor="middle" font-family="Consolas, monospace" font-size="11" fill="#26342F">{exposure}</text>'
        )
    parts.extend(
        [
            '<text x="405" y="535" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#26342F">state-visit exposure</text>',
            '<text x="46" y="335" transform="rotate(-90 46 335)" text-anchor="middle" font-family="Georgia, serif" font-size="13" fill="#26342F">log(K / K_E2048)</text>',
        ]
    )
    summary = result["summary"]["recovery"]
    endpoint_rows = {
        "S211->D211": next(row for row in records if row["cell_id"] == "extension_S211_D211" and row["exposure"] == 5120),
        "S211->D223": next(row for row in records if row["cell_id"] == "splice_S211_D223" and row["exposure"] == summary["S211_D223"]["first_measured_recovery_exposure"]),
        "S211->D227": next(row for row in records if row["cell_id"] == "splice_S211_D227" and row["exposure"] == 4096),
        "S223->D211": next(row for row in records if row["cell_id"] == "splice_S223_D211" and row["exposure"] == summary["S223_D211"]["first_measured_recovery_exposure"]),
    }
    bar_width = 42.0
    gap = (right_x1 - right_x0 - 4 * bar_width) / 3.0
    threshold_y = bar_y(1.0)
    parts.extend(
        [
            f'<line x1="{right_x0}" y1="{threshold_y:.2f}" x2="{right_x1}" y2="{threshold_y:.2f}" stroke="#26342F" stroke-width="1.5" stroke-dasharray="5 4"/>',
            f'<text x="{right_x1}" y="{threshold_y-7:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#26342F">I_U=1</text>',
        ]
    )
    for index, key in enumerate(colors):
        x = right_x0 + index * (bar_width + gap)
        value = float(endpoint_rows[key]["sensitivity_interference_ratio"])
        y = bar_y(value)
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bottom-y:.2f}" rx="4" fill="{colors[key]}"/>',
                f'<text x="{x+bar_width/2:.2f}" y="{y-8:.2f}" text-anchor="middle" font-family="Consolas, monospace" font-size="11" font-weight="bold" fill="#26342F">{value:.3f}</text>',
                f'<text x="{x+bar_width/2:.2f}" y="507" text-anchor="middle" font-family="Consolas, monospace" font-size="10" fill="#26342F">{key.replace("->", "/")}</text>',
            ]
        )
    for index, key in enumerate(colors):
        x = 104 + index * 165
        parts.extend(
            [
                f'<line x1="{x}" y1="576" x2="{x+26}" y2="576" stroke="{colors[key]}" stroke-width="4"/>',
                f'<text x="{x+33}" y="581" font-family="Consolas, monospace" font-size="11" fill="#26342F">{key}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="815" y="545" width="275" height="57" rx="8" fill="#E5EEE9" stroke="#4B6F66"/>',
            '<text x="952" y="567" text-anchor="middle" font-family="Georgia, serif" font-size="13" font-weight="bold" fill="#1F2D28">Mixed state-by-suffix control</text>',
            '<text x="952" y="587" text-anchor="middle" font-family="Georgia, serif" font-size="12" fill="#56645F">D211 delayed recovery: E5120</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line]
    source = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(result, records, source), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
