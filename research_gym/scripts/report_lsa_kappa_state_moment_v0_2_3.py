"""Render the sealed state/moment/suffix factorial result as deterministic SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    ROOT
    / "experiments"
    / "loop_schedule_kappa_state_moment_v0_2_3_recovery1"
    / "state_moment_result.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "lsa_kappa_state_moment_v0_2_3.svg"


def _fill(value: float) -> str:
    if value >= 0.7:
        return "#2F766D"
    if value >= 0.3:
        return "#78A69A"
    if value >= 0.1:
        return "#BCD0C8"
    if value >= 0.0:
        return "#E1E8E2"
    if value >= -0.2:
        return "#E9C0AB"
    return "#C7654D"


def render_svg(result: dict[str, Any]) -> str:
    summary = result["summary"]
    cells = {
        (
            int(cell["model_weight_seed"]),
            int(cell["optimizer_moment_seed"]),
            int(cell["suffix_seed"]),
        ): cell
        for cell in summary["cells"]
    }
    effects = summary["factorial_classification"]["absolute_effect_ranking"]
    ratio = float(summary["factorial_classification"]["largest_to_runner_up_ratio"])
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">',
        '<rect width="1200" height="700" fill="#E8E1D3"/>',
        '<rect x="24" y="24" width="1152" height="652" rx="20" fill="#FFFDF8" stroke="#26342F" stroke-width="1.5"/>',
        '<text x="54" y="70" font-family="Georgia, serif" font-size="26" font-weight="bold" fill="#1F2D28">Rebound is routed by checkpoint-component interactions</text>',
        '<text x="54" y="100" font-family="Georgia, serif" font-size="14" fill="#56645F">Endpoint response y = log(K4096 / K2048); green is recovery, rust is decline</text>',
        '<text x="66" y="150" font-family="Georgia, serif" font-size="18" font-weight="bold" fill="#1F2D28">Complete 2x2x2 response table</text>',
        '<text x="708" y="150" font-family="Georgia, serif" font-size="18" font-weight="bold" fill="#1F2D28">Registered factorial effects</text>',
    ]
    panel_positions = {211: 66.0, 223: 360.0}
    cell_width, cell_height = 112.0, 82.0
    for model_seed, panel_x in panel_positions.items():
        parts.extend(
            [
                f'<text x="{panel_x}" y="186" font-family="Consolas, monospace" font-size="15" font-weight="bold" fill="#26342F">model weights M{model_seed}</text>',
                f'<text x="{panel_x+145}" y="218" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#56645F">suffix D211</text>',
                f'<text x="{panel_x+261}" y="218" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#56645F">suffix D223</text>',
            ]
        )
        for row_index, optimizer_seed in enumerate((211, 223)):
            y = 232.0 + row_index * (cell_height + 10.0)
            parts.append(
                f'<text x="{panel_x+28}" y="{y+47:.1f}" font-family="Consolas, monospace" font-size="12" fill="#26342F">O{optimizer_seed}</text>'
            )
            for column_index, suffix_seed in enumerate((211, 223)):
                cell = cells[(model_seed, optimizer_seed, suffix_seed)]
                value = float(cell["endpoint_log_change"])
                x = panel_x + 89.0 + column_index * (cell_width + 4.0)
                text_color = "#FFFDF8" if value >= 0.7 or value < -0.2 else "#1F2D28"
                recovery = "recover" if cell["recovered_by_endpoint"] else "no"
                parts.extend(
                    [
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width}" height="{cell_height}" rx="8" fill="{_fill(value)}" stroke="#FFFDF8" stroke-width="2"/>',
                        f'<text x="{x+cell_width/2:.1f}" y="{y+36:.1f}" text-anchor="middle" font-family="Consolas, monospace" font-size="19" font-weight="bold" fill="{text_color}">{value:+.3f}</text>',
                        f'<text x="{x+cell_width/2:.1f}" y="{y+59:.1f}" text-anchor="middle" font-family="Consolas, monospace" font-size="10" fill="{text_color}">{recovery}</text>',
                    ]
                )
    bar_x0, bar_x1 = 708.0, 1125.0
    zero_x = 908.0
    max_effect = 0.6
    scale = (bar_x1 - zero_x) / max_effect
    parts.append(
        f'<line x1="{zero_x}" y1="185" x2="{zero_x}" y2="510" stroke="#26342F" stroke-width="1.5"/>'
    )
    labels = {
        "model_weight": "M",
        "optimizer_moment": "O",
        "suffix": "D",
        "model_weight_x_optimizer_moment": "M x O",
        "model_weight_x_suffix": "M x D",
        "optimizer_moment_x_suffix": "O x D",
        "three_way_interaction": "M x O x D",
    }
    ordered = sorted(effects, key=lambda row: list(labels).index(str(row["term"])))
    for index, row in enumerate(ordered):
        value = float(row["effect"])
        y = 198.0 + index * 43.0
        width = abs(value) * scale
        x = zero_x if value >= 0 else zero_x - width
        color = "#2F766D" if value >= 0 else "#C7654D"
        parts.extend(
            [
                f'<text x="{bar_x0+55:.1f}" y="{y+19:.1f}" text-anchor="end" font-family="Consolas, monospace" font-size="12" fill="#26342F">{labels[str(row["term"])]}</text>',
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="25" rx="4" fill="{color}"/>',
                f'<text x="{(x+width+7 if value >= 0 else x-7):.2f}" y="{y+18:.1f}" text-anchor="{"start" if value >= 0 else "end"}" font-family="Consolas, monospace" font-size="11" font-weight="bold" fill="#26342F">{value:+.3f}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="66" y="470" width="548" height="145" rx="12" fill="#EEF2EC" stroke="#769187"/>',
            '<text x="88" y="500" font-family="Georgia, serif" font-size="15" font-weight="bold" fill="#1F2D28">The M211 suffix preference reverses when AdamW state is swapped</text>',
            '<text x="88" y="528" font-family="Consolas, monospace" font-size="12" fill="#26342F">O211: D211 -0.325  |  D223 +0.268</text>',
            '<text x="88" y="551" font-family="Consolas, monospace" font-size="12" fill="#26342F">O223: D211 +0.890  |  D223 -0.089</text>',
            '<text x="88" y="581" font-family="Georgia, serif" font-size="13" fill="#56645F">Moments do not store a context-free recovery direction; they route suffix response.</text>',
            '<rect x="708" y="535" width="417" height="80" rx="12" fill="#F3E7D8" stroke="#B68556"/>',
            '<text x="916" y="563" text-anchor="middle" font-family="Georgia, serif" font-size="15" font-weight="bold" fill="#1F2D28">factorial_unresolved</text>',
            f'<text x="916" y="588" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#56645F">|M x O x D| / |M| = {ratio:.3f} &lt; 1.500 gate</text>',
            '<text x="66" y="650" font-family="Georgia, serif" font-size="12" fill="#56645F">Saturated deterministic design: zero residual degrees of freedom; effects are descriptive, not uncertainty-calibrated.</text>',
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
