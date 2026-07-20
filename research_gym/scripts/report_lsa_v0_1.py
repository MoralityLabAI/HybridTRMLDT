"""Generate deterministic SVG figures for the LSA v0.1 empirical note."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "experiments" / "loop_schedule_algebra_v0_1"
DEFAULT_SATURATION_EXPERIMENT = (
    ROOT / "experiments" / "loop_schedule_algebra_saturation_addendum_v1"
)
DEFAULT_OUTPUT = ROOT / "reports" / "figures"
BACKGROUND = "#f3efe6"
INK = "#18332c"
MUTED = "#65736d"
GRID = "#d2cabd"
TIED = "#b84b35"
UNTIED = "#19736c"
ACCENT = "#d69a2d"
STABLE = "#79a56f"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(x: float, y: float, value: str, *, size: int = 13, color: str = INK, anchor: str = "start", family: str = "Georgia, serif", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}">{escape(value)}</text>'
    )


def _svg(title: str, description: str, body: Iterable[str], *, width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def gamma_trajectory_figure(primary: dict[str, Any]) -> str:
    summary = primary["summary"]
    fits = summary["trajectory_fits"]
    intervals = summary["trajectory_bootstrap_intervals"]
    width, height = 820, 470
    left, right, top, bottom = 92, 770, 95, 385
    x_values = [left + index * (right - left) / (len(fits) - 1) for index in range(len(fits))]

    def y(value: float) -> float:
        return bottom - value / 0.4 * (bottom - top)

    body = [
        _text(54, 42, "Visit alignment is learned", size=26, weight="bold"),
        _text(54, 68, "Four-point tied exponent across common state-visit exposures", size=14, color=MUTED),
    ]
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4):
        yy = y(tick)
        body.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="{GRID}"/>')
        body.append(_text(left - 12, yy + 4, f"{tick:.1f}", size=12, color=MUTED, anchor="end", family="Consolas, monospace"))
    body.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>',
        ]
    )
    upper = " ".join(f"{x:.1f},{y(float(row['upper'])):.1f}" for x, row in zip(x_values, intervals))
    lower = " ".join(f"{x:.1f},{y(float(row['lower'])):.1f}" for x, row in reversed(list(zip(x_values, intervals))))
    body.append(f'<polygon points="{upper} {lower}" fill="{ACCENT}" opacity="0.20"/>')
    points = " ".join(f"{x:.1f},{y(float(row['gamma'])):.1f}" for x, row in zip(x_values, fits))
    body.append(f'<polyline points="{points}" fill="none" stroke="{TIED}" stroke-width="4"/>')
    for x, fit, interval in zip(x_values, fits, intervals):
        body.append(f'<line x1="{x:.1f}" y1="{y(float(interval["lower"])):.1f}" x2="{x:.1f}" y2="{y(float(interval["upper"])):.1f}" stroke="{TIED}" stroke-width="2"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y(float(fit["gamma"])):.1f}" r="6" fill="{TIED}" stroke="{BACKGROUND}" stroke-width="3"/>')
        body.append(_text(x, bottom + 24, f"{int(fit['exposure']):,}", size=11, color=MUTED, anchor="middle", family="Consolas, monospace"))
    body.extend(
        [
            _text((left + right) / 2, 446, "state-visit exposures", size=13, color=MUTED, anchor="middle"),
            _text(22, (top + bottom) / 2, "gamma", size=14, color=INK),
            _text(560, 112, "preregistered label: learned_growth", size=13, color=TIED, family="Consolas, monospace"),
            _text(560, 134, "0.130 -> 0.309", size=13, color=TIED, family="Consolas, monospace"),
            _text(560, 156, "gamma=1 is off-scale", size=12, color=MUTED, family="Consolas, monospace"),
        ]
    )
    return _svg(
        "Visit alignment grows during training",
        "The tied four-point gamma estimate rises from 0.130 at initialization to 0.309 after 4096 state-visit exposures, with seed-bootstrap intervals.",
        body,
        width=width,
        height=height,
    )


def scaling_figure(primary: dict[str, Any], external: dict[str, Any]) -> str:
    width, height = 940, 500
    body = [
        _text(52, 42, "Weight tying separates visit-alignment scaling", size=25, weight="bold"),
        _text(52, 68, "Geometric mean across three seeds at 4,096 state-visit exposures", size=14, color=MUTED),
    ]
    panels = [
        ("Two-layer residual loop", primary["summary"]["terminal_four_point_tied_fit"], primary["summary"]["terminal_four_point_untied_fit"]),
        ("Mini attention+MLP loop", external["summary"]["tied_fit"], external["summary"]["untied_fit"]),
    ]
    for panel_index, (label, tied, untied) in enumerate(panels):
        x0 = 80 + panel_index * 455
        x1 = x0 + 380
        top, bottom = 112, 405

        def y(value: float) -> float:
            return bottom - value / 3.5 * (bottom - top)

        xs = [x0 + 55 + index * 92 for index in range(4)]
        for tick in (0, 1, 2, 3):
            yy = y(tick)
            body.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{GRID}"/>')
            body.append(_text(x0 - 10, yy + 4, str(tick), size=11, color=MUTED, anchor="end", family="Consolas, monospace"))
        body.extend(
            [
                f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>',
                f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>',
                _text((x0 + x1) / 2, 95, label, size=17, anchor="middle", weight="bold"),
            ]
        )
        for rounds, x in zip((2, 4, 8, 16), xs):
            body.append(_text(x, bottom + 22, str(rounds), size=11, color=MUTED, anchor="middle", family="Consolas, monospace"))
        for fit, color in ((tied, TIED), (untied, UNTIED)):
            values = [float(fit["geometric_mean_kappa"][str(rounds)]) for rounds in (2, 4, 8, 16)]
            points = " ".join(f"{x:.1f},{y(value):.1f}" for x, value in zip(xs, values))
            body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
            for x, value in zip(xs, values):
                body.append(f'<circle cx="{x:.1f}" cy="{y(value):.1f}" r="6" fill="{color}" stroke="{BACKGROUND}" stroke-width="3"/>')
        body.append(_text(x0 + 20, 440, f"tied gamma={float(tied['gamma']):.3f}", size=12, color=TIED, family="Consolas, monospace"))
        body.append(_text(x0 + 205, 440, f"untied gamma={float(untied['gamma']):.3f}", size=12, color=UNTIED, family="Consolas, monospace"))
        if panel_index == 0:
            prediction = float(primary["summary"]["r16_holdout"]["predicted_kappa"])
            body.append(f'<circle cx="{xs[-1]:.1f}" cy="{y(prediction):.1f}" r="9" fill="none" stroke="{ACCENT}" stroke-width="3" stroke-dasharray="5 4"/>')
            body.append(_text(xs[-1] - 8, y(prediction) - 16, "v0 extrapolation", size=11, color=ACCENT, anchor="end"))
    body.extend(
        [
            _text(26, 265, "kappa", size=14),
            _text(470, 484, "loop visits R", size=13, color=MUTED, anchor="middle"),
            _text(700, 466, "tied", size=12, color=TIED, family="Consolas, monospace"),
            _text(760, 466, "untied", size=12, color=UNTIED, family="Consolas, monospace"),
        ]
    )
    return _svg(
        "Visit-alignment scaling in MLP and mini Transformer loops",
        "Tied kappa grows with loop visits in both model families while untied controls remain nearly flat. The MLP R16 point falls below the v0 extrapolation.",
        body,
        width=width,
        height=height,
    )


def ladder_figure(ladder: dict[str, Any]) -> str:
    outcomes = ladder["summary"]["outcomes"]
    width, height = 840, 540
    left, top = 300, 115
    cell_w, cell_h = 105, 42
    body = [
        _text(52, 42, "The stability boundary remains censored", size=25, weight="bold"),
        _text(52, 68, "All 96 seed-level cells are stable; every aggregate boundary is p<=0", size=14, color=MUTED),
    ]
    p_grid = (0.0, 0.05, 0.1, 0.15)
    for column, p in enumerate(p_grid):
        body.append(_text(left + column * cell_w + cell_w / 2, 100, f"p={p:g}", size=12, color=MUTED, anchor="middle", family="Consolas, monospace"))
    for row_index, outcome in enumerate(outcomes):
        yy = top + row_index * cell_h
        label = f"LR={float(outcome['learning_rate']):g}  R={int(outcome['rounds'])}"
        body.append(_text(left - 18, yy + 27, label, size=12, color=INK, anchor="end", family="Consolas, monospace"))
        for column, p in enumerate(p_grid):
            stable = bool(outcome["stable_by_p"][f"{p:g}"])
            fill = STABLE if stable else TIED
            xx = left + column * cell_w
            body.append(f'<rect x="{xx}" y="{yy}" width="{cell_w - 6}" height="{cell_h - 6}" rx="4" fill="{fill}" opacity="0.88"/>')
            body.append(_text(xx + (cell_w - 6) / 2, yy + 24, "stable" if stable else "unstable", size=11, color=BACKGROUND, anchor="middle", family="Consolas, monospace", weight="bold"))
    body.extend(
        [
            _text(52, 485, "No depth ordering is identifiable at any learning rate.", size=15, color=INK, weight="bold"),
            _text(52, 510, "Machine endpoint p=0 is not promoted to an observed boundary.", size=13, color=MUTED),
        ]
    )
    return _svg(
        "Learning-rate ladder remains left-censored",
        "All registered p values are stable for both loop counts and all four learning rates, leaving every boundary left-censored at p less than or equal to zero.",
        body,
        width=width,
        height=height,
    )


def high_loop_figure(
    primary: dict[str, Any], fit_r32: dict[str, Any], holdout_r64: dict[str, Any]
) -> str:
    """Show the nonmonotonic kappa trajectory beside its gradient-scale context."""

    rounds = (2, 4, 8, 16, 32, 64)
    tied = holdout_r64["summary"]["tied_geometric_mean_kappa"]
    untied = holdout_r64["summary"]["untied_geometric_mean_kappa"]
    predictions = holdout_r64["summary"]["sealed_predictions"]
    training = [
        *primary["summary"]["training_cells"],
        *fit_r32["summary"]["training_cells"],
        *holdout_r64["summary"]["training_cells"],
    ]
    gradients: dict[tuple[str, int], float] = {}
    for regime in ("tied", "untied"):
        for round_count in rounds:
            marker = f"_{regime}_R{round_count}_"
            values = [
                float(row["max_gradient_norm"])
                for row in training
                if marker in row["cell_id"]
            ]
            gradients[(regime, round_count)] = max(values)

    width, height = 1000, 540
    body = [
        _text(48, 40, "Alignment collapses in the high-loop regime", size=25, weight="bold"),
        _text(48, 67, "Untouched R=64 rejects both smooth extrapolators as tied gradient scale rises", size=14, color=MUTED),
    ]
    left_x0, left_x1 = 78, 470
    right_x0, right_x1 = 570, 962
    top, bottom = 115, 420
    xs_left = [left_x0 + index * (left_x1 - left_x0) / 5 for index in range(6)]
    xs_right = [right_x0 + index * (right_x1 - right_x0) / 5 for index in range(6)]

    def kappa_y(value: float) -> float:
        return bottom - value / 3.5 * (bottom - top)

    def gradient_y(value: float) -> float:
        minimum, maximum = math.log10(0.1), math.log10(300.0)
        return bottom - (math.log10(value) - minimum) / (maximum - minimum) * (bottom - top)

    for tick in (0.0, 1.0, 2.0, 3.0):
        yy = kappa_y(tick)
        body.append(f'<line x1="{left_x0}" y1="{yy:.1f}" x2="{left_x1}" y2="{yy:.1f}" stroke="{GRID}"/>')
        body.append(_text(left_x0 - 10, yy + 4, f"{tick:g}", size=11, color=MUTED, anchor="end", family="Consolas, monospace"))
    for tick in (0.1, 1.0, 10.0, 100.0):
        yy = gradient_y(tick)
        body.append(f'<line x1="{right_x0}" y1="{yy:.1f}" x2="{right_x1}" y2="{yy:.1f}" stroke="{GRID}"/>')
        body.append(_text(right_x0 - 10, yy + 4, f"{tick:g}", size=11, color=MUTED, anchor="end", family="Consolas, monospace"))
    for x0, x1 in ((left_x0, left_x1), (right_x0, right_x1)):
        body.extend(
            [
                f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>',
                f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>',
            ]
        )
    body.extend(
        [
            _text((left_x0 + left_x1) / 2, 98, "terminal kappa", size=17, anchor="middle", weight="bold"),
            _text((right_x0 + right_x1) / 2, 98, "maximum gradient norm (log scale)", size=17, anchor="middle", weight="bold"),
        ]
    )
    for round_count, left_x, right_x in zip(rounds, xs_left, xs_right):
        body.append(_text(left_x, bottom + 22, str(round_count), size=11, color=MUTED, anchor="middle", family="Consolas, monospace"))
        body.append(_text(right_x, bottom + 22, str(round_count), size=11, color=MUTED, anchor="middle", family="Consolas, monospace"))
    for regime, color in (("tied", TIED), ("untied", UNTIED)):
        values = tied if regime == "tied" else untied
        points = " ".join(
            f"{x:.1f},{kappa_y(float(values[str(round_count)])):.1f}"
            for x, round_count in zip(xs_left, rounds)
        )
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        gradient_points = " ".join(
            f"{x:.1f},{gradient_y(gradients[(regime, round_count)]):.1f}"
            for x, round_count in zip(xs_right, rounds)
        )
        body.append(f'<polyline points="{gradient_points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, round_count in zip(xs_left, rounds):
            body.append(f'<circle cx="{x:.1f}" cy="{kappa_y(float(values[str(round_count)])):.1f}" r="5.5" fill="{color}" stroke="{BACKGROUND}" stroke-width="2"/>')
        for x, round_count in zip(xs_right, rounds):
            body.append(f'<circle cx="{x:.1f}" cy="{gradient_y(gradients[(regime, round_count)]):.1f}" r="5.5" fill="{color}" stroke="{BACKGROUND}" stroke-width="2"/>')
    for name, color in (("saturating_exponential", ACCENT), ("logarithmic", MUTED)):
        predicted = float(predictions[name])
        body.append(f'<circle cx="{xs_left[-1]:.1f}" cy="{kappa_y(predicted):.1f}" r="8" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="4 3"/>')
    body.extend(
        [
            _text(57, 478, "tied", size=12, color=TIED, family="Consolas, monospace", weight="bold"),
            _text(117, 478, "untied", size=12, color=UNTIED, family="Consolas, monospace", weight="bold"),
            _text(205, 478, "open markers: sealed R64 predictions", size=12, color=MUTED, family="Consolas, monospace"),
            _text(500, 515, "loop visits R", size=13, color=MUTED, anchor="middle"),
        ]
    )
    return _svg(
        "High-loop visit alignment and gradient scale",
        "Tied kappa peaks at R16 and then falls through R64 while tied maximum gradient norm rises above 100. Untied kappa and gradients remain near their controls.",
        body,
        width=width,
        height=height,
    )
def generate(
    experiment: Path,
    output: Path,
    saturation_experiment: Path = DEFAULT_SATURATION_EXPERIMENT,
) -> tuple[Path, ...]:
    primary = _read(experiment / "primary_result.json")
    external = _read(experiment / "external_result.json")
    ladder = _read(experiment / "ladder_result.json")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "lsa_v0_1_gamma_trajectory.svg": gamma_trajectory_figure(primary),
        "lsa_v0_1_kappa_scaling.svg": scaling_figure(primary, external),
        "lsa_v0_1_boundary_ladder.svg": ladder_figure(ladder),
    }
    if (saturation_experiment / "holdout_r64_result.json").exists():
        artifacts["lsa_v0_1_high_loop.svg"] = high_loop_figure(
            primary,
            _read(saturation_experiment / "fit_r32_result.json"),
            _read(saturation_experiment / "holdout_r64_result.json"),
        )
    paths = []
    for name, payload in artifacts.items():
        path = output / name
        path.write_text(payload, encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--saturation-experiment", type=Path, default=DEFAULT_SATURATION_EXPERIMENT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in generate(
        args.experiment.resolve(),
        args.output.resolve(),
        args.saturation_experiment.resolve(),
    ):
        print(path)


if __name__ == "__main__":
    main()
