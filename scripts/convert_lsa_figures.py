"""Convert deterministic LSA SVG figures to TeX-friendly vector PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab import rl_config
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg


def convert(source: Path, destination: Path) -> Path:
    rl_config.invariant = 1
    drawing = svg2rlg(str(source))
    if drawing is None:
        raise RuntimeError(f"could not parse SVG: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(destination))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.source.resolve().glob("lsa_v0_1_*.svg"))
    if not sources:
        raise RuntimeError("no LSA SVG figures found")
    for source in sources:
        print(convert(source, args.output.resolve() / f"{source.stem}.pdf"))


if __name__ == "__main__":
    main()
