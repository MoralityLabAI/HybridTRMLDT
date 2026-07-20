"""Build a byte-stable Overleaf bundle for the LSA v0.1 empirical note."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "papers" / "loop_schedule_alignment_v0_1"
DEFAULT_OUTPUT = ROOT / "packages" / "loop_schedule_alignment_v0_1_overleaf.zip"
FIGURES = (
    "lsa_v0_1_boundary_ladder",
    "lsa_v0_1_gamma_trajectory",
    "lsa_v0_1_high_loop",
    "lsa_v0_1_kappa_scaling",
)


def package(source: Path, output: Path) -> str:
    members = [source / "README.md", source / "main.tex", source / "references.bib"]
    for stem in FIGURES:
        members.extend((source / "figures" / f"{stem}.pdf", source / "figures" / f"{stem}.svg"))
    missing = [path for path in members if not path.exists()]
    if missing:
        raise RuntimeError(f"missing paper bundle inputs: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(members, key=lambda value: value.relative_to(source).as_posix()):
            name = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(package(args.source.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
