"""Verbatim upstream code, pinned by SHA-256 in SOURCES.json and checked on import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "SOURCES.json").read_text(encoding="utf-8"))


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify() -> None:
    actual = sha256_lf(HERE / "ldt_experiment.py")
    if actual != SOURCES["ldt_experiment.py"]["sha256_lf"]:
        raise ImportError("vendored ldt_experiment.py differs from its pinned upstream hash")


verify()
from . import ldt_experiment  # noqa: E402,F401
