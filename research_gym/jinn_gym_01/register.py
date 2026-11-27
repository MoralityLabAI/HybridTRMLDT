"""Write experiments/jinn_gym_01/registration.json (config, code hashes, vendored sources).

    python -m research_gym.jinn_gym_01.register
"""
from __future__ import annotations

import json
from pathlib import Path

from . import host
from ._vendor import SOURCES
from .export import code_hashes, digest_of
from .inky import MODEL
from .runner import CFG

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    hashes = code_hashes()
    reg = {"experiment": "JINN-GYM-01", "registered": "2026-11-28", "model": MODEL, "config": CFG,
           "code_hashes": hashes, "code_digest": digest_of(hashes), "host_code_hash": host.host_code_hash(),
           "vendored_sources": SOURCES, "paid_fallback": False, "max_hours": 8.0}
    out = ROOT / "experiments" / "jinn_gym_01" / "registration.json"
    out.write_text(json.dumps(reg, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(out, reg["code_digest"])


if __name__ == "__main__":
    main()
