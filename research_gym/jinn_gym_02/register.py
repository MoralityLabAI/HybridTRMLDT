"""Write experiments/jinn_gym_02/registration.json.

    python -m research_gym.jinn_gym_02.register
"""
from __future__ import annotations

import json
from pathlib import Path

from research_gym.jinn_gym_01 import host
from research_gym.jinn_gym_01.export import digest_of
from research_gym.jinn_gym_01.inky import MODEL

from . import verifiers as V
from .campaign import CFG, code_hashes02
from .evolve import ARMS, FID_STEP, MARGIN, UTIL_STEP

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    h = code_hashes02()
    lean = V.find_lean()
    reg = {"experiment": "JINN-GYM-02", "registered": "2026-11-28", "base": "research/jinn-gym-01@5ce0517", "model": MODEL,
           "config": CFG, "noninferiority_margin": MARGIN, "fidelity_step": FID_STEP, "utility_step": UTIL_STEP,
           "arms": ARMS, "code_hashes": h, "code_digest": digest_of(h), "host_code_hash": host.host_code_hash(),
           "lean_version": V.lean_version(lean) if lean else None, "paid_fallback": False, "max_hours": 8.0}
    out = ROOT / "experiments" / "jinn_gym_02" / "registration.json"
    out.write_text(json.dumps(reg, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(out, reg["code_digest"], reg["host_code_hash"])


if __name__ == "__main__":
    main()
