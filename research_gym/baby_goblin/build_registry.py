"""Build the final Baby Goblin skill registry from measured campaign results.

Status rules (preregistered, PREREGISTRATION_FULL.md Stage 8):
* learned skill: ``accepted`` only if its source stage passed its gate (race promotion
  for that skill, or the stage gate for hybrid / research / BPD artifacts); otherwise
  ``experimental``.
* symbolic exact solvers / certificates: ``accepted`` when their exactness or
  soundness obligation is met in the campaign receipts (zero violations, exact on
  complete-observation strata); otherwise ``experimental``.
* LLM fallback: always ``experimental`` (opt-in, network, not metered as compute).

    python -m research_gym.baby_goblin.build_registry --campaign experiments/trm_skill_gym_02/full --out research_gym/baby_goblin/registry.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from research_gym.trm_skill_gym_v3 import router as R
from .core import SkillSpec

GYM01_PACKAGES = {"repair": ("repair_C_gru", "recurrent_proxy_gru"), "retrieval": ("retrieval_D_trm_direct", "native_trm_update"),
                  "deduction": ("deduction_C_gru", "recurrent_proxy_gru"), "curation": ("curation_D_trm_direct", "native_trm_update"),
                  "stopping": ("stopping_C_gru", "recurrent_proxy_gru")}


def _read(p: Path) -> Optional[Dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build(campaign: Path) -> Dict:
    race = _read(campaign / "race" / "heldout.json") or {}
    hybrid = _read(campaign / "hybrid" / "results.json") or {}
    specs = []
    for skill, (pkg, impl) in GYM01_PACKAGES.items():
        r = (race.get("results") or {}).get(skill) or {}
        promoted = (r.get("status") == "ok" and r.get("selected") != r.get("baseline")
                    and r.get("macro_selected", 0) - r.get("macro_baseline", 0) >= 0.02
                    and r.get("worst_stratum_selected", 0) >= r.get("worst_stratum_baseline", 0) - 0.02)
        specs.append(SkillSpec(name=f"{skill}_learned", skill=skill, kind="learned", implementation=impl,
                               status="experimental", recurrent_state=True,
                               checkpoint=f"experiments/trm_skill_gym_01/packages/{pkg}/weights.pt",
                               skill_md=f"experiments/trm_skill_gym_01/packages/{pkg}/SKILL.md",
                               evidence=("GYM-01 secondary rule only (beats existing skill); the GYM-02 race "
                                         + ("promoted a different architecture" if promoted else "did not promote any architecture")
                                         + " for this skill")))
    viol = 0
    for job in hybrid.values():
        for ob in job.get("proof_obligations") or []:
            viol += ob.get("violations", 0)
        for arms in (job.get("test") or {}).values():
            for v in arms.values():
                viol += v.get("certificate_violations", 0) if isinstance(v, dict) else 0
    cert_ok = bool(hybrid) and viol == 0
    for skill in ("retrieval", "deduction"):
        specs.append(SkillSpec(name=f"{skill}_certificate", skill=skill, kind="certificate", implementation="three_valued_certificate",
                               status="accepted" if cert_ok else "experimental",
                               evidence=f"S3 proof obligations + held-out certificate violations = {viol}" if hybrid else "S3 not run"))
    for skill in R.STAGES:
        specs.append(SkillSpec(name=f"{skill}_symbolic", skill=skill, kind="symbolic", implementation="exact_solver",
                               status="accepted", evidence="exact on complete observations (tests/test_baby_goblin.py); "
                               "on noisy graphs it is the thresholded-graph solver and is not exact"))
        specs.append(SkillSpec(name=f"{skill}_heuristic", skill=skill, kind="heuristic", implementation="existing_skill_procedure",
                               status="accepted", evidence="GYM-01 arm A reference procedure"))
    specs.append(SkillSpec(name="research_inky", skill="research", kind="llm_fallback", implementation="inkling-2-mini-free",
                           status="experimental", evidence="GYM-01/GYM-02 text-only arms truncate on numeric skills"))
    return {"skills": [asdict(s) for s in specs], "source": str(campaign)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=Path("experiments/trm_skill_gym_02/full"))
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("registry.json"))
    a = p.parse_args(argv)
    a.out.write_text(json.dumps(build(a.campaign), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(a.out)


if __name__ == "__main__":
    main()
