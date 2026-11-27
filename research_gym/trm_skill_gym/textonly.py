"""Text-only arm: Inky Mini executes each skill from its SKILL.md alone (no
trainable module, no verifier calls).  Uses the same frozen-evaluation instances
as the first rows of every arm's frozen_dev pool, so results pair by instance.

    python -m research_gym.trm_skill_gym.textonly --out experiments/trm_skill_gym_01/run/text_only --per-skill 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import skill_templates as S
from . import tasks as T
from .inky import InkyClient, InkyUnavailable, extract_json

SYSTEM = ("You execute an agent skill. Read the SKILL.md and the instance, reason carefully, and answer with ONLY "
          "the JSON object the skill's action schema asks for. Indices refer to candidate numbers.")


def run(out: Path, per_skill: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    client = InkyClient(max_completion_tokens=6000, reasoning_effort="medium")
    for skill in T.SKILLS:
        path = out / f"{skill}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["index"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        lo = T.ROOT_RANGES["frozen_eval"][0]
        fams = T.FAMILIES[skill]["dev"]
        for i in range(per_skill):
            if i in done:
                continue
            fam = fams[i % len(fams)]
            inst = T.make_instance(skill, fam, lo + i)
            user = S.render(skill) + "\n\n## Instance\n" + S.serialize_instance(skill, inst)
            rec = {"skill": skill, "family": fam, "root": lo + i, "index": i,
                   "prompt_sha256": hashlib.sha256((SYSTEM + user).encode()).hexdigest()}
            try:
                resp = client.chat(SYSTEM, user, temperature=0.0, seed=7)
            except (InkyUnavailable, ValueError) as exc:
                rec.update(status="unavailable", error=str(exc))
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec) + "\n")
                continue
            obj = extract_json(resp["content"]) or {}
            sel = obj.get("selected")
            batch = T.stack([inst])
            pred = torch.zeros(1, T.N)
            valid = isinstance(sel, list) and all(isinstance(x, int) and 0 <= x < T.N for x in sel)
            if valid:
                for x in sel:
                    pred[0, x] = 1
            ex = float(T.exact(skill, batch, pred * batch["mask"])[0]) if valid else 0.0
            rec.update(status="ok", parsed=valid, selected=sel if valid else None, exact=ex,
                       usage=resp["usage"], seconds=resp["seconds"], finish_reason=resp["finish_reason"],
                       reasoning_chars=resp["reasoning_chars"])
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec) + "\n")
            print(skill, i, ex, flush=True)
    (out / "ledger.json").write_text(json.dumps(client.ledger, indent=1), encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-skill", type=int, default=16)
    a = ap.parse_args()
    run(a.out, a.per_skill)


if __name__ == "__main__":
    main()
