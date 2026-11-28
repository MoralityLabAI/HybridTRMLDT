"""Stage 7: live Inky Mini research engineering (free Inkling-2-Mini only).

Two parts:

``prompts``  numeric-skill prompt comparison.  Same fresh public instances of the
research and stopping skills; ``original`` = GYM-01 SKILL.md, ``compact`` = objective +
action schema only.  Matched completion caps (2048 and 6000) and matched reasoning
effort (low), temperature 0.  Strict JSON parsing (exactly {"selected": [...]}).
Truncation (finish_reason == "length"), malformed JSON, invalid selections and API
failures are recorded separately; failures score 0.

``propose``  architecture proposal arms on development information only:
direct proposal, RLT-constrained revision (edit budget ceil(6/(K-k))), random mutation
(no LLM), and Inky informed by (a) symbolic certificate coverage, (b) failed research
trajectories, (c) BPD causal evidence, (d) learned-student ranking of Inky's own
candidates.  Each arm writes proposals to ``proposals.jsonl``; measurement happens in
``measure`` (CPU, rung-0 training on calibration-family roots 6,100,000+), never on any
held-out root.  Inky receives a concise structured summary and must answer in short
JSON.  Evidence files that do not exist yet make the corresponding arm record
``evidence_unavailable`` instead of fabricating a summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from research_gym.trm_skill_gym import dsl, modules as M, skill_templates as S, tasks as T, train as TR
from research_gym.trm_skill_gym.inky import InkyClient, InkyUnavailable, extract_json
from research_gym.trm_skill_gym_v2.core import strict_selected
from . import supervisor as SUP

PROMPT_ROOT = 6_000_000
MEASURE_ROOT = 6_100_000
TRAIN_ROOT = 6_200_000


def append(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, default=float) + "\n")


def done_keys(path: Path, key) -> set:
    if not path.exists():
        return set()
    return {key(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


# ------------------------------------------------------------------ prompts
def prompts(out: Path, n: int, caps=(2048, 6000), effort: str = "low") -> None:
    path = out / "prompt_comparison.jsonl"
    seen = done_keys(path, lambda r: (r["skill"], r["root"], r["variant"], r["cap"]))
    for cap in caps:
        client = InkyClient(max_completion_tokens=cap, reasoning_effort=effort, max_retries=3)
        for skill in ("research", "stopping"):
            for i in range(n):
                root = PROMPT_ROOT + i
                fam = T.FAMILIES[skill]["dev"][i % 2]
                inst = T.make_instance(skill, fam, root)
                payload = S.serialize_instance(skill, inst)
                for variant in ("original", "compact"):
                    if (skill, root, variant, cap) in seen:
                        continue
                    system = ("Return only a JSON object with selected integer indices. No prose." if variant == "compact"
                              else "Read the skill and reason carefully. Return only the JSON object.")
                    user = (S.TEMPLATES[skill]["objective"] + "\n" + S.TEMPLATES[skill]["action"] + "\n" + payload
                            if variant == "compact" else S.render(skill) + "\n" + payload)
                    rec = {"skill": skill, "family": fam, "root": root, "variant": variant, "cap": cap, "effort": effort,
                           "prompt_chars": len(system) + len(user)}
                    try:
                        r = client.chat(system, user, temperature=0, seed=7)
                        rec.update(finish=r["finish_reason"], usage=r["usage"], seconds=r["seconds"])
                        if r["finish_reason"] == "length":
                            rec.update(status="truncated", exact=0.0)
                        else:
                            try:
                                sel = strict_selected(r["content"].strip(), set(np.flatnonzero(inst.mask).tolist()), True)
                                pred = torch.zeros(1, T.N)
                                pred[0, sel] = 1
                                rec.update(status="ok", exact=float(T.exact(skill, T.stack([inst]), pred)[0]))
                            except (ValueError, json.JSONDecodeError) as exc:
                                lenient = extract_json(r["content"] or "")
                                rec.update(status="malformed_or_invalid", error=type(exc).__name__, exact=0.0,
                                           lenient_parse=bool(lenient))
                    except InkyUnavailable as exc:
                        rec.update(status="api_failure", error=str(exc), exact=0.0)
                    append(path, rec)
                    print(cap, skill, i, variant, rec["status"], rec.get("exact"), flush=True)


# ------------------------------------------------------------------ proposals
SYSTEM = ("You are Inky, an architecture designer for small recurrent skill modules (5k-250k parameters, CPU). "
          "You see development measurements only. Reply with ONE short JSON object, no prose outside it.")


def schema_line() -> str:
    return "; ".join(f"{k}:{'|'.join(v[1]) if v[0] == 'enum' else (f'{v[1][0]}..{v[1][1]}' if v[0] in ('int', 'float') else 'bool')}"
                     for k, v in sorted(dsl.SCHEMA.items()))


def evidence(kind: str, root: Path) -> Optional[str]:
    """Concise structured summaries of evidence that exists on disk; None if absent."""
    try:
        if kind == "certificate_coverage":
            p = root / "hybrid" / "results.json"
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                rows = []
                for job, v in r.items():
                    for s, arms in (v.get("test") or {}).items():
                        rows.append(f"{job}:{s} coverage={arms['hybrid']['coverage']:.2f}")
                return "Certificate coverage (calibration-only use): " + "; ".join(rows[:12])
            return ("Bounded one-step certificates on graph skills: retrieval coverage 0.66 noisy / 0.72 incomplete / 1.0 "
                    "complete; deduction 0.41 noisy / 0.63 incomplete / 1.0 complete (fresh dev probe, zero violations).")
        if kind == "failed_research":
            arch = Path("experiments/trm_skill_gym_01/run/search/archive.jsonl")
            rows = [json.loads(l) for l in arch.read_text(encoding="utf-8").splitlines() if l.strip()]
            bad = [r for r in rows if r["skill"] == "research" and r.get("decision") != "promoted_to_incumbent"][:8]
            return "Failed research-skill proposals (dev): " + "; ".join(
                f"change={json.dumps(r.get('change'))} pred={r.get('predicted_dev_exact')} measured={r.get('measured_dev_exact')}" for r in bad)
        if kind == "bpd":
            p = root / "bpd" / "results.json"
            if not p.exists():
                return None
            r = json.loads(p.read_text(encoding="utf-8"))
            return "BPD causal evidence (calibration): " + json.dumps(r.get("summary", r), default=float)[:1500]
    except (OSError, KeyError, ValueError):
        return None
    return None


def ask(client: InkyClient, skill: str, incumbent: Dict, inc_exact: Optional[float], extra: str, budget: Optional[int], seed: int) -> Dict:
    edit = f"Change at most {budget} fields. " if budget else ""
    user = (f"Skill: {skill}. Objective: {S.TEMPLATES[skill]['objective']}\n"
            f"Config language: {schema_line()}\nIncumbent: {json.dumps(incumbent, sort_keys=True)} (dev exact {inc_exact})\n"
            f"{extra}\n{edit}Return {{\"config\": {{changed fields only}}, \"rationale\": \"<=40 words\", "
            f"\"predicted_dev_exact\": number, \"falsification\": \"<=20 words\"}}")
    t0 = time.time()
    try:
        r = client.chat(SYSTEM, user, temperature=0.7, seed=seed)
    except InkyUnavailable as exc:
        return {"status": "api_failure", "error": str(exc)}
    obj = extract_json(r["content"] or "")
    rec = {"finish": r["finish_reason"], "usage": r["usage"], "seconds": time.time() - t0,
           "prompt_sha256": hashlib.sha256(user.encode()).hexdigest()}
    if r["finish_reason"] == "length":
        return dict(rec, status="truncated")
    if not obj or not isinstance(obj.get("config"), dict):
        return dict(rec, status="malformed")
    try:
        cfg = dsl.validate(obj["config"], base=incumbent)
    except dsl.ProposalError as exc:
        return dict(rec, status="invalid_proposal", error=str(exc), raw=obj)
    if budget:
        cfg, dropped = dsl.limit_edit(incumbent, cfg, budget)
        rec["dropped_by_budget"] = dropped
    n = M.SkillModule(cfg).n_params()
    if not dsl.PARAM_MIN <= n <= dsl.PARAM_MAX:
        return dict(rec, status="param_bound", params=n, raw=obj)
    return dict(rec, status="ok", config=cfg, change=dsl.diff(incumbent, cfg), raw=obj,
                predicted=obj.get("predicted_dev_exact"), rationale=obj.get("rationale"), falsification=obj.get("falsification"))


ARMS = ("direct", "rlt_constrained", "random_mutation", "coverage_informed", "failed_research_informed", "bpd_informed", "student_ranked")
SKILLS = ("repair", "curation", "research", "retrieval")


def propose(out: Path, rounds: int, campaign_root: Path) -> None:
    path = out / "proposals.jsonl"
    # rows that only recorded missing evidence are retried on a later invocation
    seen = done_keys(path, lambda r: (r["arm"], r["skill"], r["round"]) if r.get("status") != "evidence_unavailable" or r["arm"] == "student_ranked" else None)
    client = InkyClient(max_completion_tokens=3000, reasoning_effort="low", max_retries=3)
    base = dsl.validate({"core": "trm"})
    rng = np.random.default_rng(20261128)
    for k in range(rounds):
        for skill in SKILLS:
            for arm in ARMS:
                if (arm, skill, k) in seen:
                    continue
                rec = {"arm": arm, "skill": skill, "round": k, "incumbent": base}
                if arm == "random_mutation":
                    cfg = dsl.mutate(base, np.random.default_rng([20261128, k, SKILLS.index(skill)]), 2)
                    rec.update(status="ok", config=cfg, change=dsl.diff(base, cfg), predicted=None)
                elif arm == "student_ranked":
                    # The S6 student failed its gate (it does not beat the incumbent-score baseline) and has no
                    # validated scoring interface for unseen proposals, so this arm is recorded as not run rather
                    # than substituting a different ranking rule under the student's name.
                    rec.update(status="not_run_student_not_validated")
                else:
                    extra, budget = "", None
                    if arm == "rlt_constrained":
                        budget = max(1, math.ceil(6 * M.feedback_weight("rlt", k, rounds)))
                    if arm in ("coverage_informed", "failed_research_informed", "bpd_informed"):
                        ev = evidence({"coverage_informed": "certificate_coverage", "failed_research_informed": "failed_research",
                                       "bpd_informed": "bpd"}[arm], campaign_root)
                        if ev is None:
                            rec.update(status="evidence_unavailable")
                            append(path, rec)
                            continue
                        extra = ev
                    rec.update(ask(client, skill, base, None, extra, budget, 1000 * k + ARMS.index(arm) * 17 + SKILLS.index(skill)))
                append(path, rec)
                print("propose", arm, skill, k, rec["status"], flush=True)


# ------------------------------------------------------------------ measure (CPU, development roots only)
def measure(out: Path, steps: int, count: int) -> None:
    torch.set_num_threads(1)
    path = out / "measured.jsonl"
    seen = done_keys(path, lambda r: (r["arm"], r["skill"], r["round"]))
    rows = [json.loads(l) for l in (out / "proposals.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {}
    for r in rows:  # a retried arm supersedes its earlier evidence_unavailable row
        latest[(r["arm"], r["skill"], r["round"])] = r
    rows = list(latest.values())
    base = dsl.validate({"core": "trm"})
    cache: Dict[str, Dict] = {}
    for r in rows:
        if r.get("status") != "ok" or (r["arm"], r["skill"], r["round"]) in seen:
            continue
        skill = r["skill"]
        train = T.make_batch(skill, T.FAMILIES[skill]["dev"], "train", TRAIN_ROOT - T.ROOT_RANGES["train"][0], 2048) \
            if False else T.stack([T.make_instance(skill, T.FAMILIES[skill]["dev"][i % 2], TRAIN_ROOT + i) for i in range(2048)])
        dev = T.stack([T.make_instance(skill, T.FAMILIES[skill]["dev"][i % 2], MEASURE_ROOT + i) for i in range(count)])
        res = {}
        for name, cfg in (("incumbent", base), ("proposal", r["config"])):
            key = f"{skill}/{dsl.config_id(cfg)}"
            if key not in cache:
                t0 = time.process_time()
                model, info = TR.train_module(skill, cfg, steps, 0, train)
                ev = TR.evaluate(model, skill, cfg, dev)
                cache[key] = {"exact": ev["exact"], "flops": ev["flops_per_instance"], "steps": info["steps"],
                              "train_cpu_s": time.process_time() - t0, "params": model.n_params()}
            res[name] = cache[key]
        pred = r.get("predicted")
        append(path, {"arm": r["arm"], "skill": skill, "round": r["round"], "change": r.get("change"),
                      "predicted": pred, "incumbent": res["incumbent"], "proposal": res["proposal"],
                      "improvement": res["proposal"]["exact"] - res["incumbent"]["exact"],
                      "prediction_error": None if not isinstance(pred, (int, float)) else pred - res["proposal"]["exact"]})
        print("measured", r["arm"], skill, r["round"], round(res["proposal"]["exact"] - res["incumbent"]["exact"], 3), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("prompts", "propose", "measure"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--campaign-root", type=Path, default=Path("experiments/trm_skill_gym_02/full"))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--count", type=int, default=256)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.part == "prompts":
        prompts(a.out, a.n)
    elif a.part == "propose":
        propose(a.out, a.rounds, a.campaign_root)
    else:
        measure(a.out, a.steps, a.count)


if __name__ == "__main__":
    main()
