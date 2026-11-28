"""Baby Goblin prototype: executable skills behind one typed contract.

A *skill* is an independently replaceable expert: a learned recurrent module, a
symbolic algorithm, a certificate procedure or an LLM fallback.  Every skill
declares its observation contract (the GYM-01 candidate-set tensors), its action
schema, whether it carries recurrent state, an uncertainty estimate, a
computational budget and a verifier contract.  The registry records an explicit
acceptance status: a skill that did not pass its preregistered gate stays
``experimental``.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from research_gym.trm_skill_gym import modules as M, tasks as T
from research_gym.trm_skill_gym_v3 import hybrid as H
from research_gym.trm_skill_gym_v3 import router as R


@dataclass
class SkillResult:
    decision: np.ndarray          # 0/1 over the 16 candidate slots
    uncertainty: float            # in [0, 1]; 0 = certified / fully determined
    flops: int = 0                # model FLOPs actually executed
    host_ops: int = 0             # symbolic edge examinations / tensor MAC units
    cpu_s: float = 0.0
    verifier_calls: int = 0
    state: Optional[dict] = None  # recurrent state for skills that carry one


@dataclass
class SkillSpec:
    name: str
    skill: str                    # task family it serves (GYM-01 skill name)
    kind: str                     # learned | symbolic | certificate | heuristic | llm_fallback
    implementation: str           # native_trm_update | recurrent_proxy_gru | feedforward | exact_solver | ...
    status: str                   # accepted | experimental
    evidence: str                 # which measured result justifies the status
    recurrent_state: bool = False
    budget_flops: int = 50_000_000
    action_schema: str = '{"selected": [slot indices]}'
    verifier: str = "research_gym.trm_skill_gym.tasks.verify"
    checkpoint: Optional[str] = None
    skill_md: Optional[str] = None


class Skill:
    def __init__(self, spec: SkillSpec, fn: Callable[[T.Instance, int], SkillResult]):
        self.spec, self._fn = spec, fn

    def run(self, inst: T.Instance, budget_flops: Optional[int] = None) -> SkillResult:
        budget = budget_flops if budget_flops is not None else self.spec.budget_flops
        t0 = time.process_time()
        res = self._fn(inst, budget)
        res.cpu_s = time.process_time() - t0
        if res.flops > budget:
            raise RuntimeError(f"{self.spec.name} exceeded its FLOP budget ({res.flops} > {budget})")
        return res

    def verify(self, inst: T.Instance, decision: np.ndarray) -> bool:
        b = T.stack([inst])
        ok, _ = T.verify(self.spec.skill, b, torch.from_numpy(decision)[None] * b["mask"])
        return bool(ok[0])


def _neural(skill: str, path: Path) -> Callable:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = M.SkillModule(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cfg = ck["config"]

    def fn(inst, budget):
        b = T.stack([inst])
        M.flops_reset()
        with torch.no_grad():
            cand = b["cand"].clone()
            cand[..., T.FS:] = 0
            st = model.init_state(b)
            for _ in range(cfg["visits"]):
                logits, conf, value, st = model.visit(cand, b, st, cfg["visits"])
        return SkillResult(decision=T.decide(skill, logits, b["mask"])[0].numpy(), uncertainty=float(1 - torch.sigmoid(conf)[0]),
                           flops=M.flops_read(), state={k: v for k, v in st.items() if k != "u"})
    return fn


def _symbolic(skill: str) -> Callable:
    def fn(inst, budget):
        r = R.run_expert(skill, "symbolic", inst, {})
        return SkillResult(decision=r["dec"], uncertainty=r["unc"], host_ops=r["host"])
    return fn


def _certificate(skill: str) -> Callable:
    def fn(inst, budget):
        q = int(np.argmax(inst.cand[:, 0])) if skill == "retrieval" else None
        st, depth, work = H.certify3(skill, inst.rel, inst.mask, q)
        dec = np.zeros(T.N, np.float32)
        dec[: len(st)] = st > 0
        unres = (st == 0) & (inst.mask[: len(st)] > 0)
        return SkillResult(decision=dec, uncertainty=float(unres.mean()) if len(st) else 0.0, host_ops=work,
                           state={"status": st.tolist(), "depth": depth.tolist()})
    return fn


def _heuristic(skill: str) -> Callable:
    def fn(inst, budget):
        r = R.run_expert(skill, "heuristic", inst, {})
        return SkillResult(decision=r["dec"], uncertainty=r["unc"], host_ops=r["host"])
    return fn


def _inky(skill: str) -> Callable:
    """Opt-in LLM fallback (free Inkling-2-Mini); never called unless the router enables it."""
    def fn(inst, budget):
        from research_gym.trm_skill_gym import skill_templates as S
        from research_gym.trm_skill_gym.inky import InkyClient, extract_json
        client = InkyClient(max_completion_tokens=6000, reasoning_effort="medium")
        r = client.chat("Return only the JSON object the skill asks for.", S.render(skill) + "\n" + S.serialize_instance(skill, inst),
                        temperature=0, seed=7)
        obj = extract_json(r["content"] or "") or {}
        dec = np.zeros(T.N, np.float32)
        for i in obj.get("selected") or []:
            if isinstance(i, int) and 0 <= i < T.N:
                dec[i] = 1
        return SkillResult(decision=dec * inst.mask, uncertainty=0.5 if r["finish_reason"] == "stop" else 1.0)
    return fn


FACTORIES = {"learned": _neural, "symbolic": _symbolic, "certificate": _certificate, "heuristic": _heuristic, "llm_fallback": _inky}


class Registry:
    def __init__(self, specs: List[SkillSpec]):
        self.specs = {s.name: s for s in specs}
        self._skills: Dict[str, Skill] = {}

    @classmethod
    def load(cls, path: Path) -> "Registry":
        rows = json.loads(Path(path).read_text(encoding="utf-8"))["skills"]
        return cls([SkillSpec(**r) for r in rows])

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            s = self.specs[name]
            fn = FACTORIES[s.kind](s.skill, Path(s.checkpoint)) if s.kind == "learned" else FACTORIES[s.kind](s.skill)
            self._skills[name] = Skill(s, fn)
        return self._skills[name]

    def for_skill(self, skill: str, include_experimental: bool = True) -> List[SkillSpec]:
        return [s for s in self.specs.values() if s.skill == skill and (include_experimental or s.status == "accepted")]

    def dump(self) -> Dict:
        return {"skills": [asdict(s) for s in self.specs.values()]}


class Trajectory:
    """Structured research-trajectory log (JSONL), one row per executed skill call."""

    def __init__(self, path: Optional[Path] = None, run_id: str = "baby-goblin"):
        self.path, self.run_id, self.rows = path, run_id, []

    def log(self, **row) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": self.run_id, "step": len(self.rows), **row}
        self.rows.append(row)
        if self.path:
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, sort_keys=True, default=float) + "\n")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.rows, sort_keys=True, default=float).encode()).hexdigest()


def route(registry: Registry, skill: str, inst: T.Instance, budget_flops: int, traj: Trajectory, tau: float = 0.2,
          allow_llm: bool = False) -> SkillResult:
    """Budget-aware termination: certificate/symbolic first; stop when the executed skill is
    certain enough or the budget is spent; escalate to learned skills, then (opt-in) the LLM."""
    order = sorted(registry.for_skill(skill), key=lambda s: {"certificate": 0, "symbolic": 1, "learned": 2, "heuristic": 3, "llm_fallback": 4}[s.kind])
    spent, last = 0, None
    for spec in order:
        if spec.kind == "llm_fallback" and not allow_llm:
            continue
        if spent >= budget_flops and last is not None:
            traj.log(skill=skill, event="budget_stop", spent_flops=spent)
            break
        res = registry.get(spec.name).run(inst, max(budget_flops - spent, 0) or spec.budget_flops)
        spent += res.flops
        traj.log(skill=skill, expert=spec.name, kind=spec.kind, status=spec.status, uncertainty=res.uncertainty,
                 flops=res.flops, host_ops=res.host_ops, cpu_s=res.cpu_s)
        last = res
        # certificates/solvers stop only when fully determined; learned skills use the confidence threshold
        if res.uncertainty <= (0.0 if spec.kind in ("certificate", "symbolic") else tau):
            traj.log(skill=skill, event="terminate_confident", expert=spec.name)
            break
    return last


# ------------------------------------------------------------------ integration hooks
def datagoblin_events(traj: Trajectory) -> List[Dict]:
    """DataGoblin ``training_gym_event`` rows (schemas/training_gym_event.schema.json)."""
    ev = [{"ts": traj.rows[0]["ts"] if traj.rows else time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "start", "run_id": traj.run_id, "step": 0}]
    for r in traj.rows:
        ev.append({"ts": r["ts"], "event": "checkpoint", "run_id": traj.run_id, "step": r["step"],
                   "reason": json.dumps({k: r[k] for k in r if k not in ("ts", "run_id", "step")}, sort_keys=True, default=float)})
    ev.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "complete", "run_id": traj.run_id, "step": len(traj.rows),
               "reason": f"trajectory_sha256={traj.digest()}"})
    return ev


def pocket_goblin_tools(registry: Registry) -> List[Dict]:
    """MCP tool descriptors in the PocketGoblin bench format (name, description, inputSchema)."""
    out = []
    for s in registry.specs.values():
        out.append({"name": f"skill_{s.name}", "description": f"{s.skill} via {s.kind}/{s.implementation} ({s.status}). "
                    f"Action schema {s.action_schema}; verifier {s.verifier}.",
                    "inputSchema": {"type": "object", "properties": {
                        "cand": {"type": "array", "description": f"[{T.N}, {T.FC}] candidate features"},
                        "rel": {"type": "array", "description": f"[{T.N}, {T.N}] relation"},
                        "glob": {"type": "array", "description": f"[{T.G}] global features"},
                        "mask": {"type": "array", "description": f"[{T.N}] valid slots"},
                        "budget_flops": {"type": "integer", "minimum": 0}},
                        "required": ["cand", "rel", "glob", "mask"], "additionalProperties": False}})
    return out


def pocket_goblin_call(registry: Registry, name: str, arguments: Dict) -> Dict:
    spec = registry.specs[name[len("skill_"):]]
    inst = T._empty()
    for k in ("cand", "rel", "glob", "mask"):
        setattr(inst, k, np.asarray(arguments[k], dtype=np.float32))
    res = registry.get(spec.name).run(inst, arguments.get("budget_flops"))
    return {"selected": np.nonzero(res.decision > 0.5)[0].tolist(), "uncertainty": res.uncertainty, "flops": res.flops,
            "host_ops": res.host_ops, "status": spec.status}
