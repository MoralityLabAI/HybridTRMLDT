"""TRM-SKILL-GYM-01 campaign: bounded Inky architecture search, frozen arm
comparison, ablations, transfer, and exports.  Resumable: every job writes its
result file once and is skipped on restart; the search state is saved after
every round; Inky responses are cached by request digest.

    python -m research_gym.trm_skill_gym.campaign --out experiments/trm_skill_gym_01/run --hours 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import dsl
from . import modules as M
from . import skill_templates as S
from . import tasks as T
from . import train as TR
from .inky import InkyClient, InkyUnavailable, extract_json

PROFILE = {
    "main_steps": 800,
    "rung0_steps": 300,
    "rung1_steps": 800,
    "transfer_steps": 300,
    "main_seeds": [0, 1],
    "search_rounds_per_arm": 12,
    "proposals_per_round": 3,
    "search_eval": 512,
    "frozen_eval": 512,
    "search_fraction": 0.35,      # share of the wall-clock budget for the search phase
    "export_reserve_s": 900,
}
SMOKE = dict(PROFILE, main_steps=4, rung0_steps=2, rung1_steps=3, transfer_steps=2, main_seeds=[0],
             search_rounds_per_arm=1, proposals_per_round=2, search_eval=32, frozen_eval=32)
META_ARMS = ("meta_direct", "meta_rlt", "random_mutation")
INKY_ARMS = ("meta_direct", "meta_rlt")
F_MAX = 6  # widest edit at the terminal meta-RLT round
TRANSFER_PAIRS = [("deduction", "retrieval"), ("retrieval", "repair"), ("stopping", "research")]


def base_trm() -> Dict:
    return dsl.validate({"core": "trm"})


def arm_configs() -> Dict[str, Dict]:
    d = base_trm()
    n = M.SkillModule(d).n_params()
    return {
        "A_existing_skill": dict(d, core="heuristic"),
        "B_feedforward": M.match_hidden(dsl.validate({"core": "ff", "mixer_layers": 3}), n),
        "C_gru": M.match_hidden(dsl.validate({"core": "gru"}), n),
        "D_trm_direct": d,
        "E_trm_matched_damping": dsl.validate({"core": "trm", "feedback": "matched_damping"}),
        "F_trm_rlt": dsl.validate({"core": "trm", "feedback": "rlt"}),
        "G_trm_ldt": dsl.validate({"core": "trm", "ldt_projection": True}),
    }


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(run: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run / "campaign.log", "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


class Campaign:
    def __init__(self, run: Path, hours: float, profile: Dict, use_inky: bool = True):
        self.run = run
        self.p = profile
        run.mkdir(parents=True, exist_ok=True)
        clock = run / "clock.json"
        if clock.exists():
            self.t_start = read_json(clock)["t_start"]
        else:
            self.t_start = time.time()
            write_json(clock, {"t_start": self.t_start, "hours": hours})
        self.deadline = self.t_start + hours * 3600
        self.cache = run / "pool_cache"
        self.inky = InkyClient(max_completion_tokens=4000, reasoning_effort="medium") if use_inky else None
        self.inky_ok = bool(use_inky and self.inky.available())
        self.arms = arm_configs()

    # ------------------------------------------------------------ time
    def remaining(self) -> float:
        return self.deadline - time.time()

    def search_deadline(self) -> float:
        return self.t_start + (self.deadline - self.t_start) * self.p["search_fraction"]

    # ------------------------------------------------------------ data
    def train_pool(self, skill):
        n = TR.POOL_TRAIN if self.p is PROFILE else 256
        return TR.pool(skill, T.FAMILIES[skill]["dev"], "train", n, self.cache)

    def search_pool(self, skill, family):
        return TR.pool(skill, [family], "dev_eval", self.p["search_eval"], self.cache)

    def frozen_pools(self, skill):
        n = self.p["frozen_eval"]
        return {
            "frozen_dev": TR.pool(skill, T.FAMILIES[skill]["dev"], "frozen_eval", n, self.cache, 0),
            "frozen_withheld": TR.pool(skill, T.FAMILIES[skill]["withheld"], "frozen_eval", n, self.cache, 50_000),
        }

    # ------------------------------------------------------------ cached training for the search
    def measure(self, skill: str, family: str, cfg: Dict, steps: int, seed: int = 0) -> Dict:
        key = f"{skill}__{family}__{dsl.config_id(cfg)}__{steps}__{seed}"
        path = self.run / "search" / "measurements" / f"{key}.json"
        if path.exists():
            return read_json(path)
        n_params = M.SkillModule(cfg).n_params()
        if not dsl.PARAM_MIN <= n_params <= dsl.PARAM_MAX:
            res = {"status": "rejected_param_bound", "params": n_params}
        else:
            try:
                model, info = TR.train_module(skill, cfg, steps, seed, self.train_pool(skill))
                ev = TR.evaluate(model, skill, cfg, self.search_pool(skill, family))
                res = {"status": "ok", "params": n_params, "train": info, "dev": ev}
            except TR.RamCapExceeded as exc:
                res = {"status": "ram_cap_exceeded", "params": n_params, "error": str(exc)}
            except Exception as exc:  # recorded, never fatal for the campaign
                res = {"status": "train_error", "params": n_params, "error": f"{type(exc).__name__}: {exc}"}
        res.update({"skill": skill, "family": family, "config": cfg, "config_id": dsl.config_id(cfg), "steps": steps,
                    "implementation": dsl.implementation_label(cfg)})
        write_json(path, res)
        return res

    # ------------------------------------------------------------ Inky
    def inky_call(self, system: str, user: str, seed: int) -> Optional[Dict]:
        key = sha([system, user, seed])
        path = self.run / "search" / "inky_cache" / f"{key[:24]}.json"
        if path.exists():
            return read_json(path)
        if not self.inky_ok:
            return None
        try:
            resp = self.inky.chat(system, user, temperature=0.7, seed=seed)
        except (InkyUnavailable, ValueError) as exc:
            log(self.run, f"inky unavailable: {exc}")
            return {"error": str(exc)}
        resp["request_sha256"] = key
        write_json(path, resp)
        return resp

    def prompt(self, arm: str, skill: str, family: str, incumbent: Dict, inc_res: Dict, history: List[Dict],
               budget: Optional[int], n_props: int) -> (str, str):
        system = (
            "You are Inky, an architecture designer for small trainable recurrent skill modules (5k-250k parameters, "
            "CPU training). You propose architectures ONLY as JSON configurations in a restricted language. You never "
            "write code. Measured results come from CPU training and an independent verifier, not from you. Be "
            "quantitative: give a mathematical rationale, a numeric prediction, and a falsification criterion."
        )
        hist = []
        for h in history[-10:]:
            hist.append(json.dumps({k: h.get(k) for k in ("round", "skill", "family", "change", "predicted_dev_exact",
                                                          "measured_dev_exact", "params", "status", "decision")}, default=float))
        skill_md = S.render(skill)
        edit = (f"EDIT BUDGET this round: change at most {budget} fields relative to the incumbent (extra changes are "
                f"reverted by the host in the order you list them).\n") if budget is not None else ""
        user = f"""## Module contract
A skill module runs R visits over a candidate set. Each visit reads typed observations (24 features per candidate,
8 global features, one relation matrix) plus a host-written verifier feedback channel, updates a persistent latent
workspace z and a recurrent answer state y, and emits candidate scores, a confidence, and a predicted value of
another recurrence. Cores: trm = native TRM update (n latent steps z<-net(x+y+z), then answer step y<-net(y+z),
T cycles per visit, gradient through the last grad_cycles cycles, deep supervision across visits); gru = GRU
recurrent proxy; ff = feedforward head. feedback controls the answer update: direct (y replaced), rlt
(y <- y + eta(e-y), eta=1/(D-u)), matched_damping (constant eta with RLT's coefficient sum); the last update always
replaces. ldt_projection adds sound host certificates that clamp certified candidate scores. verifier_visits =
number of in-loop verifier calls whose structured diagnostics are fed back (each costs one verifier call).

## Configuration language (closed; unknown keys are rejected; params must be within {dsl.PARAM_MIN}-{dsl.PARAM_MAX})
{dsl.schema_doc()}

## Skill under test: {skill}, development family: {family}
{skill_md}
Training: {self.p['rung0_steps']} steps (racing rung 0), best proposal promoted to {self.p['rung1_steps']} steps, batch {TR.BATCH}.
Metric: exact success on held-out development roots of family {family}. Also tracked: params, verifier calls,
visits, FLOPs per instance.

## Incumbent architecture for {skill}
{json.dumps(incumbent, sort_keys=True)}
Incumbent measured: {json.dumps(summ(inc_res), default=float)}

## Recent experiment history (this search arm)
{chr(10).join(hist) if hist else "(none yet)"}

{edit}## Request
First diagnose the most recent result(s) (why did they succeed or fail?). Then propose {n_props} distinct
modifications of the incumbent. Return ONLY one JSON object:
{{"diagnosis": "<failure/success diagnosis of the last results>",
  "proposals": [{{"config": {{<only the fields you change>}}, "rationale": "<mathematical rationale>",
                 "predicted_dev_exact": <number 0..1>, "falsification": "<result that would refute it>"}}]}}"""
        return system, user

    # ------------------------------------------------------------ search
    def search(self) -> None:
        sdir = self.run / "search"
        state_path = sdir / "state.json"
        state = read_json(state_path) if state_path.exists() else {
            "round": {a: 0 for a in META_ARMS}, "incumbent": {a: {} for a in META_ARMS},
            "history": {a: [] for a in META_ARMS}, "archive": [], "done": False,
        }
        K = self.p["search_rounds_per_arm"]
        rng = np.random.default_rng(20261128)
        while not state["done"]:
            progressed = False
            for arm in META_ARMS:
                k = state["round"][arm]
                if k >= K:
                    continue
                if time.time() > self.search_deadline():
                    log(self.run, f"search phase deadline reached at {state['round']}")
                    state["done"] = True
                    break
                if arm in INKY_ARMS and not self.inky_ok:
                    state["round"][arm] = K
                    continue
                self.search_round(arm, k, state, np.random.default_rng([20261128, k, META_ARMS.index(arm)]))
                state["round"][arm] = k + 1
                write_json(state_path, state)
                progressed = True
            if not progressed:
                state["done"] = True
        self.final_diagnoses(state)
        write_json(state_path, state)
        self.write_pareto(state)

    def search_round(self, arm: str, k: int, state: Dict, rng) -> None:
        K = self.p["search_rounds_per_arm"]
        skill = T.SKILLS[k % len(T.SKILLS)]
        family = T.FAMILIES[skill]["dev"][(k // len(T.SKILLS)) % 2]
        inc = state["incumbent"][arm].get(skill) or base_trm()
        inc_res = self.measure(skill, family, inc, self.p["rung1_steps"])
        budget = None
        if arm == "meta_rlt":
            budget = max(1, math.ceil(F_MAX * M.feedback_weight("rlt", k, K)))
        log(self.run, f"search {arm} round {k}: {skill}/{family} incumbent dev={summ(inc_res).get('dev_exact')}")
        t0 = time.time()
        proposals, diagnosis, inky_meta = [], None, {}
        n_props = self.p["proposals_per_round"]
        if arm in INKY_ARMS:
            system, user = self.prompt(arm, skill, family, inc, inc_res, state["history"][arm], budget, n_props)
            obj, resp = None, None
            for attempt in range(2):  # one bounded format retry
                resp = self.inky_call(system, user + ("" if attempt == 0 else "\n\nReturn ONLY the JSON object."),
                                      seed=1000 * k + 17 * attempt + META_ARMS.index(arm))
                if resp is None or "error" in resp:
                    break
                obj = extract_json(resp.get("content", ""))
                if obj and isinstance(obj.get("proposals"), list):
                    break
                obj = None
            inky_meta = {"usage": (resp or {}).get("usage"), "seconds": (resp or {}).get("seconds"),
                         "request_sha256": (resp or {}).get("request_sha256"), "error": (resp or {}).get("error")}
            if obj:
                diagnosis = obj.get("diagnosis")
                for pr in obj["proposals"][:n_props]:
                    if isinstance(pr, dict):
                        proposals.append(pr)
            # attach the diagnosis to the previous round's records of this arm
            for h in reversed(state["history"][arm]):
                if h.get("failure_diagnosis") is None and h["round"] == k - 1:
                    h["failure_diagnosis"] = diagnosis
        else:
            for _ in range(n_props):
                cfg = dsl.mutate(inc, rng, 2)
                proposals.append({"config": dsl.diff(inc, cfg), "rationale": None, "predicted_dev_exact": None,
                                  "falsification": None})
        # compile + race
        entries = []
        for j, pr in enumerate(proposals):
            entry = {"arm": arm, "round": k, "skill": skill, "family": family, "proposal_index": j,
                     "incumbent": inc, "incumbent_dev_exact": summ(inc_res).get("dev_exact"),
                     "raw_config": pr.get("config"), "rationale": pr.get("rationale"),
                     "predicted_dev_exact": pr.get("predicted_dev_exact"), "falsification": pr.get("falsification"),
                     "edit_budget": budget, "failure_diagnosis": None, "inky": inky_meta if arm in INKY_ARMS else None,
                     "proposer": "inky_mini" if arm in INKY_ARMS else "random_mutation_control"}
            try:
                cfg = dsl.validate(pr.get("config") or {}, base=inc)
                dropped = []
                if budget is not None:
                    cfg, dropped = dsl.limit_edit(inc, cfg, budget)
                entry.update(config=cfg, config_id=dsl.config_id(cfg), change=dsl.diff(inc, cfg), dropped_by_budget=dropped)
                if not entry["change"]:
                    entry.update(status="no_change", decision="rejected_no_change")
                else:
                    r0 = self.measure(skill, family, cfg, self.p["rung0_steps"])
                    entry.update(rung0=summ(r0), params=r0.get("params"), status=r0["status"])
                    if r0["status"] != "ok":
                        entry["decision"] = r0["status"]
            except dsl.ProposalError as exc:
                entry.update(status="schema_rejected", decision="schema_rejected", error=str(exc), change=None)
            entries.append(entry)
        racing = [e for e in entries if e.get("status") == "ok"]
        racing.sort(key=lambda e: -(e["rung0"].get("dev_exact") or 0))
        for e in racing[1:]:
            e["decision"] = "raced_out_rung0"
        if racing:
            top = racing[0]
            r1 = self.measure(skill, family, top["config"], self.p["rung1_steps"])
            top["rung1"] = summ(r1)
            top["measured_dev_exact"] = top["rung1"].get("dev_exact")
            if r1["status"] == "ok" and (top["measured_dev_exact"] or 0) > (summ(inc_res).get("dev_exact") or 0):
                top["decision"] = "promoted_to_incumbent"
                state["incumbent"][arm][skill] = top["config"]
            else:
                top["decision"] = "rung1_not_better_than_incumbent"
        for e in racing[1:]:
            e["measured_dev_exact"] = e["rung0"].get("dev_exact")
        for e in entries:
            e["round_seconds"] = time.time() - t0
            e["round_diagnosis"] = diagnosis
            state["history"][arm].append(e)
            state["archive"].append(e)
        log(self.run, f"  {arm} round {k}: " + ", ".join(f"{e.get('decision')}:{e.get('measured_dev_exact')}" for e in entries))

    def final_diagnoses(self, state: Dict) -> None:
        for arm in INKY_ARMS:
            hist = state["history"][arm]
            if not hist or not self.inky_ok or state.get("final_diagnosis", {}).get(arm):
                continue
            rows = [json.dumps({k: h.get(k) for k in ("round", "skill", "change", "predicted_dev_exact",
                                                      "measured_dev_exact", "decision")}, default=float) for h in hist[-24:]]
            system = "You are Inky, a failure analyst for small recurrent skill modules. Answer with one JSON object."
            user = ("Search history (one row per proposal):\n" + "\n".join(rows) +
                    '\n\nReturn {"diagnosis": "<what the history shows about which changes help or hurt and why>", '
                    '"next_experiment": {"skill": "<skill>", "config": {<fields>}, "rationale": "<why>"}}')
            resp = self.inky_call(system, user, seed=999)
            obj = extract_json((resp or {}).get("content", "")) if resp and "error" not in resp else None
            state.setdefault("final_diagnosis", {})[arm] = obj or {"error": (resp or {}).get("error", "unparsed")}
            for h in hist:
                if h.get("failure_diagnosis") is None and h["round"] == hist[-1]["round"]:
                    h["failure_diagnosis"] = (obj or {}).get("diagnosis")

    def write_pareto(self, state: Dict) -> None:
        pts = []
        seen = set()
        for e in state["archive"]:
            res = e.get("rung1") or e.get("rung0")
            if not res or e.get("status") != "ok":
                continue
            key = (e["skill"], e["config_id"], "rung1" if e.get("rung1") else "rung0")
            if key in seen:
                continue
            seen.add(key)
            pts.append({"skill": e["skill"], "arm": e["arm"], "config_id": e["config_id"], "rung": key[2],
                        "config": e["config"], "dev_exact": res.get("dev_exact") or 0.0, "params": e.get("params"),
                        "verifier_calls": res.get("verifier_calls"), "visits": res.get("visits"),
                        "flops": res.get("flops_per_instance"), "implementation": dsl.implementation_label(e["config"])})
        for skill in T.SKILLS:
            sp = [p for p in pts if p["skill"] == skill]
            for p in sp:
                p["pareto"] = not any(
                    q is not p and q["dev_exact"] >= p["dev_exact"] and q["params"] <= p["params"]
                    and q["verifier_calls"] <= p["verifier_calls"] and q["visits"] <= p["visits"] and q["flops"] <= p["flops"]
                    and (q["dev_exact"] > p["dev_exact"] or q["params"] < p["params"] or q["verifier_calls"] < p["verifier_calls"]
                         or q["visits"] < p["visits"] or q["flops"] < p["flops"])
                    for q in sp)
        write_json(self.run / "search" / "pareto.json", pts)
        with open(self.run / "search" / "archive.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for e in state["archive"]:
                f.write(json.dumps(e, sort_keys=True, default=float) + "\n")

    # ------------------------------------------------------------ freeze H
    def freeze(self) -> Dict:
        path = self.run / "frozen_architectures.json"
        if path.exists():
            return read_json(path)
        state = read_json(self.run / "search" / "state.json")
        out = {}
        best_any = None
        for skill in T.SKILLS:
            cands = [e for e in state["archive"] if e["arm"] in INKY_ARMS and e["skill"] == skill and e.get("rung1")
                     and e["rung1"].get("dev_exact") is not None]
            if cands:
                b = max(cands, key=lambda e: e["rung1"]["dev_exact"])
                out[skill] = {"config": b["config"], "source": f"{b['arm']} round {b['round']}", "dev_exact": b["rung1"]["dev_exact"]}
                if best_any is None or b["rung1"]["dev_exact"] > best_any[1]:
                    best_any = (b["config"], b["rung1"]["dev_exact"], skill)
        for skill in T.SKILLS:
            if skill not in out:
                if best_any:
                    out[skill] = {"config": best_any[0], "source": f"selected_on_other_skill:{best_any[2]}", "dev_exact": None}
                else:
                    out[skill] = {"config": base_trm(), "source": "no_inky_selection_fallback_to_D", "dev_exact": None}
        frozen = {"frozen_at": time.time(), "architectures": out}
        frozen["sha256"] = sha(out)
        write_json(path, frozen)
        return frozen

    # ------------------------------------------------------------ main / ablation jobs
    def job(self, kind: str, skill: str, arm: str, cfg: Dict, seed: int, steps: int, shuffled: bool = False,
            init_state=None, freeze_core: bool = False, extra: Optional[Dict] = None) -> Optional[Dict]:
        name = f"{skill}__{arm}__s{seed}"
        path = self.run / kind / f"{name}.json"
        if path.exists():
            return read_json(path)
        if self.remaining() < self.p["export_reserve_s"]:
            return None
        pools = self.frozen_pools(skill)
        t0 = time.time()
        res = {"kind": kind, "skill": skill, "arm": arm, "seed": seed, "config": cfg, "steps": steps,
               "implementation": "heuristic_existing_skill" if cfg["core"] == "heuristic" else dsl.implementation_label(cfg),
               "shuffled_supervision": shuffled, "freeze_core": freeze_core, **(extra or {})}
        try:
            if cfg["core"] == "heuristic":
                ecfg = dict(base_trm(), ldt_projection=False)
                res["params"] = 0
                res["train"] = {"steps": 0, "train_seconds": 0.0, "examples_seen": 0, "peak_rss_delta_bytes": 0}
                for pname, pool in pools.items():
                    res[pname] = TR.evaluate(None, skill, ecfg, pool, heuristic=True)
                res["certificate_violations"] = 0
            else:
                model, info = TR.train_module(skill, cfg, steps, seed, self.train_pool(skill), shuffled=shuffled,
                                              init_state=init_state, freeze_core=freeze_core)
                res["params"] = model.n_params()
                res["train"] = info
                R = cfg["visits"]
                for pname, pool in pools.items():
                    res[pname] = TR.evaluate(model, skill, cfg, pool)
                    if kind == "main":
                        res[pname + "_2R"] = TR.evaluate(model, skill, cfg, pool, visits=2 * R)
                        res[pname + "_1"] = TR.evaluate(model, skill, cfg, pool, visits=1)
                res["certificate_violations"] = sum(T.certificate_violations(skill, p) for p in pools.values()) if cfg["ldt_projection"] else 0
                if kind in ("main", "transfer_source"):
                    ck = self.run / "checkpoints" / f"{name}.pt"
                    ck.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"config": cfg, "state_dict": model.state_dict()}, ck)
                    res["checkpoint"] = str(ck.relative_to(self.run))
            res["status"] = "ok"
        except TR.RamCapExceeded as exc:
            res.update(status="ram_cap_exceeded", error=str(exc))
        except Exception as exc:
            res.update(status="error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])
        res["wall_seconds"] = time.time() - t0
        write_json(path, res)
        log(self.run, f"{kind} {name}: {res['status']} dev={res.get('frozen_dev', {}).get('exact')} "
                      f"withheld={res.get('frozen_withheld', {}).get('exact')} ({res['wall_seconds']:.0f}s)")
        return res

    def main_arms(self, frozen: Dict, seed: int) -> None:
        for skill in T.SKILLS:
            arms = dict(self.arms)
            arms["H_inky_engineered"] = frozen["architectures"][skill]["config"]
            for arm, cfg in arms.items():
                self.job("main", skill, arm, cfg, seed, self.p["main_steps"])

    def ablations(self) -> None:
        for skill in T.SKILLS:
            self.job("ablation", skill, "D_shuffled_supervision", self.arms["D_trm_direct"], 0, self.p["main_steps"], shuffled=True)
            self.job("ablation", skill, "D_latent_reset", dict(self.arms["D_trm_direct"], latent_reset=True), 0, self.p["main_steps"])
            self.job("ablation", skill, "F_latent_reset", dict(self.arms["F_trm_rlt"], latent_reset=True), 0, self.p["main_steps"])

    def transfer(self) -> None:
        d = self.arms["D_trm_direct"]
        for src, tgt in TRANSFER_PAIRS:
            ck = self.run / "checkpoints" / f"{src}__D_trm_direct__s0.pt"
            if not ck.exists():
                continue
            state = torch.load(ck)["state_dict"]
            steps = self.p["transfer_steps"]
            self.job("transfer", tgt, f"from_{src}__scratch", d, 0, steps, extra={"source": src, "mode": "scratch"})
            self.job("transfer", tgt, f"from_{src}__shared_init", d, 0, steps, init_state=state,
                     extra={"source": src, "mode": "shared_initialization"})
            self.job("transfer", tgt, f"from_{src}__frozen_core", d, 0, steps, init_state=state, freeze_core=True,
                     extra={"source": src, "mode": "frozen_latent_module"})

    # ------------------------------------------------------------ driver
    def go(self) -> None:
        torch.set_num_threads(1)
        log(self.run, f"campaign start; inky={'on' if self.inky_ok else 'off'}; remaining {self.remaining() / 3600:.2f} h")
        receipts = self.run / "receipts.json"
        if not receipts.exists():
            src = Path(__file__).resolve().parent
            prereg = src.parents[1] / "experiments" / "trm_skill_gym_01" / "PREREGISTRATION.md"
            lf = lambda p: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            write_json(receipts, {"source_sha256_lf": {p.name: lf(p) for p in sorted(src.glob("*.py"))},
                                  "preregistration_sha256_lf": lf(prereg) if prereg.exists() else None,
                                  "profile": self.p, "torch": torch.__version__, "started": self.t_start})
        write_json(self.run / "arms.json", {k: dict(v, implementation=("heuristic_existing_skill" if v["core"] == "heuristic"
                                                                          else dsl.implementation_label(v)),
                                                    params=(0 if v["core"] == "heuristic" else M.SkillModule(v).n_params()))
                                            for k, v in self.arms.items()})
        self.search()
        frozen = self.freeze()
        log(self.run, f"frozen architectures sha256={frozen['sha256'][:16]}")
        self.main_arms(frozen, self.p["main_seeds"][0])
        self.ablations()
        self.transfer()
        for seed in self.p["main_seeds"][1:]:
            self.main_arms(frozen, seed)
        write_json(self.run / "inky_ledger.json", self.inky.ledger if self.inky else {})
        log(self.run, "campaign jobs finished; run `python -m research_gym.trm_skill_gym.report` to export")


def summ(res: Dict) -> Dict:
    if not res or res.get("status") != "ok":
        return {"status": (res or {}).get("status")}
    d = res["dev"]
    return {"dev_exact": round(d["exact"], 4), "repair_success": round(d["repair_success"], 4),
            "exact_by_visit": [round(x, 4) for x in d["exact_by_visit"]], "params": res["params"],
            "verifier_calls": d["verifier_calls_per_instance"], "visits": d["visits"],
            "flops_per_instance": d["flops_per_instance"], "ece": d.get("ece"),
            "adaptive_exact": d.get("adaptive_exact"), "adaptive_visits": d.get("adaptive_visits"),
            "train_seconds": res["train"]["train_seconds"], "peak_rss_delta_bytes": res["train"]["peak_rss_delta_bytes"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-inky", action="store_true")
    a = ap.parse_args()
    if a.hours > 8.0:
        ap.error("the wall-clock limit is eight hours")
    Campaign(a.out, a.hours, SMOKE if a.smoke else PROFILE, use_inky=not a.no_inky).go()


if __name__ == "__main__":
    main()
