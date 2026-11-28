"""MOLT-GOBLIN-03 discovery campaign: Inky Mini proposes, the CPU fixture decides.

Per iteration Inky receives the configuration grammar, public family descriptions,
the frontier with development receipts, mechanistic evidence for leading arms, the
list of already-tested canonical configurations, and its own recent hypotheses with
outcomes. It returns a hypothesis, a candidate change, a discriminating control and
predictions. Both are compiled, deduplicated by canonical digest, trained on
development roots (DEV3_BASE) and scored against arm A. Successive halving promotes
on an efficiency-adjusted score. Hidden roots and the finals-only family never appear.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from research_gym.molt_goblin_02 import inky as inky_mod

from .resources3 import install_tree_guard, lower_priority, peak_tree_rss

from .arms import ARMS, BASE, PANEL, PRINCIPAL, RUNGS, TEACHER
from .factory3 import Config, describe_grammar, logical_visits
from .gym3 import FAMILIES3, FINALS_ONLY_FAMILIES, TaskSpec, verify_spec
from .mech import specialization
from .train3 import DEV3_BASE, root_seed, run_cell, salt_of

CAMPAIGN_FAMILIES = PRINCIPAL
REFERENCE = "A_single"
SEED_ARMS = ("A_single", "B_hrm", "C_mix", "H_sparse_st", "I_anneal", "K_distill")
VISIT_PENALTY = 0.03          # per doubling of inference visits beyond arm A's 4
MAX_LIVE = 16
ADVERSARIAL_EVERY = 6
PROMOTE_EVERY = 4
MECH_FAMILY = "proof_resolve"
MECHANISMS = ("routing", "sparsity", "module_objective", "aux_loss", "composition", "distillation",
              "capacity", "loop_forcing", "sharing")

SYSTEM = (
    "You are a research scientist explaining why a soft four-module HRM mixture beat single-block recurrence in "
    "tiny models (MOLT-GOBLIN-02), and designing a cheaper architecture for Baby Goblin and DataGoblin skill modules. "
    "Propose one falsifiable experiment per turn: a hypothesis about the computational source of the advantage, a "
    "candidate configuration, and a discriminating control that differs from the candidate only in the mechanism "
    "you test. You may propose functional module objectives, auxiliary losses, sequential composition rules and "
    "distillation strategies through the grammar. You never see task data or labels; reason from receipts. "
    "Failed experiments are evidence. Reply with at most 12 short lines of reasoning, then exactly one ```json block.")


def _atomic_write(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.5)


def _diff(cfg: dict) -> dict:
    b = BASE.to_dict()
    return {k: v for k, v in cfg.items() if b.get(k) != v}


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _job(payload):
    """One training cell; top-level so spawned workers can run it. Returns (record, mech or None)."""
    cfg_d, family, seed, adv, want_mech, cache_teacher, teacher_d, threads, panels = payload
    lower_priority()
    torch.set_num_threads(threads)
    cfg = Config.from_dict(cfg_d)
    panel = {"train": panels[family]["train"], "eval": dict(panels[family]["eval"])}
    for a in adv:
        panel["eval"][f"adv_{a['id']}"] = TaskSpec(**{k: v for k, v in a.items() if k != "id"})
    t0 = time.time()
    mech = None
    try:
        r = run_cell(cfg, family, panel, seed, DEV3_BASE, teacher_cfg=Config.from_dict(teacher_d), n_eval=256,
                     extrap_loops=min(16, 2 * cfg.loops), return_model=want_mech, cache_teacher=cache_teacher)
        model = r.pop("_model", None)
        r["status"] = "ok" if r["finite"] else "nonfinite"
        if model is not None and r["status"] == "ok":
            mech = specialization(model, panel["eval"]["dev"], 256, root_seed(DEV3_BASE, family, seed, salt_of("mech")))
    except Exception as exc:
        r = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "family": family, "seed": seed}
    r["wall_s"] = time.time() - t0
    return r, mech


class Campaign:
    def __init__(self, out: Path, hours: float, threads: int, use_inky: bool, max_iterations: int = 10_000,
                 workers: int = 1):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.state_path, self.runs_path = out / "state.json", out / "runs.jsonl"
        self.max_iterations = max_iterations
        self.threads = threads
        self.pool = None
        if workers > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            self.pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        torch.set_num_threads(threads)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state["resumed"] = self.state.get("resumed", 0) + 1
        else:
            now = time.time()
            self.state = {"schema": "molt-goblin-03.campaign.v1", "started_at": now, "deadline": now + hours * 3600,
                          "iteration": 0, "archive": {}, "live": [], "history": [], "adversarial": [],
                          "rejected_adversarial": [], "cache": {}, "mech": {}, "promotions": [], "resumed": 0}
            for name in SEED_ARMS:
                self._archive(ARMS[name], {"kind": "registered_arm", "name": name})
        self.inky = inky_mod.Inky(log_path=out / "inky_requests.jsonl") if use_inky and inky_mod.available() else None
        self.state["proposer"] = "inky" if self.inky else "deterministic_fallback"
        self._save()

    def _save(self):
        self.state["peak_tree_rss_bytes"] = max(self.state.get("peak_tree_rss_bytes") or 0, peak_tree_rss() or 0)
        _atomic_write(self.state_path, self.state)

    def _archive(self, cfg: Config, origin: dict) -> str:
        cid = cfg.digest()
        if cid not in self.state["archive"]:
            self.state["archive"][cid] = {"config": cfg.to_dict(), "canonical_diff": _diff(cfg.canonical()),
                                          "origin": origin, "rung": -1, "scores": {}}
        return cid

    def _name(self, cid):
        return self.state["archive"][cid]["origin"].get("name") or f"c{cid[:6]}"

    # ------------------------------------------------------------------ evaluation
    def _payload(self, cid, family, seed, steps):
        cfg = {**self.state["archive"][cid]["config"], "steps": steps}
        adv = [a["spec"] | {"id": a["id"]} for a in self.state["adversarial"] if a["family"] == family]
        want_mech = family == MECH_FAMILY and seed == 11 and cfg["n_types"] > 1
        return (cfg, family, seed, adv, want_mech, Config.from_dict(cfg).digest() == TEACHER.digest(),
                TEACHER.to_dict(), self.threads, {f: {"train": p["train"], "eval": p["eval"]} for f, p in PANEL.items()})

    def _ensure(self, keys):
        """Train every missing (cid, family, seed, steps) cell, in parallel workers when available."""
        todo = [k for k in dict.fromkeys(keys) if k not in self.state["cache"]]
        if not todo:
            return
        payloads = [self._payload(*k) for k in todo]
        results = self.pool.map(_job, payloads) if self.pool else (_job(p) for p in payloads)
        for (cid, family, seed, steps), (r, mech) in zip(todo, results):   # checkpoint each cell as it lands
            self._record(cid, family, seed, steps, r, mech)

    def _run(self, cid, family, seed, steps):
        self._ensure([(cid, family, seed, steps)])
        return self.state["cache"][f"{cid}|{family}|{seed}|{steps}"]

    def _record(self, cid, family, seed, steps, r, mech):
        key = f"{cid}|{family}|{seed}|{steps}"
        cfg = Config.from_dict({**self.state["archive"][cid]["config"], "steps": steps})
        if mech is not None:
            self.state["mech"][cid] = {k: (_r(v, 3) if isinstance(v, float) else [_r(x, 3) for x in v] if isinstance(v, list) else v)
                                       for k, v in mech.items() if k != "spec"}
        with self.runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, **r}) + "\n")
        c = {"status": r["status"]}
        if r["status"] == "ok":
            ev = r["evals"]
            c.update({"exact": ev["dev"]["exact"], "params": r["params"], "train_cpu_s": _r(r["train_cpu_s"], 2),
                      "visits": logical_visits(cfg, hard_eval=cfg.routing in ("sparse_st", "anneal")),
                      "flops": r["cost"]["flops_per_example"],
                      "flops_hard": r["cost"].get("dispatch_hard", {}).get("flops_per_example"),
                      "ood": {k: ev[k]["exact"] for k in ev if k.startswith("ood")},
                      "adv": {k: ev[k]["exact"] for k in ev if k.startswith("adv_")},
                      "interventions": {k.split("|")[1]: ev[k]["exact"] for k in ev if "|" in k},
                      "routing_entropy": ev["dev"].get("routing_entropy"), "routing_loop_mi": ev["dev"].get("routing_loop_mi"),
                      "hard_usage": ev["dev"].get("hard_usage"),
                      "exact_by_loop": [round(x, 3) for x in ev["dev"]["exact_by_loop"]]})
        else:
            c["error"] = r.get("error", r["status"])
        self.state["cache"][key] = c
        self._save()
        return c

    def evaluate(self, cid, rung, extra: list | None = None):
        spec = RUNGS[rung]
        ref = ARMS[REFERENCE].digest()
        need = [(c, f, s, spec["steps"]) for c in [cid, ref] + list(extra or []) for f in CAMPAIGN_FAMILIES for s in spec["seeds"]]
        self._ensure(need)
        fams = {}
        for family in CAMPAIGN_FAMILIES:
            runs = [self._run(cid, family, s, spec["steps"]) for s in spec["seeds"]]
            refs = [self._run(ref, family, s, spec["steps"]) for s in spec["seeds"]]
            if any(r["status"] != "ok" for r in runs):
                fams[family] = {"status": "failed", "errors": [r.get("error") for r in runs if r["status"] != "ok"]}
                continue
            mean = lambda xs: sum(xs) / len(xs)
            ood = [mean(list(r["ood"].values())) for r in runs]
            rood = [mean(list(r["ood"].values())) for r in refs]
            fams[family] = {"exact": mean([r["exact"] for r in runs]),
                            "delta_vs_A": mean([a["exact"] - b["exact"] for a, b in zip(runs, refs)]),
                            "ood": mean(ood), "ood_delta_vs_A": mean(ood) - mean(rood),
                            "visits": runs[0]["visits"], "params": runs[0]["params"], "flops": runs[0]["flops"],
                            "train_cpu_s": mean([r["train_cpu_s"] for r in runs]),
                            "interventions": runs[0]["interventions"], "routing_entropy": runs[0]["routing_entropy"],
                            "hard_usage": runs[0]["hard_usage"], "seeds": list(spec["seeds"])}
        good = [v for v in fams.values() if "delta_vs_A" in v]
        quality = sum(v["delta_vs_A"] + 0.5 * v["ood_delta_vs_A"] for v in good) / len(good) if len(good) == len(fams) else None
        visits = good[0]["visits"] if good else None
        score = None if quality is None else quality - VISIT_PENALTY * max(0.0, math.log2(visits / 4))
        s = {"families": fams, "quality": quality, "score": score, "visits": visits,
             "mean_exact": sum(v["exact"] for v in good) / len(good) if good else None}
        a = self.state["archive"][cid]
        a["scores"][str(rung)] = s
        a["rung"] = max(a["rung"], rung)
        self._save()
        return s

    # ------------------------------------------------------------------ prompt
    def _frontier(self, k=10):
        rows = []
        for cid, a in self.state["archive"].items():
            if not a["scores"]:
                continue
            s = a["scores"][str(a["rung"])]
            rows.append({"id": self._name(cid), "rung": a["rung"], "quality": _r(s["quality"], 3), "score": _r(s["score"], 3),
                         "visits": s["visits"], "diff": a["canonical_diff"],
                         "fam": {f: {"exact": _r(v.get("exact"), 3), "ood": _r(v.get("ood"), 3)} for f, v in s["families"].items()},
                         "mech": self.state["mech"].get(cid)})
        reg = [r for r in rows if not r["id"].startswith("c")]
        cand = sorted([r for r in rows if r["id"].startswith("c")], key=lambda r: (r["rung"], r["score"] or -9), reverse=True)[:k]
        return reg + cand

    def _tested(self, limit_chars=5000):
        items = [json.dumps(a["canonical_diff"], separators=(",", ":"), sort_keys=True)
                 for a in self.state["archive"].values() if a["scores"]]
        out, n = [], 0
        for s in reversed(items):
            if n + len(s) > limit_chars:
                break
            out.append(s); n += len(s)
        return "\n".join(reversed(out))

    def _history(self, budget):
        out, n = [], 0
        for h in reversed(self.state["history"]):
            s = json.dumps({k: h.get(k) for k in ("iteration", "kind", "mechanism", "hypothesis", "candidate", "control",
                                                  "predicted", "observed", "error")}, separators=(",", ":"))
            if n + len(s) > budget:
                break
            out.append(s); n += len(s)
        return "\n".join(reversed(out))

    def _context(self):
        fams = {f: FAMILIES3[f].describe() for f in CAMPAIGN_FAMILIES}
        return (f"GRAMMAR (BASE = {json.dumps(BASE.to_dict(), separators=(',', ':'))}):\n{describe_grammar()}\n"
                f"Registered arms: A_single = BASE; B_hrm = hrm n_types 4 (MOLT-02 winner, 16 visits); C_mix = uniform mix; "
                f"H_sparse_st and I_anneal = native sparse routing (4 visits); K_distill = single-block student of B (4 visits). "
                f"Distillation always uses B_hrm as the frozen teacher.\n"
                f"DEV FAMILIES:\n{json.dumps(fams, indent=0)}\n"
                f"Metrics: exact on 256 development roots; ood = shifted specs; visits = module visits per position at inference; "
                f"quality = mean over families of (exact - A) + 0.5*(ood - A ood); score = quality - {VISIT_PENALTY}*log2(visits/4). "
                f"mech (proof_resolve, seed 11): module_cka_mean, ablate_module (exact with module i removed), permute_modules, "
                f"route_uniform/route_shuffled/route_forced (exact under routing interventions), effective ranks.\n\n"
                f"FRONTIER:\n{json.dumps(self._frontier(), separators=(',', ':'))}\n\n"
                f"ALREADY TESTED (canonical diffs from BASE; do not repeat):\n{self._tested()}\n")

    def _proposal_prompt(self):
        ctx = self._context()
        return (ctx + f"\nRECENT EXPERIMENTS:\n{self._history(max(2000, 21000 - len(ctx)))}\n\n"
                "TASK: diagnose the most recent experiment in one or two lines, say how it revises your account of where "
                "the mixture advantage comes from, then propose the next experiment. JSON keys: "
                '{"diagnosis": str, "revision": str, "hypothesis": str, "mechanism": "<one of ' + "|".join(MECHANISMS) + '>", '
                '"base": "<frontier id>", "candidate": {<grammar keys to change>}, "control": {<keys to change from base>}, '
                '"predicted": {"<family>": "up|down|flat"}, "rationale": str}. Keep width<=64 and loops<=8.')

    def _adversarial_prompt(self):
        ctx = self._context()
        return (ctx + f"\nRECENT EXPERIMENTS:\n{self._history(6000)}\n\nTASK: name the two leading explanations of the "
                "mixture advantage and invent ONE development task variant that separates them. Spec fields: family "
                f"(one of {', '.join(CAMPAIGN_FAMILIES)}), length, depth, distract, variant='clean'. Total tokens <= 96. "
                'JSON keys: {"explanations": [str, str], "spec": {...}, "prediction": str}.')

    # ------------------------------------------------------------------ proposals
    def _fallback(self):
        grid = [("routing", {"routing": "mix", "n_types": 4}, {}), ("sharing", {"routing": "hrm", "n_types": 4, "tie_modules": True}, {"routing": "hrm", "n_types": 4}),
                ("sparsity", {"routing": "sparse_st", "n_types": 4, "load_balance": 0.1}, {"routing": "sparse_st", "n_types": 4}),
                ("composition", {"routing": "sequential", "n_types": 4}, {"routing": "cycle", "n_types": 4}),
                ("module_objective", {"routing": "hrm", "n_types": 4, "functional": True, "func_coef": 0.5}, {"routing": "hrm", "n_types": 4}),
                ("aux_loss", {"routing": "hrm", "n_types": 4, "diversity": 0.5}, {"routing": "hrm", "n_types": 4}),
                ("distillation", {"distill": "hidden", "distill_coef": 1.0}, {"distill": "logits_all", "distill_coef": 1.0})]
        i = sum(h.get("kind") == "proposal" for h in self.state["history"]) % len(grid)
        m, c, k = grid[i]
        return {"hypothesis": f"fallback factorial cell ({m})", "mechanism": m, "base": REFERENCE, "candidate": c,
                "control": k, "predicted": {}, "rationale": "deterministic fallback (no Inky)"}

    def _resolve_base(self, name):
        name = str(name or REFERENCE)
        for cid, a in self.state["archive"].items():
            if self._name(cid) == name or (name.startswith("c") and cid.startswith(name[1:])):
                return dict(a["config"])
        if name in ARMS and ARMS[name] is not None:
            return ARMS[name].to_dict()
        return BASE.to_dict()

    @staticmethod
    def _compile(base, change):
        if not isinstance(change, dict):
            raise ValueError("change must be an object")
        m = {**base, **change, "steps": BASE.steps}
        if m.get("routing") == "single":
            m["n_types"], m["tie_modules"], m["functional"], m["func_coef"] = 1, False, False, 0.0
        elif m.get("n_types", 1) == 1:
            m["n_types"] = 4
        if m.get("distill", "none") == "none":
            m["distill_coef"] = 0.0
        elif not m.get("distill_coef"):
            m["distill_coef"] = 1.0
        cfg = Config.from_dict(m)
        if cfg.width > 64 or cfg.loops > 8:
            raise ValueError("width<=64 and loops<=8 in the campaign")
        return cfg

    def step(self):
        it = self.state["iteration"]
        if self.inky and it % ADVERSARIAL_EVERY == ADVERSARIAL_EVERY - 1:
            self._adversarial(it)
        rec = {"iteration": it, "kind": "proposal", "t": time.time(), "proposer": self.state["proposer"]}
        prop = None
        if self.inky:
            try:
                prompt = self._proposal_prompt()
                reply = self.inky.ask(SYSTEM, prompt, "proposal")
                if reply["ok"] and reply.get("finish_reason") == "length" and inky_mod.extract_json(reply["text"]) is None:
                    rec["truncated_first_reply"] = True
                    reply = self.inky.ask(SYSTEM, prompt, "proposal_retry_low_effort", effort="low")
            except ValueError as exc:
                reply = {"ok": False, "error_type": f"ContextGuard: {exc}"[:200]}
            rec["request"] = {k: reply.get(k) for k in ("ok", "seconds", "usage", "finish_reason", "error_type", "http_status")}
            if reply["ok"]:
                rec["reply_text"] = reply["text"][-4000:]
                prop = inky_mod.extract_json(reply["text"])
                if prop is None:
                    rec["error"] = "no JSON object in reply"
            else:
                rec["error"] = f"request failed: {reply.get('error_type')}"
        else:
            prop = self._fallback()
        if prop is not None:
            rec.update({k: prop.get(k) for k in ("diagnosis", "revision", "hypothesis", "mechanism", "base", "candidate",
                                                  "control", "predicted", "rationale")})
            try:
                base = self._resolve_base(prop.get("base"))
                cand = self._compile(base, prop.get("candidate") or {})
                ctl = self._compile(base, prop.get("control") or {})
                cid = cand.digest()
                if cid in self.state["archive"] and self.state["archive"][cid]["scores"]:
                    rec["error"] = f"duplicate of already-tested {self._name(cid)} (canonical digest)"
                else:
                    cid = self._archive(cand, {"kind": "candidate", "iteration": it, "mechanism": prop.get("mechanism")})
                    kid = self._archive(ctl, {"kind": "control", "iteration": it, "of": cid})
                    s_c = self.evaluate(cid, 0, extra=[kid])
                    s_k = self.evaluate(kid, 0) if kid != cid else s_c
                    rec["candidate_id"], rec["control_id"] = self._name(cid), self._name(kid)
                    rec["observed"] = self._observed(s_c, s_k)
                    rec["verdict"] = self._verdict(prop.get("predicted") or {}, s_c)
                    if cid not in self.state["live"]:
                        self.state["live"].append(cid)
            except Exception as exc:
                rec["error"] = f"invalid proposal: {type(exc).__name__}: {exc}"[:400]
        self.state["history"].append(rec)
        self.state["iteration"] = it + 1
        if (it + 1) % PROMOTE_EVERY == 0:
            self._promote()
        self._save()

    @staticmethod
    def _observed(s_c, s_k):
        o = {}
        for f, v in s_c["families"].items():
            k = s_k["families"].get(f, {})
            o[f] = {"exact": _r(v.get("exact")), "minus_A": _r(v.get("delta_vs_A")),
                    "minus_control": _r(v["exact"] - k["exact"]) if "exact" in v and "exact" in k else None,
                    "ood": _r(v.get("ood")), "status": v.get("status", "ok")}
        o.update({"quality": _r(s_c["quality"]), "score": _r(s_c["score"]), "visits": s_c["visits"]})
        return o

    @staticmethod
    def _verdict(pred, s_c):
        out = {}
        for f, d in pred.items():
            dv = s_c["families"].get(f, {}).get("delta_vs_A")
            if dv is None or not isinstance(d, str):
                continue
            obs = "up" if dv > 0.03 else "down" if dv < -0.03 else "flat"
            out[f] = {"predicted": d, "observed": obs, "match": d.lower().startswith(obs)}
        return out

    def _adversarial(self, it):
        try:
            prompt = self._adversarial_prompt()
            reply = self.inky.ask(SYSTEM, prompt, "adversarial_task")
            if reply["ok"] and reply.get("finish_reason") == "length" and inky_mod.extract_json(reply["text"]) is None:
                reply = self.inky.ask(SYSTEM, prompt, "adversarial_task_retry_low_effort", effort="low")
        except ValueError as exc:
            reply = {"ok": False, "error_type": f"ContextGuard: {exc}"[:200]}
        rec = {"iteration": it, "kind": "adversarial_task", "t": time.time(),
               "request": {k: reply.get(k) for k in ("ok", "seconds", "usage", "error_type")}}
        if reply["ok"]:
            prop = inky_mod.extract_json(reply["text"]) or {}
            rec.update({"explanations": prop.get("explanations"), "prediction": prop.get("prediction"), "spec": prop.get("spec")})
            try:
                sp = prop.get("spec") or {}
                spec = TaskSpec(family=str(sp["family"]), length=int(sp["length"]), depth=int(sp.get("depth", 0)),
                                distract=int(sp.get("distract", 0)), variant=str(sp.get("variant", "clean")))
                if spec.family not in CAMPAIGN_FAMILIES or spec.family in FINALS_ONLY_FAMILIES:
                    raise ValueError("variant must come from a campaign development family")
                FAMILIES3[spec.family].validate(spec)
                v = verify_spec(spec, n=128, seed=424242 + it)
                rec["verification"] = v
                if not v["ok"]:
                    raise ValueError("verification failed: " + "; ".join(v["reasons"])[:200])
                adv = {"id": f"a{len(self.state['adversarial'])}", "family": spec.family, "iteration": it,
                       "spec": {"family": spec.family, "length": spec.length, "depth": spec.depth, "distract": spec.distract,
                                "variant": spec.variant}, "explanations": prop.get("explanations"), "prediction": prop.get("prediction")}
                self.state["adversarial"].append(adv)
                rec["accepted"] = adv["id"]
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                self.state["rejected_adversarial"].append(rec)
        else:
            rec["error"] = f"request failed: {reply.get('error_type')}"
        self.state["history"].append(rec)
        self._save()

    def _promote(self):
        arch = self.state["archive"]
        for rung, cap in ((0, 2), (1, 1)):
            pool = [c for c in self.state["live"] if arch[c]["rung"] == rung and arch[c]["scores"].get(str(rung), {}).get("score") is not None]
            pool.sort(key=lambda c: arch[c]["scores"][str(rung)]["score"], reverse=True)
            for cid in pool[:min(cap, max(1, len(pool) // 3)) if pool else 0]:
                if time.time() > self.state["deadline"]:
                    return
                s = self.evaluate(cid, rung + 1)
                self.state["promotions"].append({"id": self._name(cid), "from": rung, "to": rung + 1, "score": s["score"],
                                                 "iteration": self.state["iteration"]})
        live = self.state["live"]
        if len(live) > MAX_LIVE:
            live.sort(key=lambda c: (arch[c]["rung"], arch[c]["scores"].get(str(arch[c]["rung"]), {}).get("score") or -9), reverse=True)
            self.state["live"] = live[:MAX_LIVE]

    def run(self):
        # Teacher first so the distillation arm and later students reuse its cached checkpoints.
        self.evaluate(ARMS["B_hrm"].digest(), 0)
        rest = [ARMS[n].digest() for n in SEED_ARMS if n != "B_hrm"]
        if rest:
            self.evaluate(rest[0], 0, extra=rest[1:])
            for cid in rest[1:]:
                self.evaluate(cid, 0)
        while time.time() < self.state["deadline"] and self.state["iteration"] < self.max_iterations:
            self.step()
        self.state["finished_at"] = time.time()
        self._save()
        if self.pool:
            self.pool.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-iterations", type=int, default=10_000)
    ap.add_argument("--no-inky", action="store_true")
    ap.add_argument("--ram-gib", type=float, default=3.0, help="hard total for the driver and all workers")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    lower_priority()
    if a.hours > 6.0:
        ap.error("campaign budget is capped at 6 hours")
    a.out.mkdir(parents=True, exist_ok=True)
    cap = install_tree_guard(a.ram_gib, receipt_path=a.out / "ram_breach.json")
    (a.out / "resource_cap.json").write_text(json.dumps(cap, indent=1), encoding="utf-8")
    Campaign(a.out, a.hours, a.threads, not a.no_inky, a.max_iterations, workers=a.workers).run()


if __name__ == "__main__":
    main()
