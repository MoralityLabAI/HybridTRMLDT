"""MOLT-GOBLIN-02 overnight discovery loop.

Each iteration: show Inky Mini the frontier and receipts, get a hypothesis with a concrete
configuration, a predicted effect and a discriminating control, compile both into the
factory, train on CPU, evaluate against matched baseline arms on development roots, and
return the outcome (including failures) in the next prompt. A bounded population advances
through successive-halving rungs. Every ADVERSARIAL_EVERY iterations Inky proposes a task
variant; it is used only after independent verification by the exact generator/solver.
Inky sees family descriptions, configurations and aggregate scores only; never labels,
examples or hidden roots. State is checkpointed after every training run.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from . import inky as inky_mod
from . import resources
from .factory import MoltConfig, describe_schema
from .gym import DEV_FAMILIES, FAMILIES, TaskSpec, verify_spec
from .panel import BASE, BASELINES, PANEL, RUNGS
from .train import DEV_BASE, train_and_eval

REFERENCE = "prefix"
MAX_LIVE = 16
ADVERSARIAL_EVERY = 6
PROMOTE_EVERY = 4
RESEARCH_SPACE = (
    "1 full previous-loop memory", "2 sparse top-k loop retrieval", "3 learned retention/eviction",
    "4 low-rank compressed loop memory", "5 confidence-weighted memory attention",
    "6 error-conditioned recurrence", "7 typed deduce/construct/repair/verify loops",
    "8 hierarchical HRM routing among loop types", "9 LDT-certified pruning of candidates",
    "10 adaptive halting from estimated future utility", "11 direct/damped/RLT answer-state updates",
    "12 BPD-style reusable loop subspaces (update_rank)")

SYSTEM = (
    "You are a research scientist designing Mixture-of-Loops recurrent architectures for tiny models "
    "(Baby Goblin and TRM skill modules). You propose one falsifiable experiment at a time. "
    "Keep standard MoLT memory mechanisms (memory, topk, rank, window, conf_bias, loop_keys) separate from "
    "heterogeneous module routing (routing, n_types, route_hard_eval, slow_period) so effects can be attributed: "
    "change one mechanism family per experiment unless you state a factorial reason. "
    "Every proposal needs a discriminating control config that differs from the candidate in exactly the "
    "mechanism you claim matters. You never see task labels; reason from the aggregate receipts. "
    "Negative results are informative; say what they refute. Reply with reasoning in at most 12 short lines, "
    "then exactly one fenced ```json block.")


def _diff(cfg: dict) -> dict:
    b = BASE.to_dict()
    return {k: v for k, v in cfg.items() if b.get(k) != v}


def _atomic_write(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    # Windows denies the rename while a reader or a sync client holds the target open.
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.5)


class Campaign:
    def __init__(self, out: Path, hours: float, threads: int, use_inky: bool, max_iterations: int = 10_000):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.state_path = out / "state.json"
        self.results_path = out / "runs.jsonl"
        self.threads = threads
        self.max_iterations = max_iterations
        torch.set_num_threads(threads)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state["resumed"] = self.state.get("resumed", 0) + 1
        else:
            now = time.time()
            self.state = {"schema": "molt-goblin-02.campaign.v1", "started_at": now, "deadline": now + hours * 3600,
                          "iteration": 0, "archive": {}, "live": [], "history": [], "adversarial": [],
                          "rejected_adversarial": [], "cache": {}, "promotions": [], "resumed": 0}
            for name, cfg in BASELINES.items():
                self._archive(cfg, origin={"kind": "baseline", "name": name})
        self.inky = inky_mod.Inky(log_path=out / "inky_requests.jsonl") if use_inky and inky_mod.available() else None
        self.state["proposer"] = "inky" if self.inky else "deterministic_fallback"
        self._save()

    # ------------------------------------------------------------------ bookkeeping
    def _save(self):
        self.state["inky_usage"] = self.inky.usage if self.inky else None
        _atomic_write(self.state_path, self.state)

    def _archive(self, cfg: MoltConfig, origin: dict) -> str:
        cid = cfg.digest()
        if cid not in self.state["archive"]:
            self.state["archive"][cid] = {"config": cfg.to_dict(), "diff": _diff(cfg.to_dict()), "origin": origin,
                                          "rung": -1, "scores": {}}
        return cid

    def _name(self, cid: str) -> str:
        o = self.state["archive"][cid]["origin"]
        return o.get("name") or f"c{cid[:6]}"

    def _run(self, cid: str, family: str, seed: int, steps: int) -> dict:
        key = f"{cid}|{family}|{seed}|{steps}"
        if key in self.state["cache"]:
            return self.state["cache"][key]
        cfg = MoltConfig.from_dict({**self.state["archive"][cid]["config"], "steps": steps})
        p = PANEL[family]
        evals = dict(p["eval"])
        for adv in self.state["adversarial"]:
            if adv["family"] == family:
                evals[f"adv_{adv['id']}"] = TaskSpec(**adv["spec"])
        t0 = time.time()
        try:
            r = train_and_eval(cfg, family, p["train"], evals, seed, n_eval=256, base=DEV_BASE,
                               extrap_loops=min(12, 2 * cfg.loops))
            r["wall_s"] = time.time() - t0
            r["status"] = "ok" if r["finite"] else "nonfinite"
        except Exception as exc:  # recorded as a failed intervention, never silently dropped
            r = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "config_digest": cid,
                 "family": family, "seed": seed}
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, **r}) + "\n")
        compact = {"status": r["status"]}
        if r["status"] == "ok":
            ev = r["evals"]
            compact.update({
                "exact": ev["dev"]["exact"], "params": r["params"], "train_cpu_s": round(r["train_cpu_s"], 2),
                "visits": ev["dev"]["mean_block_visits"], "bytes": ev["dev"]["retained_bytes_per_example"],
                "ood": {k: ev[k]["exact"] for k in ev if k.startswith("ood")},
                "adv": {k: ev[k]["exact"] for k in ev if k.startswith("adv_")},
                "interventions": {k.split("|")[1]: ev[k]["exact"] for k in ev if "|" in k},
                "fixes": ev["dev"].get("loop_fixes"), "breaks": ev["dev"].get("loop_breaks"),
                "routing_entropy": ev["dev"].get("routing_entropy")})
        else:
            compact["error"] = r.get("error", r["status"])
        self.state["cache"][key] = compact
        self._save()
        return compact

    def evaluate(self, cid: str, rung: int) -> dict:
        spec = RUNGS[rung]
        fam_scores = {}
        for family in PANEL:
            runs = [self._run(cid, family, s, spec["steps"]) for s in spec["seeds"]]
            refs = [self._run(self._ref_id(), family, s, spec["steps"]) for s in spec["seeds"]]
            ok = [r for r in runs if r["status"] == "ok"]
            if len(ok) < len(runs):
                fam_scores[family] = {"status": "failed", "errors": [r.get("error") for r in runs if r["status"] != "ok"]}
                continue
            mean = lambda xs: sum(xs) / len(xs)
            deltas = [a["exact"] - b["exact"] for a, b in zip(runs, refs) if b["status"] == "ok"]
            oods = [mean(list(r["ood"].values())) for r in runs if r["ood"]]
            ref_oods = [mean(list(r["ood"].values())) for r in refs if r.get("ood")]
            fam_scores[family] = {"exact": mean([r["exact"] for r in runs]), "delta_vs_prefix": mean(deltas) if deltas else None,
                                  "ood": mean(oods) if oods else None,
                                  "ood_delta_vs_prefix": (mean(oods) - mean(ref_oods)) if oods and ref_oods else None,
                                  "visits": mean([r["visits"] for r in runs]), "bytes": mean([r["bytes"] for r in runs]),
                                  "train_cpu_s": mean([r["train_cpu_s"] for r in runs]), "params": runs[0]["params"],
                                  "interventions": runs[0]["interventions"], "seeds": list(spec["seeds"])}
        good = [v for v in fam_scores.values() if v.get("delta_vs_prefix") is not None]
        agg = None
        if len(good) == len(PANEL):
            agg = sum(v["delta_vs_prefix"] + 0.5 * (v["ood_delta_vs_prefix"] or 0.0) for v in good) / len(good)
        summary = {"families": fam_scores, "score": agg,
                   "mean_exact": (sum(v["exact"] for v in good) / len(good)) if good else None}
        a = self.state["archive"][cid]
        a["scores"][str(rung)] = summary
        a["rung"] = max(a["rung"], rung)
        self._save()
        return summary

    def _ref_id(self) -> str:
        return BASELINES[REFERENCE].digest()

    # ------------------------------------------------------------------ prompts
    def _frontier(self, k: int = 8) -> list[dict]:
        rows = []
        for cid, a in self.state["archive"].items():
            if not a["scores"]:
                continue
            r = str(a["rung"])
            s = a["scores"][r]
            rows.append({"id": self._name(cid), "rung": a["rung"], "score": s["score"], "diff": a["diff"],
                         "families": {f: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                          for kk, vv in v.items() if kk in ("exact", "ood", "visits", "bytes", "train_cpu_s", "status")}
                                      for f, v in s["families"].items()}})
        rows.sort(key=lambda x: (x["score"] is not None, x["rung"], x["score"] or -9), reverse=True)
        base_rows = [r for r in rows if not r["id"].startswith("c")]
        cand = [r for r in rows if r["id"].startswith("c")][:k]
        return base_rows + cand

    def _history_text(self, budget_chars: int) -> str:
        out = []
        for h in reversed(self.state["history"]):
            item = {k: h.get(k) for k in ("iteration", "kind", "mechanism", "hypothesis", "candidate", "control",
                                          "predicted", "observed", "verdict", "error")}
            s = json.dumps(item, separators=(",", ":"))
            if sum(len(x) for x in out) + len(s) > budget_chars:
                break
            out.append(s)
        return "\n".join(reversed(out))

    def _context(self) -> str:
        fam = {f: FAMILIES[f].describe() for f in PANEL}
        specs = {f: {"train": PANEL[f]["train"].key(), "evals": {k: v.key() for k, v in PANEL[f]["eval"].items()}} for f in PANEL}
        frontier = json.dumps(self._frontier(), separators=(",", ":"))
        return (f"CONFIG SCHEMA (BASE = {json.dumps(BASE.to_dict(), separators=(',', ':'))}):\n{describe_schema()}\n\n"
                f"DEV FAMILIES (public descriptions):\n{json.dumps(fam, indent=0)}\nPANEL SPECS:{json.dumps(specs)}\n"
                f"Metrics: exact = fraction of examples fully correct (256 dev roots per eval); ood = mean exact on shifted specs "
                f"(ood_* evals also run at 2x loops); visits = loop-module visits per example; bytes = retained memory per example; "
                f"score = mean over families of (exact - prefix baseline) + 0.5*(ood - prefix ood).\n"
                f"Research space: {'; '.join(RESEARCH_SPACE)}.\n\nFRONTIER:\n{frontier}\n")

    def _proposal_prompt(self) -> str:
        ctx = self._context()
        hist = self._history_text(max(2000, 22000 - len(ctx)))
        return (ctx + f"\nRECENT EXPERIMENTS (oldest first, includes failures):\n{hist}\n\n"
                "TASK: First diagnose the most recent experiment in one or two lines and state how it revises your "
                "hypothesis. Then propose the next experiment. JSON keys: "
                '{"diagnosis": str, "revision": str, "hypothesis": str, "mechanism": "<one research-space item number>", '
                '"base": "<frontier id to start from>", "candidate": {<config keys to change>}, '
                '"control": {<config keys to change from base for the discriminating control>}, '
                '"predicted": {"<family>": "up|down|flat", ...}, "rationale": str}. '
                "Do not repeat a tested configuration. Keep width<=48 and loops<=6 so a run fits the CPU budget.")

    def _adversarial_prompt(self) -> str:
        ctx = self._context()
        hist = self._history_text(6000)
        return (ctx + f"\nRECENT EXPERIMENTS:\n{hist}\n\n"
                "TASK: Name the two leading explanations of the frontier results and invent ONE task variant of a DEV "
                "family that should separate them (a matched variant where one explanation predicts a gain and the "
                "other predicts none). Spec fields: family, length, depth, distract, variant. Allowed variants: "
                "pointer_chase clean|shuffled; revision clean|reordered; repair flagged|unflagged; proof_deps clean. "
                "Total tokens must stay <= 96. JSON keys: "
                '{"explanations": [str, str], "spec": {"family":..., "length":..., "depth":..., "distract":..., "variant":...}, '
                '"prediction": str}.')

    # ------------------------------------------------------------------ proposals
    def _fallback_proposal(self) -> dict:
        grid = [("1", {"memory": "full", "loss": "all"}, {"memory": "prev", "loss": "all"}),
                ("2", {"memory": "topk", "topk": 4}, {"memory": "full"}),
                ("3", {"memory": "retention"}, {"memory": "full"}),
                ("4", {"memory": "lowrank", "rank": 4}, {"memory": "full"}),
                ("5", {"memory": "full", "conf_bias": True}, {"memory": "full"}),
                ("6", {"memory": "prev", "error_feedback": True}, {"memory": "prev"}),
                ("7", {"routing": "cycle", "n_types": 4}, {}),
                ("8", {"routing": "hrm", "n_types": 4}, {"routing": "mix", "n_types": 4}),
                ("9", {"ldt_mask": True}, {}),
                ("10", {"halt": "utility", "loss": "all"}, {"loss": "all"}),
                ("11", {"update": "damped"}, {}),
                ("12", {"update_rank": 4}, {})]
        i = len([h for h in self.state["history"] if h.get("kind") == "proposal"]) % len(grid)
        m, cand, ctl = grid[i]
        return {"hypothesis": f"deterministic factorial cell for research item {m}", "mechanism": m, "base": REFERENCE,
                "candidate": cand, "control": ctl, "predicted": {}, "rationale": "fallback grid (no Inky)",
                "diagnosis": None, "revision": None}

    def _resolve_base(self, name) -> dict:
        name = str(name or REFERENCE)
        for cid, a in self.state["archive"].items():
            if self._name(cid) == name or cid.startswith(name.lstrip("c")) and name.startswith("c"):
                return dict(a["config"])
        if name in BASELINES:
            return BASELINES[name].to_dict()
        return BASELINES[REFERENCE].to_dict()

    def _compile(self, base: dict, change: dict) -> MoltConfig:
        if not isinstance(change, dict):
            raise ValueError("config change must be an object")
        merged = {**base, **change}
        if merged.get("routing") == "single":
            merged["n_types"] = 1
        elif merged.get("n_types", 1) == 1 and merged.get("routing") in ("cycle", "mix", "hrm"):
            merged["n_types"] = 4
        merged["steps"] = BASE.steps
        cfg = MoltConfig.from_dict(merged)
        if cfg.width > 48 or cfg.loops > 6:
            raise ValueError("width<=48 and loops<=6 under the CPU budget")
        return cfg

    # ------------------------------------------------------------------ main loop
    def step(self) -> None:
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
                    rec["truncated_first_reply"] = {k: reply.get(k) for k in ("usage", "effort", "max_completion_tokens")}
                    reply = self.inky.ask(SYSTEM, prompt, "proposal_retry_low_effort", effort="low")
            except ValueError as exc:   # context guard
                reply = {"ok": False, "error_type": f"ContextGuard: {exc}"[:200], "http_status": None}
            rec["request"] = {k: reply.get(k) for k in ("ok", "seconds", "usage", "finish_reason", "error_type", "http_status")}
            if reply["ok"]:
                rec["reply_text"] = reply["text"][-4000:]
                prop = inky_mod.extract_json(reply["text"])
                if prop is None:
                    rec["error"] = "no JSON object in reply"
            else:
                rec["error"] = f"request failed: {reply.get('error_type')} {reply.get('http_status')}"
        else:
            prop = self._fallback_proposal()
        if prop is not None:
            rec.update({k: prop.get(k) for k in ("diagnosis", "revision", "hypothesis", "mechanism", "base",
                                                  "candidate", "control", "predicted", "rationale")})
            try:
                base = self._resolve_base(prop.get("base"))
                cand = self._compile(base, prop.get("candidate") or {})
                ctl = self._compile(base, prop.get("control") or {})
                cid = cand.digest()
                if cid in self.state["archive"] and self.state["archive"][cid]["scores"]:
                    rec["error"] = f"duplicate of already-tested {self._name(cid)}"
                    rec["duplicate_of"] = self._name(cid)
                else:
                    cid = self._archive(cand, {"kind": "candidate", "iteration": it, "mechanism": prop.get("mechanism")})
                    kid = self._archive(ctl, {"kind": "control", "iteration": it, "of": cid})
                    s_c = self.evaluate(cid, 0)
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
    def _observed(s_c, s_k) -> dict:
        obs = {}
        for f, v in s_c["families"].items():
            k = s_k["families"].get(f, {})
            obs[f] = {"cand_exact": _r(v.get("exact")), "cand_minus_prefix": _r(v.get("delta_vs_prefix")),
                      "cand_minus_control": _r(v["exact"] - k["exact"]) if "exact" in v and "exact" in k else None,
                      "cand_ood": _r(v.get("ood")), "visits": _r(v.get("visits")), "bytes": _r(v.get("bytes")),
                      "status": v.get("status", "ok")}
        obs["score"] = _r(s_c["score"])
        return obs

    @staticmethod
    def _verdict(pred: dict, s_c) -> dict:
        out = {}
        for f, d in pred.items():
            v = s_c["families"].get(f, {})
            dv = v.get("delta_vs_prefix")
            if dv is None or not isinstance(d, str):
                continue
            obs = "up" if dv > 0.03 else "down" if dv < -0.03 else "flat"
            out[f] = {"predicted": d, "observed": obs, "match": d.lower().startswith(obs)}
        return out

    def _adversarial(self, it: int) -> None:
        try:
            prompt = self._adversarial_prompt()
            reply = self.inky.ask(SYSTEM, prompt, "adversarial_task")
            if reply["ok"] and reply.get("finish_reason") == "length" and inky_mod.extract_json(reply["text"]) is None:
                reply = self.inky.ask(SYSTEM, prompt, "adversarial_task_retry_low_effort", effort="low")
        except ValueError as exc:
            reply = {"ok": False, "error_type": f"ContextGuard: {exc}"[:200]}
        rec = {"iteration": it, "kind": "adversarial_task", "t": time.time(),
               "request": {k: reply.get(k) for k in ("ok", "seconds", "usage", "error_type", "http_status")}}
        if reply["ok"]:
            prop = inky_mod.extract_json(reply["text"]) or {}
            rec.update({"explanations": prop.get("explanations"), "prediction": prop.get("prediction"),
                        "spec": prop.get("spec")})
            try:
                sp = prop.get("spec") or {}
                spec = TaskSpec(family=str(sp["family"]), length=int(sp["length"]), depth=int(sp.get("depth", 0)),
                                distract=int(sp.get("distract", 0)), variant=str(sp.get("variant", "clean")))
                if spec.family not in DEV_FAMILIES or spec.family not in PANEL:
                    raise ValueError("variant must come from a panel development family")
                FAMILIES[spec.family].validate(spec)
                v = verify_spec(spec, n=128, seed=424242 + it)
                rec["verification"] = v
                if not v["ok"]:
                    raise ValueError("verification failed: " + "; ".join(v["reasons"])[:200])
                adv = {"id": f"a{len(self.state['adversarial'])}", "family": spec.family,
                       "spec": {"family": spec.family, "length": spec.length, "depth": spec.depth,
                                "distract": spec.distract, "variant": spec.variant}, "iteration": it,
                       "explanations": prop.get("explanations"), "prediction": prop.get("prediction")}
                self.state["adversarial"].append(adv)
                rec["accepted"] = adv["id"]
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                self.state["rejected_adversarial"].append(rec)
        else:
            rec["error"] = f"request failed: {reply.get('error_type')}"
        self.state["history"].append(rec)
        self._save()

    def _promote(self) -> None:
        arch = self.state["archive"]
        for rung, cap in ((0, 2), (1, 1)):
            pool = [cid for cid in self.state["live"] if arch[cid]["rung"] == rung and arch[cid]["scores"].get(str(rung), {}).get("score") is not None]
            pool.sort(key=lambda c: arch[c]["scores"][str(rung)]["score"], reverse=True)
            n = min(cap, max(1, len(pool) // 3)) if pool else 0
            for cid in pool[:n]:
                if time.time() > self.state["deadline"]:
                    return
                s = self.evaluate(cid, rung + 1)
                self.state["promotions"].append({"id": self._name(cid), "from": rung, "to": rung + 1, "score": s["score"],
                                                 "iteration": self.state["iteration"]})
        # bounded population: drop the weakest rung-0 members beyond MAX_LIVE
        live = self.state["live"]
        if len(live) > MAX_LIVE:
            live.sort(key=lambda c: (arch[c]["rung"], arch[c]["scores"].get(str(arch[c]["rung"]), {}).get("score") or -9), reverse=True)
            self.state["live"] = live[:MAX_LIVE]

    def run(self) -> None:
        # Reference and baseline arms first, at rung 0, so every candidate has a matched comparison.
        for name, cfg in BASELINES.items():
            self.evaluate(cfg.digest(), 0)
        while time.time() < self.state["deadline"] and self.state["iteration"] < self.max_iterations:
            self.step()
        # Baselines also enter the top rung so finalists compare at equal seeds.
        for name in ("prefix", "tied", "prev"):
            if time.time() < self.state["deadline"] + 1800:
                self.evaluate(BASELINES[name].digest(), len(RUNGS) - 1)
        self.state["finished_at"] = time.time()
        self._save()


def _r(x):
    return None if x is None else round(float(x), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=5.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-iterations", type=int, default=10_000)
    ap.add_argument("--no-inky", action="store_true")
    ap.add_argument("--ram-gib", type=float, default=3.0)
    a = ap.parse_args()
    cap = resources.install_cap(a.ram_gib)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "resource_cap.json").write_text(json.dumps(cap, indent=1), encoding="utf-8")
    Campaign(a.out, a.hours, a.threads, not a.no_inky, a.max_iterations).run()


if __name__ == "__main__":
    main()
