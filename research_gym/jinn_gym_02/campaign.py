"""JINN-GYM-02 campaign: utility-constrained persona evolution and compositional routing.

    python -m research_gym.jinn_gym_02.campaign --out DIR --hours 8 --key-file PATH
    python -m research_gym.jinn_gym_02.campaign --out DIR --hours 1 --dry-run --small

Phases: BUILD (verifier-backed task banks) -> INIT (GYM-01 frozen personas as roots) -> MUTATE
(matched mutation arms, utility-constrained selection on fresh selection batches) -> RECOMB ->
FREEZE (boundary audit) -> ROUTER (TRM, baselines, self-models, Inky self-predictions) -> HELDOUT
(untouched family) -> COMPACT -> REPORT. Inky calls go through one worker thread (one request in
flight) that also prefetches neutral priors. Resumable from state.json and the JSONL logs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research_gym.jinn_gym_01 import host, metrics
from research_gym.jinn_gym_01.export import append_jsonl, code_hashes as code_hashes01, digest_of
from research_gym.jinn_gym_01.inky import MODEL, CredentialRejected, InkyClient, Usage, load_key
from research_gym.jinn_gym_01.module import decide, host_flags, prior_from_response
from research_gym.jinn_gym_01.prompts import extract_json, parse_decision, decision_prompt, persona_card
from research_gym.jinn_gym_01.runner import RamGuard, write_json_atomic
from research_gym.jinn_gym_01.schema import (MAX_PATCH_OPS, SpecError, apply_patch, canonical, spec_hash, validate)
from research_gym.jinn_gym_01.tags import (ABSTAIN, BOUNDARY_TAGS, COMPOSITION_PAIRS, PERSONA_BRIEFS, PERSONA_TAGS,
                                           PERSONAS, SIGNATURES)

from . import compact as K
from . import domains as D
from . import evolve as E
from . import router as R
from . import verifiers as V

CFG = {
    "dev_per_domain": 20, "sel_per_domain": 200, "held_per_domain": 60,
    "panel_per_domain": 15, "recomb_per_domain": 20, "batch_per_domain": 10,
    "mutation_end_fraction": 0.62, "report_margin_s": 900,
    "decision_max_tokens": 1200, "patch_max_tokens": 2500, "aux_max_tokens": 1200,
    "decision_temperature": 0.0, "patch_temperature": 0.8, "reasoning_effort": "low", "context_limit": 12_288,
    "ram_cap_bytes": 2 * 1024 ** 3, "python_build_workers": 4, "rerun_check_fraction": 0.05,
    "trm_hidden": 64, "trm_steps": 4, "trm_epochs": 400, "seed": 20261128,
}
SMALL = {"dev_per_domain": 3, "sel_per_domain": 26, "held_per_domain": 6, "panel_per_domain": 3,
         "recomb_per_domain": 3, "batch_per_domain": 4, "trm_epochs": 60}
SPLIT_PRIORITY = {0: 1, 2: 1, 1: 3}


class Stop(Exception):
    pass


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# =========================================================================== Inky worker
class InkyWorker(threading.Thread):
    """One request in flight. Priority queue: patches and routing before prefetch."""

    def __init__(self, client: InkyClient, out: Path, cache: dict):
        super().__init__(daemon=True)
        self.client, self.out, self.cache = client, out, cache
        self.q: queue.PriorityQueue = queue.PriorityQueue()
        self.lock = threading.Lock()
        self.pending: dict[str, cf.Future] = {}
        self.seq = itertools.count()
        self.fatal: BaseException | None = None
        self.stop_flag = threading.Event()

    def submit(self, key: str, prompt: str, max_tokens: int, purpose: str, temperature: float, priority: int) -> cf.Future:
        with self.lock:
            if key in self.cache:
                f = cf.Future()
                f.set_result(self.cache[key])
                return f
            if key in self.pending:
                return self.pending[key]
            f = cf.Future()
            self.pending[key] = f
        self.q.put((priority, next(self.seq), key, prompt, max_tokens, purpose, temperature))
        return f

    def run(self):
        while not self.stop_flag.is_set():
            try:
                pr, _, key, prompt, mt, purpose, temp = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            fut = self.pending.get(key)
            if key in self.cache:
                fut.set_result(self.cache[key])
                continue
            try:
                r = self.client.complete(prompt, max_tokens=mt, purpose=purpose, temperature=temp)
                entry = {"key": key, "purpose": purpose, "text": r["text"], "finish_reason": r["finish_reason"],
                         "response_model": r["response_model"], "prompt_tokens": r["prompt_tokens"],
                         "completion_tokens": r["completion_tokens"], "seconds": r["seconds"],
                         "attempts": r["attempts"], "t": time.time()}
                with self.lock:
                    append_jsonl(self.out / "responses.jsonl", entry)
                    self.cache[key] = entry
                fut.set_result(entry)
            except CredentialRejected as exc:
                self.fatal = exc
                fut.set_exception(exc)
                self.stop_flag.set()
            except Exception as exc:  # attempts exhausted or stop requested: empty response
                fut.set_result({"key": key, "text": "", "failed": True, "error": str(exc)[:80]})
            finally:
                with self.lock:
                    self.pending.pop(key, None)


# =========================================================================== campaign
class Campaign:
    def __init__(self, out: Path, hours: float, client: InkyClient, dry_run: bool, small: bool = False):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("bank", "frozen", "modules", "export", "lean_tmp"):
            (out / sub).mkdir(exist_ok=True)
        self.cfg = dict(CFG)
        if small:
            self.cfg.update(SMALL)
        self.dry_run = dry_run
        self.ram = RamGuard(self.cfg["ram_cap_bytes"])
        self._signal = None
        sp = out / "state.json"
        if sp.exists():
            self.state = json.loads(sp.read_text(encoding="utf-8"))
            self.state["resumes"] += 1
            self.state["stop_reason"] = None
            for k, v in (self.state.get("usage") or {}).items():
                if hasattr(client.usage, k):
                    setattr(client.usage, k, v)
        else:
            now = time.time()
            self.state = {"schema": "jinn-gym-02.state.v1", "started_at": now, "deadline": now + hours * 3600,
                          "hours": hours, "phase": "BUILD", "round": 0, "lineages": {}, "frozen": False,
                          "resumes": 0, "stop_reason": None, "dry_run": dry_run, "small": small,
                          "host_code_hash_start": host.host_code_hash(), "cpu_seconds": {}}
        self.client = client
        client.stop_flag = lambda: self._signal is not None or (out / "STOP").exists()
        self.cache = {e["key"]: e for e in read_jsonl(out / "responses.jsonl")}
        self.worker = InkyWorker(client, out, self.cache)
        self.lean = V.find_lean()
        self.hashes = code_hashes02()
        if not (out / "manifest.json").exists():
            write_json_atomic(out / "manifest.json", {
                "experiment": "JINN-GYM-02", "model": MODEL, "dry_run": dry_run, "small": small, "config": self.cfg,
                "hours": hours, "code_hashes": self.hashes, "code_digest": digest_of(self.hashes),
                "host_code_hash": self.state["host_code_hash_start"], "lean": self.lean,
                "lean_version": V.lean_version(self.lean) if self.lean else None, "paid_fallback": False,
                "concurrency": 1, "weights_frozen": True, "python": sys.version.split()[0]})
        self.tasks: dict[str, dict] = {"dev": {}, "sel": {}, "held": {}}
        self.factory = None

    # ------------------------------------------------------------------ infra
    def save(self):
        self.state["usage"] = self.client.usage.as_dict()
        self.state["ram_peak_increment_bytes"] = self.ram.peak_increment
        write_json_atomic(self.out / "state.json", self.state)

    def check(self):
        if self._signal:
            raise Stop(f"signal_{self._signal}")
        if (self.out / "STOP").exists():
            raise Stop("stop_file")
        if time.time() >= self.state["deadline"]:
            raise Stop("deadline")
        if self.ram.exceeded():
            raise Stop("ram_cap")
        if self.worker.fatal:
            raise self.worker.fatal

    def timed(self, name, t0):
        self.state["cpu_seconds"][name] = self.state["cpu_seconds"].get(name, 0.0) + (time.time() - t0)

    def wait(self, fut: cf.Future) -> dict:
        while True:
            self.check()
            try:
                return fut.result(timeout=1.0)
            except cf.TimeoutError:
                continue

    def neutral_future(self, task, priority=1):
        return self.worker.submit(f"neutral|{task.task_id}", decision_prompt(task), self.cfg["decision_max_tokens"],
                                  "decision_neutral", self.cfg["decision_temperature"], priority)

    def neutral(self, task) -> tuple[dict, dict]:
        e = self.wait(self.neutral_future(task))
        parsed = parse_decision(task, e["text"])
        return parsed, prior_from_response(task, parsed if parsed["parsed"] else None)

    # ------------------------------------------------------------------ BUILD
    def split_specs(self):
        c = self.cfg
        for fam, name, n in ((0, "dev", c["dev_per_domain"]), (2, "sel", c["sel_per_domain"]), (1, "held", c["held_per_domain"])):
            yield fam, name, n

    def load_banks(self):
        for _, name, _ in self.split_specs():
            for d in read_jsonl(self.out / "bank" / f"{name}.jsonl"):
                t = D.task_from_dict(d)
                self.tasks[name][t.task_id] = t

    def build(self):
        t0 = time.time()
        self.load_banks()
        lock = threading.Lock()
        reruns = []

        def store(name, t):
            with lock:
                append_jsonl(self.out / "bank" / f"{name}.jsonl", D.task_to_dict(t))
                self.tasks[name][t.task_id] = t
            self.neutral_future(t, SPLIT_PRIORITY[t.family])

        jobs = []
        for fam, name, n in self.split_specs():
            for dom in D.DOMAINS:
                for i in range(n):
                    tid = f"{dom}-f{fam}-{D.SPLIT_OF_FAMILY[fam]}-{i:04d}"
                    if tid in self.tasks[name]:
                        self.neutral_future(self.tasks[name][tid], SPLIT_PRIORITY[fam])
                    else:
                        jobs.append((fam, name, dom, i))
        lean_jobs = [j for j in jobs if j[2] == "lean_proof"]
        other = [j for j in jobs if j[2] != "lean_proof"]

        def run(j):
            fam, name, dom, i = j
            t = D.make_task(dom, fam, i, self.lean, self.out / "lean_tmp")
            if not D.verify_task(t):
                raise RuntimeError(f"verification failed for {t.task_id}")
            store(name, t)
            return t.task_id

        def lean_loop():
            for j in lean_jobs:
                if self._signal or (self.out / "STOP").exists():
                    return
                run(j)
        th = threading.Thread(target=lean_loop)
        th.start()
        with cf.ThreadPoolExecutor(self.cfg["python_build_workers"]) as ex:
            for f in cf.as_completed([ex.submit(run, j) for j in other]):
                f.result()
        while th.is_alive():
            th.join(timeout=1.0)
            self.check()
        # rerun spot checks: rebuild a seeded sample from scratch and compare utilities
        rng = random.Random(self.cfg["seed"])
        allt = [t for name in self.tasks for t in self.tasks[name].values()]
        sample = rng.sample(allt, max(3, int(self.cfg["rerun_check_fraction"] * len(allt))))
        ok = sum(D.rerun_check(t, self.lean, self.out / "lean_tmp") for t in sample)
        self.state["rerun_check"] = {"n": len(sample), "agree": ok}
        self.state["bank_sizes"] = {k: len(v) for k, v in self.tasks.items()}
        self.timed("build", t0)
        self.state["phase"] = "INIT"
        self.save()

    # ------------------------------------------------------------------ splits of the selection family
    def sel_lists(self):
        c = self.cfg
        per = defaultdict(list)
        for t in sorted(self.tasks["sel"].values(), key=lambda t: t.task_id):
            per[t.domain].append(t)
        panel, recomb, batches = [], [], []
        pools = {}
        for d in D.DOMAINS:
            L = per[d]
            panel += L[: c["panel_per_domain"]]
            recomb += L[c["panel_per_domain"]: c["panel_per_domain"] + c["recomb_per_domain"]]
            pools[d] = L[c["panel_per_domain"] + c["recomb_per_domain"]:]
        n_rounds = min(len(v) for v in pools.values()) // c["batch_per_domain"]
        for r in range(n_rounds):
            b = []
            for d in D.DOMAINS:
                b += pools[d][r * c["batch_per_domain"]: (r + 1) * c["batch_per_domain"]]
            batches.append(b)
        return panel, recomb, batches

    # ------------------------------------------------------------------ INIT
    def roots(self) -> dict:
        root = Path(__file__).resolve().parents[2] / "experiments" / "jinn_gym_01" / "outputs" / "run_20261128" / "frozen"
        out = {}
        for p in PERSONAS:
            spec = json.loads((root / f"{p}.json").read_text(encoding="utf-8"))["spec"]
            errs = validate(spec, p)
            if errs:
                raise RuntimeError(f"GYM-01 frozen spec for {p} fails validation: {errs[:2]}")
            out[p] = spec
        return out

    def init(self):
        self.factory = self.factory or E.ModuleFactory(list(self.tasks["dev"].values()), self.out / "modules")
        roots = self.roots()
        for p, spec in roots.items():
            h = spec_hash(spec)
            (self.out / "modules" / "specs").mkdir(exist_ok=True)
            write_json_atomic(self.out / "modules" / "specs" / f"{h}.json", spec)
            self.state["lineages"].setdefault(p, {a: h for a in E.ARMS})
            self.state.setdefault("roots", {})[p] = h
        self.state["phase"] = "MUTATE"
        self.save()

    def spec_of(self, h: str) -> dict:
        return json.loads((self.out / "modules" / "specs" / f"{h}.json").read_text(encoding="utf-8"))

    def put_spec(self, spec: dict) -> str:
        h = spec_hash(spec)
        p = self.out / "modules" / "specs" / f"{h}.json"
        if not p.exists():
            write_json_atomic(p, spec)
        return h

    # ------------------------------------------------------------------ evaluation helpers
    def priors_for(self, tasks):
        out = {}
        for t in tasks:
            parsed, prior = self.neutral(t)
            out[t.task_id] = (parsed, prior)
        return out

    def rows(self, mod, persona, tasks, priors, arm="D"):
        return [E.module_row(t, persona, arm, priors[t.task_id][1], [mod["learned"]]) for t in tasks]

    def nrows(self, persona, tasks, priors):
        return [E.neutral_row(t, persona, priors[t.task_id][0]) for t in tasks]

    def archive_item(self, persona, mod, source, panel, priors, extra=None) -> dict:
        pr = self.rows(mod, persona, panel, priors)
        st = E.batch_stats(pr, self.nrows(persona, panel, priors))
        item = {"persona": persona, "hash": mod["hash"], "spec_hash": mod["spec_hash"], "source": source,
                "panel": st, "head_flat": K.flat(mod["head"]).round(6).tolist(), **(extra or {})}
        append_jsonl(self.out / "archive.jsonl", item)
        return item

    # ------------------------------------------------------------------ MUTATE
    def patch_prompt(self, persona, spec, diag, examples) -> str:
        ops_doc = ('{"op":"set_weight","feature":NAME,"weight":X} | {"op":"add_tag","feature":NAME,"tag":TAG,"sign":"+" or "-"} | '
                   '{"op":"remove_tag","tag":TAG} | {"op":"set_abstain_threshold","value":X} | '
                   '{"op":"set_abstain_if_no_ranked_action","value":BOOL} | '
                   '{"op":"set_evidence","field":"min_evidence_items"|"abstain_on_missing_inputs"|"require_failure_log_for_repair","value":V} | '
                   '{"op":"add_transition","from":PHASE,"to":PHASE} | {"op":"remove_transition","from":PHASE,"to":PHASE} | '
                   '{"op":"add_critical","tag":TAG} | {"op":"remove_critical","tag":TAG} | {"op":"set_ranking","ranking":[TAGS]} | '
                   '{"op":"add_feature","name":NAME,"weight":X}')
        st = diag["stats"]
        return "\n\n".join([
            f"You maintain the formal specification of the '{persona}' research persona ({PERSONA_BRIEFS[persona]}). "
            "An independent verifier (Lean proof checks, Python unit tests, simulated experiments) scored it on a batch of research tasks. "
            "Propose a MINIMAL patch.",
            "SELECTION RULE: a patch is accepted only if the persona shows its distinctive behavior more often AND its verified "
            f"utility stays within {E.MARGIN:.2f} of neutral Inky on the same tasks. If utility is already more than "
            f"{E.MARGIN:.2f} below neutral, a patch is accepted when it closes that gap.",
            f"BATCH RESULT: persona fidelity {st['fidelity']:.2f}; normalized utility {st['utility']:.3f} vs neutral "
            f"{st['utility_neutral']:.3f} (gap {st['gap']:+.3f}); abstain rate {st['abstain_rate']:.2f}.",
            "ABSTENTION-AWARE DIAGNOSTICS (counts): " + canonical(diag["codes"]) +
            ". justified_abstention = abstained when abstaining was best; excessive_caution = abstained when acting was better; "
            "unsupported_action = acted when abstaining was best; successful_recovery / failed_recovery = after a reported failure, "
            "picked / missed the best action. Abstain reasons: " + canonical(diag["abstain_reasons"]),
            "EXAMPLES:\n" + "\n".join(canonical(x) for x in examples[:3]),
            "CURRENT SPECIFICATION:\n" + canonical(spec),
            f"Allowed operations (at most {MAX_PATCH_OPS}):\n" + ops_doc,
            "Allowed tags: " + ", ".join(PERSONA_TAGS) + ". Host-boundary tags (" + ", ".join(BOUNDARY_TAGS)
            + ") may never be scored positively, and authorization is not part of the persona.",
            'Return only JSON: {"ops": [...], "rationale": "at most 30 words"}.'])

    def mutate(self):
        self.factory = self.factory or E.ModuleFactory(list(self.tasks["dev"].values()), self.out / "modules")
        panel, recomb, batches = self.sel_lists()
        end = self.state["started_at"] + self.cfg["mutation_end_fraction"] * self.state["hours"] * 3600
        self.state["mutation_rounds_available"] = len(batches)
        selector = E.LearnedSelector()
        for m in read_jsonl(self.out / "mutations.jsonl"):
            if m.get("features") is not None:
                selector.add(np.array(m["features"]), m["accepted"])
        panel_priors = self.priors_for(panel)
        archived = {(a["persona"], a["hash"]) for a in read_jsonl(self.out / "archive.jsonl")}
        while self.state["round"] < len(batches) and time.time() < end:
            t0 = time.time()
            r = self.state["round"]
            batch = batches[r]
            bp = self.priors_for(batch)
            selector.fit()
            rng = random.Random(E._seed("round", r, self.cfg["seed"]))
            # Inky arm: compute parent diagnostics first and queue the patch requests
            parents, futs = {}, {}
            for p in PERSONAS:
                for arm in E.ARMS:
                    h = self.state["lineages"][p][arm]
                    mod = self.factory.build(p, self.spec_of(h))
                    rows = self.rows(mod, p, batch, bp)
                    nr = self.nrows(p, batch, bp)
                    diag = E.batch_diagnostics(rows, nr)
                    parents[(p, arm)] = (mod, rows, diag)
                    if (p, mod["hash"]) not in archived:
                        self.archive_item(p, mod, "root" if r == 0 and h == self.state["roots"][p] else "lineage", panel, panel_priors)
                        archived.add((p, mod["hash"]))
                    if arm == "inky":
                        ex = [{"domain": x["domain"], "executed_tags": x["executed_tags"], "utility": round(x["utility"], 3),
                               "best": round(x["best_utility"], 3), "codes": E.task_codes(x)} for x in rows if E.task_codes(x)][:3]
                        futs[p] = self.worker.submit(f"patch02|r{r}|{p}|{h}", self.patch_prompt(p, mod["spec"], diag, ex),
                                                     self.cfg["patch_max_tokens"], "mutation_patch", self.cfg["patch_temperature"], 0)
            for p in PERSONAS:
                for arm in E.ARMS:
                    self.check()
                    mod, prow, diag = parents[(p, arm)]
                    rule, text = None, None
                    if arm == "random":
                        ops = E.random_mutation(mod["spec"], rng, p)
                    elif arm == "abstention":
                        ops, rule = E.abstention_repair(mod["spec"], diag)
                    elif arm == "heuristic":
                        ops = E.heuristic_mutation(mod["spec"], p, r)
                    elif arm == "inky":
                        e = self.wait(futs[p])
                        text = e["text"]
                        obj = extract_json(text)
                        ops = obj.get("ops") if isinstance(obj, dict) else None
                    else:
                        cands = [E.random_mutation(mod["spec"], random.Random(rng.random()), p) for _ in range(10)]
                        cands += [E.abstention_repair(mod["spec"], diag)[0], E.heuristic_mutation(mod["spec"], p, r)]
                        cands = [c for c in cands if c]
                        ops = max(cands, key=lambda c: selector.score(E.op_features(c, diag))) if cands else None
                    self.evaluate_candidate(r, p, arm, mod, prow, diag, ops, rule, text, batch, bp, panel, panel_priors,
                                            selector, archived)
            self.state["round"] = r + 1
            self.timed("mutate", t0)
            self.save()
        self.state["phase"] = "RECOMB"
        self.save()

    def evaluate_candidate(self, r, p, arm, mod, prow, diag, ops, rule, text, batch, bp, panel, panel_priors, selector, archived):
        nr = self.nrows(p, batch, bp)
        parent = E.batch_stats(prow, nr)
        rec = {"round": r, "persona": p, "arm": arm, "parent": mod["hash"], "ops": ops, "rule": rule,
               "diagnostics": diag, "parent_stats": parent, "batch": [t.task_id for t in batch],
               "inky_text": (text or "")[:3000] if arm == "inky" else None}
        failure, child = None, None
        if not ops:
            failure = "no_candidate"
        else:
            try:
                cs = apply_patch(mod["spec"], ops)
                errs = validate(cs, p)
                if errs:
                    failure = "invalid: " + "; ".join(errs[:2])
                elif spec_hash(cs) == mod["spec_hash"]:
                    failure = "no_change"
                else:
                    child = cs
            except SpecError as exc:
                failure = f"invalid: {exc}"
        rec["features"] = E.op_features(ops, diag).tolist() if ops else None
        if child is not None:
            ch = self.put_spec(child)
            cm = self.factory.build(p, child)
            crow = self.rows(cm, p, batch, bp)
            cst = E.batch_stats(crow, nr)
            audit = host.boundary_audit(batch, {"child": lambda t: decide(t, bp[t.task_id][1], [cm["learned"]])})
            ok, why = E.accept(parent, cst)
            if not audit["passed"]:
                ok, why = False, "boundary_audit_failed"
            rec.update(child=ch, child_stats=cst, accepted=ok, reason=why, boundary_audit=audit)
            if (p, cm["hash"]) not in archived:
                self.archive_item(p, cm, f"mutation:{arm}", panel, panel_priors, {"round": r})
                archived.add((p, cm["hash"]))
            if ok:
                self.state["lineages"][p][arm] = ch
        else:
            rec.update(child=None, accepted=False, reason=failure)
        if rec["features"] is not None:
            selector.add(np.array(rec["features"]), rec["accepted"])
        append_jsonl(self.out / "mutations.jsonl", rec)

    # ------------------------------------------------------------------ RECOMB
    def best_variant(self, p) -> dict:
        items = [a for a in read_jsonl(self.out / "archive.jsonl") if a["persona"] == p]
        return E.freeze_choice(items)

    def module_for_item(self, item) -> dict:
        spec = self.spec_of(item["spec_hash"])
        ho = item.get("head_override")
        return self.factory.build(item["persona"], spec, ho)

    def recombine(self):
        t0 = time.time()
        self.factory = self.factory or E.ModuleFactory(list(self.tasks["dev"].values()), self.out / "modules")
        panel, recomb, _ = self.sel_lists()
        pp = self.priors_for(panel)
        rp = self.priors_for(recomb)
        done = {(x["recipient"], x["donor"], x["operator"]) for x in read_jsonl(self.out / "recombination.jsonl")}
        best = {p: self.best_variant(p) for p in PERSONAS}
        mods = {p: self.module_for_item(best[p]) for p in PERSONAS}
        pairs = [(a, b) for a, b in COMPOSITION_PAIRS] + [(b, a) for a, b in COMPOSITION_PAIRS]
        for rec_p, don_p in pairs:
            for op in ("feature", "ranking", "params"):
                if (rec_p, don_p, op) in done:
                    continue
                self.check()
                rm, dm = mods[rec_p], mods[don_p]
                if op == "params":
                    ho = E.recombine_params(rm["head"], dm["head"], don_p)
                    cm = self.factory.build(rec_p, rm["spec"], ho)
                else:
                    cs = (E.recombine_feature if op == "feature" else E.recombine_ranking)(rm["spec"], dm["spec"], rec_p)
                    if cs is None:
                        append_jsonl(self.out / "recombination.jsonl", {"recipient": rec_p, "donor": don_p, "operator": op, "valid": False})
                        continue
                    self.put_spec(cs)
                    cm = self.factory.build(rec_p, cs)
                    ho = None
                rows = {}
                for who, m in (("child", cm), ("recipient", rm), ("donor", dm)):
                    rows[who + "_own"] = self.rows(m, rec_p, recomb, rp)
                    rows[who + "_donor"] = self.rows(m, don_p, recomb, rp)
                S = {k: metrics.summarize(v) for k, v in rows.items()}
                f = lambda k: S[k]["fidelity"] or 0.0  # noqa: E731
                u = lambda k: S[k]["normalized_utility"]  # noqa: E731
                item = self.archive_item(rec_p, cm, f"recomb:{op}:{don_p}", panel, pp,
                                         {"head_override": ho, "donor": don_p} if ho else {"donor": don_p})
                x = {"recipient": rec_p, "donor": don_p, "operator": op, "valid": True, "child": cm["hash"],
                     "transfer": f("child_donor") - f("recipient_donor"),
                     "interference": f("recipient_own") - f("child_own"),
                     "complementarity": u("child_own") - max(u("recipient_own"), u("donor_own")),
                     "child_fid_own": f("child_own"), "child_fid_donor": f("child_donor"),
                     "recipient_fid_own": f("recipient_own"), "recipient_fid_donor": f("recipient_donor"),
                     "donor_fid_donor": f("donor_donor"), "utilities": {k: u(k) for k in ("child_own", "recipient_own", "donor_own")},
                     "panel": item["panel"]}
                append_jsonl(self.out / "recombination.jsonl", x)
        self.timed("recomb", t0)
        self.state["phase"] = "FREEZE"
        self.save()

    # ------------------------------------------------------------------ FREEZE
    def frozen_modules(self) -> dict:
        self.factory = self.factory or E.ModuleFactory(list(self.tasks["dev"].values()), self.out / "modules")
        out = {}
        for p in PERSONAS:
            fz = json.loads((self.out / "frozen" / f"{p}.json").read_text(encoding="utf-8"))
            out[p] = self.factory.build(p, fz["spec"], fz.get("head_override"))
        return out

    def freeze(self):
        self.factory = self.factory or E.ModuleFactory(list(self.tasks["dev"].values()), self.out / "modules")
        arch = read_jsonl(self.out / "archive.jsonl")
        front = {}
        for p in PERSONAS:
            items = [a for a in arch if a["persona"] == p]
            ch = E.freeze_choice(items)
            front[p] = [{"hash": i["hash"], "source": i["source"], "panel": i["panel"]} for i in E.pareto_front(items)]
            write_json_atomic(self.out / "frozen" / f"{p}.json", {"persona": p, "hash": ch["hash"], "spec_hash": ch["spec_hash"],
                                                                "source": ch["source"], "panel": ch["panel"],
                                                                "spec": self.spec_of(ch["spec_hash"]),
                                                                "head_override": ch.get("head_override")})
        write_json_atomic(self.out / "pareto_front.json", front)
        fm = self.frozen_modules()
        held = list(self.tasks["held"].values())
        fns = {f"learned:{p}": (lambda t, m=fm[p]: decide(t, prior_from_response(t, None), [m["learned"]])) for p in PERSONAS}
        fns.update({f"rubric:{p}": (lambda t, m=fm[p]: decide(t, prior_from_response(t, None), [m["rubric"]])) for p in PERSONAS})
        for a, b in COMPOSITION_PAIRS:
            fns[f"comp:{a}+{b}"] = (lambda t, x=fm[a], y=fm[b]: decide(t, prior_from_response(t, None), [x["learned"], y["learned"]]))
        audit = host.boundary_audit(held, fns)
        forced = dict(fm[PERSONAS[1]]["spec"])
        forced = json.loads(json.dumps(forced))
        forced["critical_tags"] = []
        forced["preference_features"].append({"name": "boundary_lover", "weight": 10.0,
                                              "positive_tags": list(BOUNDARY_TAGS), "negative_tags": []})
        fmod = E.ModuleFactory([], self.out / "modules").build(PERSONAS[1], forced)
        adv = host.boundary_audit(held, {"forced": lambda t: decide(t, prior_from_response(t, None), [fmod["learned"]])})
        write_json_atomic(self.out / "cca_audit_freeze.json", {
            "heldout_tasks": len(held), "audit": audit, "forced_boundary_lover": adv,
            "validator_rejects_forced": bool(validate(forced, PERSONAS[1])),
            "host_code_hash_start": self.state["host_code_hash_start"], "host_code_hash_freeze": host.host_code_hash(),
            "host_code_unchanged": self.state["host_code_hash_start"] == host.host_code_hash()})
        self.state["frozen"] = True
        self.state["frozen_at"] = time.time()
        self.state["phase"] = "ROUTER"
        self.save()

    # ------------------------------------------------------------------ options for routing
    def options(self, fm) -> list[tuple[str, list]]:
        opts = [(p, [fm[p]["learned"]]) for p in PERSONAS]
        opts += [(f"{a}+{b}", [fm[a]["learned"], fm[b]["learned"]]) for a, b in COMPOSITION_PAIRS]
        opts.append(("inky", None))
        return opts

    def option_matrix(self, tasks, priors, fm):
        opts = self.options(fm)
        X, U = [], []
        for t in tasks:
            parsed, prior = priors[t.task_id]
            X.append(R.features(t, prior, parsed["parsed"]))
            row = []
            for name, mods in opts:
                if mods is None:
                    rr = E.neutral_row(t, PERSONAS[0], parsed)
                else:
                    rr = E.module_row(t, PERSONAS[0], name, prior, mods)
                row.append(rr["normalized_utility"])
            U.append(row)
        return np.array(X), np.array(U), [o[0] for o in opts]

    def selfmodel_prompt(self, p, spec) -> str:
        return "\n\n".join([
            "You are the research persona described below. Predict your own behavior on NEW tasks before seeing them.",
            persona_card(spec),
            "TASK DOMAINS: lean_proof = choose a proof step for a Lean 4 goal (some goals are false); python_repair = choose a step "
            "toward fixing a failing Python function; design_sim = choose an experimental design for an effect that may be zero.",
            "METRIC DEFINITIONS (exact): " + "; ".join(f"{k}: {v}" for k, v in R.SELF_METRICS.items()),
            'Return only JSON: {"lean_proof": {"success": p, "failure": p, "abstain": p, "compute": c}, "python_repair": {...}, '
            '"design_sim": {...}} with probabilities in [0,1] and compute in [1,3].'])

    def train_router(self):
        t0 = time.time()
        fm = self.frozen_modules()
        sel = sorted(self.tasks["sel"].values(), key=lambda t: t.task_id)
        dev = sorted(self.tasks["dev"].values(), key=lambda t: t.task_id)
        sp, dp = self.priors_for(sel), self.priors_for(dev)
        Xs, Us, names = self.option_matrix(sel, sp, fm)
        Xd, Ud, _ = self.option_matrix(dev, dp, fm)
        res = {"options": names, "n_train": len(sel), "n_val": len(dev)}
        import torch
        torch.set_num_threads(2)
        for steps in (self.cfg["trm_steps"], 1):
            m = R.TRMRouter(Xs.shape[1], Us.shape[1], self.cfg["trm_hidden"], steps, seed=self.cfg["seed"])
            info = m.fit(Xs, Us, Xd, Ud, epochs=self.cfg["trm_epochs"])
            torch.save({"state": [p.detach() for p in m.params], "mu": m.mu, "sd": m.sd, "steps": steps},
                       self.out / f"router_trm_k{steps}.pt")
            res[f"trm_k{steps}"] = info
        res["static_best"] = names[int(Us.mean(0).argmax())]
        res["domain_best"] = {d: names[int(Us[[t.domain == d for t in sel]].mean(0).argmax())] for d in D.DOMAINS}
        res["option_mean_utility_selection"] = dict(zip(names, Us.mean(0).round(4).tolist()))
        write_json_atomic(self.out / "router.json", res)
        # self-models (learned) on selection rows of each frozen persona
        sm = {}
        for p in PERSONAS:
            rows = [E.module_row(t, p, "D", sp[t.task_id][1], [fm[p]["learned"]]) for t in sel]
            X = np.array([R.features(t, sp[t.task_id][1], sp[t.task_id][0]["parsed"]) for t in sel])
            sm[p] = R.SelfModel().fit(X, rows)
        self._selfmodels = sm
        # Inky self-predictions, elicited after freeze and before any held-out exposure
        futs = {p: self.worker.submit(f"selfpred|{p}|{fm[p]['hash']}", self.selfmodel_prompt(p, fm[p]["spec"]),
                                      self.cfg["aux_max_tokens"], "selfmodel_elicit", 0.0, 0) for p in PERSONAS}
        elic = {}
        for p, f in futs.items():
            e = self.wait(f)
            elic[p] = {"raw": e["text"][:2000], "parsed": extract_json(e["text"])}
        write_json_atomic(self.out / "selfmodel_elicited.json", elic)
        self.timed("router", t0)
        self.state["phase"] = "HELDOUT"
        self.save()

    def load_router(self, n_in, n_out, steps):
        import torch
        m = R.TRMRouter(n_in, n_out, self.cfg["trm_hidden"], steps, seed=self.cfg["seed"])
        ck = torch.load(self.out / f"router_trm_k{steps}.pt", weights_only=False)
        with torch.no_grad():
            for p, s in zip(m.params, ck["state"]):
                p.copy_(s)
        m.mu, m.sd = ck["mu"], ck["sd"]
        return m

    def route_prompt(self, task) -> str:
        names = "\n".join(f"- {p}: {PERSONA_BRIEFS[p]}" for p in PERSONAS)
        return decision_prompt(task).split("Return only a JSON object")[0] + (
            "Instead of choosing an action, choose which research persona module should handle this task:\n" + names +
            '\n- none: use your own judgment.\nReturn only JSON: {"route": persona name or "none"}.')

    # ------------------------------------------------------------------ HELDOUT
    def heldout(self):
        if not self.state.get("frozen"):
            raise RuntimeError("held-out evaluation before freeze is not allowed")
        t0 = time.time()
        fm = self.frozen_modules()
        root_mods = {p: self.factory.build(p, self.spec_of(self.state["roots"][p])) for p in PERSONAS}
        arm_mods = {(p, a): self.factory.build(p, self.spec_of(self.state["lineages"][p][a])) for p in PERSONAS for a in E.ARMS}
        held = sorted(self.tasks["held"].values(), key=lambda t: (int(t.task_id.rsplit("-", 1)[1]), t.domain))
        rinfo = json.loads((self.out / "router.json").read_text(encoding="utf-8"))
        names = rinfo["options"]
        opts = dict(self.options(fm))
        done = {r["task_id"] for r in read_jsonl(self.out / "routing_rows.jsonl")}
        route_f = {t.task_id: self.worker.submit(f"route|{t.task_id}", self.route_prompt(t), self.cfg["aux_max_tokens"],
                                                 "route_inky", 0.0, 0) for t in held if t.task_id not in done}
        trm = {k: None for k in (self.cfg["trm_steps"], 1)}
        for t in held:
            if t.task_id in done:
                continue
            self.check()
            parsed, prior = self.neutral(t)
            x = R.features(t, prior, parsed["parsed"])
            for k in trm:
                if trm[k] is None:
                    trm[k] = self.load_router(len(x), len(names), k)
            for p in PERSONAS:
                rows = [E.neutral_row(t, p, parsed),
                        E.module_row(t, p, "FINAL", prior, [fm[p]["learned"]]),
                        E.module_row(t, p, "FINAL_C", prior, [fm[p]["rubric"]]),
                        E.module_row(t, p, "ROOT", prior, [root_mods[p]["learned"]])]
                rows += [E.module_row(t, p, f"ARM_{a}", prior, [arm_mods[(p, a)]["learned"]]) for a in E.ARMS]
                for rr in rows:
                    append_jsonl(self.out / "heldout_rows.jsonl", rr)
            # option utilities on this task
            util = {}
            for name in names:
                mods = opts[name]
                rr = E.neutral_row(t, PERSONAS[0], parsed) if mods is None else E.module_row(t, PERSONAS[0], name, prior, mods)
                util[name] = {"u": rr["normalized_utility"], "executed": rr["executed"], "unauth": rr["unauthorized_execution"]}
            choice = {f"trm_k{k}": names[int(trm[k].predict(x[None])[0].argmax())] for k in trm}
            choice["static_best"] = rinfo["static_best"]
            choice["domain_best"] = rinfo["domain_best"][t.domain]
            fl = host_flags(t)
            choice["rule"] = ("repair_specialist" if fl["failure_reported"] else
                              {"lean_proof": "formalist", "python_repair": "repair_specialist", "design_sim": "experimentalist"}[t.domain])
            ri = self.wait(route_f[t.task_id])
            ro = extract_json(ri["text"])
            rname = ro.get("route") if isinstance(ro, dict) else None
            choice["inky_route"] = rname if rname in PERSONAS else "inky"
            choice["oracle"] = max(names, key=lambda n: util[n]["u"])
            rec = {"task_id": t.task_id, "domain": t.domain, "template": t.hidden.get("template"),
                   "features": x.round(5).tolist(), "choices": choice,
                   "utility": {k: util[v]["u"] for k, v in choice.items()},
                   "uniform": float(np.mean([util[n]["u"] for n in names])),
                   "unauthorized": {k: util[v]["unauth"] for k, v in choice.items()},
                   "inky_route_raw": (ri.get("text") or "")[:300], "option_utility": {n: util[n]["u"] for n in names}}
            append_jsonl(self.out / "routing_rows.jsonl", rec)
        self.selfmodel_eval(fm, held)
        self.timed("heldout", t0)
        self.state["phase"] = "COMPACT"
        self.save()

    def selfmodel_eval(self, fm, held):
        """Learned self-models (fit on selection rows) and Inky's elicited self-predictions, scored on held-out."""
        sel = sorted(self.tasks["sel"].values(), key=lambda t: t.task_id)
        sp, hp = self.priors_for(sel), self.priors_for(held)
        elic = json.loads((self.out / "selfmodel_elicited.json").read_text(encoding="utf-8"))
        out = {"definitions": R.SELF_METRICS, "learned": {}, "elicited": {}}
        for p in PERSONAS:
            rows_s = [E.module_row(t, p, "D", sp[t.task_id][1], [fm[p]["learned"]]) for t in sel]
            Xs = np.array([R.features(t, sp[t.task_id][1], sp[t.task_id][0]["parsed"]) for t in sel])
            rows_h = [E.module_row(t, p, "D", hp[t.task_id][1], [fm[p]["learned"]]) for t in held]
            Xh = np.array([R.features(t, hp[t.task_id][1], hp[t.task_id][0]["parsed"]) for t in held])
            out["learned"][p] = R.SelfModel().fit(Xs, rows_s).score(Xh, rows_h)
            T = R.self_targets(rows_h)
            pe = (elic.get(p) or {}).get("parsed") or {}
            per = {}
            for d in D.DOMAINS:
                idx = [i for i, t in enumerate(held) if t.domain == d]
                obs = {k: float(T[k][idx].mean()) for k in T}
                pred = pe.get(d) if isinstance(pe.get(d), dict) else {}
                per[d] = {"observed": obs, "predicted": {k: pred.get(k) for k in T},
                          "abs_error": {k: (abs(float(pred[k]) - obs[k]) if isinstance(pred.get(k), (int, float)) else None) for k in T}}
            out["elicited"][p] = per
        write_json_atomic(self.out / "selfmodel_eval.json", out)

    # ------------------------------------------------------------------ COMPACT
    def compactify(self):
        t0 = time.time()
        fm = self.frozen_modules()
        arch = read_jsonl(self.out / "archive.jsonl")
        M = np.array([a["head_flat"] for a in arch])
        res = {"population": K.spectrum(M)}
        by_p = defaultdict(list)
        for a in arch:
            by_p[a["persona"]].append(a["head_flat"])
        ang = {}
        for a, b in itertools.combinations(PERSONAS, 2):
            if len(by_p[a]) > 2 and len(by_p[b]) > 2:
                ang[f"{a}|{b}"] = K.principal_angles(np.array(by_p[a]), np.array(by_p[b]))
        res["principal_angles_deg"] = ang
        res["per_persona_effective_rank"] = {p: K.spectrum(np.array(v))["effective_rank"] for p, v in by_p.items() if len(v) > 2}
        held = sorted(self.tasks["held"].values(), key=lambda t: t.task_id)
        hp = self.priors_for(held)
        full = {p: self.rows(fm[p], p, held, hp) for p in PERSONAS}
        ranks = [1, 2, 3, 4, 6, 8, 12, 16, 24]
        res["low_rank"] = []
        for r in ranks:
            if r >= min(M.shape):
                break
            mu, Vb = K.basis(M, r)
            agree, fid, util, fid_full, util_full = [], [], [], [], []
            for p in PERSONAS:
                vec = K.project(K.flat(fm[p]["head"]), mu, Vb)
                m = self.factory.build(p, fm[p]["spec"], K.unflat(vec, fm[p]["head"]))
                rows = self.rows(m, p, held, hp)
                agree.append(np.mean([a["executed"] == b["executed"] for a, b in zip(rows, full[p])]))
                s, sf = metrics.summarize(rows), metrics.summarize(full[p])
                fid.append(s["fidelity"] or 0.0)
                fid_full.append(sf["fidelity"] or 0.0)
                util.append(s["normalized_utility"])
                util_full.append(sf["normalized_utility"])
            res["low_rank"].append({"rank": r, "behavioral_agreement": float(np.mean(agree)),
                                    "min_agreement": float(np.min(agree)),
                                    "fidelity_retention": float(np.sum(fid) / max(1e-9, np.sum(fid_full))),
                                    "utility_retention": float(np.sum(util) / max(1e-9, np.sum(util_full))),
                                    "per_persona_agreement": dict(zip(PERSONAS, map(float, agree)))})
        write_json_atomic(self.out / "compact.json", res)
        self.timed("compact", t0)
        self.state["phase"] = "REPORT"
        self.save()

    # ------------------------------------------------------------------ main
    def run(self) -> int:
        def handler(signum, frame):
            self._signal = signum
        for s in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, s):
                try:
                    signal.signal(getattr(signal, s), handler)
                except (ValueError, OSError):
                    pass
        if os.name == "nt" and not self.dry_run:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            except Exception:
                pass
        self.worker.start()
        code = 0
        try:
            self.load_banks()
            steps = [("BUILD", self.build), ("INIT", self.init), ("MUTATE", self.mutate), ("RECOMB", self.recombine),
                     ("FREEZE", self.freeze), ("ROUTER", self.train_router), ("HELDOUT", self.heldout),
                     ("COMPACT", self.compactify)]
            for name, fn in steps:
                if self.state["phase"] == name:
                    fn()
        except Stop as s:
            self.state["stop_reason"] = str(s)
        except CredentialRejected as exc:
            self.state["stop_reason"] = f"credential_rejected: {exc}"
            code = 2
        except Exception as exc:
            self.state["stop_reason"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
            (self.out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            code = 1
        finally:
            self.worker.stop_flag.set()
            self.state["ended_at"] = time.time()
            self.save()
            try:
                from .report import build_report
                build_report(self.out)
            except Exception:
                (self.out / "report_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return code


def code_hashes02() -> dict:
    here = Path(__file__).resolve().parent
    out = code_hashes01()
    for p in sorted(here.glob("*.py")):
        out[p.relative_to(here.parent.parent).as_posix()] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args(argv)
    if a.hours > 8.0:
        ap.error("the registered maximum campaign budget is eight hours")
    if a.dry_run:
        from .fakes import FakeTransport02
        client = InkyClient("dry-run", Usage(), transport=FakeTransport02(CFG["seed"]), min_interval=0.0,
                            effort=CFG["reasoning_effort"], context_limit=CFG["context_limit"])
    else:
        client = InkyClient(load_key(a.key_file), Usage(), effort=CFG["reasoning_effort"], context_limit=CFG["context_limit"])
    c = Campaign(a.out, a.hours, client, a.dry_run, a.small)
    if c.state.get("dry_run") != a.dry_run:
        ap.error("output directory belongs to a run with a different dry-run setting")
    return c.run()


if __name__ == "__main__":
    sys.exit(main())
