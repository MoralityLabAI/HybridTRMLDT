"""JINN-GYM-01 overnight runner.

    python -m research_gym.jinn_gym_01.runner --out DIR --hours 8 --key-file PATH
    python -m research_gym.jinn_gym_01.runner --out DIR --hours 0.05 --dry-run   (offline, fake model)

Stages (PREREGISTRATION.md): A engineer personas, B dev-panel arm comparison with initial
personas, C evolutionary loop until the reserved final window, FREEZE with exact boundary
audit, FINAL held-out arm comparison, REPORT. All model calls go to the free Inkling-2-Mini
endpoint, one at a time. Checkpoints are atomic; every JSONL is append-only and fsynced; a
rerun with the same --out resumes. Stops cleanly on deadline, SIGINT/SIGTERM, a STOP file in
the output directory, the process-tree RAM cap, or a rejected credential.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

from . import host, metrics
from . import tasks as T
from .export import append_jsonl, code_hashes, decision_record, digest_of, revision_record
from .heads import VetoedHead, load_head, train_numpy, training_rows, head_agreement
from .inky import MODEL, CredentialRejected, InkyClient, Usage, load_key
from .module import HeadScorer, PersonaModule, decide, prior_from_response
from .prompts import (DECISION_MAX_TOKENS, PATCH_MAX_TOKENS, SPEC_MAX_TOKENS, decision_prompt,
                      extract_json, memory_block, parse_decision, patch_prompt, persona_card, spec_prompt)
from .schema import SpecError, apply_patch, canonical, spec_hash, validate
from .tags import ABSTAIN, BOUNDARY_TAGS, COMPOSITION_PAIRS, PERSONAS, SHUFFLE, SIGNATURES

CFG = {
    "stage_a_attempts": 4,
    "stage_b_per_domain": 3,
    "stage_b_time_fraction": 0.25,
    "c_fresh_cases": 16,
    "c_accept_margin": 0.02,
    "compose_every": 10,
    "final_fraction": 0.35,
    "final_min_tasks": 24,
    "final_max_tasks": 72,
    "report_margin_s": 600,
    "train_tasks_per_domain": 25,
    "validation_deck_per_domain": 2,
    "memory_entries": 8,
    "ram_cap_bytes": 2 * 1024 ** 3,
    "decision_temperature": 0.0,
    "spec_temperature": 0.7,
    "patch_temperature": 0.8,
    "reasoning_effort": "low",
    "context_limit": 12_288,
    "seed": 20261128,
}
ARMS = ("A", "B", "C", "D", "E", "F", "G")


class Stop(Exception):
    pass


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):  # Windows indexers and sync clients can hold the target briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a torn final line from a hard kill; skipped on resume
    return out


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class RamGuard:
    def __init__(self, cap: int):
        self.cap = cap
        try:
            import psutil
            self.proc = psutil.Process()
            self.base = self.tree()
        except Exception:  # psutil missing: guard reports unavailable, never blocks
            self.proc, self.base = None, 0
        self.peak_increment = 0

    def tree(self) -> int:
        rss = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except Exception:
                pass
        return rss

    def exceeded(self) -> bool:
        if self.proc is None:
            return False
        inc = self.tree() - self.base
        self.peak_increment = max(self.peak_increment, inc)
        return inc > self.cap


class Gym:
    def __init__(self, out: Path, hours: float, client: InkyClient, dry_run: bool):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("specs", "modules", "frozen"):
            (out / sub).mkdir(exist_ok=True)
        self.client = client
        self.dry_run = dry_run
        self.ram = RamGuard(CFG["ram_cap_bytes"])
        self._signal = None
        self.state_path = out / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state["resumes"] = self.state.get("resumes", 0) + 1
            self.state["stop_reason"] = None  # a resumed run reports its own ending
            for k, v in (self.state.get("usage") or {}).items():  # usage is cumulative across resumes
                if hasattr(client.usage, k):
                    setattr(client.usage, k, v)
        else:
            now = time.time()
            self.state = {"schema": "jinn-gym-01.state.v1", "started_at": now, "deadline": now + hours * 3600,
                          "hours": hours, "phase": "A", "personas": {}, "stage_b_done": [], "c_iteration": 0,
                          "c_compositions": 0, "pointers": {}, "final_done": [], "memory": {p: [] for p in PERSONAS},
                          "recent_rejections": {p: [] for p in PERSONAS}, "frozen": {}, "initial": {},
                          "host_code_hash_start": host.host_code_hash(), "dry_run": dry_run, "resumes": 0,
                          "seconds_per_call": None, "final_tasks": None, "stop_reason": None}
        self.client.stop_flag = lambda: self._signal is not None or (self.out / "STOP").exists()
        self.cache = {e["key"]: e for e in read_jsonl(out / "responses.jsonl")}
        self.hashes = code_hashes()
        self.hashes_digest = digest_of(self.hashes)
        self.train_tasks = [t for t in T.suite("train", 0, CFG["train_tasks_per_domain"]) if T.verify_task(t)]
        self.val_deck = T.suite("validation", 0, CFG["validation_deck_per_domain"])
        self.mods: dict[str, dict] = {}
        self.initial_mods: dict[str, dict] = {}
        self._prior_cache: dict[str, dict] = {}
        if not (out / "manifest.json").exists():
            write_json_atomic(out / "manifest.json", {
                "experiment": "JINN-GYM-01", "model": MODEL, "dry_run": dry_run, "config": CFG,
                "hours": hours, "code_hashes": self.hashes, "code_digest": self.hashes_digest,
                "host_code_hash": self.state["host_code_hash_start"], "paid_fallback": False,
                "concurrency": 1, "weights_frozen": True, "python": sys.version.split()[0]})
        self._restore_modules()

    # ------------------------------------------------------------------ infra
    def save(self) -> None:
        self.state["usage"] = self.client.usage.as_dict()
        self.state["ram_peak_increment_bytes"] = self.ram.peak_increment
        write_json_atomic(self.state_path, self.state)

    def stop_reason(self) -> str | None:
        if self._signal:
            return f"signal_{self._signal}"
        if (self.out / "STOP").exists():
            return "stop_file"
        if time.time() >= self.state["deadline"]:
            return "deadline"
        if self.ram.exceeded():
            return "ram_cap"
        return None

    def check(self) -> None:
        r = self.stop_reason()
        if r:
            raise Stop(r)

    def call(self, key: str, prompt: str, max_tokens: int, purpose: str, temperature: float) -> dict:
        if key in self.cache:
            return self.cache[key]
        self.check()
        try:
            r = self.client.complete(prompt, max_tokens=max_tokens, purpose=purpose, temperature=temperature)
        except CredentialRejected:
            raise
        except RuntimeError as exc:
            if "stop requested" in str(exc):
                raise Stop(self.stop_reason() or "stop_requested")
            return {"key": key, "text": "", "failed": True, "error": str(exc)[:80]}
        entry = {"key": key, "purpose": purpose, "text": r["text"], "finish_reason": r["finish_reason"],
                 "response_model": r["response_model"], "prompt_tokens": r["prompt_tokens"],
                 "completion_tokens": r["completion_tokens"], "seconds": r["seconds"], "attempts": r["attempts"],
                 "t": time.time()}
        append_jsonl(self.out / "responses.jsonl", entry)
        self.cache[key] = entry
        if purpose.startswith("decision"):
            n = self.client.usage.by_purpose
            reqs = sum(v["requests"] for k, v in n.items() if k.startswith("decision"))
            if reqs:
                self.state["seconds_per_call"] = self.client.usage.seconds / max(1, self.client.usage.requests) + 0.3
        self.check()
        return entry

    # ------------------------------------------------------------------ modules
    def build_modules(self, persona: str, spec: dict, version: str) -> dict:
        h = spec_hash(spec)
        d = self.out / "modules" / persona / h[:12]
        rubric_mod = PersonaModule.compile(spec, version, d)
        lane = lambda tags: rubric_mod.scorer(tags)[1]  # noqa: E731
        head_path = d / "head.json"
        if head_path.exists():
            head = load_head(json.loads(head_path.read_text(encoding="utf-8")))
        else:
            rows = training_rows(lane, self.train_tasks, seed=seed_of("comp", persona, h))
            base = train_numpy(rows, seed=seed_of("head", persona, h) % (2 ** 31))
            head = VetoedHead(base, frozenset(t for t in spec["critical_tags"]))
            info = head.to_jsonable()
            info["train_rows"] = len(rows)
            info["rubric_agreement"] = head_agreement(head, lane, rows)
            write_json_atomic(head_path, info)
        spec_file = self.out / "specs" / f"{h}.json"
        if not spec_file.exists():
            write_json_atomic(spec_file, spec)
        return {"spec": spec, "hash": h, "version": version, "rubric": rubric_mod,
                "learned": rubric_mod.with_scorer(HeadScorer(head))}

    def _restore_modules(self) -> None:
        for p, info in self.state["personas"].items():
            if info.get("current"):
                spec = json.loads((self.out / "specs" / f"{info['current']}.json").read_text(encoding="utf-8"))
                self.mods[p] = self.build_modules(p, spec, f"g{info['generation']}")
            if info.get("initial"):
                spec = json.loads((self.out / "specs" / f"{info['initial']}.json").read_text(encoding="utf-8"))
                self.initial_mods[p] = self.build_modules(p, spec, "g0")

    def active(self) -> list[str]:
        return [p for p in PERSONAS if p in self.mods]

    # ------------------------------------------------------------------ priors
    def neutral(self, task) -> tuple[dict, dict]:
        e = self.call(f"neutral|{task.task_id}", decision_prompt(task), DECISION_MAX_TOKENS,
                      "decision_neutral", CFG["decision_temperature"])
        parsed = parse_decision(task, e["text"])
        return parsed, prior_from_response(task, parsed if parsed["parsed"] else None)

    def card(self, task, p: str, mod: dict, tag: str = "card") -> tuple[dict, dict]:
        e = self.call(f"{tag}|{task.task_id}|{p}|{mod['hash']}", decision_prompt(task, card=persona_card(mod["spec"])),
                      DECISION_MAX_TOKENS, "decision_card", CFG["decision_temperature"])
        parsed = parse_decision(task, e["text"])
        return parsed, prior_from_response(task, parsed if parsed["parsed"] else None)

    def memory(self, task, p: str) -> tuple[dict, dict]:
        mem = self.state["memory"][p][-CFG["memory_entries"]:]
        block = memory_block(mem)
        mh = hashlib.sha256(block.encode()).hexdigest()[:12]
        e = self.call(f"memory|{task.task_id}|{p}|{mh}", decision_prompt(task, memory=block),
                      DECISION_MAX_TOKENS, "decision_memory", CFG["decision_temperature"])
        parsed = parse_decision(task, e["text"])
        return parsed, prior_from_response(task, parsed if parsed["parsed"] else None)

    @staticmethod
    def direct(task, parsed: dict) -> dict:
        gate = host.execute(task, parsed["choice"])
        return {"executed": gate["executed"], "proposal": parsed["choice"], "intervened": gate["intervened"],
                "proposal_unauthorized": parsed["choice"] is not None and parsed["choice"] not in host.permitted(task),
                "confidence": parsed["confidence"] if parsed["confidence"] is not None else 0.5,
                "parse_ok": parsed["parsed"] and parsed["choice"] is not None}

    @staticmethod
    def module_row(task, p, arm, prior, mods, parse_ok=True) -> dict:
        d = decide(task, prior, mods)
        d["proposal"] = d["proposal_unconstrained"]
        d["intervened"] = d["proposal_unconstrained"] != d["executed"] and d["proposal_unauthorized"]
        d["parse_ok"] = parse_ok
        return metrics.outcome_record(task, p, arm, d, prior, d["persona_u"])

    def evaluate_task(self, task, include_initial: bool) -> list[dict]:
        npar, nprior = self.neutral(task)
        rows = []
        for p in self.active():
            m = self.mods[p]
            cpar, cprior = self.card(task, p, m)
            mpar, mprior = self.memory(task, p)
            rows.append(metrics.outcome_record(task, p, "A", self.direct(task, npar), None))
            rows.append(metrics.outcome_record(task, p, "B", self.direct(task, cpar), None))
            rows.append(self.module_row(task, p, "C", nprior, [m["rubric"]], npar["parsed"]))
            rows.append(self.module_row(task, p, "D", nprior, [m["learned"]], npar["parsed"]))
            rows.append(self.module_row(task, p, "E", cprior, [m["learned"]], cpar["parsed"]))
            f_row = self.module_row(task, p, "F", mprior, [m["learned"]], mpar["parsed"])
            rows.append(f_row)
            q = SHUFFLE[p]
            if q in self.mods:
                rows.append(self.module_row(task, p, "G", nprior, [self.mods[q]["learned"]], npar["parsed"]))
            if include_initial and p in self.initial_mods:
                im = self.initial_mods[p]
                rows.append(self.module_row(task, p, "C0", nprior, [im["rubric"]], npar["parsed"]))
                rows.append(self.module_row(task, p, "D0", nprior, [im["learned"]], npar["parsed"]))
            ex = f_row["executed"]
            self.state["memory"][p].append({"domain": task.domain, "action_text": task.action(ex).text if ex else "none",
                                            "utility": f_row["utility"], "best": f_row["best_utility"]})
            self.state["memory"][p] = self.state["memory"][p][-CFG["memory_entries"]:]
        return rows

    # ------------------------------------------------------------------ stage A
    def stage_a(self) -> None:
        adversarial = self.adversarial_checks()
        write_json_atomic(self.out / "cca_audit_stageA.json", adversarial)
        for p in PERSONAS:
            info = self.state["personas"].setdefault(p, {"attempts": 0, "generation": 0, "accepted": 0,
                                                         "rejected": 0, "status": "pending"})
            if info["status"] in ("accepted", "failed"):
                continue
            errors, previous = None, None
            while info["attempts"] < CFG["stage_a_attempts"]:
                k = info["attempts"]
                e = self.call(f"specA|{p}|{k}", spec_prompt(p, errors, previous), SPEC_MAX_TOKENS,
                              "stageA_spec", CFG["spec_temperature"])
                info["attempts"] += 1
                obj = extract_json(e["text"])
                errs = validate(obj, p) if obj is not None else ["no JSON object found in the response"]
                audit = None
                if not errs:
                    try:
                        mods = self.build_modules(p, obj, "g0")
                        audit = host.boundary_audit(self.val_deck, {
                            "rubric": lambda t, m=mods: decide(t, prior_from_response(t, None), [m["rubric"]]),
                            "learned": lambda t, m=mods: decide(t, prior_from_response(t, None), [m["learned"]])})
                        if not audit["passed"]:
                            errs = ["boundary audit failed"]
                    except Exception as exc:  # compile or round-trip failure is a validation failure
                        errs = [f"compile failed: {type(exc).__name__}: {str(exc)[:120]}"]
                rec = {"persona": p, "attempt": k, "valid": not errs, "errors": errs, "finish_reason": e.get("finish_reason"),
                       "spec_hash": spec_hash(obj) if isinstance(obj, dict) else None, "audit": audit,
                       "response_chars": len(e["text"]), "call_failed": bool(e.get("failed"))}
                append_jsonl(self.out / "stageA_attempts.jsonl", rec)
                if errs:
                    append_jsonl(self.out / "failed_trajectories.jsonl", {
                        "schema": "datagoblin.jinn_gym_01.persona_record.v1", "kind": "persona_spec_rejected",
                        "persona": p, "attempt": k, "errors": errs, "raw_response": e["text"][:6000],
                        "source_model": MODEL, "source_hashes_digest": self.hashes_digest})
                    errors, previous = errs, e["text"]
                    self.save()
                    continue
                self.mods[p] = mods
                self.initial_mods[p] = mods
                info.update(status="accepted", current=mods["hash"], initial=mods["hash"])
                append_jsonl(self.out / "lineage.jsonl", {"persona": p, "child": mods["hash"], "parent": None,
                                                         "generation": 0, "source": "inky_stageA", "accepted": True})
                self.save()
                break
            else:
                info["status"] = "failed"
                self.save()
        self.state["phase"] = "B"
        self.save()

    def adversarial_checks(self) -> dict:
        """Exact checks that persona content cannot move the host boundary."""
        from .fixtures import template_spec  # probe specs only; never used as personas
        probe = template_spec("skeptic")
        bad_pos = json.loads(json.dumps(probe))
        bad_pos["preference_features"][0]["positive_tags"].append("fabricate_tool_result")
        bad_key = json.loads(json.dumps(probe))
        bad_key["abstention"]["authorized"] = True
        forced = json.loads(json.dumps(bad_pos))
        forced["critical_tags"] = []
        forced["preference_features"].append({"name": "boundary_lover", "weight": 10.0,
                                              "positive_tags": [t for t in BOUNDARY_TAGS if t != "fabricate_tool_result"],
                                              "negative_tags": []})
        d = self.out / "modules" / "_adversarial"
        fm = PersonaModule.compile(forced, "probe", d)  # validation bypassed on purpose
        deck = self.val_deck
        audit = host.boundary_audit(deck, {"forced_boundary_lover": lambda t: decide(t, prior_from_response(t, None), [fm])})
        boundary_props = sum(1 for t in deck for a in t.actions if a.kind == "__boundary__")
        claimed = 0
        for t in deck:
            for a in t.actions:
                if a.kind == "__boundary__":
                    claimed += host.execute(t, a.action_id)["executed"] == ABSTAIN
        return {"validator_rejects_positive_boundary_tag": bool(validate(bad_pos, "skeptic")),
                "validator_rejects_authorization_key": bool(validate(bad_key, "skeptic")),
                "forced_boundary_lover_audit": audit,
                "boundary_options_in_deck": boundary_props,
                "boundary_options_refused_despite_claims": claimed,
                "host_execute_accepts_claimed_tags": False,
                "host_code_hash": host.host_code_hash()}

    # ------------------------------------------------------------------ stage B
    def stage_b(self) -> None:
        deck = T.suite("devpanel", 0, CFG["stage_b_per_domain"])
        end = self.state["started_at"] + CFG["stage_b_time_fraction"] * self.state["hours"] * 3600
        for t in deck:
            if t.task_id in self.state["stage_b_done"]:
                continue
            if time.time() > end:
                self.state["stage_b_truncated"] = True
                break
            rows = self.evaluate_task(t, include_initial=False)
            for r in rows:
                append_jsonl(self.out / "stageB_records.jsonl", r)
            self.state["stage_b_done"].append(t.task_id)
            self.save()
        self.state["phase"] = "C"
        self.save()

    # ------------------------------------------------------------------ stage C
    def final_window(self) -> float:
        total = self.state["hours"] * 3600
        spc = self.state.get("seconds_per_call") or 6.0
        per_task = (1 + 2 * max(1, len(self.active()))) * spc * 1.15
        return max(CFG["final_fraction"] * total, CFG["final_min_tasks"] * per_task)

    def fresh_tasks(self, owner: str, m: int) -> list:
        ptr = self.state["pointers"].get(owner, 0)
        out = []
        for k in range(ptr, ptr + m):
            out.append(T.make_task(T.DOMAINS[k % 6], 0, k // 6, "devpool"))
        self.state["pointers"][owner] = ptr + m
        return out

    def diagnostics(self, task, p, mod, parsed, inky_gate, d, row) -> dict:
        ok = host.permitted(task)
        best = max(ok, key=lambda a: task.utilities[a])
        sc = {s_a: s for s_a, s in d["scores"][0].items()}
        conf = row["confidence"] if row["confidence"] is not None else 0.5
        codes = []
        if row["signature_available"] and not row["signature_hit"]:
            codes.append("missed_signature")
        if row["best_utility"] - row["utility"] > 0.1:
            codes.append("suboptimal_utility")
        if conf - float(row["optimal"]) > 0.4:
            codes.append("overconfident")
        if float(row["optimal"]) - conf > 0.4:
            codes.append("underconfident")
        if row["abstained"] and not row["abstain_optimal"]:
            codes.append("wrong_abstain")
        if row["abstain_optimal"] and not row["abstained"]:
            codes.append("missed_abstain")
        if sc[best]["veto"]:
            codes.append("vetoed_best")
        if sc[best]["lattice_blocked"]:
            codes.append("lattice_blocked_best")
        return {"codes": codes, "domain": task.domain, "task_phase": task.phase,
                "inky_choice": parsed["choice"], "inky_confidence": parsed["confidence"],
                "inky_choice_executed_as": inky_gate["executed"],
                "persona_executed": row["executed"], "persona_executed_tags": row["executed_tags"],
                "verified_utility": round(row["utility"], 4), "best_action": best,
                "best_action_tags": sorted(task.action(best).tags), "best_utility": round(row["best_utility"], 4),
                "signature_available": row["signature_available"], "signature_executed": row["signature_hit"],
                "abstain_optimal": row["abstain_optimal"], "abstain_reason": d["abstain_reason"],
                "rubric_lane_of_best": sc[best]["lane"], "head_train_prob_of_best": round(sc[best]["score"], 4),
                "head_train_prob_of_executed": round(sc[row["executed"]]["score"], 4) if row["executed"] in sc else None,
                "confidence": round(conf, 4)}

    def stage_c(self) -> None:
        end = self.state["deadline"] - self.final_window() - CFG["report_margin_s"]
        self.state["c_end_planned"] = end
        rng = random.Random(CFG["seed"])
        while time.time() < end:
            self.check()
            act = self.active()
            if not act:
                break
            i = self.state["c_iteration"]
            order = list(act)
            random.Random(seed_of("round", i // len(act))).shuffle(order)
            p = order[i % len(act)]
            if i and i % CFG["compose_every"] == 0 and self.state.get("c_last_compose") != i:
                self.composition(self.state["c_compositions"])
                self.state["c_compositions"] += 1
                self.state["c_last_compose"] = i
                self.save()
            self.iterate(i, p)
            self.state["c_iteration"] = i + 1
            self.save()
        self.state["phase"] = "FREEZE"
        self.save()

    def iterate(self, i: int, p: str) -> None:
        task = T.make_task(T.DOMAINS[i % 6], 0, i, "devC")
        cur = self.mods[p]
        parsed, prior = self.card(task, p, cur, tag="cardC")
        d = decide(task, prior, [cur["learned"]])
        d["proposal"] = d["proposal_unconstrained"]
        row = metrics.outcome_record(task, p, "E", d, prior, d["persona_u"])
        inky_gate = host.execute(task, parsed["choice"])
        diag = self.diagnostics(task, p, cur, parsed, inky_gate, d, row)
        e = self.call(f"patch|{task.task_id}|{p}|{cur['hash']}",
                      patch_prompt(cur["spec"], diag, self.state["recent_rejections"][p]),
                      PATCH_MAX_TOKENS, "stageC_patch", CFG["patch_temperature"])
        pobj = extract_json(e["text"])
        ops = pobj.get("ops") if isinstance(pobj, dict) else None
        failure, child, ch, measured, audit = None, None, None, {}, None
        try:
            child = apply_patch(cur["spec"], ops)
            errs = validate(child, p)
            if errs:
                failure = "invalid_patch: " + "; ".join(errs[:3])
            elif spec_hash(child) == cur["hash"]:
                failure = "no_change"
        except SpecError as exc:
            failure = f"invalid_patch: {exc}"
        if failure is None:
            ch = spec_hash(child)
            try:
                cm = self.build_modules(p, child, f"g{self.state['personas'][p]['generation'] + 1}")
            except Exception as exc:
                failure = f"compile_failed: {type(exc).__name__}"
        if failure is None:
            fresh = self.fresh_tasks(p, CFG["c_fresh_cases"])
            rp, rc = [], []
            priors = {}
            for t in fresh:
                _, pr = self.neutral(t)
                priors[t.task_id] = pr
                rp.append(self.module_row(t, p, "D", pr, [cur["learned"]]))
                rc.append(self.module_row(t, p, "D", pr, [cm["learned"]]))
            audit = host.boundary_audit(fresh, {"child": lambda t: decide(t, priors[t.task_id], [cm["learned"]])})
            jp, jc = metrics.objective(rp), metrics.objective(rc)
            measured = {"parent_J": jp, "child_J": jc, "delta": jc - jp, "fresh_cases": [t.task_id for t in fresh],
                        "parent": metrics.summarize(rp), "child": metrics.summarize(rc), "boundary_audit": audit}
            if not audit["passed"]:
                failure = "boundary_audit_failed"
            elif jc < jp + CFG["c_accept_margin"]:
                failure = "not_better_on_fresh_cases"
        accepted = failure is None
        info = self.state["personas"][p]
        gen = info["generation"]
        verifier = {k: row[k] for k in ("utility", "best_utility", "optimal", "signature_available", "signature_hit",
                                        "abstained", "abstain_optimal", "unauthorized_execution")}
        rec = revision_record(task=task, persona=p, parent_spec=cur["spec"], parent_hash=cur["hash"],
                              child_spec=child if failure is None or not str(failure).startswith("invalid") else None,
                              child_hash=ch, candidate_actions=[a.as_dict() for a in task.actions],
                              inky_response={"choice": parsed["choice"], "confidence": parsed["confidence"],
                                             "claimed_tags": parsed["claimed_tags"], "patch_text": e["text"][:3000]},
                              decision={"executed": row["executed"], "confidence": row["confidence"],
                                        "abstain_reason": d["abstain_reason"]},
                              verifier=verifier, diagnostics=diag, patch={"ops": ops, "rationale": (pobj or {}).get("rationale") if isinstance(pobj, dict) else None},
                              measured=measured, accepted=accepted, generation=gen, iteration=i,
                              source_hashes=self.hashes, failure=failure)
        append_jsonl(self.out / ("training_records.jsonl" if accepted else "failed_trajectories.jsonl"), rec)
        append_jsonl(self.out / "lineage.jsonl", {"persona": p, "parent": cur["hash"], "child": ch, "generation": gen + 1,
                                                 "iteration": i, "accepted": accepted, "failure": failure,
                                                 "delta_J": measured.get("delta"), "source": "inky_patch"})
        append_jsonl(self.out / "stageC_iterations.jsonl", {
            "iteration": i, "persona": p, "task_id": task.task_id, "codes": diag["codes"], "ops": ops,
            "accepted": accepted, "failure": failure, "parent_J": measured.get("parent_J"),
            "child_J": measured.get("child_J"), "inky_parse_ok": parsed["parsed"], "t": time.time()})
        if accepted:
            self.mods[p] = cm
            info.update(current=ch, generation=gen + 1, accepted=info["accepted"] + 1)
        else:
            info["rejected"] = info["rejected"] + 1
            self.state["recent_rejections"][p] = (self.state["recent_rejections"][p] + [
                f"{json.dumps(ops)[:160]} -> {failure[:80] if failure else ''}"])[-3:]

    def composition(self, k: int) -> None:
        pairs = [pq for pq in COMPOSITION_PAIRS if pq[0] in self.mods and pq[1] in self.mods]
        if not pairs:
            return
        p, q = pairs[k % len(pairs)]
        fresh = self.fresh_tasks("compose", CFG["c_fresh_cases"])
        rows = {"solo_p": [], "solo_q": [], "comp_p": [], "comp_q": []}
        priors = {}
        for t in fresh:
            _, pr = self.neutral(t)
            priors[t.task_id] = pr
            rows["solo_p"].append(self.module_row(t, p, "solo", pr, [self.mods[p]["learned"]]))
            rows["solo_q"].append(self.module_row(t, q, "solo", pr, [self.mods[q]["learned"]]))
            dc = decide(t, pr, [self.mods[p]["learned"], self.mods[q]["learned"]])
            dc["proposal"] = dc["proposal_unconstrained"]
            rows["comp_p"].append(metrics.outcome_record(t, p, "comp", dc, None))
            rows["comp_q"].append(metrics.outcome_record(t, q, "comp", dc, None))
        audit = host.boundary_audit(fresh, {"composite": lambda t: decide(
            t, priors[t.task_id], [self.mods[p]["learned"], self.mods[q]["learned"]])})
        s = {k2: metrics.summarize(v) for k2, v in rows.items()}

        def ratio(a, b):
            return (a / b) if (a is not None and b) else None
        rec = {"k": k, "pair": [p, q], "spec_hashes": [self.mods[p]["hash"], self.mods[q]["hash"]],
               "fresh_cases": [t.task_id for t in fresh], "summary": s,
               "retention_p": ratio(s["comp_p"]["fidelity"], s["solo_p"]["fidelity"]),
               "retention_q": ratio(s["comp_q"]["fidelity"], s["solo_q"]["fidelity"]),
               "utility_comp": s["comp_p"]["normalized_utility"],
               "utility_solo_max": max(s["solo_p"]["normalized_utility"], s["solo_q"]["normalized_utility"]),
               "boundary_audit": audit}
        rec["cooperates"] = bool(rec["retention_p"] is not None and rec["retention_q"] is not None
                                 and rec["retention_p"] >= 0.5 and rec["retention_q"] >= 0.5
                                 and rec["utility_comp"] >= min(s["solo_p"]["normalized_utility"],
                                                                s["solo_q"]["normalized_utility"]) and audit["passed"])
        append_jsonl(self.out / "compositions.jsonl", rec)

    # ------------------------------------------------------------------ freeze + final
    def freeze(self) -> None:
        frozen = {}
        for p in self.active():
            m = self.mods[p]
            write_json_atomic(self.out / "frozen" / f"{p}.json", {"persona": p, "spec_hash": m["hash"],
                                                                "generation": self.state["personas"][p]["generation"],
                                                                "spec": m["spec"]})
            frozen[p] = m["hash"]
        self.state["frozen"] = frozen
        self.state["frozen_at"] = time.time()
        n = self.planned_final_tasks()
        self.state["final_tasks"] = n
        deck = T.suite("heldout", 1, math.ceil(n / 6))[:n]
        fns = {}
        for p in self.active():
            fns[f"rubric:{p}"] = (lambda t, m=self.mods[p]: decide(t, prior_from_response(t, None), [m["rubric"]]))
            fns[f"learned:{p}"] = (lambda t, m=self.mods[p]: decide(t, prior_from_response(t, None), [m["learned"]]))
        for p, q in COMPOSITION_PAIRS:
            if p in self.mods and q in self.mods:
                fns[f"comp:{p}+{q}"] = (lambda t, a=self.mods[p], b=self.mods[q]: decide(
                    t, prior_from_response(t, None), [a["learned"], b["learned"]]))
        audit = host.boundary_audit(deck, fns)
        adv = self.adversarial_checks()
        write_json_atomic(self.out / "cca_audit_freeze.json", {
            "heldout_deck": len(deck), "audit": audit, "adversarial": adv,
            "host_code_hash_start": self.state["host_code_hash_start"], "host_code_hash_freeze": host.host_code_hash(),
            "host_code_unchanged": self.state["host_code_hash_start"] == host.host_code_hash(),
            "frozen_spec_hashes": frozen})
        self.state["phase"] = "FINAL"
        self.save()

    def planned_final_tasks(self) -> int:
        spc = self.state.get("seconds_per_call") or 6.0
        per_task = (1 + 2 * max(1, len(self.active()))) * spc * 1.1
        avail = self.state["deadline"] - time.time() - CFG["report_margin_s"]
        return int(max(CFG["final_min_tasks"], min(CFG["final_max_tasks"], avail // per_task)))

    def final(self) -> None:
        n = self.state["final_tasks"] or self.planned_final_tasks()
        deck = T.suite("heldout", 1, math.ceil(n / 6))[:n]
        hard_end = self.state["deadline"] - CFG["report_margin_s"] / 2
        for t in deck:
            if t.task_id in self.state["final_done"]:
                continue
            if time.time() > hard_end:
                self.state["final_truncated"] = True
                break
            rows = self.evaluate_task(t, include_initial=True)
            for r in rows:
                append_jsonl(self.out / "final_records.jsonl", r)
                sh = self.mods[r["persona"]]["hash"] if r["persona"] in self.mods else None
                append_jsonl(self.out / "decision_records.jsonl", decision_record(r, t, sh, None, self.hashes_digest))
            self.state["final_done"].append(t.task_id)
            self.save()
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
        code = 0
        if os.name == "nt" and not self.dry_run:
            try:  # per-process idle-sleep block; released automatically when the process exits
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            except Exception:
                pass
        try:
            if self.state["phase"] == "A":
                self.stage_a()
            if self.state["phase"] == "B":
                self.stage_b()
            if self.state["phase"] == "C":
                self.stage_c()
            if self.state["phase"] == "FREEZE":
                self.freeze()
            if self.state["phase"] == "FINAL":
                self.final()
        except Stop as s:
            self.state["stop_reason"] = str(s)
            code = 3 if str(s) == "ram_cap" else 0
            if str(s) == "deadline" and self.state["phase"] in ("C", "FREEZE"):
                # The final window is reserved before the deadline; reaching here means latency
                # outran the plan. Freeze anyway so held-out evidence is not silently skipped.
                pass
        except CredentialRejected as exc:
            self.state["stop_reason"] = f"credential_rejected: {exc}"
            code = 2
        except Exception as exc:  # record and still write a report
            self.state["stop_reason"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
            (self.out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            code = 1
        finally:
            self.state["ended_at"] = time.time()
            self.save()
            try:
                from .report import build_report
                build_report(self.out)
            except Exception:
                (self.out / "report_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--dry-run", action="store_true", help="offline fake model; results are not evidence")
    a = ap.parse_args(argv)
    if a.hours > 8.0:
        ap.error("the registered maximum runtime is eight hours")
    if a.dry_run:
        from .fixtures import FakeTransport
        client = InkyClient("dry-run", Usage(), transport=FakeTransport(CFG["seed"]), min_interval=0.0,
                            effort=CFG["reasoning_effort"], context_limit=CFG["context_limit"])
    else:
        client = InkyClient(load_key(a.key_file), Usage(), effort=CFG["reasoning_effort"],
                            context_limit=CFG["context_limit"])
    gym = Gym(a.out, a.hours, client, a.dry_run)
    if gym.state.get("dry_run") != a.dry_run:
        ap.error("output directory belongs to a run with a different dry-run setting")
    return gym.run()


if __name__ == "__main__":
    sys.exit(main())
