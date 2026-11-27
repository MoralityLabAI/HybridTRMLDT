"""Fresh native benchmark for the COG-13K table interval receiver.

Reuse J's frozen search/audit and K's compiled receiver without modifying either.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import re
import resource
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
JDIR = HERE.parent / "cog13j_native_progressive"
KDIR = HERE.parent / "cog13k_table_interval_receiver"
sys.path.insert(0, str(JDIR))
import native as jn
import transport as jt
sys.path.insert(0, str(KDIR))
import benchmark as kb

ARMS = ("frequency", "local_dense", "shared_dense", "decimal_progressive", "table_progressive")
FAMILIES = ("nat", "ring", "finset", "mixed", "logic", "modular")
LIMIT = 45_000_000_000
C_SHA = jt.SHA
H_SHA = jn.H_SHA
J_SHA = kb.J_SHA
TABLE_SHA = "0af5f6f1ffb02dd43cf81ec3a8a29fa50fdc8f262ba97f88088e154ff2ac4757"
SETUP = {"frequency": 128, "local_dense": 4424, "shared_dense": 4424,
         "decimal_progressive": 5960, "table_progressive": 5960 + 32784}
BASE_ROUTE = jt.route
BASE_RECEIVE = jt.receive
BASE_J_SOURCE = jn.source
MENU = jt.MENU


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def dump(path, obj):
    jn.dump(path, obj)


def registry():
    roots, refs = [], {}
    for s in range(2):
        for i in range(12):
            n = 6103 + 397*s + 29*i
            depth = 7 + i % 3 if s else 3 + i % 2
            family = FAMILIES[i % len(FAMILIES)]
            leaves, steps = [], []
            for k in range(n, n + depth):
                if family == "nat":
                    binders = "(a b c : ℕ) (hab : a ≤ b) (hbc : b ≤ c)"
                    expr = f"a + {k} ≤ c + {k+2}"
                    tactics = ["omega"]
                elif family == "ring":
                    binders = "(x y : ℚ)"
                    expr = f"(x + y + {k})^2 = x*x + 2*x*y + y*y + {2*k}*x + {2*k}*y + {k*k}"
                    tactics = ["ring"]
                elif family == "finset":
                    binders = "(x : ℕ) (s t u : Finset ℕ)"
                    expr = ("x ∈ (s ∪ t) ∩ u ↔ (x ∈ s ∨ x ∈ t) ∧ x ∈ u"
                            if k % 2 else "x ∈ s ∩ (t ∪ u) ↔ x ∈ s ∧ (x ∈ t ∨ x ∈ u)")
                    tactics = ["simp"]
                elif family == "mixed":
                    binders = "(a b : ℕ) (hab : a ≤ b) (P Q : Prop) (hp : P) (hq : Q)"
                    expr = f"P → (Q ∧ a + {k} ≤ b + {k+2})"
                    tactics = ["intro hnew", "constructor", "assumption", "omega"]
                elif family == "logic":
                    binders = "(P Q R : Prop)"
                    expr = ("(P ∧ Q) ∧ R ↔ P ∧ (Q ∧ R)" if k % 2
                            else "P ∧ Q ↔ Q ∧ P")
                    tactics = ["aesop"]
                else:
                    binders = "(a : ℕ)"
                    expr = f"(a + {k}) % 7 ≤ 6"
                    tactics = ["omega"]
                leaves.append("(" + expr + ")")
                steps.extend((["constructor"] if k != n+depth-1 else []) + tactics)
            # Finset and propositional variants include a visible, unused
            # binder to separate exact source strings. Structural similarity
            # is reported, rather than presented as semantic novelty.
            if family in ("finset", "logic"):
                binders += f" (tag{n} : ℕ)"
            root = {"id": f"cog13l_{s}_{i}", "split": "deeper" if s else "ordinary",
                    "family": family, "depth": depth,
                    "statement": binders + " : " + " ∧ ".join(leaves)}
            roots.append(root)
            refs[root["id"]] = steps
    need(len(roots) == len({r["statement"] for r in roots}) == 24, "root roster")
    need(all(a in MENU for steps in refs.values() for a in steps), "reference grammar")
    return roots, refs


def source(c, actions, probe=False):
    need(re.fullmatch(r"cog13l_[01]_(?:[0-9]|1[01])", c["id"]) is not None,
         "source identity")
    need(all(a in MENU for a in actions), "source grammar")
    return ("import Mathlib\n" + ("import StateCacheJ\n" if probe else "") +
            "set_option maxHeartbeats 100000000\n" +
            f"theorem {c['id']} {c['statement']} := by\n" +
            ("  cog13j_probe\n" if probe else "".join("  "+a+"\n" for a in actions)) +
            f"#print axioms {c['id']}\n")


def read_registry(path, member, digest):
    need(jt.sha(path.read_bytes()) == digest, "prior archive pin")
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(member))


def check_registry(roots, old, h, old_j):
    prior = old["cases"] + read_registry(h, "registry.json", H_SHA) + read_registry(
        old_j, "native/registry.json", J_SHA)
    need(len(prior) == 56, "prior root count")
    need(not {r["statement"] for r in roots} & {r["statement"] for r in prior},
         "prior statement overlap")
    return {"newRoots": len(roots), "priorRoots": len(prior), "exactOverlap": 0,
            "semanticIndependenceClaimed": False}


def compile_receiver(out):
    lib = out / "receiver.so"
    cmd = ["gcc", "-O3", "-std=c11", "-fPIC", "-shared", "-fno-fast-math",
           "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
           str(KDIR / "receiver.c"), "-o", str(lib), "-lm"]
    t0 = time.monotonic_ns()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run(cmd, check=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return lib, {"argv": cmd, "sourceSha256": jt.sha((KDIR/"receiver.c").read_bytes()),
                 "childCpuNs": round(1e9*((after.ru_utime-before.ru_utime) +
                                          (after.ru_stime-before.ru_stime))),
                 "wallNs": time.monotonic_ns()-t0}


class Adapter:
    def __init__(self, lib, model):
        self.fast = kb.Receiver(lib, model)
        need(jt.sha(self.fast.table_bytes) == TABLE_SHA, "table identity")
        self.checks = self.fast.test()
        self.routes = 0

    def route(self, goals, options, arm, model, freq):
        use_table = arm == "table_progressive"
        self.fast.raw.clear()
        jt.receive = self.fast.receive if use_table else BASE_RECEIVE
        try:
            r = BASE_ROUTE(goals, options, "progressive" if arm.endswith("progressive") else arm,
                           model, freq)
        finally:
            jt.receive = BASE_RECEIVE
        r["tableIntervals"] = [dict(x) for x in self.fast.raw] if use_table else []
        self.routes += 1
        return r

    def install(self):
        jn.source = source
        jn.route = self.route
        jn.LIMIT = LIMIT

    def restore(self):
        jn.source = BASE_J_SOURCE
        jn.route = BASE_ROUTE
        jn.LIMIT = 60_000_000_000
        jt.receive = BASE_RECEIVE

    def audit_intervals(self, episode, model, freq):
        count = certificates = 0
        for e in episode["events"]:
            if "failure" in e:
                continue
            route = e["route"]
            need(route["denseAgreement"] is not False, "dense action mismatch")
            if episode["arm"] != "table_progressive":
                need(route["tableIntervals"] == [], "unexpected interval receipt")
                continue
            # Recompute the recorded packet trace and intervals. The dense
            # score vector is computed outside the timed search and is never
            # passed into the table receiver.
            replay = self.route(e["before"], e["options"], episode["arm"], model, freq)
            need(replay["tableIntervals"] == route["tableIntervals"], "forged table interval")
            packets = [s for s in route["stages"] if s.get("mode") in (6, 7) and "packet" in s]
            need(len(packets) == len(route["tableIntervals"]), "interval packet binding")
            ids = [MENU.index(a) for a in e["options"]]
            scores = jt.dense(model, jt.goal_vector(e["before"]), ids)
            for p, q in zip(packets, route["tableIntervals"], strict=True):
                count += 1
                if q.get("invalid"):
                    need(p["reply"] == 255, "unsupported interval received credit")
                    continue
                bounds = q["bounds"]
                need(len(bounds) == 2*len(scores) and q["code"] == p["reply"], "bound shape/reply")
                need(all(math.isfinite(x) for x in bounds), "nonfinite interval")
                need(all(lo-1e-10 <= x <= hi+1e-10 and lo <= hi for lo, hi, x in
                         zip(bounds[::2], bounds[1::2], scores, strict=True)),
                     "dense score outside interval")
                if q["code"] != 255:
                    need(q["code"] == jt.winner(scores), "false table certificate")
                    certificates += 1
        return {"tableIntervalsChecked": count, "tableCertificates": certificates}


def audit_episode(adapter, episode, model, freq, folder=None):
    need(episode["reason"] != "evidence_error", "search evidence error")
    t = time.process_time_ns()
    # J replays the policy, cache, and certificates and checks raw files.
    v = jn.audit(episode, model, freq, folder)
    v.update(adapter.audit_intervals(episode, model, freq))
    stages = [s for e in episode["events"] if "route" in e for s in e["route"]["stages"]]
    v["fallbacks"] = sum(s.get("mode") == 0 for s in stages
                         if episode["arm"].endswith("progressive"))
    v["senderAbstentions"] = sum("abstainBeforeSend" in s for s in stages)
    v["offlineAuditCpuNs"] = time.process_time_ns() - t
    return v


def fixture(adapter, model, freq):
    results = {}
    for arm in ARMS:
        r = jn.search(registry()[0][0], arm, model, freq, jn.Fixture())
        m = audit_episode(adapter, r, model, freq)
        need(r["solved"], "fixture completion")
        results[arm] = (r, m)
    import copy
    original = results["table_progressive"][0]
    def rejects(mutate):
        r = copy.deepcopy(original)
        mutate(r)
        try:
            audit_episode(adapter, r, model, freq)
        except (ValueError, KeyError, TypeError):
            return True
        return False
    need(rejects(lambda r: r["events"][0]["reply"].update(executed=999)), "forged cache")
    need(rejects(lambda r: r.update(certificates=[])), "forged certificate")
    need(rejects(lambda r: r["events"][0]["route"].update(bytes=0)), "forged bytes")
    def forge(r):
        e = next(e for e in r["events"] if e["route"]["tableIntervals"])
        e["route"]["tableIntervals"][0]["code"] = 999
    need(rejects(forge), "forged interval")
    return {"fixture": True, "arms": len(results),
            "decisions": sum(m.get("decisions", 0) for _, m in results.values()),
            "forgeryRejected": ["cache", "certificate", "bytes", "interval"],
            "tableTests": adapter.checks}


def preflight(lean, cwd, folder, roots, refs):
    folder.mkdir()
    file = folder/"Reference.lean"
    statements = [source(r, refs[r["id"]]).removeprefix("import Mathlib\n") for r in roots]
    file.write_text("import Mathlib\n" + "\n".join(statements), encoding="utf-8")
    start = time.monotonic_ns()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (folder/"reference.log").open("xb") as log:
        p = subprocess.run([lean, str(file)], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           timeout=240, check=False)
    output = (folder/"reference.log").read_text(errors="replace")
    ok = [r["id"] for r in roots if jn.accepted(p.returncode, output, r["id"])]
    bad = folder/"Negative.lean"
    bad.write_text("import Mathlib\ntheorem cog13l_bad : True := by\n  exact cog13l_missing_identifier\n",
                   encoding="utf-8")
    with (folder/"negative.log").open("xb") as log:
        n = subprocess.run([lean, str(bad)], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           timeout=30, check=False)
    neglog = (folder/"negative.log").read_text(errors="replace")
    negative = n.returncode != 0 and "cog13l_missing_identifier" in neglog and bool(
        re.search(r"error(?:\([^\n]*\))?:", neglog))
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {"passed": len(ok) == len(roots) and negative, "referenceRoots": ok,
               "referenceScriptLengths": {k: len(v) for k,v in refs.items()},
               "referenceSourceSha256": jt.sha(file.read_bytes()), "referenceReturnCode": p.returncode,
               "negativeRejected": negative, "negativeReturnCode": n.returncode,
               "childCpuNs": round(1e9*((after.ru_utime-before.ru_utime) +
                                        (after.ru_stime-before.ru_stime))),
               "wallNs": time.monotonic_ns()-start}
    dump(folder/"receipt.json", receipt)
    need(receipt["passed"], "reference preflight did not certify complete cohort")
    return receipt


def trajectory(r):
    return [{"actions": e["actions"],
             "reply": None if not e.get("reply") else
             {"accepted": e["reply"]["result"]["accepted"],
              "goals": e["reply"]["result"].get("goals") if e["reply"]["result"]["accepted"] else None}}
            for e in r["events"] if "actions" in e]


def summarize(rows, setup_ns):
    out = {}
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        c = Counter()
        for r in rr:
            c.update(r["metrics"])
        credited = [r["root"] for r in rr if r["solved"]]
        setup = setup_ns if arm == "table_progressive" else 0
        cpu = c["nativeChildCpuNs"] + c["hostCpuNs"] + setup
        families = {}
        for split in ("ordinary", "deeper"):
            families[split] = {f: {"roots": sum(r["split"] == split and r["family"] == f for r in rr),
                                   "solved": sum(r["split"] == split and r["family"] == f and r["solved"] for r in rr)}
                               for f in FAMILIES}
        out[arm] = {**dict(c), "roots": len(rr), "solved": len(credited), "solvedRoots": credited,
                    "setupCpuNs": setup, "setupBytes": SETUP[arm], "setupAndRecurringBytes": SETUP[arm]+c["bytes"],
                    "chargedCpuNs": cpu, "routingCpuNs": c["senderNs"]+c["receiverNs"],
                    "proofsPerCpuHour": len(credited)*3.6e12/cpu if cpu else None, "families": families}
    return out


def matrix(args, adapter, roots, old, model, freq, out):
    roster = check_registry(roots, old, args.old_h, args.old_j)
    dump(out/"registry_check.json", roster)
    pre = preflight(args.lean, args.cwd, out/"preflight", roots, registry()[1])
    rows, errors, common, initial, traces = [], [], {}, {}, {}
    for i, c in enumerate(roots):
        for arm in ARMS[i%len(ARMS):] + ARMS[:i%len(ARMS)]:
            folder = out/(c["id"]+"_"+arm)
            folder.mkdir()
            r = None
            metrics = {}
            reason = "infrastructure"
            host_start = time.process_time_ns()
            wall_start = time.monotonic_ns()
            child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            try:
                r = jn.search(c, arm, model, freq, jn.Native(args.lean, args.cwd, folder))
                dump(folder/"episode.json", r)
                reason = r["reason"]
                metrics = audit_episode(adapter, r, model, freq, folder)
                if r["initial"]:
                    need(initial.setdefault(c["id"], r["initial"]) == r["initial"], "root observation")
                for e in r["events"]:
                    if not e.get("reply"):
                        continue
                    key = (c["id"], tuple(e["actions"]))
                    v = e["reply"]["result"]
                    verdict = {"accepted": v["accepted"], "goals": v.get("goals") if v["accepted"] else None}
                    need(common.setdefault(key, verdict) == verdict, "common native verdict")
                traces[(c["id"], arm)] = trajectory(r)
            except Exception as e:
                errors.append({"root": c["id"], "arm": arm, "error": repr(e)})
                dump(folder/"audit_error.json", errors[-1])
                # Retain observed costs when an audit fails. Do not grant proof
                # credit or treat a missing audit metric as zero work.
                if r is not None:
                    metrics["hostCpuNs"] = r["hostCpuNs"]
                    metrics["nativeChildCpuNs"] = sum(p["cpuNs"] for p in r["processes"])
                    metrics["wallNs"] = r["elapsedNs"]
                else:
                    end = resource.getrusage(resource.RUSAGE_CHILDREN)
                    metrics["hostCpuNs"] = time.process_time_ns()-host_start
                    metrics["wallNs"] = time.monotonic_ns()-wall_start
                    metrics["nativeChildCpuNs"] = round(1e9*((end.ru_utime-child_start.ru_utime) +
                                                             (end.ru_stime-child_start.ru_stime)))
                reason = "evidence_error"
            rows.append({"root": c["id"], "split": c["split"], "family": c["family"], "arm": arm,
                         "solved": bool(r and r["solved"] and metrics and reason != "evidence_error"),
                         "reason": reason, "metrics": metrics})
            with (out/"rows.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1], ensure_ascii=False)+"\n")
    totals = summarize(rows, adapter.fast.setup_ns)
    d = totals["decimal_progressive"]
    t = totals["table_progressive"]
    b = totals["shared_dense"]
    unknowns = sum(r["reason"] in ("query_unknown", "startup_unknown", "evidence_error", "infrastructure") for r in rows)
    complete = len(rows) == 120 and len({(r["root"],r["arm"]) for r in rows}) == 120 and not errors
    clean = complete and not unknowns
    divergences = [c["id"] for c in roots if traces.get((c["id"], "decimal_progressive")) !=
                   traces.get((c["id"], "table_progressive"))]
    gates = {"clean": bool(clean),
             "sameDenseRootSet": bool(clean and b["solved"] and all(set(totals[a]["solvedRoots"]) ==
                                       set(b["solvedRoots"]) for a in ARMS[1:])),
             "trajectoryParity": bool(clean and not divergences),
             "bytes": bool(clean and b.get("bytes",0) and d.get("bytes",0) and
                           t.get("bytes",0) <= min(1.05*d["bytes"], 0.5*b["bytes"])),
             "receiver5x": bool(clean and t.get("receiverNs",0)>0 and
                                5*t["receiverNs"] <= d.get("receiverNs",0)),
             "proofCpu": bool(clean and d["chargedCpuNs"]>0 and t["chargedCpuNs"] <= 1.05*d["chargedCpuNs"])}
    result = {"schema": "cog13l.native.v1", "complete": complete, "unknowns": unknowns, "errors": errors,
              "gates": gates, "trajectoryDivergences": divergences, "roster": roster,
              "setupTableCpuNs": adapter.fast.setup_ns, "tableSha256": TABLE_SHA,
              "preflight": pre, "totals": totals, "rows": rows,
              "sourceArchives": {"C": C_SHA, "H": H_SHA, "J": J_SHA}}
    dump(out/"result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}, indent=2))
    return complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--old-h", type=Path)
    p.add_argument("--old-j", type=Path)
    p.add_argument("--lean")
    p.add_argument("--cwd", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fixture", action="store_true")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.process_time_ns()
    old, model, freq = jt.load(args.archive)
    load_ns = time.process_time_ns()-t0
    roots, refs = registry()
    dump(out/"registry.json", roots)
    lib, build = compile_receiver(out)
    adapter = Adapter(lib, model)
    dump(out/"setup.json", {"commonArchiveLoadCpuNs": load_ns, "compiler": build,
                           "tableCpuNs": adapter.fast.setup_ns, "tableBytes": len(adapter.fast.table_bytes),
                           "tableSha256": TABLE_SHA, "tableTests": adapter.checks})
    adapter.install()
    try:
        f = fixture(adapter, model, freq)
        dump(out/"fixture.json", f)
        if args.fixture:
            print(json.dumps(f, indent=2))
            return 0
        need(args.old_h and args.old_j and args.lean and args.cwd, "native arguments")
        return 0 if matrix(args, adapter, roots, old, model, freq, out) else 1
    finally:
        adapter.restore()


if __name__ == "__main__":
    raise SystemExit(main())
