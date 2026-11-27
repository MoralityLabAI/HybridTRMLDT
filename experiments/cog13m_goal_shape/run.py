"""COG-13M: frozen goal-shape prior, matched native search and width control."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

from audit_l import SHA as L_SHA, audit as audit_l

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cog13l", HERE.parent/"cog13l_native_table_routing"/"run.py")
l = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l)
jn, jt = l.jn, l.jt
MENU = jt.MENU
ARMS = ("frequency", "dense", "dense_wide", "shape_frequency", "shape_dense", "shape_dense_wide")
LIMIT = 45_000_000_000
WIDE = {"dense_wide", "shape_dense_wide"}
BASE = {"frequency", "shape_frequency"}


def need(x, msg):
    if not x:
        raise ValueError(msg)


def registry():
    roots, refs = [], {}
    for s in range(2):
        for i, family in enumerate(["modular"]*6+["nat","nat","ring","finset","mixed","logic"]):
            n = 10127+613*s+53*i
            depth = 7+i%4 if s else 3+i%3
            leaves, steps = [], []
            for j in range(depth):
                k=n+j*3
                if family=="modular":
                    m=(5,7,9,11,13,17)[i]
                    b="(a : ℕ)"
                    expr=f"(a + {k}) % {m} < {m}" if j%2 else f"(a + {k}) % {m} ≤ {m-1}"
                    t=["omega"]
                elif family=="nat":
                    b="(a b c : ℕ) (hab : a ≤ b) (hbc : b ≤ c)"
                    expr=f"a + {k} ≤ c + {k+3}"
                    t=["omega"]
                elif family=="ring":
                    b="(x y : ℚ)"
                    expr=f"(x - y + {k})^2 = x*x - 2*x*y + y*y + {2*k}*x - {2*k}*y + {k*k}"
                    t=["ring"]
                elif family=="finset":
                    b=f"(x : ℕ) (s t u : Finset ℕ) (tag{n} : ℕ)"
                    expr="x ∈ u ∩ (s ∪ t) ↔ x ∈ u ∧ (x ∈ s ∨ x ∈ t)"
                    t=["simp"]
                elif family=="mixed":
                    b="(a b : ℕ) (hab : a ≤ b) (P Q : Prop) (hp : P) (hq : Q)"
                    expr=f"Q → (P ∧ a + {k} ≤ b + {k+4})"
                    t=["intro hnew","constructor","assumption","omega"]
                else:
                    b=f"(P Q R : Prop) (tag{n} : ℕ)"
                    expr="(P ∧ Q) ∧ R ↔ Q ∧ (R ∧ P)"
                    t=["aesop"]
                leaves.append("("+expr+")")
                steps.extend((["constructor"] if j<depth-1 else [])+t)
            r={"id":f"cog13m_{s}_{i}","split":"deeper" if s else "ordinary","family":family,
               "depth":depth,"statement":b+" : "+" ∧ ".join(leaves)}
            roots.append(r)
            refs[r["id"]]=steps
    need(len(roots)==len({r["statement"] for r in roots})==24,"registry")
    need(all(a in MENU for v in refs.values() for a in v),"reference grammar")
    return roots,refs


def source(c, actions, probe=False):
    need(re.fullmatch(r"cog13m_[01]_(?:[0-9]|1[01])", c["id"]) and all(a in MENU for a in actions),"source")
    return ("import Mathlib\n"+("import StateCacheJ\n" if probe else "")+
            "set_option maxHeartbeats 100000000\n"+f"theorem {c['id']} {c['statement']} := by\n"+
            ("  cog13j_probe\n" if probe else "".join("  "+a+"\n" for a in actions))+
            f"#print axioms {c['id']}\n")


def target(goal):
    return goal.rsplit("⊢",1)[-1].strip()


def unwrap(t):
    while t.startswith("(") and t.endswith(")"):
        d=0
        end=None
        for i,ch in enumerate(t):
            d+=(ch=="(")-(ch==")")
            if d==0:
                end=i
                break
        if end!=len(t)-1:
            break
        t=t[1:-1].strip()
    return t


def shape(goal):
    t=unwrap(target(goal))
    if t.startswith(("∀","∃")):
        return "binder"
    d=0
    ops=set()
    for ch in t:
        if ch in "([{⟨":
            d+=1
        elif ch in ")]}⟩":
            d=max(0,d-1)
        elif d==0 and ch in "↔→∧∨=≤≥<>":
            ops.add(ch)
    for op, name in (("↔","iff"),("→","imp"),("∨","or"),("∧","and")):
        if op in ops:
            return name
    # The rule relies on display syntax, not proof semantics. No action is
    # excluded. This conservative recognition does not classify every atom.
    if ("ℕ" in goal or "ℤ" in goal or "%" in t) and (
        ops & set("≤≥<>") or re.search(r"\.(?:le|lt)\b",t)):
        return "nat_order"
    if "=" in ops and ("ℚ" in goal or "ℝ" in goal):
        return "ring_eq"
    return "other"


def priority(goals):
    shapes=[shape(g) for g in goals]
    first=shapes[0]
    if first=="nat_order":
        return (["all_goals omega","omega"] if len(shapes)>1 and all(x=="nat_order" for x in shapes)
                else ["omega","all_goals omega"]), ["constructor"]
    if first=="and":
        return ["aesop","constructor"], []
    if first in ("imp","binder"):
        return ["intro hnew","aesop"], []
    if first=="iff":
        return ["simp","aesop","constructor"], []
    if first=="or":
        return ["aesop","simp"], []
    if first=="ring_eq":
        return (["all_goals ring","ring"] if len(shapes)>1 and all(x=="ring_eq" for x in shapes)
                else ["ring","all_goals ring"]), ["constructor"]
    return [], []


def route(goals,options,arm,model,freq):
    start=time.process_time_ns()
    ids=[MENU.index(x) for x in options]
    need(ids==sorted(set(ids)) and ids,"canonical menu")
    scores=[freq[a] for a in options] if arm in BASE else jt.dense(model,jt.goal_vector(goals),ids)
    base=jt.winner(scores)
    prefer,defer=priority(goals) if arm.startswith("shape") else ([],[])
    def order(i):
        a=options[i]
        return (0,prefer.index(a),i) if a in prefer else (2,-scores[i],i) if a in defer else (1,-scores[i],i)
    top=min(range(len(options)),key=order)
    return {"action":options[top],"bytes":1,"senderNs":time.process_time_ns()-start,
            "receiverNs":0,"auditNs":0,"denseAgreement":None,
            "stages":[{"mode":"local_shape" if arm.startswith("shape") else "local_base",
                       "reply":top,"baseReply":base,"targetShape":shape(goals[0]),
                       "prefer":prefer,"defer":defer}]}


def advance_for(arm):
    cap=16 if arm in WIDE else 8
    def advance(stack,n,actions,r):
        if not r["accepted"] or r["goals"]==n["goals"]:
            return False
        if not r["goals"]:
            return True
        if len(actions)<12 and len(r["goals"])<=cap and not any(x["goals"]==r["goals"] for x in stack):
            stack.append(jn.node(r["goals"],actions))
        return False
    return advance


def configure(arm):
    jn.source=source
    jn.route=route
    jn.advance=advance_for(arm)
    jn.LIMIT=LIMIT


def audit(r,model,freq,folder=None):
    configure(r["arm"])
    need(r["reason"]!="evidence_error","search error")
    v=jn.audit(r,model,freq,folder)
    v.update(atomicConstructors=0,wideResults=0,changedChoices=0)
    for e in r["events"]:
        if "route" not in e:
            continue
        stage=e["route"]["stages"][0]
        v["changedChoices"]+=stage["reply"]!=stage["baseReply"]
        if not e.get("reply"):
            continue
        q=e["reply"]["result"]
        v["wideResults"]+=q["accepted"] and len(q.get("goals",[]))>8
        v["atomicConstructors"]+=q["accepted"] and e["actions"][-1]=="constructor" and shape(e["before"][0])=="nat_order"
    return v


def fixture(model,freq):
    g=["a : ℕ\n⊢ (a + 17) % 7 ≤ 6","a : ℕ\n⊢ (a + 18) % 7 ≤ 6"]
    opts=list(MENU)
    need(route(g,opts,"shape_dense",model,freq)["action"]=="all_goals omega","all atomic")
    need(route(g[:1],opts,"shape_frequency",model,freq)["action"]=="omega","atomic omega")
    need(shape("a : ℕ\n⊢ (a%7 ≤ 6) ∧ (a%7 ≤ 6)")=="and","connective parser")
    need(shape("a : ℕ\n⊢ ((a + 2) % 7).le 5")=="nat_order","dot comparison")
    n=jn.node(["root"])
    x={"accepted":True,"goals":["g"+str(i) for i in range(9)]}
    a,b=[n],[n]
    advance_for("dense")(a,n,["aesop"],x)
    advance_for("dense_wide")(b,n,["aesop"],x)
    need(len(a)==1 and len(b)==2,"width control")
    total=0
    import copy
    for arm in ARMS:
        configure(arm)
        r=jn.search(registry()[0][0],arm,model,freq,jn.Fixture())
        v=audit(r,model,freq)
        need(r["solved"],"fixture proof")
        total+=v["decisions"]
        mutations=[
            lambda z:z["events"][0]["reply"].update(executed=999),
            lambda z:z.update(certificates=[]),
            lambda z:z["events"][0]["route"]["stages"][0].update(reply=999),
        ]
        for f in mutations:
            q=copy.deepcopy(r);f(q)
            try:audit(q,model,freq)
            except (ValueError,KeyError,TypeError):continue
            raise ValueError("forged receipt accepted")
    return {"fixture":True,"arms":6,"probes":total,"priorityChecks":4,"widthCheck":True,"negativeChecks":18}


def matrix(args,roots,refs,model,freq,out):
    old,_,_=jt.load(args.archive)
    prior=old["cases"]+l.read_registry(args.old_h,"registry.json",l.H_SHA)+l.read_registry(
        args.old_j,"native/registry.json",l.J_SHA)+l.read_registry(args.old_l,"native/registry.json",L_SHA)
    need(len(prior)==80 and not {r["statement"] for r in roots}&{r["statement"] for r in prior},"prior overlap")
    jn.source=source
    l.source=source
    pre=l.preflight(args.lean,args.cwd,out/"preflight",roots,refs)
    rows=[];errors=[];initial={};common={}
    for i,c in enumerate(roots):
        for arm in ARMS[i%6:]+ARMS[:i%6]:
            configure(arm)
            folder=out/(c["id"]+"_"+arm);folder.mkdir()
            r=None;metrics={};reason="infrastructure"
            host=time.process_time_ns();wall=time.monotonic_ns();child=resource.getrusage(resource.RUSAGE_CHILDREN)
            try:
                r=jn.search(c,arm,model,freq,jn.Native(args.lean,args.cwd,folder))
                l.dump(folder/"episode.json",r)
                metrics=audit(r,model,freq,folder);reason=r["reason"]
                if r["initial"]:need(initial.setdefault(c["id"],r["initial"])==r["initial"],"initial divergence")
                for e in r["events"]:
                    if not e.get("reply"):continue
                    v=e["reply"]["result"];k=(c["id"],tuple(e["actions"]))
                    q=(v["accepted"],v.get("goals") if v["accepted"] else None)
                    need(common.setdefault(k,q)==q,"native script divergence")
            except Exception as e:
                errors.append({"root":c["id"],"arm":arm,"error":repr(e)})
                l.dump(folder/"audit_error.json",errors[-1]);reason="evidence_error"
                if r:
                    metrics.update(hostCpuNs=r["hostCpuNs"],wallNs=r["elapsedNs"],nativeChildCpuNs=sum(p["cpuNs"] for p in r["processes"]))
                else:
                    end=resource.getrusage(resource.RUSAGE_CHILDREN)
                    metrics.update(hostCpuNs=time.process_time_ns()-host,wallNs=time.monotonic_ns()-wall,
                                   nativeChildCpuNs=round(1e9*((end.ru_utime-child.ru_utime)+(end.ru_stime-child.ru_stime))))
            row={"root":c["id"],"arm":arm,"family":c["family"],"split":c["split"],
                 "solved":bool(r and r["solved"] and reason!="evidence_error"),"reason":reason,"metrics":metrics}
            rows.append(row)
            with (out/"rows.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
    totals={}
    for arm in ARMS:
        rr=[r for r in rows if r["arm"]==arm]
        m=Counter()
        for r in rr:m.update(r["metrics"])
        solved=[r["root"] for r in rr if r["solved"]];mod=[r["root"] for r in rr if r["solved"] and r["family"]=="modular"]
        cpu=m["hostCpuNs"]+m["nativeChildCpuNs"]
        totals[arm]={**dict(m),"roots":len(rr),"solved":len(solved),"solvedRoots":solved,
                     "modularSolved":len(mod),"modularRootsSolved":mod,"setupBytes":128 if arm in BASE else 4424,
                     "chargedCpuNs":cpu,"proofsPerCpuHour":len(solved)*3.6e12/cpu if cpu else None}
    unknown=sum(r["reason"] in ("query_unknown","startup_unknown","evidence_error","infrastructure") for r in rows)
    complete=len(rows)==len({(r["root"],r["arm"]) for r in rows})==144 and not errors
    clean=complete and unknown==0
    d=totals["dense"];s=totals["shape_dense"];w=totals["shape_dense_wide"]
    nonmod=set(d["solvedRoots"])-set(d["modularRootsSolved"])
    gates={"clean":clean,"shapeRepair":clean and s["modularSolved"]>=d["modularSolved"]+2 and nonmod<=set(s["solvedRoots"]),
           "width":clean and w["modularSolved"]>s["modularSolved"]}
    contrasts={}
    for a,b in [("shape_dense","dense"),("dense_wide","dense"),("shape_dense_wide","shape_dense"),
                ("shape_dense_wide","dense_wide"),("shape_frequency","frequency"),("shape_dense","shape_frequency")]:
        x,y=set(totals[a]["solvedRoots"]),set(totals[b]["solvedRoots"])
        contrasts[a+"_vs_"+b]={"gains":sorted(x-y),"losses":sorted(y-x)}
    result={"schema":"cog13m.native.v1","complete":complete,"unknowns":unknown,"errors":errors,"gates":gates,
            "preflight":pre,"priorRoots":len(prior),"totals":totals,"contrasts":contrasts,"rows":rows}
    l.dump(out/"result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("--old-h",type=Path);p.add_argument("--old-j",type=Path)
    p.add_argument("--old-l",type=Path,required=True);p.add_argument("--lean");p.add_argument("--cwd",type=Path)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--fixture",action="store_true")
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    _,model,freq=jt.load(a.archive);roots,refs=registry()
    l.dump(out/"registry.json",roots);l.dump(out/"l_failure_audit.json",audit_l(a.old_l))
    f=fixture(model,freq);l.dump(out/"fixture.json",f);print(json.dumps(f))
    if a.fixture:return 0
    need(a.old_h and a.old_j and a.lean and a.cwd,"native parameters")
    return 0 if matrix(a,roots,refs,model,freq,out) else 1


if __name__=="__main__":
    raise SystemExit(main())
