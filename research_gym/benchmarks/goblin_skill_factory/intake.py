"""Stage audited Skill Factory receipts, keeping training admission disabled.

Consumes the nine same-source CPU workflow artifact directories. Replays native
audits, separates public observations from evaluator targets, groups all linked
roots, and rejects known evaluation lineage. No tokenizer or SFT export here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import torch

from . import audit as A, gym as G, molt_memory as MO, certified_skill as C
from . import persona as P, loop_clocks as L, research_repair as R
from . import mcp_geometry as MCP, certificate_search as CS, goblinbench as B

SCHEMA="goblinbench.staged_rollout.v1"
SOURCES={
 "GSF-01":("gsf-01-development","report.json","train_rollouts.jsonl",6144,A.audit),
 "GSF-02":("gsf-02-molt-development","molt_report.json","molt_rollouts.jsonl",1024,MO.audit),
 "GSF-03":("gsf-03-native-repair-development","repair_report.json","repair_rollouts.jsonl",2304,C.audit),
 "GSF-04":("gsf-04-dual-ldt-public-development","persona_report.json","persona_rollouts.jsonl",1280,P.audit),
 "GSF-05":("gsf-05-loop-clocks-development","clock_report.json","clock_rollouts.jsonl",1280,L.audit),
 "GSF-06":("gsf-06-research-repair-development","research_report.json","research_rollouts.jsonl",90,R.audit),
 "GSF-07":("gsf-07-mcp-geometry-development","mcp_report.json","mcp_rollouts.jsonl.gz",512,MCP.audit),
 "GSF-08":("gsf-08-certificate-search-development","certificate_report.json","certificate_rollouts.jsonl.gz",1920,CS.audit),
 "GB-SF-09":("gsf-09-bench-development","bench_report.json","bench_rollouts.jsonl",8,B.audit),
}
RAW_SCHEMAS={"GSF-01":G.SCHEMA+".rollout","GSF-02":MO.SCHEMA+".rollout",
 "GSF-03":C.SCHEMA+".rollout","GSF-04":P.SCHEMA+".rollout","GSF-05":L.SCHEMA+".rollout",
 "GSF-06":R.SCHEMA+".rollout","GSF-07":MCP.SCHEMA+".rollout","GSF-08":CS.SCHEMA+".rollout",
 "GB-SF-09":B.SCHEMA+".episode"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    op=gzip.open if path.suffix==".gz" else open
    with op(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                raise ValueError("blank receipt")
            yield json.loads(line)


def k(namespace,root):
    return B.lineage(namespace,root)


def rows_keys(c):
    return {k("gsf01-world",c["root"]),k("gsf01-rows",B.digest(c["rows"]))}


def molt_keys(c):
    return {k("GSF-02",c["root"]),k("molt-world",B.digest({a:v for a,v in c.items() if a not in ("root","split")}))}


def clock_keys(c):
    return {k("GSF-05",c["root"]),k("clock-graph",B.digest(c["maps"]))}


def meta_keys(root,family,cfg):
    keys={k("GSF-06",root)}
    for role,split in (("fit","train"),("cal","cal"),("stress_cal","shift"),
                       ("eval","dev"),("stress_eval","shift")):
        n=cfg["train_n"] if role=="fit" else cfg["eval_n"]
        for c in G.dataset(family,split,R.seed_for(root,role),n):
            keys.update(rows_keys(c))
    return keys


def held_keys(reports):
    held=set()
    c=reports["GSF-01"]["config"]
    for family in G.FAMILIES:
        for seed in c["seeds"]:
            for split in ("cal","dev","shift"):
                for x in G.dataset(family,split,seed,c["eval_n"]):
                    held.update(rows_keys(x))
    for gym,mod,keyfn in (("GSF-02",MO,molt_keys),("GSF-05",L,clock_keys)):
        c=reports[gym]["config"]
        for seed in c["seeds"]:
            for split in ("cal","dev","shift"):
                for i in range(c["eval_n"]):
                    held.update(keyfn(mod.world(split,seed,i)))
    for gym in ("GSF-03","GSF-08"):
        c=reports[gym]["config"]
        parts=C.partitions(c["train_n"],c["eval_n"])
        for split in ("cal","dev","shift"):
            held.update(k("cca-state",C.state_root(s)) for s in parts[split])
    for split in ("cal","dev","shift"):
        held.update(B.tag_key(tags) for tags in P.tag_pool(split))
    r=reports["GSF-06"]
    for family in G.FAMILIES:
        for regime in R.REGIMES:
            for i in range(r["roots_per_stratum"]):
                held.update(meta_keys(R.root_id(family,regime,"dev",i),family,r["config"]))
    c=reports["GSF-07"]["config"]
    for split in ("dev","shift"):
        for i in range(c["eval_n"]):
            held.add(k("mcp02",MCP.world(split,i)["group_id"]))
    for c in reports["GB-SF-09"]["cells"]:
        if c["role"]=="development":
            held.update(k("GB-SF-09",c["root"]) for _ in (0,))
            for role in ("cal","final"):
                held.update(c["lineage"][role])
    return held


def raw_digest(gym,x):
    # MCP's canonical digest uses ensure_ascii=False; other native producers
    # use the default JSON escape convention.
    return MCP.digest(x) if gym=="GSF-07" else B.digest(x)


def normalize(gym,row,report):
    x=dict(row)
    receipt=x.pop("receipt_hash")
    if (x.get("schema")!=RAW_SCHEMAS[gym] or x.get("training_ready") is not False
            or x.get("split",x.get("role"))!="trace_train" or raw_digest(gym,x)!=receipt):
        raise ValueError("source receipt or custody mismatch")
    root=x["root"]
    kind=x.get("target_kind","research_decision")
    oracle_only=bool(x.get("uses_private_oracle",False))
    keys={k(gym,root)}
    if gym=="GSF-01":
        keys.update(rows_keys(x["public"]))
        public=x["public"]
        trace=x["visits"]
        targets={"verified_action":x["repair_target"],"grade":x["receipt"]}
    elif gym=="GSF-02":
        keys.update(molt_keys(x["public"]))
        public=x["public"]
        trace=x["visits"]
        targets={"retention_mask":x["receipt"]["target_mask"],"grade":x["receipt"]}
    elif gym=="GSF-03":
        keys.add(k("cca-state",root))
        public={"case":x["public"],"budget":x["budget"]}
        ep=x["episode"]
        trace={a:v for a,v in ep.items() if a!="grade"}
        targets={"repair_action":x["repair_target"],"grade":ep["grade"]}
    elif gym=="GSF-04":
        p=x["public"]
        keys.update(B.tag_key(t) for e in p["evidence"].values() for t in e["possible_tags"])
        # Never expose future interventions to the initial decision.
        public=[]
        trace=[]
        targets=[]
        for phase in x["phases"]:
            evidence,forbidden,halt=P.phase_public(p,phase["phase"])
            public.append({"phase":phase["phase"],"frame":phase["frame"],
                           "evidence":evidence,"forbidden":sorted(forbidden),"halted":halt})
            trace.append({a:v for a,v in phase.items() if a not in ("target","target_probabilities","regret","tv","optimal")})
            targets.append({a:phase[a] for a in ("target","target_probabilities","regret","tv","optimal")})
    elif gym=="GSF-05":
        keys.update(clock_keys(x["public"]))
        public=x["public"]
        trace={"visits":x["visits"],"word":x["word"],"policy":x["policy"],"threshold":x["threshold"]}
        targets={"answer":x["receipt"]["target"],"grade":x["receipt"]}
    elif gym=="GSF-06":
        keys.update(meta_keys(root,x["family"],report["config"]))
        public=x["public"]
        trace=x["turns"]
        targets=x["result"]
    elif gym=="GSF-07":
        keys.add(k("mcp02",root))
        public=[d["public"] for d in x["public_decisions"]]
        trace={"chosen_indices":[d["choice_index"] for d in x["public_decisions"]],
               "decisions":x["result"]["decisions"],"history":x["result"]["history"],
               "handoff":x["result"]["handoff"]}
        targets={"oracle_usefulness":x["evaluator_supervision"],"native_result":x["result"]}
    elif gym=="GSF-08":
        keys.add(k("cca-state",root))
        public=[q["public_observation"] for q in x["queries"]]
        trace=[{a:v for a,v in q.items() if a not in ("grade","public_observation")} for q in x["queries"]]
        targets={"grades":[q["grade"] for q in x["queries"]],"work":x["work"]}
    else:
        keys.update(x["lineage"]["cal"])
        keys.update(x["lineage"]["final"])
        public=x["public"]
        trace=x["turns"]
        targets=x["result"]
    # This is an envelope, not an instruction/completion pair. Public event
    # lists are temporal observations and must only be rendered up to a turn.
    return {"schema":SCHEMA,"gym":gym,"source_root":root,"lineage":sorted(keys),
            "source_receipt_hash":receipt,"source_schema":x["schema"],"target_kind":kind,
            "public_observations":public,"observed_execution":trace,"evaluator_only":targets,
            "oracle_reference_only":oracle_only,"visibility":"public_synthetic",
            "admission":"staged_requires_corpus_review","training_ready":False}


class Groups:
    def __init__(self):
        self.parent={}

    def root(self,x):
        self.parent.setdefault(x,x)
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x

    def link(self,keys):
        roots=sorted({self.root(x) for x in keys})
        for r in roots[1:]:
            self.parent[r]=roots[0]


def stage(root,out):
    torch.set_num_threads(1)
    reports={}
    audit_receipts={}
    manifests={}
    for gym,(name,rep,trace,n,auditor) in SOURCES.items():
        path=root/name
        reports[gym]=json.loads((path/rep).read_text(encoding="utf-8"))
        actual=auditor(path)
        expected=json.loads((path/"audit.json").read_text(encoding="utf-8"))
        if actual!=expected or actual.get("status")!="audited_development_only" or actual.get("rollouts",actual.get("research_rollouts"))!=n:
            raise ValueError(f"native audit or expected count mismatch: {gym}")
        audit_receipts[gym]=actual
        manifests[gym]={"report_sha256":sha(path/rep),"trace_sha256":sha(path/trace),
                        "audit_sha256":sha(path/"audit.json"),"artifact":name}
    held=held_keys(reports)
    groups=Groups()
    counts=Counter()
    source_receipts=set()
    # Two passes avoid keeping all native decision tapes in RAM.
    def records():
        for gym,(name,rep,trace,n,_) in SOURCES.items():
            for row in lines(root/name/trace):
                yield normalize(gym,row,reports[gym])
    for r in records():
        rid=(r["gym"],r["source_receipt_hash"])
        if rid in source_receipts or held.intersection(r["lineage"]):
            raise ValueError(f"duplicate or held lineage: {r['gym']} / {r['source_root']}")
        source_receipts.add(rid)
        groups.link(r["lineage"])
        counts[r["gym"]]+=1
    if dict(counts)!={g:s[3] for g,s in SOURCES.items()}:
        raise ValueError("incomplete inventory")
    out.mkdir(parents=True,exist_ok=False)
    target=out/"staged_rollouts.jsonl.gz"
    grouped=Counter()
    with gzip.open(target,"wt",encoding="utf-8") as f:
        for r in records():
            root_id=groups.root(r["lineage"][0])
            group_id=B.digest(["goblinbench-lineage-v1",root_id])
            grouped[group_id]+=1
            r["lineage_group"]=group_id
            r["staged_sha256"]=B.digest(r)
            f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
    registry={"schema":"goblinbench-held-lineage.v1","held_lineage":sorted(held)}
    (out/"held_lineage.json").write_text(json.dumps(registry,sort_keys=True),encoding="utf-8")
    report={"schema":SCHEMA,"status":"staged_not_admitted","source_manifest":B.source_manifest(),
            "source_artifacts":manifests,"source_audits":audit_receipts,
            "records":sum(counts.values()),"counts":dict(counts),"lineage_groups":len(grouped),
            "largest_group":max(grouped.values()),"held_lineage_keys":len(held),
            "staged_sha256":sha(target),"held_lineage_sha256":sha(out/"held_lineage.json"),
            "training_ready":False,"admitted_records":0}
    (out/"intake_receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=stage(a.artifacts,a.out)
    print(json.dumps({k:r[k] for k in ("status","records","counts","lineage_groups","largest_group","held_lineage_keys","admitted_records")},sort_keys=True))
