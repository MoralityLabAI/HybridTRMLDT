"""Local-only fit of Jinn/Beast lane heads from DataGoblin's pinned private intake.

Never fetches source data or writes it into Git. Uses original TRAIN rows only,
partitioned again by canonical tag-set root. Original source heldouts stay excluded.
This is policy emulation, not a learned authorization gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import torch

from .persona import CLASSES, LEARNED, digest, fit_arrays, heads_from

SOURCE="jinn-beast-audited-tags-v1"
REVISION="01512c40be082ad88aa69594abca43a2b41908cc"
DATA_SHA="2a4594c672ed0ca2439bf28e09612474dc6e1e576d32a175d0653be2b24f0151"
POLICY_SHA={"jinn":"85e544c35fe17f67019564a6d3e59dfe697e1fdfbf6ce294e24f0ef32e618576",
            "beast":"8828fbe417f099ba6ac1e418adae5f1685cd7f672e34b2de9ff018b2bd4e45ff"}
FRAMES=("jinn","beast")
SPLITS=("train","cal","dev","trace_train")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def semantic(tags):
    return sha(canonical(sorted(tags)).encode())


def exact_target(tags,policy,low,high):
    dims={name:Fraction(w) for name,w in policy["dimensions"].items()}
    if not dims or sum(dims.values())!=1 or any(v<=0 for v in dims.values()):
        raise ValueError("invalid dimensions")
    active=sorted(set(tags)&set(policy["tags"]))
    signs={name:set() for name in dims}
    for tag in active:
        spec=policy["tags"][tag]
        value=Fraction(spec["value"])
        if value not in (-1,1):
            raise ValueError("invalid sign")
        signs[spec["dimension"]].add(value)
    conflict=sorted(d for d,v in signs.items() if len(v)>1)
    critical=sorted(set(active)&set(policy["critical_tags"]))
    if not active or conflict:
        score,lane=Fraction(1,2),"hold"
    else:
        score=(1+sum(dims[d]*sum(v) for d,v in signs.items()))/2
        if critical:
            score,lane=min(score,Fraction(policy["critical_cap"])),"reject"
        else:
            lane="train" if score>=high else "reject" if score<=low else "hold"
    return {"lane":lane,"score":round(float(score),6),"critical_hits":critical,
            "annotation_conflicts":conflict,"relevant_tags":active}


def local_only(path):
    path=path.resolve()
    if any((p/".git").exists() for p in (path,*path.parents)):
        raise ValueError("private data/artifact directory must be outside Git")
    return path


def load_intake(source):
    source=local_only(source)
    receipt=json.loads((source/"RECEIPT.json").read_text(encoding="utf-8"))
    if (receipt.get("schema")!="research-v2-specialist-receipt-v1"
            or receipt.get("source_id")!=SOURCE or receipt.get("source_revision")!=REVISION
            or receipt.get("visibility")!="private" or receipt.get("use")!="internal_specialist_training"
            or receipt.get("train_only") is not True or receipt.get("all_source_labels_recomputed") is not True
            or receipt.get("source_files",{}).get("dataset",{}).get("sha256")!=DATA_SHA):
        raise ValueError("wrong source or privacy receipt")
    for frame in FRAMES:
        if receipt["source_files"][frame]["sha256"]!=POLICY_SHA[frame]:
            raise ValueError("policy source pin mismatch")
    payload={}
    for name in ("train.jsonl","held_roots.json"):
        data=(source/name).read_bytes()
        if receipt["files"][name]!={"sha256":sha(data),"bytes":len(data)}:
            raise ValueError("intake file mismatch")
        payload[name]=data
    held=json.loads(payload["held_roots.json"])
    if held["schema"]!="research-v2-held-roots-v1" or SOURCE not in held["allow_source_ids"]:
        raise ValueError("invalid held registry")
    held_root=set(held["held_root_ids"])
    held_sem=set(held["held_semantic_sha256"])
    grouped={}
    policies={}
    seen=set()
    for line in payload["train.jsonl"].splitlines():
        r=json.loads(line)
        tags=r["input"]["tags"]
        sem=semantic(tags)
        frame=r["policy_frame"]
        root="jinn-beast:"+sem
        if (r["schema"]!="research-v2-specialist-record-v1" or r["source_id"]!=SOURCE
                or r["source_revision"]!=REVISION or r["source_sha256"]!=DATA_SHA
                or r["split"]!="train" or r["visibility"]!="private"
                or r["use"]!="internal_specialist_training" or frame not in FRAMES
                or r["policy_sha256"]!=POLICY_SHA[frame]):
            raise ValueError("source row mismatch")
        if r["root_id"] in held_root or sem in held_sem:
            raise ValueError("held-root collision")
        if r["variant"]!="original":
            continue
        if (len(set(tags))!=len(tags) or r["root_id"]!=root or r["semantic_sha256"]!=sem
                or r["original_row_id"]!="mb-"+sem[:16] or (root,frame) in seen):
            raise ValueError("root identity mismatch")
        seen.add((root,frame))
        policy=r["input"]["policy"]
        if frame in policies and policies[frame]!=policy:
            raise ValueError("inconsistent policy frame")
        policies[frame]=policy
        low,high=Fraction(r["input"]["reject_maximum"]),Fraction(r["input"]["train_minimum"])
        if (low,high)!=(Fraction(3,10),Fraction(7,10)):
            raise ValueError("threshold mismatch")
        if exact_target(tags,policy,low,high)!=r["target"]:
            raise ValueError("independent target mismatch")
        grouped.setdefault(root,{})[frame]=r
    if not grouped or any(set(v)!=set(FRAMES) for v in grouped.values()):
        raise ValueError("incomplete paired roots")
    if len(grouped)!=receipt["admitted_train_roots"]:
        raise ValueError("intake root count mismatch")
    vocab=tuple(sorted(set().union(*(set(p["tags"]) for p in policies.values()))))
    if any(not set(r["input"]["tags"])<=set(vocab) for v in grouped.values() for r in v.values()):
        raise ValueError("unknown source tag")
    return grouped,vocab,sha((source/"RECEIPT.json").read_bytes())


def partition(root):
    b=int(digest(["gsf04-private",root]),16)%10
    return "train" if b<6 else "cal" if b==6 else "dev" if b==7 else "trace_train"


def metrics(rows,heads):
    n=correct=critical=critical_missed=0
    for r in rows:
        head=heads["p"+str(FRAMES.index(r["policy_frame"]))]
        pred=head.lane(frozenset(r["input"]["tags"]))
        n+=1
        correct+=pred==r["target"]["lane"]
        target=r["target"]
        if target["critical_hits"] and not target["annotation_conflicts"]:
            critical+=1
            critical_missed+=pred!="reject"
    return {"n":n,"lane_accuracy":correct/n,"critical_nonconflict":critical,
            "critical_missed_raw":critical_missed}


def run(source,out,steps=300,batch=64,seeds=(137,163)):
    if not (1<=steps<=1024 and 1<=batch<=128):
        raise ValueError("budget")
    out=local_only(out)
    grouped,vocab,source_receipt=load_intake(source)
    parts={s:[] for s in SPLITS}
    for root,rows in sorted(grouped.items()):
        parts[partition(root)].extend(rows[f] for f in FRAMES)
    if any(not rows for rows in parts.values()):
        raise ValueError("empty private partition")
    os.umask(0o077)
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.set_num_threads(1)
    train=parts["train"]
    x=torch.tensor([[float(t in r["input"]["tags"]) for t in vocab] for r in train])
    frames=torch.tensor([FRAMES.index(r["policy_frame"]) for r in train])
    y=torch.tensor([CLASSES.index(r["target"]["lane"]) for r in train])
    soft=torch.nn.functional.one_hot(y,num_classes=3).float()
    report={"schema":"hybrid.gsf.dual_ldt.private.v1","visibility":"private",
            "use":"internal_specialist_training","source_receipt_sha256":source_receipt,
            "training_ready":False,"source_heldouts_used":False,
            "counts":{s:len(rows) for s,rows in parts.items()},
            "root_hashes":{s:digest(sorted({r["root_id"] for r in rows})) for s,rows in parts.items()},
            "cells":[]}
    with (out/"private_rollouts.jsonl").open("w",encoding="utf-8") as trace:
        for seed in seeds:
            for arm in LEARNED:
                rec,info=fit_arrays(arm,seed,x,frames,soft,steps,batch,vocab)
                p=out/f"{seed}-{arm}.json"
                p.write_text(json.dumps(rec,sort_keys=True,allow_nan=False),encoding="utf-8")
                heads=heads_from(rec)
                row={"seed":seed,"arm":arm,**info,"artifact":p.name,"artifact_sha256":sha(p.read_bytes())}
                for split in ("cal","dev","trace_train"):
                    row[split]=metrics(parts[split],heads)
                for r in parts["trace_train"]:
                    frame=r["policy_frame"]
                    predicted=heads["p"+str(FRAMES.index(frame))].lane(frozenset(r["input"]["tags"]))
                    item={"schema":"hybrid.gsf.dual_ldt.private.rollout.v1","visibility":"private",
                          "use":"internal_specialist_training","training_ready":False,
                          "root":r["root_id"],"split":"trace_train","frame":frame,"seed":seed,"arm":arm,
                          "weights_hash":info["weights_hash"],"source_record_id":r["record_id"],
                          "public_within_private_task":r["input"],"predicted_lane":predicted,
                          "verified_target":r["target"],"correct":predicted==r["target"]["lane"]}
                    item["receipt_hash"]=digest(item)
                    trace.write(json.dumps(item,sort_keys=True,allow_nan=False)+"\n")
                report["cells"].append(row)
    report["trace_sha256"]=sha((out/"private_rollouts.jsonl").read_bytes())
    (out/"private_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def selftest():
    p={"dimensions":{"a":"1"},"tags":{"good":{"dimension":"a","value":"1"},
       "bad":{"dimension":"a","value":"-1"}},"critical_tags":["bad"],"critical_cap":"0.2"}
    def lane(tags):
        return exact_target(tags,p,Fraction(3,10),Fraction(7,10))["lane"]
    assert lane(["good"])=="train" and lane(["bad"])=="reject"
    assert lane(["good","bad"])=="hold" and lane([])=="hold"
    print("GSF-04 private target grammar selftest passed (synthetic only)")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--intake",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    if a.selftest:
        selftest()
    if a.intake and a.out:
        r=run(a.intake,a.out)
        print(json.dumps({"status":"private_local_only","cells":len(r["cells"])}))
    elif not a.selftest:
        p.error("provide --intake and --out, or --selftest")
