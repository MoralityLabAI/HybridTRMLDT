"""FG-08: actual batched causal verification, wave admission and RAM receipts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import torch

from . import gym as G

SCHEMA="hybrid.gsf.floodgates.fg08.v1"
OFFSET=40000
RANK=4
MODES=("first","deep")
POLICIES=("serial","first4","deep4")
PANELS=("dev","shift","trace_train")
RESERVE=2*G.LOOPS*G.WIDTH*(G.PREFIX+G.GENERATE-1)*4
PROFILES={"b1":(1,1048576),"b4":(4,1048576),"b16":(16,1048576),
          "b32":(32,1048576),"pressure":(32,131072)}
DEFAULT=dict(G.DEFAULT,seeds=[137,163],eval_n=32,repeats=3,profiles=list(PROFILES))


def manifest():
    here=Path(__file__).resolve().parent
    return {n:hashlib.sha256((here/n).read_bytes()).hexdigest()
            for n in ("gym.py","fg08_serving.py","FG08_PROTOCOL.md")}


def validate(cfg):
    G.validate({k:v for k,v in cfg.items() if k not in ("repeats","profiles")})
    if type(cfg.get("repeats")) is not int or not 1<=cfg["repeats"]<=5:
        raise ValueError("invalid repeat count")
    ps=cfg.get("profiles")
    if not isinstance(ps,list) or not ps or len(ps)!=len(set(ps)) or any(p not in PROFILES for p in ps):
        raise ValueError("invalid profiles")


def partitions(cfg):
    n={"base_train":cfg["base_n"],"adapter_train":cfg["adapter_n"]}
    sets={s:[G.world(s,OFFSET+i) for i in range(n.get(s,cfg["eval_n"]))] for s in G.SPLITS}
    seen=set()
    for ws in sets.values():
        keys={tuple(w["prefix"]) for w in ws}
        if seen&keys or len(keys)!=len(ws):
            raise ValueError("prefix lineage overlap")
        seen.update(keys)
    return sets


def hashes(sets):
    return {s:G.digest([w["root"] for w in ws]) for s,ws in sets.items()}


def capacity(profile):
    maximum,budget=profile
    cap=min(maximum,budget//RESERVE)
    if cap<1:
        raise ValueError("committed KV reservation cannot fit one request")
    return cap


def size(cache):
    return G.cache_size(cache)


def select_cache(cache,i,n):
    return tuple((k[i:i+1,:n].clone(),v[i:i+1,:n].clone()) for k,v in cache)


def batch_cache(requests):
    if not requests:
        raise ValueError("empty cache batch")
    lengths={len(r["tokens"])-1 for r in requests}
    if len(lengths)!=1:
        raise ValueError("mixed prefix lengths")
    n=next(iter(lengths))
    if any(len(r["cache"])!=G.LOOPS or any(k.shape!=v.shape or k.shape[0]!=1 or k.shape[1]!=n
           for k,v in r["cache"]) for r in requests):
        raise ValueError("committed cache shape mismatch")
    return tuple((torch.cat([r["cache"][d][0] for r in requests],dim=0),
                  torch.cat([r["cache"][d][1] for r in requests],dim=0)) for d in range(G.LOOPS))


def resolve(draft,predictions,remaining):
    if len(predictions)!=len(draft)+1 or remaining<1:
        raise ValueError("invalid verification shape")
    a=0
    while a<len(draft) and draft[a]==predictions[a]:
        a+=1
    emitted=(draft[:a]+[predictions[a]])[:remaining]
    return a,emitted


def percentile(values,p):
    a=sorted(values)
    return a[max(0,math.ceil(len(a)*p)-1)]


@torch.no_grad()
def serve(model,adapter,policy,worlds,profile):
    """One fresh workload. Reference judging happens outside the timed region."""
    if policy not in POLICIES or not worlds or (policy!="serial" and adapter is None):
        raise ValueError("invalid serving workload")
    cap=capacity(profile)
    budget=profile[1]
    work=Counter()
    hist=Counter()
    groups=[]
    output=[]
    ttft=[]
    completion=[]
    queue_completion=[]
    waves=[]
    start=time.perf_counter()
    cpu_start=time.process_time()

    def committed(active,reserved):
        live=sum(size(r["cache"]) for r in active if r["cache"] is not None)
        if live>reserved or reserved>budget:
            raise AssertionError("persistent committed KV cap exceeded")
        work["peak_committed_kv_bytes"]=max(work["peak_committed_kv_bytes"],live)

    for offset in range(0,len(worlds),cap):
        wave=worlds[offset:offset+cap]
        admitted=time.perf_counter()
        reserved=len(wave)*RESERVE
        work["waves"]+=1
        work["peak_reserved_kv_bytes"]=max(work["peak_reserved_kv_bytes"],reserved)
        _,prefill=model(torch.tensor([w["prefix"][:-1] for w in wave]))
        work["prefill_calls"]+=1
        work["prefill_token_block_visits"]+=len(wave)*(G.PREFIX-1)*G.LOOPS
        work["peak_prefill_kv_bytes"]=max(work["peak_prefill_kv_bytes"],size(prefill))
        active=[]
        for i,w in enumerate(wave):
            kv=select_cache(prefill,i,G.PREFIX-1)
            work["logical_cache_scatter_bytes"]+=size(kv)
            active.append({"world":w,"ordinal":offset+i,"tokens":list(w["prefix"]),
                           "cache":kv,"steps":[],"first_time":None,"done_time":None})
        del prefill
        committed(active,reserved)
        while any(r["cache"] is not None for r in active):
            buckets=defaultdict(list)
            for r in active:
                if r["cache"] is not None:
                    buckets[len(r["tokens"])].append(r)
            # A bucket is scheduled once per round. Requests advanced by an
            # earlier bucket wait until the next round; no future lookahead.
            for length,requests in sorted(buckets.items()):
                remaining=G.PREFIX+G.GENERATE-length
                b=len(requests)
                past=batch_cache(requests)
                stacked=size(past)
                work["logical_cache_stack_bytes"]+=stacked
                work["peak_stacked_kv_bytes"]=max(work["peak_stacked_kv_bytes"],stacked)
                anchor=torch.tensor([[r["tokens"][-1]] for r in requests])
                k=0 if policy=="serial" else min(G.BLOCK,remaining)
                if k:
                    mode="first" if policy=="first4" else "deep"
                    proposal=model.draft(anchor,past,adapter,k,mode)
                    draft=proposal.argmax(-1)
                    x=torch.cat((anchor,draft),dim=1)
                    drafts=draft.tolist()
                    work["draft_calls"]+=1
                    work["draft_token_block_visits"]+=b*(k+1)
                    work["draft_slots"]+=b*k
                    work["logical_draft_prefix_kv_bytes"]+=sum(
                        z.numel()*z.element_size() for z in (past[0] if mode=="first" else past[-1]))
                    del proposal,draft
                else:
                    x=anchor
                    drafts=[[] for _ in requests]
                z,extended=model(x,past)
                predictions=z[:,-1].argmax(-1).tolist()
                provisional=size(extended)
                work["peak_provisional_kv_bytes"]=max(work["peak_provisional_kv_bytes"],provisional)
                work["target_calls"]+=1
                work["target_token_block_visits"]+=b*(k+1)*G.LOOPS
                work["serialized_decode_block_calls"]+=G.LOOPS+bool(k)
                work["request_steps"]+=b
                hist[str(b)]+=1
                group_id=len(groups)
                accepted=[]
                advances=[]
                scattered=0
                for i,r in enumerate(requests):
                    a,emitted=resolve(drafts[i],predictions[i],remaining)
                    r["tokens"].extend(emitted)
                    keep=len(r["tokens"])-1
                    if keep>extended[0][0].shape[1]:
                        raise AssertionError("committed beyond provisional cache")
                    done=len(r["tokens"])==G.PREFIX+G.GENERATE
                    kv=None if done else select_cache(extended,i,keep)
                    if kv is not None:
                        scattered+=size(kv)
                    r["cache"]=kv
                    work["discarded_cache_positions"]+=extended[0][0].shape[1]-keep
                    work["accepted_prefix_tokens"]+=a
                    work["emitted_tokens"]+=len(emitted)
                    r["steps"].append({"group":group_id,"batch":b,"draft":drafts[i],
                          "target_predictions":predictions[i],"accepted_prefix":a,"emitted":emitted,
                          "cache_before":length-1,"cache_provisional":extended[0][0].shape[1],
                          "cache_after":keep,"released":done})
                    accepted.append(a)
                    advances.append(len(emitted))
                    if done:
                        work["completed_requests"]+=1
                work["logical_cache_scatter_bytes"]+=scattered
                work["peak_group_scattered_kv_bytes"]=max(work["peak_group_scattered_kv_bytes"],scattered)
                groups.append({"group":group_id,"wave_offset":offset,"prefix_length":length,
                               "requests":[r["ordinal"] for r in requests],"batch":b,"draft_length":k,
                               "accepted_prefix":accepted,"advanced":advances})
                committed(active,reserved)
                del z,extended,past,x,anchor
                now=time.perf_counter()
                for r in requests:
                    if r["first_time"] is None:
                        r["first_time"]=now
                    if r["cache"] is None:
                        r["done_time"]=now
        wave_end=time.perf_counter()
        waves.append(wave_end-admitted)
        for r in active:
            if r["first_time"] is None or r["done_time"] is None:
                raise AssertionError("missing completion timestamp")
            ttft.append(r["first_time"]-admitted)
            completion.append(r["done_time"]-admitted)
            queue_completion.append(r["done_time"]-start)
            output.append({"root":r["world"]["root"],"index":r["world"]["index"],
                           "prefix":r["world"]["prefix"],"verified_output":r["tokens"][G.PREFIX:],
                           "steps":r["steps"]})
        committed(active,reserved)
        del active
    elapsed=time.perf_counter()-start
    cpu=time.process_time()-cpu_start
    if work["completed_requests"]!=len(worlds) or work["emitted_tokens"]!=len(worlds)*G.GENERATE:
        raise AssertionError("request/token conservation")
    if work["target_calls"]!=len(groups) or sum(int(k)*v for k,v in hist.items())!=work["request_steps"]:
        raise AssertionError("batch call conservation")
    if work["serialized_decode_block_calls"]!=G.LOOPS*work["target_calls"]+work["draft_calls"]:
        raise AssertionError("serialized block conservation")
    summary={"requests":len(worlds),"admission_cap":cap,"profile":list(profile),
             "work":dict(work),"actual_batch_histogram":dict(hist),
             "schedule_sha256":G.digest(groups),"schedule":groups}
    timing={"wall_seconds":elapsed,"cpu_seconds":cpu,"tokens_per_second":work["emitted_tokens"]/elapsed,
            "cpu_tokens_per_second":work["emitted_tokens"]/cpu,"wave_p50_seconds":statistics.median(waves),
            "ttft_p50_seconds":statistics.median(ttft),"ttft_p95_seconds":percentile(ttft,.95),
            "completion_p50_seconds":statistics.median(completion),"completion_p95_seconds":percentile(completion,.95),
            "queue_completion_p50_seconds":statistics.median(queue_completion),
            "queue_completion_p95_seconds":percentile(queue_completion,.95)}
    return summary,output,timing


def grade(summary,rows,worlds,refs):
    if [r["root"] for r in rows]!=[w["root"] for w in worlds]:
        raise AssertionError("request order or coverage")
    correct=0
    exact=0
    for r,w in zip(rows,worlds,strict=True):
        ref=refs[w["root"]][G.PREFIX:]
        if r["verified_output"]!=ref:
            raise AssertionError(f"batched greedy mismatch {w['root']}")
        truth=G.truth(w)[G.PREFIX:]
        n=sum(a==b for a,b in zip(r["verified_output"],truth,strict=True))
        r.update(reference_output=ref,truth_output=truth,truth_correct_tokens=n,greedy_parity=True)
        correct+=n
        exact+=r["verified_output"]==truth
    summary.update(truth_accuracy=correct/(len(rows)*G.GENERATE),truth_exact_sequences=exact,
                   greedy_mismatches=0)
    return summary,rows


def timing_summary(samples,repeats):
    if len(samples)!=repeats or not samples:
        raise ValueError("timing sample coverage")
    keys=set(samples[0])
    for s in samples:
        if set(s)!=keys or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in s.values()):
            raise ValueError("invalid measured time")
    return {k:statistics.median(s[k] for s in samples) for k in sorted(keys)}


def peak_rss():
    try:
        import resource
    except ImportError:
        return None
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return int(value)*1024
    if sys.platform=="darwin":
        return int(value)
    return None


def load_models(out,target,adapters,policy):
    m=G.load(G.Target(),out,target)
    adapter=None
    if policy!="serial":
        mode="first" if policy=="first4" else "deep"
        row=adapters[mode]
        if row["target_hash"]!=target["weights_hash"] or row["rank"]!=RANK:
            raise ValueError("adapter binding")
        adapter=G.load(G.DraftAdapter(RANK),out,row)
    return m,adapter


def worker(spec):
    torch.set_num_threads(1)
    cfg=spec["config"]
    validate(cfg)
    policy=spec["policy"]
    profile=PROFILES[spec["profile"]]
    out=Path(spec["weights_dir"])
    model,adapter=load_models(out,spec["target"],spec["adapters"],policy)
    sets=partitions(cfg)
    refs={s:{w["root"]:G.serial(model,w["prefix"]) for w in sets[s]} for s in PANELS}
    # Warmup is untimed for the report and does not reuse workload state.
    warm=sets["dev"][:capacity(profile)]
    s,r,_=serve(model,adapter,policy,warm,profile)
    grade(s,r,warm,refs["dev"])
    panels={}
    trace_rows=[]
    for split in PANELS:
        stable=None
        samples=[]
        for _ in range(cfg["repeats"]):
            s,r,t=serve(model,adapter,policy,sets[split],profile)
            grade(s,r,sets[split],refs[split])
            samples.append(t)
            if stable is None:
                stable=(s,r)
            elif stable!=(s,r):
                raise AssertionError("repeat decision or work mismatch")
        s,r=stable
        panels[split]={"ledger":s,"timings":samples,"median":timing_summary(samples,cfg["repeats"])}
        if split=="trace_train":
            trace_rows=r
    if G.weight_hash(model)!=spec["target"]["weights_hash"]:
        raise AssertionError("frozen target mutated")
    if adapter is not None and G.weight_hash(adapter)!=spec["adapters"]["first" if policy=="first4" else "deep"]["weights_hash"]:
        raise AssertionError("frozen adapter mutated")
    return {"panels":panels,"trace_rows":trace_rows,"peak_rss_bytes":peak_rss()}


def record(seed,policy,profile,target_hash,adapter_hash,row):
    r={"schema":SCHEMA+".rollout","split":"trace_train","root":row["root"],
       "seed":seed,"policy":policy,"profile":profile,"target_weights_hash":target_hash,
       "adapter_weights_hash":adapter_hash,**row,"training_ready":False}
    r["receipt_hash"]=G.digest(r)
    return r


def comparisons(cells):
    by={(c["seed"],c["policy"],c["profile"]):c for c in cells}
    results=[]
    for seed,policy,profile in sorted(by):
        if policy=="serial":
            continue
        a=by[seed,policy,profile]
        b=by[seed,"serial",profile]
        for split in PANELS:
            x,y=a["panels"][split],b["panels"][split]
            u,v=x["ledger"]["work"],y["ledger"]["work"]
            results.append({"seed":seed,"policy":policy,"profile":profile,"split":split,
                 "tokens_per_second_ratio":x["median"]["tokens_per_second"]/y["median"]["tokens_per_second"],
                 "cpu_tokens_per_second_ratio":x["median"]["cpu_tokens_per_second"]/y["median"]["cpu_tokens_per_second"],
                 "target_call_ratio":v["target_calls"]/u["target_calls"],
                 "decode_block_call_ratio":v["serialized_decode_block_calls"]/u["serialized_decode_block_calls"],
                 "decode_token_work_ratio":(u["target_token_block_visits"]+u.get("draft_token_block_visits",0))/v["target_token_block_visits"],
                 "completion_p95_ratio":x["median"]["completion_p95_seconds"]/y["median"]["completion_p95_seconds"]})
    return results


def run(out,cfg=None):
    cfg=dict(DEFAULT if cfg is None else cfg)
    validate(cfg)
    torch.set_num_threads(1)
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sets=partitions(cfg)
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "source_manifest":manifest(),"split_hashes":hashes(sets),"reserve_bytes_per_request":RESERVE,
            "targets":[],"adapters":[],"cells":[]}
    path=out/"fg08_rollouts.jsonl"
    with tempfile.TemporaryDirectory(prefix="fg08-worker-") as tmp,path.open("w",encoding="utf-8") as trace:
        tmp=Path(tmp)
        for seed in cfg["seeds"]:
            model,t=G.fit_target("final_only",seed,sets["base_train"],cfg)
            t.update(seed=seed,**G.save(out,f"{seed}-target.pt",model))
            report["targets"].append(t)
            adapters={}
            for mode in MODES:
                a,info=G.fit_adapter(model,seed,mode,RANK,sets["adapter_train"],cfg)
                info.update(seed=seed,mode=mode,rank=RANK,**G.save(out,f"{seed}-{mode}.pt",a))
                adapters[mode]=info
                report["adapters"].append(info)
            for policy,profile in itertools.product(POLICIES,cfg["profiles"]):
                spec={"config":cfg,"policy":policy,"profile":profile,"weights_dir":str(out),
                      "target":t,"adapters":adapters}
                sp=tmp/"spec.json"
                result_path=tmp/"result.json"
                if result_path.exists():
                    result_path.unlink()
                sp.write_text(json.dumps(spec,allow_nan=False),encoding="utf-8")
                cmd=[sys.executable,"-m",__name__ if __name__!="__main__" else
                     "research_gym.benchmarks.floodgates_gym.fg08_serving",
                     "--worker-spec",str(sp),"--worker-out",str(result_path)]
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=180,check=False)
                if p.returncode:
                    raise RuntimeError(f"serving worker failed {seed}/{policy}/{profile}: {p.stderr[-4000:]}")
                r=json.loads(result_path.read_text(encoding="utf-8"))
                ah=None if policy=="serial" else adapters["first" if policy=="first4" else "deep"]["weights_hash"]
                report["cells"].append({"seed":seed,"policy":policy,"profile":profile,
                      "target_weights_hash":t["weights_hash"],"adapter_weights_hash":ah,
                      "peak_rss_bytes":r["peak_rss_bytes"],"panels":r["panels"]})
                for row in r["trace_rows"]:
                    obj=record(seed,policy,profile,t["weights_hash"],ah,row)
                    trace.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n")
                print(json.dumps({"finished_cell":[seed,policy,profile],"peak_rss_bytes":r["peak_rss_bytes"]}),flush=True)
    report["paired"]=comparisons(report["cells"])
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"fg08_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    torch.set_num_threads(1)
    r=json.loads((out/"fg08_report.json").read_text(encoding="utf-8"))
    cfg=r["config"]
    validate(cfg)
    sets=partitions(cfg)
    if (r["schema"]!=SCHEMA or r["source_manifest"]!=manifest() or r["split_hashes"]!=hashes(sets)
        or r["reserve_bytes_per_request"]!=RESERVE):
        raise ValueError("source or split binding")
    tr={x["seed"]:x for x in r["targets"]}
    ar={(x["seed"],x["mode"]):x for x in r["adapters"]}
    cells={(x["seed"],x["policy"],x["profile"]):x for x in r["cells"]}
    if (len(tr)!=len(r["targets"]) or set(tr)!=set(cfg["seeds"])
        or len(ar)!=len(r["adapters"]) or set(ar)!=set(itertools.product(cfg["seeds"],MODES))
        or len(cells)!=len(r["cells"]) or set(cells)!=set(itertools.product(cfg["seeds"],POLICIES,cfg["profiles"]))):
        raise ValueError("cell coverage")
    path=out/"fg08_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=r["trace_sha256"]:
        raise ValueError("trace hash")
    stored=defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        bound=row.pop("receipt_hash")
        key=(row["seed"],row["policy"],row["profile"])
        if (G.digest(row)!=bound or row["schema"]!=SCHEMA+".rollout" or row["split"]!="trace_train"
            or row["training_ready"] is not False or key not in cells):
            raise ValueError("trace custody")
        stored[key].append(bound)
    seen=0
    for seed in cfg["seeds"]:
        model=G.load(G.Target(),out,tr[seed])
        adapters={}
        for mode in MODES:
            c=ar[seed,mode]
            if c["target_hash"]!=tr[seed]["weights_hash"] or c["rank"]!=RANK:
                raise ValueError("adapter identity")
            adapters[mode]=G.load(G.DraftAdapter(RANK),out,c)
        refs={w["root"]:G.serial(model,w["prefix"]) for w in sets["trace_train"]}
        for policy,profile in itertools.product(POLICIES,cfg["profiles"]):
            c=cells[seed,policy,profile]
            mode="first" if policy=="first4" else "deep"
            ah=None if policy=="serial" else ar[seed,mode]["weights_hash"]
            if c["target_weights_hash"]!=tr[seed]["weights_hash"] or c["adapter_weights_hash"]!=ah:
                raise ValueError("cell weights")
            rss=c["peak_rss_bytes"]
            if rss is not None and (type(rss) is not int or rss<=0):
                raise ValueError("invalid RSS receipt")
            for split in PANELS:
                p=c["panels"][split]
                if timing_summary(p["timings"],cfg["repeats"])!=p["median"]:
                    raise ValueError("timing summary mismatch")
                if p["ledger"]["greedy_mismatches"]!=0 or p["ledger"]["profile"]!=list(PROFILES[profile]):
                    raise ValueError("panel parity or profile")
            summary,rows,_=serve(model,None if policy=="serial" else adapters[mode],
                                 policy,sets["trace_train"],PROFILES[profile])
            grade(summary,rows,sets["trace_train"],refs)
            if summary!=c["panels"]["trace_train"]["ledger"]:
                raise ValueError("batch work or scheduler replay")
            expected=[record(seed,policy,profile,tr[seed]["weights_hash"],ah,row)["receipt_hash"] for row in rows]
            if stored[seed,policy,profile]!=expected:
                raise ValueError("exact trace replay")
            seen+=len(expected)
    if seen!=len(cells)*len(sets["trace_train"]) or comparisons(r["cells"])!=r["paired"]:
        raise ValueError("trace or comparison coverage")
    return {"status":"audited_development_only","targets":len(tr),"adapters":len(ar),
            "cells":len(cells),"rollouts":seen,"greedy_mismatches":0}


def selftest():
    torch.set_num_threads(1)
    for j in range(G.BLOCK+1):
        p=list(range(G.BLOCK+1))
        d=p[:-1].copy()
        if j<len(d):
            d[j]=9
        a,e=resolve(d,p,G.GENERATE)
        assert a==j and e==p[:j+1]
    assert resolve([1,2],[1,2,3],1)==(2,[1])
    assert RESERVE==19456 and capacity(PROFILES["pressure"])==6
    torch.manual_seed(137)
    m=G.Target().eval().requires_grad_(False)
    a=G.DraftAdapter(RANK).eval().requires_grad_(False)
    worlds=[G.world("dev",OFFSET+i) for i in range(4)]
    refs={w["root"]:G.serial(m,w["prefix"]) for w in worlds}
    for policy,profile in itertools.product(POLICIES,((1,1048576),(4,2*RESERVE),(4,1048576))):
        s,r,t=serve(m,None if policy=="serial" else a,policy,worlds,profile)
        grade(s,r,worlds,refs)
        assert s["work"]["peak_reserved_kv_bytes"]<=profile[1]
        assert s["work"]["peak_committed_kv_bytes"]<=s["work"]["peak_reserved_kv_bytes"]
        assert s["requests"]==4 and s["greedy_mismatches"]==0
        assert timing_summary([t],1)==t
    print("FG-08 batched greedy parity, forced rejection and KV reservation selftests passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--worker-spec",type=Path)
    p.add_argument("--worker-out",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.worker_spec:
        if not a.worker_out:
            p.error("worker output required")
        spec=json.loads(a.worker_spec.read_text(encoding="utf-8"))
        a.worker_out.write_text(json.dumps(worker(spec),allow_nan=False),encoding="utf-8")
        return
    if a.selftest:
        selftest()
    if a.out:
        cfg=dict(DEFAULT,base_steps=4,adapter_steps=4,batch=4,base_n=16,adapter_n=8,
                 eval_n=4,seeds=[137],repeats=2,profiles=["b1","b4","pressure"]) if a.smoke else None
        r=run(a.out,cfg)
        print(json.dumps({"status":r["status"],"cells":len(r["cells"])}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
