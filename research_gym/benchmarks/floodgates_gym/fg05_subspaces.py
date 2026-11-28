"""FG-05: BPD-inspired output-gradient initialization of draft-only LoRA.

Prospective protocol: FG05_PROTOCOL.md. Real CPU fits; no native BPD wrapper
or pretrained-language-model acceleration claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random

import torch
from torch import nn

from . import gym as G

SCHEMA = "hybrid.gsf.floodgates.fg05.v1"
RANK = 4
MODES = ("first", "deep")
ARMS = ("ordinary", "guided", "haar", "shuffled")
SITES = ("q", "v", "o")
PANELS = ("cal", "dev", "shift", "trace_train")
OFFSET = 10000
CUTS_PER_ROOT = 2
DEFAULT = dict(G.DEFAULT, seeds=[137, 163])


def digest(x):
    return G.digest(x)


def source_manifest():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ("gym.py", "fg05_subspaces.py", "FG05_PROTOCOL.md")}


def validate(cfg):
    G.validate(cfg)
    if cfg["adapter_n"] < 12:
        raise ValueError("need separate discovery and fitting roots")


def partitions(cfg):
    n = {"base_train": cfg["base_n"], "adapter_train": cfg["adapter_n"]}
    sets = {s: [G.world(s, OFFSET+i) for i in range(n.get(s, cfg["eval_n"]))]
            for s in G.SPLITS}
    pool = sets.pop("adapter_train")
    cut = len(pool)//3
    sets["discovery"] = pool[:cut]
    sets["adapter_fit"] = pool[cut:]
    seen = set()
    for worlds in sets.values():
        keys = {tuple(w["prefix"]) for w in worlds}
        if seen & keys or len(keys) != len(worlds):
            raise ValueError("prefix lineage collision")
        seen.update(keys)
    return sets


def split_hashes(sets):
    return {s: digest([w["root"] for w in worlds]) for s, worlds in sets.items()}


@torch.no_grad()
def teacher_data(model, worlds):
    seq = G.teacher_stream(model, torch.tensor([w["prefix"] for w in worlds]))
    _, caches = model(seq)
    n = len(worlds)
    work = {"teacher_stream_block_token_visits": n*(G.PREFIX-1+G.GENERATE)*G.LOOPS,
            "teacher_cache_block_token_visits": n*(G.PREFIX+G.GENERATE)*G.LOOPS}
    return seq, caches, work


def data_at(seq, caches, ids, cut):
    past = tuple((k[ids, :cut-1], v[ids, :cut-1]) for k,v in caches)
    return seq[ids, cut-1:cut], past, seq[ids, cut:cut+G.BLOCK]


def canonical_basis(u):
    # Sign orientation is for reproducible initialization, not a geometric
    # uniqueness claim if the eigenspectrum is degenerate.
    pivot = u.abs().argmax(0)
    signs = torch.sign(u[pivot, torch.arange(u.shape[1])])
    signs[signs == 0] = 1
    return u * signs


def leading(cov):
    values, vectors = torch.linalg.eigh(cov)
    values = values.flip(0)
    u = canonical_basis(vectors.flip(1)[:, :RANK])
    return values, u


def discover(model, seed, mode, data, worlds):
    seq, caches, teacher_work = data
    before = G.weight_hash(model)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("discovery requires a frozen target")
    # fit_target freezes parameters after the optimizer step, but their
    # last training gradients are still attached. Clear these old buffers
    # before the probe so the postcondition detects new gradients only.
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed+700)
    probe = G.DraftAdapter(RANK)
    sums = {site: torch.zeros(G.WIDTH,G.WIDTH,dtype=torch.float64) for site in SITES}
    counts = Counter()
    probe_packets = []
    deltas = {}

    def hook(site):
        def add_perturbation(_module, _inputs, output):
            delta = torch.zeros_like(output, requires_grad=True)
            deltas[site] = delta
            return output+delta
        return add_perturbation

    handles = [getattr(model.block, site).register_forward_hook(hook(site)) for site in SITES]
    try:
        with torch.enable_grad():
            for i,w in enumerate(worlds):
                rng = random.Random(int(digest([SCHEMA,seed,w["root"],"cuts"]),16))
                for cut in rng.sample(range(G.PREFIX,G.PREFIX+G.GENERATE-G.BLOCK+1), CUTS_PER_ROOT):
                    anchor,past,targets = data_at(seq,caches,torch.tensor([i]),cut)
                    deltas.clear()
                    logits = model.draft(anchor,past,probe,G.BLOCK,mode)
                    if set(deltas) != set(SITES):
                        raise AssertionError("missing projection site")
                    log_probs = logits.log_softmax(-1)
                    probe_packets.append([w["root"],cut])
                    for slot in range(G.BLOCK):
                        label = int(targets[0,slot])
                        grads = torch.autograd.grad(log_probs[0,slot,label],
                                                    tuple(deltas[s] for s in SITES),
                                                    retain_graph=slot+1<G.BLOCK)
                        for site,grad in zip(SITES,grads,strict=True):
                            a = grad.detach().reshape(-1,G.WIDTH).double()
                            sums[site] += a.T@a
                            counts[site] += len(a)
    finally:
        for h in handles:
            h.remove()
    if G.weight_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("discovery mutated frozen target")
    sites = {}
    bases = {}
    for site in SITES:
        cov = sums[site]/counts[site]
        values,u = leading(cov)
        if not torch.isfinite(cov).all() or not torch.isfinite(values).all():
            raise ValueError("nonfinite discovery")
        projector = u@u.T
        if not torch.allclose(projector@projector,projector,atol=1e-10,rtol=1e-10):
            raise AssertionError("projector parity")
        gap = float(values[RANK-1]-values[RANK])
        scale = max(abs(float(values[0])),1e-30)
        sites[site] = {"covariance":cov.tolist(),"eigenvalues":values.tolist(),
                       "basis":u.tolist(),"gradient_vectors":counts[site],
                       "boundary_gap":gap,"relative_boundary_gap":gap/scale,
                       "weak_boundary_gap":gap/scale<1e-6}
        bases[site] = u
    return bases, {"schema":SCHEMA+".discovery","seed":seed,"mode":mode,
                   "target_hash":before,"discovery_roots_sha256":digest([w["root"] for w in worlds]),
                   "probe_packets_sha256":digest(probe_packets),"probe_contexts":len(probe_packets),
                   "slot_backward_calls":len(probe_packets)*G.BLOCK,
                   "probe_forward_block_token_visits":len(probe_packets)*(G.BLOCK+1),
                   "teacher_data_work_shared_across_modes":teacher_work,"sites":sites}


def initial_adapter(seed,arm,bases):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(seed+700)
    adapter = G.DraftAdapter(RANK)
    if arm != "ordinary":
        with torch.no_grad():
            for i,site in enumerate(SITES):
                u = bases[site]
                rng = torch.Generator().manual_seed(seed+800+i)
                if arm == "haar":
                    u = canonical_basis(torch.linalg.qr(
                        torch.randn(G.WIDTH,RANK,generator=rng,dtype=torch.float64),
                        mode="reduced")[0])
                elif arm == "shuffled":
                    u = u[torch.randperm(G.WIDTH,generator=rng)]
                if not torch.allclose(u.T@u,torch.eye(RANK,dtype=u.dtype),rtol=1e-9,atol=1e-9):
                    raise AssertionError("initial factor not orthonormal")
                layer = getattr(adapter,site)
                layer.a.weight.zero_()
                layer.b.weight.copy_(u.to(layer.b.weight.dtype))
    for site in SITES:
        layer = getattr(adapter,site)
        if torch.count_nonzero(layer.b.weight@layer.a.weight):
            raise AssertionError("nonzero initial physical update")
    return adapter


def span_overlap(adapter,bases):
    result = {}
    for site in SITES:
        b = getattr(adapter,site).b.weight.detach().double()
        u,s,_ = torch.linalg.svd(b,full_matrices=False)
        effective = int((s > s[0]*1e-9).sum()) if float(s[0]) else 0
        actual = u[:,:effective]
        result[site] = {"effective_rank":effective,
                        "overlap":float((bases[site].T@actual).square().sum()/RANK),
                        "singular_values":s.tolist()}
    return result


def fit(model, seed, mode, arm, bases, data, cfg):
    seq,caches,teacher_work = data
    before = G.weight_hash(model)
    adapter = initial_adapter(seed,arm,bases)
    init = G.weight_hash(adapter)
    mask_hash = digest(adapter.mask.detach().tolist())
    opt = torch.optim.AdamW(adapter.parameters(),lr=.01)
    rng = random.Random(seed+301)
    batches = []
    for _ in range(cfg["adapter_steps"]):
        ids = torch.tensor([rng.randrange(len(seq)) for _ in range(cfg["batch"])])
        cut = rng.randrange(G.PREFIX,G.PREFIX+G.GENERATE-G.BLOCK+1)
        batches.append([ids.tolist(),cut])
        anchor,past,targets = data_at(seq,caches,ids,cut)
        logits = model.draft(anchor,past,adapter,G.BLOCK,mode)
        loss = nn.functional.cross_entropy(logits.reshape(-1,G.VOCAB),targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    adapter.eval()
    after = G.weight_hash(adapter)
    if before!=G.weight_hash(model) or init==after:
        raise AssertionError("frozen target or adapter fit invariant")
    return adapter,{"target_hash":before,"initial_hash":init,"weights_hash":after,
                    "initial_mask_hash":mask_hash,
                    "parameters":sum(p.numel() for p in adapter.parameters()),
                    "updates":cfg["adapter_steps"],"batch_sha256":digest(batches),
                    "train_draft_block_token_visits":cfg["adapter_steps"]*cfg["batch"]*(G.BLOCK+1),
                    "teacher_data_work_shared_across_arms":teacher_work,
                    "last_loss":float(loss.detach()),"final_span":span_overlap(adapter,bases)}


def measured(model,adapter,mode,worlds,reference):
    stats,rows = G.evaluate(model,adapter,mode,G.BLOCK,worlds,reference)
    eligible = [0]*G.BLOCK
    accepted = [0]*G.BLOCK
    for row in rows:
        for step in row["draft_steps"]:
            for j in range(len(step["draft"])):
                eligible[j]+=1
                accepted[j]+=step["accepted_prefix"]>=j+1
    stats["acceptance_survival"] = {"eligible":eligible,"accepted":accepted}
    stats["total_block_work"] = stats["verify_block_token_visits"]+stats["draft_block_token_visits"]
    stats["target_call_reduction"] = stats["serial_verify_calls"]/stats["verify_calls"]
    return stats,rows


def stable_stats(stats):
    return {k:v for k,v in stats.items() if k!="draft_verify_seconds"}


def pair_delta(left,right):
    if [r["public"]["root"] for r in left]!=[r["public"]["root"] for r in right]:
        raise ValueError("unpaired roots")
    counts=Counter()
    for a,b in zip(left,right,strict=True):
        # Positive means guided saved work relative to its comparator.
        av,bv=a["work"],b["work"]
        d=bv["verify_calls"]-av["verify_calls"]
        w=(bv["verify_block_token_visits"]+bv["draft_block_token_visits"]
           -av["verify_block_token_visits"]-av["draft_block_token_visits"])
        counts["n"]+=1
        counts["target_calls_saved"]+=d
        counts["block_token_visits_saved"]+=w
        counts["wins"]+=d>0
        counts["ties"]+=d==0
        counts["losses"]+=d<0
    return dict(counts)


def record(seed,mode,arm,fit,row):
    r={"schema":SCHEMA+".rollout","split":"trace_train","root":row["public"]["root"],
       "seed":seed,"mode":mode,"arm":arm,"rank":RANK,
       "target_weights_hash":fit["target_hash"],"adapter_weights_hash":fit["weights_hash"],
       **row,"training_ready":False}
    r["receipt_hash"]=digest(r)
    return r


def run(out,cfg=None):
    cfg=dict(DEFAULT if cfg is None else cfg)
    validate(cfg)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    sets=partitions(cfg)
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "source_manifest":source_manifest(),"split_hashes":split_hashes(sets),
            "targets":[],"discoveries":[],"cells":[],"paired":[]}
    path=out/"fg05_rollouts.jsonl"
    with path.open("w",encoding="utf-8") as trace:
        for seed in cfg["seeds"]:
            model,info=G.fit_target("final_only",seed,sets["base_train"],cfg)
            info.update(seed=seed,**G.save(out,f"{seed}-target.pt",model))
            report["targets"].append(info)
            refs={s:{w["root"]:G.serial(model,w["prefix"]) for w in sets[s]} for s in PANELS}
            discovery=teacher_data(model,sets["discovery"])
            fitting=teacher_data(model,sets["adapter_fit"])
            for mode in MODES:
                bases,d=discover(model,seed,mode,discovery,sets["discovery"])
                artifact=out/f"{seed}-{mode}-discovery.json"
                artifact.write_text(json.dumps(d,sort_keys=True,allow_nan=False),encoding="utf-8")
                report["discoveries"].append({"seed":seed,"mode":mode,
                      "artifact":artifact.name,"artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
                      "discovery_hash":digest(d)})
                paired={}
                init_logits=None
                init_hashes=[]
                fit_hashes=[]
                for arm in ARMS:
                    initial=initial_adapter(seed,arm,bases)
                    a,p,_=data_at(fitting[0],fitting[1],torch.tensor([0]),G.PREFIX)
                    with torch.no_grad():
                        z=model.draft(a,p,initial,G.BLOCK,mode)
                    if init_logits is None:
                        init_logits=z
                    elif not torch.equal(init_logits,z):
                        raise AssertionError("different zero-update predictions")
                    adapter,fit_info=fit(model,seed,mode,arm,bases,fitting,cfg)
                    init_hashes.append(fit_info["initial_mask_hash"])
                    fit_hashes.append(fit_info["batch_sha256"])
                    row={"seed":seed,"mode":mode,"arm":arm,"rank":RANK,**fit_info,
                         **G.save(out,f"{seed}-{mode}-{arm}.pt",adapter),"panels":{}}
                    paired[arm]={}
                    for split in PANELS:
                        stats,rows=measured(model,adapter,mode,sets[split],refs[split])
                        row["panels"][split]=stats
                        paired[arm][split]=rows
                        if split=="trace_train":
                            for r in rows:
                                trace.write(json.dumps(record(seed,mode,arm,fit_info,r),
                                                       sort_keys=True,allow_nan=False)+"\n")
                    report["cells"].append(row)
                if len(set(init_hashes))!=1 or len(set(fit_hashes))!=1:
                    raise AssertionError("unmatched mask initialization or batch draws")
                for control in ("ordinary","haar","shuffled"):
                    for split in PANELS:
                        report["paired"].append({"seed":seed,"mode":mode,"control":control,"split":split,
                            **pair_delta(paired["guided"][split],paired[control][split])})
            if G.weight_hash(model)!=info["weights_hash"]:
                raise AssertionError("target mutation")
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"fg05_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def audit(out):
    torch.set_num_threads(1)
    report=json.loads((out/"fg05_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    validate(cfg)
    sets=partitions(cfg)
    if report["source_manifest"]!=source_manifest() or report["split_hashes"]!=split_hashes(sets):
        raise ValueError("source or split binding")
    target_rows={c["seed"]:c for c in report["targets"]}
    discoveries={(c["seed"],c["mode"]):c for c in report["discoveries"]}
    cells={(c["seed"],c["mode"],c["arm"]):c for c in report["cells"]}
    if (len(target_rows)!=len(report["targets"]) or set(target_rows)!=set(cfg["seeds"])
            or len(discoveries)!=len(report["discoveries"]) or set(discoveries)!=set(itertools.product(cfg["seeds"],MODES))
            or len(cells)!=len(report["cells"]) or set(cells)!=set(itertools.product(cfg["seeds"],MODES,ARMS))):
        raise ValueError("cell coverage")
    targets={}
    adapters={}
    refs={}
    for seed in cfg["seeds"]:
        model=G.load(G.Target(),out,target_rows[seed])
        targets[seed]=model
        refs[seed]={w["root"]:G.serial(model,w["prefix"]) for w in sets["trace_train"]}
        data=teacher_data(model,sets["discovery"])
        for mode in MODES:
            c=discoveries[seed,mode]
            path=out/c["artifact"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=c["artifact_sha256"]:
                raise ValueError("discovery artifact")
            stored=json.loads(path.read_text(encoding="utf-8"))
            if digest(stored)!=c["discovery_hash"]:
                raise ValueError("discovery content")
            bases,replayed=discover(model,seed,mode,data,sets["discovery"])
            if {k:v for k,v in stored.items() if k!="sites"}!={k:v for k,v in replayed.items() if k!="sites"}:
                raise ValueError("discovery lineage/work")
            stored_bases={}
            for site in SITES:
                u=torch.tensor(stored["sites"][site]["basis"],dtype=torch.float64)
                v=bases[site]
                if not torch.allclose(u.T@u,torch.eye(RANK,dtype=u.dtype),atol=1e-9,rtol=1e-9):
                    raise ValueError("stored basis orthonormality")
                for field in ("covariance","eigenvalues"):
                    left=torch.tensor(stored["sites"][site][field],dtype=torch.float64)
                    right=torch.tensor(replayed["sites"][site][field],dtype=torch.float64)
                    if not torch.allclose(left,right,atol=1e-10,rtol=1e-7):
                        raise ValueError("discovery numeric replay")
                if stored["sites"][site]["gradient_vectors"]!=replayed["sites"][site]["gradient_vectors"]:
                    raise ValueError("gradient count replay")
                if not torch.allclose(u@u.T,v@v.T,atol=1e-6,rtol=1e-6):
                    raise ValueError("discovered projector replay")
                stored_bases[site]=u
            for arm in ARMS:
                key=(seed,mode,arm)
                c=cells[key]
                if c["target_hash"]!=G.weight_hash(model) or c["rank"]!=RANK:
                    raise ValueError("target or rank binding")
                if G.weight_hash(initial_adapter(seed,arm,stored_bases))!=c["initial_hash"]:
                    raise ValueError("adapter initialization replay")
                adapters[key]=G.load(G.DraftAdapter(RANK),out,c)
    path=out/"fg05_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace digest")
    worlds={w["root"]:w for w in sets["trace_train"]}
    seen=set()
    all_rows={}
    for line in path.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        bound=r.pop("receipt_hash")
        key=(r["seed"],r["mode"],r["arm"])
        ident=(key,r["root"])
        if (digest(r)!=bound or r["schema"]!=SCHEMA+".rollout" or r["split"]!="trace_train"
                or r["training_ready"] is not False or key not in cells or r["root"] not in worlds
                or ident in seen):
            raise ValueError("trace custody")
        w=worlds[r["root"]]
        _,rows=measured(targets[key[0]],adapters[key],key[1],[w],refs[key[0]])
        replay=record(*key,cells[key],rows[0])
        if replay["receipt_hash"]!=bound:
            raise ValueError("exact decoded-token or acceptance replay")
        all_rows.setdefault(key,[]).append(rows[0])
        seen.add(ident)
    if len(seen)!=len(cells)*len(worlds):
        raise ValueError("missing rollout")
    for seed,mode in itertools.product(cfg["seeds"],MODES):
        for control in ("ordinary","haar","shuffled"):
            wanted={"seed":seed,"mode":mode,"control":control,"split":"trace_train",
                    **pair_delta(all_rows[seed,mode,"guided"],all_rows[seed,mode,control])}
            if wanted not in report["paired"]:
                raise ValueError("paired delta replay")
    return {"status":"audited_development_only","targets":len(targets),
            "discoveries":len(discoveries),"adapters":len(adapters),
            "rollouts":len(seen),"greedy_mismatches":0}


def selftest():
    torch.set_num_threads(1)
    cov=torch.diag(torch.arange(1,G.WIDTH+1,dtype=torch.float64))
    _,u=leading(cov)
    q=torch.linalg.qr(torch.randn(G.WIDTH,G.WIDTH,dtype=torch.float64,
                                  generator=torch.Generator().manual_seed(11)))[0]
    _,v=leading(q.T@cov@q)
    assert torch.allclose(v@v.T,q.T@u@u.T@q,atol=1e-10,rtol=1e-10)
    cfg=dict(DEFAULT,base_steps=2,adapter_steps=2,batch=4,base_n=16,adapter_n=12,eval_n=4,seeds=[137])
    sets=partitions(cfg)
    model,_=G.fit_target("final_only",137,sets["base_train"],cfg)
    data=teacher_data(model,sets["discovery"])
    bases,d=discover(model,137,"first",data,sets["discovery"])
    assert d["slot_backward_calls"]==len(sets["discovery"])*CUTS_PER_ROOT*G.BLOCK
    masks=set()
    for arm in ARMS:
        a=initial_adapter(137,arm,bases)
        masks.add(digest(a.mask.detach().tolist()))
    assert len(masks)==1
    print("FG-05 gradient projector, rotation and matched zero-update selftests passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--selftest",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.selftest:
        selftest()
    if a.out:
        cfg=dict(DEFAULT,base_steps=4,adapter_steps=4,batch=4,base_n=16,adapter_n=12,eval_n=4,seeds=[137]) if a.smoke else None
        r=run(a.out,cfg)
        print(json.dumps({"status":r["status"],"targets":len(r["targets"]),"adapters":len(r["cells"])}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
