"""GSF-10 / FG-01: CPU masked-block self-speculation with real per-loop KV.

Independent synthetic causal target, draft-only low-rank parameters, greedy
verification with physical cache truncation. No pretrained LM or speed claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn

SCHEMA = "hybrid.gsf.floodgates.v1"
VOCAB = 11
WIDTH = 32
LOOPS = 4
PREFIX = 8
GENERATE = 12
MAX_POS = 64
BLOCK = 4
SEEDS = (137,)
SUPERVISION = ("final_only", "all_depth")
ARMS = (("first", 4), ("deep", 4), ("null", 4), ("deep", 8))
SPLITS = ("base_train", "adapter_train", "cal", "dev", "shift", "trace_train")
DEFAULT = dict(base_steps=240, adapter_steps=180, batch=16, base_n=256,
               adapter_n=96, eval_n=24, seeds=list(SEEDS))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def weight_hash(model):
    return digest({k: v.detach().cpu().tolist() for k,v in model.state_dict().items()})


def root(split, index):
    return digest([SCHEMA, split, index])


def world(split, index):
    if split not in SPLITS:
        raise ValueError(split)
    rng = random.Random(int(root(split,index),16))
    prefix = [rng.randrange(VOCAB) for _ in range(PREFIX)]
    return {"root":root(split,index),"split":split,"index":index,"prefix":prefix,
            "period":3 if split=="shift" else 1+prefix[0]%2}


def truth(w):
    tokens=list(w["prefix"])
    for _ in range(GENERATE):
        tokens.append(tokens[-w["period"]])
    return tokens


def partitions(cfg):
    n={"base_train":cfg["base_n"],"adapter_train":cfg["adapter_n"]}
    sets={s:[world(s,i) for i in range(n.get(s,cfg["eval_n"]))] for s in SPLITS}
    seen=set()
    for s in SPLITS:
        keys={tuple(w["prefix"]) for w in sets[s]}
        if len(keys)!=len(sets[s]) or seen&keys:
            raise ValueError("prefix lineage collision")
        seen.update(keys)
    return sets


def validate(cfg):
    limits={"base_steps":(1,1024),"adapter_steps":(1,1024),"batch":(1,64),
            "base_n":(8,1024),"adapter_n":(8,512),"eval_n":(4,128)}
    if set(cfg)!=set(limits)|{"seeds"} or any(type(cfg[k]) is not int or not lo<=cfg[k]<=hi
                                             for k,(lo,hi) in limits.items()):
        raise ValueError("invalid resource bounds")
    if not 1<=len(cfg["seeds"])<=4 or len(set(cfg["seeds"]))!=len(cfg["seeds"]) or any(
        type(s) is not int for s in cfg["seeds"]):
        raise ValueError("invalid seeds")


class LoRA(nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.a=nn.Linear(WIDTH,rank,bias=False)
        self.b=nn.Linear(rank,WIDTH,bias=False)
        nn.init.zeros_(self.b.weight)
        self.scale=1./rank

    def forward(self,x):
        return self.b(self.a(x))*self.scale


class DraftAdapter(nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.rank=rank
        self.q=LoRA(rank)
        self.v=LoRA(rank)
        self.o=LoRA(rank)
        self.mask=nn.Parameter(torch.empty(WIDTH))
        nn.init.normal_(self.mask,std=.1)


class SharedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(WIDTH)
        self.q=nn.Linear(WIDTH,WIDTH,bias=False)
        self.k=nn.Linear(WIDTH,WIDTH,bias=False)
        self.v=nn.Linear(WIDTH,WIDTH,bias=False)
        self.o=nn.Linear(WIDTH,WIDTH,bias=False)
        self.ff=nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,2*WIDTH),
                              nn.GELU(),nn.Linear(2*WIDTH,WIDTH))

    def forward(self,h,past=None,adapter=None):
        x=self.norm(h)
        q,k,v=self.q(x),self.k(x),self.v(x)
        if adapter is not None:
            q=q+adapter.q(x)
            v=v+adapter.v(x)
        offset=0 if past is None else past[0].shape[1]
        if past is not None:
            k=torch.cat((past[0],k),dim=1)
            v=torch.cat((past[1],v),dim=1)
        qi=torch.arange(h.shape[1],device=h.device)+offset
        ki=torch.arange(k.shape[1],device=h.device)
        scores=(q@k.transpose(-1,-2))/math.sqrt(WIDTH)
        scores=scores.masked_fill(ki[None,:]>qi[:,None],float("-inf"))
        msg=scores.softmax(-1)@v
        update=self.o(msg)
        if adapter is not None:
            update=update+adapter.o(msg)
        h=h+update
        return h+self.ff(h),(k,v)


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(VOCAB,WIDTH)
        self.position=nn.Embedding(MAX_POS,WIDTH)
        self.block=SharedBlock()
        self.head=nn.Linear(WIDTH,VOCAB)

    def embed(self,tokens,offset):
        pos=torch.arange(offset,offset+tokens.shape[1],device=tokens.device)
        return self.embedding(tokens)+self.position(pos)

    def forward(self,tokens,past=None):
        offset=0 if past is None else past[0][0].shape[1]
        if past is not None and (len(past)!=LOOPS or any(
            k.shape!=v.shape or k.shape[0]!=tokens.shape[0] or k.shape[1]!=offset
            for k,v in past)):
            raise ValueError("invalid target cache")
        h=self.embed(tokens,offset)
        caches=[]
        logits=[]
        for i in range(LOOPS):
            h,kv=self.block(h,None if past is None else past[i])
            caches.append(kv)
            logits.append(self.head(h))
        return torch.stack(logits,dim=1),tuple(caches)

    def draft(self,anchor,past,adapter,k,cache_mode):
        if not 1<=k<=BLOCK or cache_mode not in ("first","deep","null"):
            raise ValueError("invalid draft")
        offset=past[0][0].shape[1]
        positions=torch.arange(offset+1,offset+k+1,device=anchor.device)
        masks=adapter.mask[None,None,:]+self.position(positions)[None,:,:]
        h=torch.cat((self.embed(anchor,offset),masks.expand(anchor.shape[0],-1,-1)),dim=1)
        selected=past[0] if cache_mode=="first" else past[-1]
        if cache_mode=="null":
            selected=tuple(torch.zeros_like(v) for v in selected)
        h,_=self.block(h,selected,adapter)
        return self.head(h[:,1:])


def crop(past,n):
    if any(not 0<=n<=k.shape[1] for k,v in past):
        raise ValueError("invalid cache truncation")
    # Clone to prevent stale suffix storage from surviving in a view.
    return tuple((k[:,:n].clone(),v[:,:n].clone()) for k,v in past)


def cache_size(past):
    return sum(x.numel()*x.element_size() for kv in past for x in kv)


def fit_target(supervision,seed,worlds,cfg):
    torch.manual_seed(seed)
    model=Target()
    before=weight_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    data=torch.tensor([truth(w) for w in worlds])
    rng=torch.Generator().manual_seed(seed+103)
    for _ in range(cfg["base_steps"]):
        idx=torch.randint(len(data),(cfg["batch"],),generator=rng)
        seq=data[idx]
        z,_=model(seq[:,:-1])
        y=seq[:,PREFIX:]
        z=z[:,:,PREFIX-1:]
        if supervision=="final_only":
            loss=nn.functional.cross_entropy(z[:,-1].reshape(-1,VOCAB),y.reshape(-1))
        else:
            loss=nn.functional.cross_entropy(z.reshape(-1,VOCAB),y[:,None,:].expand(-1,LOOPS,-1).reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    model.requires_grad_(False)
    after=weight_hash(model)
    if after==before:
        raise AssertionError("target fit unchanged")
    return model,{"initial_hash":before,"weights_hash":after,"updates":cfg["base_steps"],
                  "parameters":sum(p.numel() for p in model.parameters()),
                  "train_block_token_visits":cfg["base_steps"]*cfg["batch"]*(PREFIX+GENERATE-1)*LOOPS,
                  "last_loss":float(loss.detach())}


@torch.no_grad()
def teacher_stream(model,prefixes):
    tokens=prefixes.clone()
    _,past=model(tokens[:,:-1])
    for _ in range(GENERATE):
        z,past=model(tokens[:,-1:],past)
        tokens=torch.cat((tokens,z[:,-1,-1].argmax(-1,keepdim=True)),dim=1)
    return tokens


def fit_adapter(model,seed,mode,rank,worlds,cfg):
    frozen=weight_hash(model)
    torch.manual_seed(seed+700)
    adapter=DraftAdapter(rank)
    before=weight_hash(adapter)
    # Train on target-generated suffixes from independent adapter-training roots.
    with torch.no_grad():
        prefixes=torch.tensor([w["prefix"] for w in worlds])
        seq=teacher_stream(model,prefixes)
        _,full_cache=model(seq)
    rng=random.Random(seed+301)
    opt=torch.optim.AdamW(adapter.parameters(),lr=.01)
    for _ in range(cfg["adapter_steps"]):
        ids=torch.tensor([rng.randrange(len(seq)) for _ in range(cfg["batch"])])
        cut=rng.randrange(PREFIX,PREFIX+GENERATE-BLOCK+1)
        # Prefix caches contain only positions strictly before the anchor.
        past=tuple((k[ids,:cut-1],v[ids,:cut-1]) for k,v in full_cache)
        anchor=seq[ids,cut-1:cut]
        targets=seq[ids,cut:cut+BLOCK]
        logits=model.draft(anchor,past,adapter,BLOCK,mode)
        loss=nn.functional.cross_entropy(logits.reshape(-1,VOCAB),targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    adapter.eval()
    if weight_hash(model)!=frozen or weight_hash(adapter)==before:
        raise AssertionError("frozen target or adapter fit violation")
    return adapter,{"initial_hash":before,"weights_hash":weight_hash(adapter),
                    "target_hash":frozen,"parameters":sum(p.numel() for p in adapter.parameters()),
                    "updates":cfg["adapter_steps"],
                    "train_draft_block_token_visits":cfg["adapter_steps"]*cfg["batch"]*(BLOCK+1),
                    "teacher_stream_block_token_visits":len(worlds)*(PREFIX-1+GENERATE)*LOOPS,
                    "teacher_cache_build_block_token_visits":len(worlds)*(PREFIX+GENERATE)*LOOPS,
                    "last_loss":float(loss.detach())}


@torch.no_grad()
def verify_step(model,tokens,past,drafts,remaining):
    """Cache owns all committed tokens except the current anchor."""
    if past[0][0].shape[1]!=len(tokens)-1 or not drafts or remaining<1:
        raise ValueError("invalid verifier state")
    x=torch.tensor([[tokens[-1],*drafts]])
    z,extended=model(x,past)
    predictions=z[0,-1].argmax(-1).tolist()
    accepted=0
    while accepted<len(drafts) and drafts[accepted]==predictions[accepted]:
        accepted+=1
    emitted=(drafts[:accepted]+[predictions[accepted]])[:remaining]
    new_tokens=tokens+emitted
    keep=len(new_tokens)-1
    committed=crop(extended,keep)
    return new_tokens,committed,{"draft":list(drafts),"target_predictions":predictions,
         "accepted_prefix":accepted,"emitted":emitted,"cache_before":len(tokens)-1,
         "cache_provisional":extended[0][0].shape[1],"cache_after":keep,
         "discarded_cache_positions":extended[0][0].shape[1]-keep,
         "verify_block_token_visits":LOOPS*len(x[0]),
         "verify_calls":1,"draft_block_token_visits":len(drafts)+1,
         "peak_provisional_kv_bytes":cache_size(extended)}


@torch.no_grad()
def serial(model,prefix):
    tokens=list(prefix)
    _,past=model(torch.tensor([tokens[:-1]]))
    for _ in range(GENERATE):
        z,past=model(torch.tensor([[tokens[-1]]]),past)
        token=int(z[0,-1,-1].argmax())
        # Independent cold forward checks the ordinary cache path.
        cold,_=model(torch.tensor([tokens]))
        if token!=int(cold[0,-1,-1].argmax()) or not torch.allclose(
            z[0,-1,-1],cold[0,-1,-1],rtol=1e-5,atol=1e-5):
            raise AssertionError("ordinary cached/cold parity")
        tokens.append(token)
    return tokens


@torch.no_grad()
def generate(model,adapter,mode,prefix,k):
    tokens=list(prefix)
    _,past=model(torch.tensor([tokens[:-1]]))
    steps=[]
    work=Counter()
    start=time.perf_counter()
    while len(tokens)<PREFIX+GENERATE:
        remaining=PREFIX+GENERATE-len(tokens)
        proposal=model.draft(torch.tensor([[tokens[-1]]]),past,adapter,min(k,remaining),mode)
        drafts=proposal[0].argmax(-1).tolist()
        tokens,past,step=verify_step(model,tokens,past,drafts,remaining)
        steps.append(step)
        work["draft_calls"]+=1
        for key in ("verify_calls","verify_block_token_visits","draft_block_token_visits","accepted_prefix"):
            work[key]+=step[key]
        work["draft_tokens"]+=len(drafts)
        work["emitted_tokens"]+=len(step["emitted"])
        work["discarded_cache_positions"]+=step["discarded_cache_positions"]
        work["peak_provisional_kv_bytes"]=max(work["peak_provisional_kv_bytes"],step["peak_provisional_kv_bytes"])
    seconds=time.perf_counter()-start
    return tokens,steps,dict(work),seconds


def evaluate(model,adapter,mode,k,worlds,reference):
    rows=[]
    total=Counter()
    elapsed=0.
    for w in worlds:
        tokens,steps,work,sec=generate(model,adapter,mode,w["prefix"],k)
        ref=reference[w["root"]]
        if tokens!=ref:
            raise AssertionError(f"greedy parity failure {w['root']}")
        for name,value in work.items():
            total[name]=max(total[name],value) if name.startswith("peak_") else total[name]+value
        correct=sum(a==b for a,b in zip(tokens[PREFIX:],truth(w)[PREFIX:]))
        total["truth_correct_tokens"]+=correct
        total["truth_exact_sequences"]+=tokens==truth(w)
        elapsed+=sec
        rows.append({"public":w,"draft_steps":steps,"work":work,"verified_output":tokens[PREFIX:],
                     "reference_output":ref[PREFIX:],"truth_output":truth(w)[PREFIX:],
                     "greedy_parity":True,"truth_correct_tokens":correct})
    return {"n":len(rows),**dict(total),
            "serial_verify_calls":len(rows)*GENERATE,
            "serial_block_token_visits":len(rows)*GENERATE*LOOPS,
            "prefill_block_token_visits":len(rows)*(PREFIX-1)*LOOPS,
            "emitted_per_verification":total["emitted_tokens"]/total["verify_calls"],
            "truth_accuracy":total["truth_correct_tokens"]/(len(rows)*GENERATE),
            "draft_verify_seconds":elapsed},rows


def choose_k(cal):
    # CAL only: minimize target calls, then actual token-block visits.
    return min(cal,key=lambda k:(cal[k]["verify_calls"],cal[k]["verify_block_token_visits"]+
                                 cal[k]["draft_block_token_visits"],k))


def save(out,name,model):
    p=out/name
    torch.save(model.state_dict(),p)
    return {"artifact":name,"artifact_sha256":hashlib.sha256(p.read_bytes()).hexdigest()}


def run(out,cfg=None):
    cfg=DEFAULT.copy() if cfg is None else cfg
    validate(cfg)
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    sets=partitions(cfg)
    report={"schema":SCHEMA,"status":"public_development_only","config":cfg,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "split_hashes":{s:digest([w["root"] for w in ws]) for s,ws in sets.items()},
            "targets":[],"cells":[]}
    path=out/"floodgates_rollouts.jsonl"
    with path.open("w",encoding="utf-8") as f:
        for seed,supervision in itertools.product(cfg["seeds"],SUPERVISION):
            model,info=fit_target(supervision,seed,sets["base_train"],cfg)
            info.update(save(out,f"{seed}-{supervision}-target.pt",model),
                        seed=seed,supervision=supervision)
            report["targets"].append(info)
            refs={s:{w["root"]:serial(model,w["prefix"]) for w in sets[s]}
                  for s in ("cal","dev","shift","trace_train")}
            for mode,rank in ARMS:
                adapter,fit=fit_adapter(model,seed,mode,rank,sets["adapter_train"],cfg)
                row={"seed":seed,"supervision":supervision,"mode":mode,"rank":rank,**fit,
                     **save(out,f"{seed}-{supervision}-{mode}-{rank}.pt",adapter)}
                cal={k:evaluate(model,adapter,mode,k,sets["cal"],refs["cal"])[0] for k in (1,2,4)}
                k=choose_k(cal)
                row["calibration"]={str(n):m for n,m in cal.items()}
                row["selected_k"]=k
                row["panels"]={}
                for s in ("dev","shift","trace_train"):
                    stats,rows=evaluate(model,adapter,mode,k,sets[s],refs[s])
                    row["panels"][s]=stats
                    if s=="trace_train":
                        for r in rows:
                            rec={"schema":SCHEMA+".rollout","split":s,"root":r["public"]["root"],
                                 "seed":seed,"supervision":supervision,"mode":mode,"rank":rank,"k":k,
                                 "target_weights_hash":info["weights_hash"],"adapter_weights_hash":fit["weights_hash"],
                                 **r,"training_ready":False}
                            rec["receipt_hash"]=digest(rec)
                            f.write(json.dumps(rec,sort_keys=True,allow_nan=False)+"\n")
                if weight_hash(model)!=info["weights_hash"]:
                    raise AssertionError("target mutated")
                report["cells"].append(row)
    report["trace_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"floodgates_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def load(model,out,receipt):
    path=out/receipt["artifact"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt["artifact_sha256"]:
        raise ValueError("artifact digest")
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    model.eval()
    model.requires_grad_(False)
    if weight_hash(model)!=receipt["weights_hash"]:
        raise ValueError("weight digest")
    return model


def audit(out):
    torch.set_num_threads(1)
    report=json.loads((out/"floodgates_report.json").read_text(encoding="utf-8"))
    cfg=report["config"]
    validate(cfg)
    if report["source_sha256"]!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("source digest")
    sets=partitions(cfg)
    if report["split_hashes"]!={s:digest([w["root"] for w in ws]) for s,ws in sets.items()}:
        raise ValueError("split digest")
    targets={(r["seed"],r["supervision"]):load(Target(),out,r) for r in report["targets"]}
    if len(report["targets"])!=len(targets) or set(targets)!=set(itertools.product(cfg["seeds"],SUPERVISION)):
        raise ValueError("target coverage")
    cells={(r["seed"],r["supervision"],r["mode"],r["rank"]):r for r in report["cells"]}
    expected={(s,b,m,r) for s,b in targets for m,r in ARMS}
    if set(cells)!=expected or len(cells)!=len(report["cells"]):
        raise ValueError("adapter coverage")
    adapters={}
    refs={}
    for key,row in cells.items():
        tkey=key[:2]
        model=targets[tkey]
        if row["target_hash"]!=weight_hash(model):
            raise ValueError("target binding")
        adapter=load(DraftAdapter(key[3]),out,row)
        adapters[key]=adapter
        if tkey not in refs:
            refs[tkey]={s:{w["root"]:serial(model,w["prefix"]) for w in sets[s]}
                        for s in ("cal","trace_train")}
        cal={k:evaluate(model,adapter,key[2],k,sets["cal"],refs[tkey]["cal"])[0] for k in (1,2,4)}
        if choose_k(cal)!=row["selected_k"]:
            raise ValueError("CAL selection replay")
        for k,m in cal.items():
            if {a:v for a,v in m.items() if a!="draft_verify_seconds"}!={
                a:v for a,v in row["calibration"][str(k)].items() if a!="draft_verify_seconds"}:
                raise ValueError("CAL metric mismatch")
    path=out/"floodgates_rollouts.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report["trace_sha256"]:
        raise ValueError("trace inventory")
    seen=set()
    roots={w["root"]:w for w in sets["trace_train"]}
    for line in path.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        bound=r.pop("receipt_hash")
        key=(r["seed"],r["supervision"],r["mode"],r["rank"])
        ident=(key,r["root"])
        if (digest(r)!=bound or r["schema"]!=SCHEMA+".rollout" or r["split"]!="trace_train"
                or r["training_ready"] is not False or ident in seen or key not in cells
                or r["root"] not in roots or r["public"]!=roots[r["root"]]
                or r["target_weights_hash"]!=cells[key]["target_hash"]
                or r["adapter_weights_hash"]!=cells[key]["weights_hash"] or r["k"]!=cells[key]["selected_k"]):
            raise ValueError("receipt custody")
        w=roots[r["root"]]
        _,rows=evaluate(targets[key[:2]],adapters[key],key[2],r["k"],[w],refs[key[:2]]["trace_train"])
        if any(r[a]!=v for a,v in rows[0].items()):
            raise ValueError("draft or verification replay")
        seen.add(ident)
    if len(seen)!=len(cells)*len(roots):
        raise ValueError("missing rollout")
    return {"status":"audited_development_only","targets":len(targets),"adapters":len(cells),
            "rollouts":len(seen),"greedy_mismatches":0}


@torch.no_grad()
def selftest():
    torch.manual_seed(137)
    model=Target().eval()
    prefix=world("dev",0)["prefix"]
    _,past=model(torch.tensor([prefix[:-1]]))
    ref=serial(model,prefix)
    # Exhaust first mismatch positions and the all-accepted/bonus case.
    for j in range(BLOCK+1):
        d=ref[PREFIX:PREFIX+BLOCK]
        if j<BLOCK:
            d[j]=(d[j]+1)%VOCAB
        tokens,kv,step=verify_step(model,prefix,past,d,GENERATE)
        assert step["accepted_prefix"]==j
        assert tokens==ref[:len(tokens)]
        assert all(k.shape[1]==len(tokens)-1 for k,v in kv)
        assert step["discarded_cache_positions"]==BLOCK-j
        # Fresh cache replay verifies rollback; values close, next ID exact.
        _,fresh=model(torch.tensor([tokens[:-1]]))
        for pair1,pair2 in zip(kv,fresh):
            for a,b in zip(pair1,pair2):
                assert torch.allclose(a,b,rtol=1e-5,atol=1e-5)
        z,_=model(torch.tensor([[tokens[-1]]]),kv)
        assert int(z[0,-1,-1].argmax())==ref[len(tokens)]
    tokens,kv,step=verify_step(model,prefix,past,ref[PREFIX:PREFIX+BLOCK],1)
    assert tokens==ref[:PREFIX+1] and kv[0][0].shape[1]==PREFIX
    # Causality: modifying later tokens cannot change prefix cache.
    x=torch.tensor([ref])
    y=x.clone()
    y[:,PREFIX:]=(y[:,PREFIX:]+1)%VOCAB
    _,c1=model(x)
    _,c2=model(y)
    for a,b in zip(crop(c1,PREFIX),crop(c2,PREFIX)):
        for u,v in zip(a,b):
            assert torch.equal(u,v)
    print("GSF-10 greedy verification, cache rollback and causal prefix checks passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.selftest:
        selftest()
    if a.out:
        cfg=dict(DEFAULT,base_steps=4,adapter_steps=4,batch=4,base_n=16,adapter_n=8,eval_n=4) if a.smoke else None
        r=run(a.out,cfg)
        print(json.dumps({"status":r["status"],"targets":len(r["targets"]),"adapters":len(r["cells"])}))
    if a.audit:
        print(json.dumps(audit(a.audit),sort_keys=True))
    if not (a.selftest or a.out or a.audit):
        p.error("choose --selftest, --out or --audit")


if __name__=="__main__":
    main()
