"""CPU spatial retrieval proxy; native Storyworld witness replay is a separate track."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PAGE=32
CAPACITIES=(16384,24576,32768,65536)
LOCAL=(-4,-3,-2,-1,1,2,3,4)
HIER=(-32,-8,-2,-1,1,2,8,32)
ARMS=('scan','row','random','hilbert','hier','adaptive')


def hilbert_xy(side,d):
    x=y=0; s=1
    while s<side:
        rx=(d//2)&1; ry=(d^rx)&1
        if ry==0:
            if rx: x,y=s-1-x,s-1-y
            x,y=y,x
        x+=s*rx; y+=s*ry; d//=4; s*=2
    return x,y


def order(n,layout,seed):
    width=128 if n<=32768 else 256
    if n%width or n%PAGE: raise ValueError('invalid capacity')
    height=n//width
    if layout=='row': return np.arange(n)
    if layout=='random': return np.random.default_rng(seed).permutation(n)
    if layout!='hilbert': raise ValueError('invalid layout')
    side=1<<(max(width,height)-1).bit_length()
    xy=(hilbert_xy(side,d) for d in range(side*side))
    return np.array([y*width+x for x,y in xy if x<width and y<height])


@dataclass(frozen=True)
class Root:
    anchor:int; target:int; hint_x:int; hint_y:int


class World:
    def __init__(self,n,seed,layout):
        self.n=n; self.width=128 if n<=32768 else 256; self.height=n//self.width
        self.pages=n//PAGE; self.seed=seed
        rng=np.random.default_rng(seed)
        self.keys=rng.permutation(n//2); self.values=rng.integers(0,2**30,n//2,dtype=np.int64)
        self.stale=rng.random(n)
        self.cells=order(n,layout,seed+901).reshape(-1,PAGE)
        self.inv=np.empty(n,dtype=int); self.inv[self.cells.ravel()]=np.arange(n)//PAGE
        self.centers=np.stack([(self.cells%self.width).mean(1),(self.cells//self.width).mean(1)],axis=1)
    def roots(self,count,salt):
        rng=np.random.default_rng(self.seed+salt)
        for _ in range(count):
            a=int(rng.integers(self.n)); x,y=a%self.width,a//self.width
            dx,dy=map(int,rng.integers(-12,13,2)); tx=int(np.clip(x+dx,0,self.width-1)); ty=int(np.clip(y+dy,0,self.height-1))
            yield Root(a,(ty*self.width+tx)//2,int(np.sign(tx-x)),int(np.sign(ty-y)))
    def probe(self,p,key,corruption):
        cells=self.cells[p]; matched=cells[self.keys[cells//2]==key]
        good=matched[self.stale[matched]>=corruption]
        return (int(self.values[good[0]//2]) if len(good) else None, bool(np.any(self.stale[matched]<corruption)),len(good))


def options(w,p,seen,offsets):
    candidates=np.asarray([p+d for d in offsets]); valid=np.array([0<=q<w.pages and q not in seen for q in candidates])
    return candidates,valid


def observe(w,r,p,a,hit,stale,remaining,valid):
    x,y=r.anchor%w.width,r.anchor//w.width; cx,cy=w.centers[p]
    return np.asarray([p/w.pages,a/w.pages,(p-a)/w.pages,remaining/32,hit,stale,r.hint_x,r.hint_y,x/w.width,y/w.height,(cx-x)/w.width,(cy-y)/w.height,*valid],dtype=np.float32)


class Controller(nn.Module):
    def __init__(self):
        super().__init__(); self.fast=nn.GRUCell(20+64,64); self.slow=nn.GRUCell(64,64); self.head=nn.Linear(64,9)
    def initial(self,b): return torch.zeros(b,64),torch.zeros(b,64)
    def forward(self,x,h,s,t):
        h=self.fast(torch.cat((x,s),-1),h)
        if t%4==0: s=self.slow(h,s)
        y=self.head(h); return y[:,:8],y[:,8],h,s


def trace(w,r,offsets,budget=16):
    a=p=int(w.inv[r.anchor]); seen=set(); key=w.keys[r.target]; goal=w.inv[2*r.target:2*r.target+2]; rows=[]
    for t in range(budget):
        seen.add(p); value,stale,_=w.probe(p,key,0); hit=value is not None
        cand,valid=options(w,p,seen,offsets); action=-1
        if not hit and valid.any():
            action=min(np.flatnonzero(valid),key=lambda i:(min(abs(cand[i]-goal)),abs(offsets[i]),i))
        rows.append((observe(w,r,p,a,hit,stale,budget-t,valid),int(action),float(hit)))
        if action<0: break
        p=int(cand[action])
    return rows


def train(w,offsets,seed,count,epochs):
    torch.manual_seed(seed); model=Controller(); opt=torch.optim.AdamW(model.parameters(),lr=.002)
    traces=[trace(w,r,offsets) for r in w.roots(count,37)]
    x=np.zeros((count,16,20),np.float32); a=np.full((count,16),-1,np.int64); y=np.zeros((count,16),np.float32); m=np.zeros((count,16),bool)
    for i,rows in enumerate(traces):
        for t,(xx,aa,yy) in enumerate(rows): x[i,t]=xx; a[i,t]=aa; y[i,t]=yy; m[i,t]=True
    x,a,y,m=map(torch.from_numpy,(x,a,y,m)); rng=np.random.default_rng(seed+12)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(count),math.ceil(count/32)):
            h,s=model.initial(len(idx)); loss=0.
            for t in range(16):
                logits,stop,h,s=model(x[idx,t],h,s,t); mask=m[idx,t]; targets=a[idx,t]; legal=x[idx,t,12:].bool()
                active=mask&(targets>=0)
                if active.any(): loss=loss+F.cross_entropy(logits[active].masked_fill(~legal[active],-1e9),targets[active],reduction='sum')
                if mask.any(): loss=loss+F.binary_cross_entropy_with_logits(stop[mask],y[idx,t][mask],pos_weight=torch.tensor(4.),reduction='sum')
            opt.zero_grad(); (loss/m[idx].sum()).backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    return model.eval()


@torch.inference_mode()
def rollout(w,r,arm,model,budget,corruption):
    a=p=int(w.inv[r.anchor]); seen=set(); key=w.keys[r.target]; best=None; stale_seen=False; early=False; matched=0; distance=0.; visits=[]
    offsets=HIER if arm in ('hier','adaptive') else LOCAL
    if model is not None: h,s=model.initial(1)
    for t in range(budget):
        seen.add(p); visits.append(p); v,stale,good=w.probe(p,key,corruption); matched+=good; stale_seen|=stale
        if v is not None and best is None: best=v
        cx,cy=w.centers[p]; distance+=abs(cx-r.anchor%w.width)+abs(cy-r.anchor//w.width)
        if arm=='scan':
            if best is not None: break
            next_page=next((q for d in range(1,budget+1) for q in (a+d,a-d) if 0<=q<w.pages and q not in seen),None)
            if next_page is None: break
            p=next_page; continue
        cand,valid=options(w,p,seen,offsets)
        x=torch.from_numpy(observe(w,r,p,a,v is not None,stale,budget-t,valid))[None]
        logits,stop,h,s=model(x,h,s,t)
        if arm=='adaptive' and stop.item()>=0:
            early=best is None; break
        if not valid.any(): break
        p=int(cand[logits[0].masked_fill(~torch.from_numpy(valid),-1e9).argmax().item()])
    success=best is not None and best==int(w.values[r.target])
    return dict(success=success,value=best,probes=len(visits),records_read=PAGE*len(visits),matched=matched,stale_seen=stale_seen,recovered=stale_seen and success,premature_stop=early,mean_center_distance=float(distance/len(visits)),pages=visits)


def native_replay(path,w,arm,model,budget,corruption):
    from MCP.mcp05.hilbert_storyworld import START,transition,classify,replay_path
    data=json.loads(Path(path).read_text()); out=[]
    for k,(ending,entry) in enumerate(sorted(data['exact']['secret_gates'].items())):
        route=entry['witness']; state=START; clues=frozenset(); i=0
        for si,j in route:
            if si!=i: raise ValueError('invalid archived witness route')
            advance,state,clues=transition(data['scenes'][si]['options'][j],data['math'],state,clues); i+=advance
        if i<len(data['scenes']) or ending not in classify(data['math'],state,clues): raise ValueError('archived witness fails native replay')
        order_control=entry.get('order_witness')
        if order_control:
            baseline,facts=replay_path(data['scenes'],data['math'],order_control['witness'])
            swapped,same_facts=replay_path(data['scenes'],data['math'],order_control['witness'],swap=order_control['swap_decisions'])
            if facts!=same_facts or ending not in classify(data['math'],baseline,facts) or ending in classify(data['math'],swapped,same_facts): raise ValueError('native swap control fails')
        state=START; clues=frozenset(); i=0; receipts=[]
        for r,(si,j) in zip(w.roots(len(route),9000+k),route,strict=True):
            old=w.values[r.target]; w.values[r.target]=3*si+j
            try: receipt=rollout(w,r,arm,model,budget,corruption)
            finally: w.values[r.target]=old
            receipts.append(receipt)
            if not receipt['success']: break
            ri,rj=divmod(receipt['value'],3)
            if ri!=i or (ri,rj)!=(si,j): break
            advance,state,clues=transition(data['scenes'][ri]['options'][rj],data['math'],state,clues); i+=advance
        out.append(dict(ending=ending,success=len(receipts)==len(route) and all(r['success'] for r in receipts) and i>=len(data['scenes']) and ending in classify(data['math'],state,clues),queries=receipts,swap_control_verified=bool(order_control)))
    return out


def run(args):
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    if args.threads<1: raise ValueError('threads must be positive')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False); start=time.perf_counter(); summary=[]; native=[]; paired={}; checkpoints={}
    seeds=(107,) if args.smoke else (107,109,113); sizes=(16384,32768) if args.smoke else CAPACITIES
    budgets=(8,16) if args.smoke else (4,8,16,32); corruptions=(0.,.15) if args.smoke else (0.,.05,.15)
    count=64 if args.smoke else 1024; epochs=2 if args.smoke else 8; roots=8 if args.smoke else 128
    namespace=700000 if args.smoke else 2000000
    with (out/'roots.jsonl').open('w') as f:
        for seed in seeds:
            models={}
            for arm in ARMS[1:-1]:
                layout='hilbert' if arm=='hier' else arm
                model=train(World(16384,10000+seed,layout),HIER if arm=='hier' else LOCAL,seed,count,epochs); models[arm]=model
                cp=out/f'{seed}-{arm}.pt'; torch.save(model.state_dict(),cp); checkpoints[cp.name]=hashlib.sha256(cp.read_bytes()).hexdigest()
            models['adaptive']=models['hier']; models['scan']=None
            for arm in ARMS:
                layout=arm if arm in ('row','random') else 'hilbert'
                for n in sizes:
                    w=World(n,namespace+100*seed+n,layout); rr=list(w.roots(roots,331))
                    for c in corruptions:
                        for b in budgets:
                            cell_start=time.perf_counter(); results=[]
                            for idx,r in enumerate(rr):
                                rec=rollout(w,r,arm,models[arm],b,c); results.append(rec)
                                f.write(json.dumps(dict(seed=seed,size=n,root=idx,arm=arm,budget=b,corruption=c,**rec))+'\n')
                            row=dict(seed=seed,size=n,arm=arm,budget=b,corruption=c,n=roots,solved=sum(r['success'] for r in results),mean_probes=float(np.mean([r['probes'] for r in results])),mean_records=float(np.mean([r['records_read'] for r in results])),retrieval_precision=sum(r['matched'] for r in results)/sum(r['records_read'] for r in results),mean_center_distance=float(np.mean([r['mean_center_distance'] for r in results])),premature=sum(r['premature_stop'] for r in results),recovered=sum(r['recovered'] for r in results),wall_seconds=time.perf_counter()-cell_start); summary.append(row)
                            if n==32768 and b==16 and c==0 and arm in ('hilbert','random'): paired[seed,arm]=[int(r['success']) for r in results]
                if args.native_world and Path(args.native_world).is_file():
                    w=World(32768,namespace+800000+seed,layout)
                    native.append(dict(seed=seed,arm=arm,results=native_replay(args.native_world,w,arm,models[arm],16,0.)))
    delta={str(s):float(np.mean(np.array(paired[s,'hilbert'])-np.array(paired[s,'random']))) for s in seeds}
    try:
        import resource,sys
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:
        try:
            import psutil
            peak_rss_bytes=psutil.Process().memory_info().peak_wset
        except (ImportError,AttributeError): peak_rss_bytes=None
    result=dict(mode='smoke' if args.smoke else 'registered_proxy',parameters=sum(p.numel() for p in Controller().parameters()),peak_process_rss_bytes=peak_rss_bytes,native_world_sha256=hashlib.sha256(Path(args.native_world).read_bytes()).hexdigest() if native else None,seed_deltas=delta,primary_delta=float(np.mean(list(delta.values()))),gate='NOT_EVALUATED_SMOKE' if args.smoke else ('PASS_POINT_ONLY' if np.mean(list(delta.values()))>=.05 else 'FAIL'),native_status='witness_replay_only' if native else 'not_run',native=native,checkpoints=checkpoints,rows=summary,wall_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k not in ('rows','native','checkpoints')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--out',required=True); p.add_argument('--threads',type=int,default=2); p.add_argument('--native-world',default=str(Path(__file__).resolve().parents[1]/'results/mcp05/inky-hilbert-16k/storyworld.hilbert.json')); run(p.parse_args())
