"""Frozen state and tied-matrix causal decomposition pilot.

Fit a behavioral orthogonal basis from independent trajectories, select interventions
on calibration, evaluate once on disjoint roots. The weight control uses exact SVD
of one physically shared linear matrix, not a trained full BPD factorization.
"""
import contextlib
import numpy as np
import torch

from research_gym.trm_skill_gym import modules as M, tasks as T
from .runner import pools, evaluate

FIT, SELECT, EVAL = 1_900_000, 2_000_000, 2_100_000


def orthogonal_random(h, rank, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((h, rank)))
    return torch.tensor(q, dtype=torch.float32)


def remove_subspace(z, center, q):
    return z - ((z-center) @ q) @ q.T


class Probe(M.SkillModule):
    basis = None
    record = False
    energy = None
    states = None

    def visit(self, cand, batch, st, planned_visits):
        if st["u"] > 0:
            if self.record:
                self.states.append(st["z"].detach().clone())
            if self.basis is not None:
                z = st["z"]
                z2 = remove_subspace(z, self.z0, self.basis)
                m = batch["mask"][..., None]
                self.energy.append(float(((z-z2).square()*m).sum()))
                st = dict(st, z=z2)
        return super().visit(cand, batch, st, planned_visits)


def load_probe(source, skill, seed):
    ck = torch.load(source/"checkpoints"/f"{skill}__F_trm_rlt__s{seed}.pt",
                    map_location="cpu", weights_only=True)
    m = Probe(ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["config"]


@torch.no_grad()
def fit_basis(m, skill, cfg, fit):
    """Difference of trace-normalized success/failure second moments of z-z0.

    Same basis is used at each recurrence boundary. Labels enter only through
    host grading on fit roots, never through a model observation.
    """
    h = m.H
    moments = [torch.zeros(h,h),torch.zeros(h,h)]
    counts = [0,0]
    pooled = torch.zeros(h,h)
    for b in fit.values():
        m.record=True
        m.states=[]
        out=M.episode(m,skill,b,cfg)
        final=out["exact"][-1].bool()
        for z in m.states:
            dz=z-m.z0
            v=dz[batch_mask(b)]
            pooled+=v.T@v
            for ok in (0,1):
                sel=(final==bool(ok))[:,None] & batch_mask(b)
                x=dz[sel]
                if x.numel():
                    x=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                    moments[ok]+=x.T@x
                    counts[ok]+=len(x)
        m.record=False
        m.states=[]
    contrast=moments[0]/max(1,counts[0])-moments[1]/max(1,counts[1])
    ev,q=torch.linalg.eigh(contrast)
    _,pca=torch.linalg.eigh(pooled)
    return q.flip(1),pca.flip(1),{"failure_slots":counts[0],"success_slots":counts[1],
                                      "contrast_eigenvalues":ev.flip(0).tolist()}


def batch_mask(b):
    return b["mask"]>0


def paired_delta(base, treatment):
    d=np.asarray(treatment)-np.asarray(base)
    rng=np.random.default_rng(77)
    boot=d[rng.integers(0,len(d),(1000,len(d)))].mean(1)
    return {"delta":float(d.mean()),"root_bootstrap_95":np.quantile(boot,[.025,.975]).tolist()}


def macro(rows):
    return float(np.mean([v["exact"] for v in rows.values()]))


def state_eval(m,skill,cfg,p, q):
    m.basis=q
    m.energy=[]
    stats={}
    predictions={}
    for k,b in p.items():
        before=len(m.energy)
        a,r=evaluate(skill,m,cfg,b)
        r["removed_squared_norm_per_instance"]=sum(m.energy[before:])/len(a)
        stats[k]=r
        predictions[k]=a
    m.basis=None
    return stats,predictions


def select_directions(m,skill,cfg,cal,basis,rank):
    base=macro(state_eval(m,skill,cfg,cal,None)[0])
    scores=[]
    for i in range(basis.shape[1]):
        score=macro(state_eval(m,skill,cfg,cal,basis[:,i:i+1])[0])-base
        scores.append(score)
    ids=sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:rank]
    return ids,scores


def latent_study(m,skill,cfg,fit,cal,test,seed,n_null):
    basis,pca,fit_info=fit_basis(m,skill,cfg,fit)
    rank=max(1,m.H//4)
    ids,scores=select_directions(m,skill,cfg,cal,basis,rank)
    coord,coord_scores=select_directions(m,skill,cfg,cal,torch.eye(m.H),rank)
    treatments={"keep":None,"full":torch.eye(m.H),"behavior":basis[:,ids],
                "pca":pca[:,:rank],"coordinate":torch.eye(m.H)[:,coord]}
    treatments.update({f"haar_{i}":orthogonal_random(m.H,rank,seed*1000+i+100)
                       for i in range(n_null)})
    result={}
    base=None
    for k,q in treatments.items():
        stats,pred=state_eval(m,skill,cfg,test,q)
        if k=="keep":
            base=pred
        result[k]={"strata":stats,"macro":macro(stats),
                   "paired":{f:paired_delta(base[f],pred[f]) for f in pred}}
    result["haar_summary"]={"mean":float(np.mean([result[f"haar_{i}"]["macro"] for i in range(n_null)])),
                            "max":max(result[f"haar_{i}"]["macro"] for i in range(n_null))}
    return {"rank":rank,"fit":fit_info,"selected":ids,"selection_effects":scores,
            "coordinate_ids":coord,"coordinate_selection_effects":coord_scores,"test":result}


class TiedWriteProbe:
    """One physical weight, shared SVD component directions, call-indexed means.

    With no selected components the original output is returned unchanged.
    A selected component's centered write is subtracted; exact residual and
    bias remain. No weight mutation, re-fit or norm rescaling.
    """
    def __init__(self,linear):
        self.linear=linear
        u,s,vh=torch.linalg.svd(linear.weight.detach(),full_matrices=False)
        self.u,self.s,self.v=u,s,vh.T
        self.means={}
        self.n={}
        self.mask=None
        self.call=0
        self.selected=()
        self.only=None
        self.fitting=False
        self.energy=0.
    def begin(self,mask):
        self.mask=mask>0
        self.call=0
        self.energy=0.
    def hook(self,mod,inputs,out):
        x=inputs[0]
        j=self.call
        self.call+=1
        if self.fitting:
            v=x[self.mask]
            self.means[j]=self.means.get(j,torch.zeros_like(x[0,0]))+v.sum(0)
            self.n[j]=self.n.get(j,0)+len(v)
        if not self.selected or (self.only is not None and j!=self.only):
            return out
        k=list(self.selected)
        mu=self.means[j]/max(1,self.n[j])
        delta=(((x-mu)@self.v[:,k])*self.s[k])@self.u[:,k].T
        delta=delta*self.mask[...,None]
        self.energy+=float(delta.square().sum())
        return out-delta
    @contextlib.contextmanager
    def installed(self):
        h=self.linear.register_forward_hook(self.hook)
        try:
            yield
        finally:
            h.remove()


@torch.no_grad()
def write_study(m,skill,cfg,fit,cal,test,seed,n_null):
    linear=m.net_z.layers[0]["b"]
    probe=TiedWriteProbe(linear)
    recon=(probe.u*probe.s)@probe.v.T
    residual=float((recon-linear.weight).abs().max())
    def run(p,ids=(),only=None,fit_means=False):
        probe.selected=tuple(ids)
        probe.only=only
        probe.fitting=fit_means
        stats={}
        preds={}
        with probe.installed():
            for k,b in p.items():
                probe.begin(b["mask"])
                a,r=evaluate(skill,m,cfg,b)
                r.update(removed_write_squared_norm_per_instance=probe.energy/len(a),
                         physical_matrix_calls=probe.call)
                stats[k]=r
                preds[k]=a
        probe.fitting=False
        return stats,preds
    fit_stats,_=run(fit,fit_means=True)
    base_cal=macro(run(cal)[0])
    effects=[macro(run(cal,[i])[0])-base_cal for i in range(len(probe.s))]
    rank=max(1,len(effects)//4)
    selected=sorted(range(len(effects)),key=lambda i:(-effects[i],i))[:rank]
    controls={"keep":(),"selected":selected,"top_singular":list(range(rank))}
    rng=np.random.default_rng(seed+19)
    for i in range(n_null):
        controls[f"random_{i}"]=rng.choice(len(effects),rank,replace=False).tolist()
    results={}
    base=None
    for k,ids in controls.items():
        s,p=run(test,ids)
        if k=="keep":
            base=p
        results[k]={"macro":macro(s),"strata":s,"paired":{f:paired_delta(base[f],p[f]) for f in p}}
    # Single-invocation and all-pass edits share exactly the same factors.
    results["per_call"]={str(j):macro(run(test,selected,j)[0]) for j in sorted(probe.means)}
    return {"matrix":"net_z.layers.0.b","shared_with_answer":m.net_y is m.net_z,
            "svd_max_error":residual,"rank":rank,"selected":selected,
            "selection_effects":effects,"fit_calls":len(probe.means),"test":results,
            "null_kind":"rank-matched singular components; removed energy reported, not equalized"}


def run(source,skills,seeds,count,n_null):
    out={}
    for skill in skills:
        fit=pools(skill,FIT,count)
        cal=pools(skill,SELECT,count)
        test=pools(skill,EVAL,count,True)
        for seed in seeds:
            m,cfg=load_probe(source,skill,seed)
            latent=latent_study(m,skill,cfg,fit,cal,test,seed,n_null)
            weights=write_study(m,skill,cfg,fit,cal,test,seed,n_null)
            out[f"{skill}/s{seed}"]={"latent":latent,"tied_weight":weights}
    return out
