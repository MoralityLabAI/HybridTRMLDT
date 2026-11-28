"""Small supervised architecture performance student; hold out entire skills.

Predict only observed metrics at their respective racing rung. All 108 proposals
remain in the receipt, including unavailable targets. No future results, diagnoses,
or promotion decision enter features. A ridge is a pilot, not LLM distillation.
"""
from itertools import combinations
import numpy as np

from research_gym.trm_skill_gym import dsl
from .core import distill_rows


def encode_config(cfg):
    v=[]
    for key,(kind,spec,default) in dsl.SCHEMA.items():
        x=(cfg or {}).get(key,default)
        if kind=="enum":
            v.extend(float(x==a) for a in spec)
        elif kind=="bool":
            v.append(float(bool(x)))
        else:
            v.append((float(x)-spec[0])/(spec[1]-spec[0]))
    return np.array(v)


def features(row):
    x=row["input"]
    inc=encode_config(x["incumbent"])
    prop=encode_config(x["config"] or x["incumbent"])
    prior=x["incumbent_dev_exact"]
    return np.r_[1,inc,prop-inc,float(prior or 0),float(prior is not None)]


def fit_predict(x,y,xt):
    # Fixed regularization, no held-out tuning. Scale using training rows only.
    mu=x.mean(0)
    scale=np.maximum(x.std(0),.05)
    x=(x-mu)/scale
    xt=(xt-mu)/scale
    b=np.linalg.solve(x.T@x+10*np.eye(x.shape[1]),x.T@(y-y.mean()))
    return np.clip(xt@b+y.mean(),0,1)


def observed(row,rung):
    m=row["target"].get(rung)
    v=(m or {}).get("dev_exact")
    return None if v is None else float(v)


def group(row):
    x=row["input"]
    return x["skill"],x["family"],x["arm"],x["round"]


def score(rows,y,pred):
    pairs=correct=0.
    for i,j in combinations(range(len(rows)),2):
        if group(rows[i])!=group(rows[j]) or y[i]==y[j]:
            continue
        pairs+=1
        sign=np.sign(pred[i]-pred[j])
        correct+=float(sign==np.sign(y[i]-y[j]))+.5*float(sign==0)
    return {"n":len(y),"mae":float(np.abs(y-pred).mean()),
            "bias":float((pred-y).mean()),"rmse":float(np.sqrt(np.mean((pred-y)**2))),
            "pair_count":int(pairs),"pair_accuracy":correct/pairs if pairs else None}


def run(archive):
    rows=list(distill_rows(archive))
    result={"n":len(rows),"splits":{s:sum(r["split"]==s for r in rows) for s in ("train","test")},"rungs":{}}
    for rung in ("rung0","rung1"):
        a=[r for r in rows if r["split"]=="train" and observed(r,rung) is not None]
        b=[r for r in rows if r["split"]=="test" and observed(r,rung) is not None]
        if not a or not b:
            result["rungs"][rung]={"status":"insufficient_labels","train":len(a),"test":len(b)}
            continue
        y=np.array([observed(r,rung) for r in a])
        yt=np.array([observed(r,rung) for r in b])
        pred=fit_predict(np.stack([features(r) for r in a]),y,np.stack([features(r) for r in b]))
        inc=np.array([r["input"]["incumbent_dev_exact"] if r["input"]["incumbent_dev_exact"] is not None else y.mean() for r in b])
        result["rungs"][rung]={"train":len(a),"test":len(b),
            "missing":len(rows)-len(a)-len(b),
            "ridge":score(b,yt,pred),"train_mean":score(b,yt,np.full(len(b),y.mean())),
            "incumbent":score(b,yt,inc),
            "predictions":[{"id":r["id"],"skill":r["input"]["skill"],"actual":float(t),
                            "predicted":float(p),"incumbent":float(i)} for r,t,p,i in zip(b,yt,pred,inc)]}
    return result
