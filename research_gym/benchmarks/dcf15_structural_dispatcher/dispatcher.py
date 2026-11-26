"""Typed reusable exact Boolean factor elimination plan, with scope-bound abstention."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Step:
    inputs: tuple
    axes: tuple
    union: tuple
    eliminate: int

@dataclass(frozen=True)
class Plan:
    scopes: tuple
    query: int
    steps: tuple
    remaining: tuple
    peak_scope: int
    work: int


def compile_plan(scopes, query):
    scopes=tuple(tuple(s) for s in scopes)
    if any(tuple(sorted(set(s)))!=s for s in scopes): raise ValueError('scopes must be sorted and unique')
    variables=set().union(*map(set,scopes))
    if query not in variables: raise ValueError('query missing')
    live=dict(enumerate(scopes)); steps=[]; peak=max(map(len,scopes)); work=0
    for _ in range(len(variables)-1):
        def candidate(v):
            ids=tuple(i for i,s in live.items() if v in s); u=tuple(sorted(set().union(*(set(live[i]) for i in ids))))
            return len(u),v,ids,u
        _,v,ids,u=min(candidate(v) for v in variables if v!=query)
        axes=tuple(tuple(2 if x in live[i] else 1 for x in u) for i in ids)
        steps.append(Step(ids,axes,u,u.index(v))); peak=max(peak,len(u)); work+=len(ids)*(1<<len(u))
        for i in ids: del live[i]
        live[len(scopes)+len(steps)-1]=tuple(x for x in u if x!=v); variables.remove(v)
    return Plan(scopes,query,tuple(steps),tuple(live),peak,work)


def execute(plan, factors, cap=16):
    if plan.peak_scope>cap: return None
    if len(factors)!=len(plan.scopes): raise ValueError('factor count')
    live={}
    for i,(s,f) in enumerate(zip(plan.scopes,factors)):
        f=np.asarray(f,dtype=float)
        if f.shape!=(2,)*len(s) or not np.isfinite(f).all() or np.any(f<0): raise ValueError('invalid factor')
        live[i]=f
    for t,step in enumerate(plan.steps):
        z=np.ones((2,)*len(step.union))
        for i,shape in zip(step.inputs,step.axes): z*=live.pop(i).reshape(shape)
        z=z.sum(axis=step.eliminate); scale=z.max()
        if scale<=0: raise ValueError('zero evidence mass')
        live[len(plan.scopes)+t]=z/scale
    z=np.ones(2)
    for i in plan.remaining: z*=live[i]
    if z.sum()<=0: raise ValueError('zero evidence mass')
    return float(z[1]/z.sum())


def decide(plan,factors,cap=16,threshold=.8):
    p=execute(plan,factors,cap)
    action=None if p is None else 1 if p>=threshold else 0 if p<=1-threshold else None
    return dict(action=action,posterior=p,status='scope_abstain' if p is None else 'exact_algorithm_fp64',peak_scope=plan.peak_scope,work_proxy=plan.work)
