import json,time,itertools,statistics
import numpy as np
from dispatcher import compile_plan,execute,decide

def graph(n,family,rng):
    if family=='chain': return [(i,i+1) for i in range(n-1)]
    if family=='star': return [(0,i) for i in range(1,n)]
    if family=='cycle': return [(i,i+1) for i in range(n-1)]+[(0,n-1)]
    return [(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<min(.25,3/n)]
def brute(scopes,fs,n,q):
    x=(np.arange(1<<n)[:,None]>>np.arange(n))&1; w=np.ones(len(x))
    for s,f in zip(scopes,fs): w*=f[tuple(x[:,i] for i in s)]
    return float(w[x[:,q]==1].sum()/w.sum())
def main():
    rows=[]; checks=[]; dataset=[]
    for n,family,seed in itertools.product((8,16,32),('chain','star','cycle','random'),(1801,1807,1811)):
        rng=np.random.default_rng(seed+n); scopes=tuple((i,) for i in range(n))+tuple(graph(n,family,rng));q=n-1;plan=compile_plan(scopes,q); fslist=[[rng.uniform(.1,1,(2,)*len(s)) for s in scopes] for _ in range(24)]
        if plan.peak_scope>16: continue
        expected=[execute(plan,fs) for fs in fslist]
        if n==8:
            for fs,p in zip(fslist,expected):
                err=abs(brute(scopes,fs,n,q)-p);assert err<1e-12;checks.append(err)
        times={'cached':[],'replan':[]}
        for rep in range(5):
            for arm in (('cached','replan') if rep%2==0 else ('replan','cached')):
                start=time.perf_counter(); pp=[execute(plan if arm=='cached' else compile_plan(scopes,q),fs) for fs in fslist];times[arm].append(time.perf_counter()-start);assert pp==expected
        start=time.perf_counter();compile_plan(scopes,q);prep=time.perf_counter()-start
        rows.append(dict(n=n,family=family,seed=seed,peak_scope=plan.peak_scope,work=plan.work,prep_s=prep,times=times,cached_s=statistics.median(times['cached']),replan_s=statistics.median(times['replan'])))
        for i,fs in enumerate(fslist):
            dataset.append(dict(schema='dcf15-v1',split='test' if seed==1811 else 'development',group=f'{n}-{family}-{seed}',query=q,scopes=scopes,factors=[f.tolist() for f in fs],receipt=decide(plan,fs),reference_p=expected[i]))
    assert decide(compile_plan(((0,1),(1,2)),0),[np.ones((2,2))]*2,cap=1)['status']=='scope_abstain'
    open('receipt.json','w').write(json.dumps(dict(rows=rows,brute_checks=len(checks),max_error=max(checks)),indent=2));open('frames.jsonl','w').write(''.join(json.dumps(x)+'\n' for x in dataset));print(len(rows),len(dataset),len(checks),max(checks),statistics.median(r['replan_s']/r['cached_s'] for r in rows));print('mean ms',statistics.mean(r['cached_s'] for r in rows)*1000,statistics.mean(r['replan_s'] for r in rows)*1000)
if __name__=='__main__':main()
