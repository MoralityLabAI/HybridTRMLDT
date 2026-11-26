"""COG-13J native paired proof routing; fixture uses no Lean."""
import argparse
import json
import os
import re
import resource
import signal
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

from transport import ARMS, MENU, load, need, route, sha

H_SHA='f20b8ce3b6f1fe6d8e12816899024027c921bcb2065ddefd6c578c7d4ec80f8d'
AXIOMS={'propext','Classical.choice','Quot.sound'}
LIMIT=60_000_000_000
SETUP={'frequency':128,'local_dense':4424,'shared_dense':4424,'half_exact':5960,'progressive':5960}


def dump(path,x):
    with path.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')


def cases():
    out=[]
    for s in range(2):
        for i in range(8):
            n=2101+311*s+31*i;d=(6+i%3) if s else (2+i%2);f=i%4;gs=[]
            if f==0:
                b='(a b c : ℕ) (h : a ≤ b) (hc : b ≤ c)'
                for j in range(d):gs.append(f'a + {n+j} ≤ c + {n+j+2}')
            elif f==1:
                b='(x y : ℚ)'
                for j in range(d):
                    k=n+j;gs.append(f'(x + y - {k})^2 = x*x + 2*x*y + y*y - {2*k}*x - {2*k}*y + {k*k}')
            elif f==2:
                b='(x : ℕ) (s t u : Finset ℕ)'
                for j in range(d):
                    k=n+j;gs.append(f'((x ∈ s ∩ (t ∪ u) ↔ x ∈ (s ∩ t) ∪ (s ∩ u)) ∧ ({k} : ℕ) ≤ {k+2})')
            else:
                b='(a b : ℕ) (h : a ≤ b) (P Q : Prop) (hp : P) (hq : Q)'
                for j in range(d):gs.append(f'P → (Q ∧ a + {n+j} ≤ b + {n+j+3})' if j%2 else f'P ∧ (Q → a + {n+j} ≤ b + {n+j+1})')
            out.append({'id':f'cog13j_{s}_{i}','split':'deeper' if s else 'ordinary','family':('nat','ring','finset','mixed')[f],'depth':d,'statement':b+' : '+' ∧ '.join('('+g+')' for g in gs)})
    need(len({c['statement'] for c in out})==16,'registry duplicate');return out


def source(c,actions,probe=False):
    need(re.fullmatch(r'cog13j_[01]_[0-7]',c['id']) and all(a in MENU for a in actions),'source grammar')
    return 'import Mathlib\n'+('import StateCacheJ\n' if probe else '')+'set_option maxHeartbeats 100000000\n'+f"theorem {c['id']} {c['statement']} := by\n"+('  cog13j_probe\n' if probe else ''.join('  '+a+'\n' for a in actions))+f"#print axioms {c['id']}\n"


def axioms(log,name):
    if f"'{name}' does not depend on any axioms" in log:return []
    m=re.search(re.escape(f"'{name}' depends on axioms: ")+r'\[([^\]]*)\]',log)
    return [x.strip() for x in m.group(1).split(',') if x.strip()] if m else None


def accepted(code,log,name):
    a=axioms(log,name)
    return code==0 and a is not None and set(a)<=AXIOMS and not re.search(r"sorryAx|error(?:\([^\n]*\))?:|declaration uses 'sorry'",log)


def cache_step(cache,actions,number,r):
    hit=max((e for e in cache if actions[:len(e['actions'])]==e['actions']),key=lambda x:len(x['actions']),default=None)
    restored=len(hit['actions']) if hit else 0
    out=[dict(e) for e in cache if e!=hit]
    if hit:out.append({**hit,'touched':number,'hits':hit['hits']+1})
    inserted=False;evicted=None
    if r['accepted'] and r['goals'] and not any(e['actions']==actions for e in out):
        out.append({'actions':list(actions),'touched':number,'hits':0});inserted=True
        if len(out)>16:
            v=min(out,key=lambda e:e['touched']);evicted=v['actions'];out.remove(v)
    return {'restored':restored,'executed':len(actions)-restored,'inserted':inserted,'evicted':evicted,'cache':out}


def validate(cache,actions,number,reply):
    need(reply['id']==number and type(reply['result']['accepted']) is bool,'native reply identity')
    r=reply['result'];need(not r['accepted'] or isinstance(r['goals'],list) and all(isinstance(g,str) for g in r['goals']),'native goals')
    t=cache_step(cache,actions,number,r)
    need(all(reply[k]==v for k,v in t.items()),'native cache transition')
    return t['cache']


def node(goals,prefix=()):return {'goals':goals,'prefix':list(prefix),'tried':[]}
def next_node(stack):
    while stack and len(stack[-1]['tried'])==len(MENU):stack.pop()
    return stack[-1] if stack else None


def advance(stack,n,actions,r):
    if not r['accepted'] or r['goals']==n['goals']:return False
    if not r['goals']:return True
    if len(actions)<12 and len(r['goals'])<=8 and not any(x['goals']==r['goals'] for x in stack):stack.append(node(r['goals'],actions))
    return False


class Native:
    def __init__(self,lean,cwd,folder):self.lean=lean;self.cwd=cwd;self.folder=folder;self.processes=[]
    def launch(self,sub,src,env):
        folder=self.folder/sub;folder.mkdir();file=folder/'source.lean';file.write_text(src,encoding='utf-8');log=folder/'stdout.log'
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic_ns()
        with log.open('xb') as f:p=subprocess.Popen([self.lean,str(file)],cwd=self.cwd,env={**os.environ,'LEAN_NUM_THREADS':'1',**env},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        return {'proc':p,'folder':folder,'log':log,'file':file,'before':before,'start':start}
    def stop(self,s,end):
        p=s['proc']
        try:p.wait(timeout=max(0,min(.3,(end-time.monotonic_ns())/1e9)))
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=2)
        after=resource.getrusage(resource.RUSAGE_CHILDREN);before=s['before']
        r={'folder':s['folder'].name,'returncode':p.returncode,'cpuNs':round(1e9*((after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime))),'wallNs':time.monotonic_ns()-s['start'],'sourceHash':sha(s['file'].read_bytes()),'logHash':sha(s['log'].read_bytes())}
        self.processes.append(r);dump(s['folder']/'process.json',r);return r
    def open(self,c,end):
        folder=self.folder/'session';s=self.launch('session',source(c,[],True),{'COG13J_MAILBOX':str(folder),'COG13J_CAP':'16','COG13J_POLICY':'lru'})
        def reply(n,deadline):
            path=folder/f'response-{n}.json'
            while time.monotonic_ns()<deadline:
                if path.exists():return json.loads(path.read_text())
                if s['proc'].poll() is not None:raise RuntimeError('Lean exited before reply '+str(n))
                time.sleep(.005)
            raise TimeoutError('native reply budget')
        def request(n,actions,stop=False):
            p=folder/f'request-{n}.json';tmp=p.with_suffix('.tmp');dump(tmp,{'id':n,'actions':actions,'stop':stop});tmp.replace(p)
        try:
            init=reply(0,min(end,time.monotonic_ns()+20_000_000_000));need(init['ready'] and init['goals'],'native startup')
        except Exception:self.stop(s,end);raise
        class Session:
            initial=init['goals'];number=0
            def query(self,actions,n,deadline):
                need(n==self.number+1,'request sequence');self.number=n;request(n,actions)
                return reply(n,min(deadline,time.monotonic_ns()+10_000_000_000))
            def close(self,deadline):
                if s['proc'].poll() is None:request(self.number+1,[],True)
                return outer.stop(s,deadline)
        outer=self;return Session()
    def certify(self,c,actions,rep,end):
        s=self.launch(f'certificate_{rep}',source(c,actions),{});p=s['proc']
        try:p.wait(timeout=max(0,min(20,(end-time.monotonic_ns())/1e9)))
        except subprocess.TimeoutExpired:pass
        process=self.stop(s,end);log=s['log'].read_text(errors='replace')
        return {**process,'accepted':accepted(process['returncode'],log,c['id']) and time.monotonic_ns()<=end,'axioms':axioms(log,c['id'])}


class Fixture:
    def __init__(self):self.processes=[]
    def open(self,c,end):
        class Session:
            initial=['P : Prop\nhp : P\n⊢ P ∧ P'];cache=[]
            def query(self,actions,n,deadline):
                r=({'accepted':True,'goals':['P : Prop\nhp : P\n⊢ P']} if actions==['constructor'] else {'accepted':True,'goals':[]} if actions==['constructor','assumption'] else {'accepted':False,'error':'fixture rejection'})
                t=cache_step(self.cache,actions,n,r);self.cache=t['cache'];return {'id':n,'result':r,**t}
            def close(self,deadline):return None
        return Session()
    def certify(self,c,actions,rep,end):return {'accepted':True,'sourceHash':sha(json.dumps(actions).encode()),'fixture':True,'folder':f'certificate_{rep}'}


def search(c,arm,m,freq,backend):
    start=time.monotonic_ns();end=start+LIMIT;cpu=time.process_time_ns();session=None;initial=None;events=[];certs=[];cache=[];closed=None;reason='probe_budget';solved=False
    try:
        session=backend.open(c,end);initial=session.initial;stack=[node(initial)]
        while len(events)<48 and time.monotonic_ns()<end:
            n=next_node(stack)
            if n is None:reason='search_exhausted';break
            opts=[a for a in MENU if a not in n['tried']];r=route(n['goals'],opts,arm,m,freq);action=r['action'];actions=n['prefix']+[action]
            need(r['denseAgreement'] is not False,'dense routing disagreement');n['tried'].append(action)
            event={'id':len(events)+1,'before':n['goals'],'options':opts,'actions':actions,'route':r,'reply':None,'error':None};events.append(event)
            if time.monotonic_ns()>=end:event['error']='routing_budget';reason='deadline';break
            try:event['reply']=session.query(actions,event['id'],end)
            except Exception as e:event['error']=str(e);reason='query_unknown';break
            cache=validate(cache,actions,event['id'],event['reply']);event['elapsedNs']=time.monotonic_ns()-start
            if event['elapsedNs']>=LIMIT:reason='deadline';break
            if advance(stack,n,actions,event['reply']['result']):
                closed=actions;session.close(end);session=None;reason='certificate_rejected_or_budget'
                for rep in range(2):
                    if time.monotonic_ns()>=end:break
                    cert=backend.certify(c,actions,rep,end);cert['elapsedNs']=time.monotonic_ns()-start;certs.append(cert)
                    if not cert['accepted']:break
                solved=len(certs)==2 and all(x['accepted'] and x['elapsedNs']<=LIMIT for x in certs) and certs[0]['sourceHash']==certs[1]['sourceHash'] and time.monotonic_ns()<=end
                if solved:reason='certified'
                break
        if reason=='probe_budget' and time.monotonic_ns()>=end:reason='deadline'
    except Exception as e:reason='startup_unknown' if initial is None else 'evidence_error';events.append({'failure':str(e)})
    finally:
        if session is not None:session.close(end)
    elapsed=time.monotonic_ns()-start
    if solved and elapsed>LIMIT:solved=False;reason='deadline_cleanup'
    return {'case':c,'arm':arm,'initial':initial,'events':events,'certificates':certs,'closed':closed,'solved':solved,'reason':reason,'elapsedNs':elapsed,'hostCpuNs':time.process_time_ns()-cpu,'processes':backend.processes}


def audit(r,m,freq,folder=None):
    c=r['case'];arm=r['arm'];stack=[node(r['initial'])] if r['initial'] else [];cache=[];closed=None;counts=Counter();last=0
    for e in r['events']:
        if 'failure' in e:need(not r['solved'],'failed episode credit');continue
        n=next_node(stack);need(n and closed is None and e['id']==counts['decisions']+1,'search tape')
        opts=[a for a in MENU if a not in n['tried']];rr=route(n['goals'],opts,arm,m,freq)
        need(e['before']==n['goals'] and e['options']==opts and e['actions']==n['prefix']+[rr['action']],'search policy')
        need(all(e['route'][k]==rr[k] for k in ('action','bytes','stages','denseAgreement')) and rr['denseAgreement'] is not False,'router replay')
        n['tried'].append(rr['action']);counts['decisions']+=1
        for k in ('bytes','senderNs','receiverNs','auditNs'):counts[k]+=e['route'][k]
        counts['refinements']+=sum(s.get('reply')==255 for s in rr['stages'])
        counts['fallbacks']+=sum(s.get('mode')==0 for s in rr['stages'] if arm in ('half_exact','progressive'))
        if e['reply'] is None:need(e['error'] and not r['solved'],'unknown credit');break
        cache=validate(cache,e['actions'],e['id'],e['reply']);counts['executed']+=e['reply']['executed'];counts['restored']+=e['reply']['restored'];counts['replied']+=1
        need(e['elapsedNs']>=last,'clock');last=e['elapsedNs']
        if folder:
            req=json.loads((folder/'session'/f"request-{e['id']}.json").read_text());rep=json.loads((folder/'session'/f"response-{e['id']}.json").read_text())
            need(req=={'id':e['id'],'actions':e['actions'],'stop':False} and rep==e['reply'],'raw mailbox')
        if last<LIMIT and advance(stack,n,e['actions'],e['reply']['result']):closed=e['actions']
    need(counts['decisions']<=48 and closed==r['closed'],'closed proof/attempt budget')
    certs=r['certificates'];need(len(certs)<=2 and len({x['folder'] for x in certs})==len(certs),'distinct certs')
    if r['solved']:need(closed and len(certs)==2 and all(x['accepted'] and x['elapsedNs']<=LIMIT for x in certs) and certs[0]['sourceHash']==certs[1]['sourceHash'] and r['elapsedNs']<=LIMIT,'uncertified credit')
    if folder:
        if r['initial']:need(json.loads((folder/'session'/'response-0.json').read_text())['goals']==r['initial'],'initial goals')
        for p in r['processes']:
            d=folder/p['folder'];need(sha((d/'source.lean').read_bytes())==p['sourceHash'] and sha((d/'stdout.log').read_bytes())==p['logHash'],'process hashes')
        for cert in certs:
            d=folder/cert['folder'];src=(d/'source.lean').read_bytes();log=(d/'stdout.log').read_bytes()
            need(src.decode()==source(c,closed) and sha(src)==cert['sourceHash'] and sha(log)==cert['logHash'],'certificate bytes')
            if cert['accepted']:need(accepted(cert['returncode'],log.decode(errors='replace'),c['id']),'certificate log')
    counts['nativeChildCpuNs']=sum(p['cpuNs'] for p in r['processes']);counts['hostCpuNs']=r['hostCpuNs'];counts['wallNs']=r['elapsedNs'];return dict(counts)


def matrix(archive,old_h,lean,cwd,out):
    old,m,f=load(archive);need(sha(old_h.read_bytes())==H_SHA,'COG-13H archive pin')
    with zipfile.ZipFile(old_h) as z:h=json.loads(z.read('registry.json'))
    registry=cases();need(len(h)==16 and not {c['statement'] for c in registry}&{c['statement'] for c in old['cases']+h},'prior statement collision')
    out.mkdir();dump(out/'registry.json',registry);rows=[];errors=[];common={};inits={}
    for i,c in enumerate(registry):
        for arm in ARMS[i%5:]+ARMS[:i%5]:
            folder=out/(c['id']+'_'+arm);folder.mkdir();r=search(c,arm,m,f,Native(lean,cwd,folder));dump(folder/'episode.json',r)
            metrics={}
            try:
                need(r['reason']!='evidence_error','execution evidence failure');metrics=audit(r,m,f,folder)
                if r['initial']:need(inits.setdefault(c['id'],r['initial'])==r['initial'],'initial divergence')
                for e in r['events']:
                    if not e.get('reply'):continue
                    v=e['reply']['result'];v={'accepted':v['accepted'],'goals':v.get('goals') if v['accepted'] else None};k=(c['id'],tuple(e['actions']));need(common.setdefault(k,v)==v,'common native verdict mismatch')
            except Exception as e:errors.append({'root':c['id'],'arm':arm,'error':str(e)})
            rows.append({'root':c['id'],'split':c['split'],'arm':arm,'solved':r['solved'],'reason':r['reason'],'metrics':metrics})
    totals={a:dict(Counter()) for a in ARMS}
    for a in ARMS:
        selected=[r for r in rows if r['arm']==a];t=Counter()
        for r in selected:t.update(r['metrics'])
        totals[a]={**dict(t),'roots':len(selected),'solved':sum(r['solved'] for r in selected),'setupBytes':SETUP[a],'solvedRoots':[r['root'] for r in selected if r['solved']]}
    unknown=sum(r['reason'] in ('query_unknown','startup_unknown','evidence_error') for r in rows);complete=len(rows)==80 and len({(r['root'],r['arm']) for r in rows})==80 and not errors
    b=totals['shared_dense'];p=totals['progressive'];clean=complete and unknown==0
    gate={'clean':clean,'sameCertifiedRoots':clean and all(set(totals[a]['solvedRoots'])==set(b['solvedRoots']) for a in ARMS[1:]),'halfBytes':clean and b.get('bytes',0)>0 and 2*p.get('bytes',0)<=b['bytes']}
    result={'schema':'cog13j.native.v1','sourceArchiveSha':sha(archive.read_bytes()),'oldHArchiveSha':H_SHA,'complete':complete,'unknown':unknown,'errors':errors,'gates':gate,'totals':totals,'rows':rows};dump(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return complete


def fixture(archive):
    _,m,f=load(archive);out=[]
    for arm in ARMS:
        r=search(cases()[0],arm,m,f,Fixture());v=audit(r,m,f);need(r['solved'],'fixture completion');out.append((r,v))
    good=out[-1][0];bad=json.loads(json.dumps(good));bad['events'][0]['reply']['executed']+=1
    def reject(x):
        try:audit(x,m,f)
        except ValueError:return True
        return False
    need(reject(bad),'forged cache accepted');bad=json.loads(json.dumps(good));bad['certificates']=[];need(reject(bad),'forged cert accepted')
    bad=json.loads(json.dumps(good));bad['events'][0]['route']['bytes']+=1;need(reject(bad),'forged bytes accepted')
    return {'fixture':True,'arms':len(out),'probes':sum(v['decisions'] for _,v in out),'cacheForgeryRejected':True,'certificateForgeryRejected':True,'byteForgeryRejected':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--fixture',action='store_true');p.add_argument('--old-h',type=Path);p.add_argument('--lean');p.add_argument('--cwd',type=Path);p.add_argument('--out',type=Path);args=p.parse_args()
    if args.fixture:print(json.dumps(fixture(args.archive)))
    else:
        need(args.old_h and args.lean and args.cwd and args.out,'native arguments');raise SystemExit(0 if matrix(args.archive,args.old_h,args.lean,args.cwd,args.out.resolve()) else 1)
