"""Native router codec: exact common-goal baseline and progressive intervals."""
import decimal as dec
import functools
import hashlib
import json
import math
import struct
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

SHA = 'd99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea'
MENU = ['assumption','rfl','constructor','intro hnew','simp','simp_all','omega','linarith','ring','norm_num','positivity','aesop','exact h','all_goals assumption','all_goals omega','all_goals ring']
ARMS = ('frequency','local_dense','shared_dense','half_exact','progressive')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    need(sha(Path(path).read_bytes()) == SHA, 'archive hash')
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('export_manifest.json'))
        for name, digest in manifest['hashes'].items():
            need(sha(z.read(name)) == digest, 'member hash')
        r = json.loads(z.read('result.json'))
    train = r['teacher']['rows']
    need(len(train) == 224 and len({x['root'] for x in train}) == 8 and all(x['split'] == 'train' for x in train), 'train custody')
    f = {a: (1+sum(x['target'] for x in train if x['action']==a))/(2+sum(x['action']==a for x in train)) for a in MENU}
    m = r['fit']['model']
    need(len(m['W'])==len(m['R'])==len(m['b'])==len(m['v'])==12 and all(len(w)==32 for w in m['W']) and all(len(w)==12 for w in m['R']), 'model shape')
    need(all(math.isfinite(v) for v in [*m['b'],*m['v'],m['o'],*(v for w in m['W']+m['R'] for v in w)]), 'nonfinite weights')
    return r,m,f


def f32(x):
    return struct.unpack('<f', struct.pack('<f',x))[0]


def dot(a,b):
    v=0.0
    for x,y in zip(a,b,strict=True):
        v+=x*y
    return v


def goal_vector(goals):
    ts=[];word=''
    for c in ' '.join(goals):
        if c=='_' or unicodedata.category(c)[0] in ('L','N'):
            word+=c
        else:
            if word:ts.append(word);word=''
            if not c.isspace():ts.append(c)
    if word:ts.append(word)
    x=[0.0]*13
    for t in ts:
        h=2166136261;b=t.encode('utf-16-le')
        for i in range(0,len(b),2):h=((h^int.from_bytes(b[i:i+2],'little'))*16777619)&0xffffffff
        x[h%13]+=1 if h&256 else -1
    return [f32(v/math.sqrt(max(1,len(ts)))) for v in x]+[f32(min(1,len(goals)/8)),f32(min(1,len(ts)/128)),1.0]


def forward(m,p):
    h=[0.0]*12
    for _ in range(3):h=[math.tanh(p[i]+dot(row,h)+m['b'][i]) for i,row in enumerate(m['R'])]
    return dot(m['v'],h)+m['o']


def dense(m,g,ids):
    return [forward(m,[dot(w,[float(j==i) for j in range(16)]+list(g)) for w in m['W']]) for i in ids]


def winner(scores):
    need(scores and all(math.isfinite(v) for v in scores),'nonfinite scores')
    return max(range(len(scores)),key=lambda i:scores[i])


def down(v):return math.nextafter(v,-math.inf)
def up(v):return math.nextafter(v,math.inf)
def add(x,y):return down(x[0]+y[0]),up(x[1]+y[1])
def mul(k,x):
    l,h=x if k>=0 else x[::-1]
    return down(k*l),up(k*h)
def isum(b,w,xs):
    v=(b,b)
    for k,x in zip(w,xs,strict=True):v=add(v,mul(k,x))
    return v


@functools.lru_cache(maxsize=200000)
def tanh_bound(x):
    if x<0:
        l,h=tanh_bound(-x);return -h,-l
    with dec.localcontext() as c:
        c.prec=48;c.rounding=dec.ROUND_HALF_EVEN
        e=dec.Decimal.from_float(2*x).exp();el,eh=e.next_minus(),e.next_plus()
        one,two=dec.Decimal(1),dec.Decimal(2)
        c.rounding=dec.ROUND_FLOOR;dl=one+el
        c.rounding=dec.ROUND_CEILING;rh=two/dl
        c.rounding=dec.ROUND_FLOOR;lo=one-rh
        c.rounding=dec.ROUND_CEILING;dh=one+eh
        c.rounding=dec.ROUND_FLOOR;rl=two/dh
        c.rounding=dec.ROUND_CEILING;hi=one-rl
        return max(-1.0,down(float(lo))),min(1.0,up(float(hi)))


def envelope(c,r):return [(down(x-y),up(x+y)) for x,y in zip(c,r,strict=True)]


def fp32_up(x):
    v=f32(x)
    if v<x:v=struct.unpack('<f',struct.pack('<I',struct.unpack('<I',struct.pack('<f',v))[0]+1))[0]
    return v


def decode(packet):
    need(len(packet)>=4,'short header')
    ver,mode,mask=struct.unpack('<BBH',packet[:4]);need(ver==1 and mask and mode in (0,6,7),'header')
    ids=[i for i in range(16) if mask&(1<<i)]
    if mode==0:
        need(len(packet)==68,'exact length');return ids,struct.unpack('<16f',packet[4:]),None
    if mode==6:
        need(len(packet)==20,'int8 length');scale,*q=struct.unpack('<f12b',packet[4:]);need(scale>0 and -128 not in q,'scale/code')
        c=[scale*x for x in q];r=[up(scale/2+1e-12)]*12
    else:
        need(len(packet)==28,'half length');c=list(struct.unpack('<12e',packet[4:]));r=[up(abs(x)*2**-10+2**-24+1e-12) for x in c]
    need(all(math.isfinite(x) for x in c+r),'nonfinite decoded')
    return ids,c,r


def encode(g,ids,m,mode):
    header=struct.pack('<BBH',1,mode,sum(1<<i for i in ids))
    if mode==0:return header+struct.pack('<16f',*g)
    c=[dot(w[16:],g) for w in m['W']]
    if mode==6:
        scale=fp32_up(max(max(abs(x) for x in c)/127,2**-30));q=[max(-127,min(127,round(x/scale))) for x in c]
        p=header+struct.pack('<f12b',scale,*q)
    else:p=header+struct.pack('<12e',*c)
    decoded,v,r=decode(p);need(decoded==ids,'mask')
    exact=[isum(0.0,w[16:],[(x,x) for x in g]) for w in m['W']]
    need(all(l<=cl<=ch<=h for (l,h),(cl,ch) in zip(envelope(v,r),exact,strict=True)),'containment')
    return p


def receive(packet,m):
    ids,c,r=decode(packet)
    if r is None:
        return winner(dense(m,c,ids)),None
    scores=[];ranges=[]
    bounds=envelope(c,r)
    for a in ids:
        scores.append(forward(m,[w[a]+x for w,x in zip(m['W'],c,strict=True)]))
        pre=[add(x,(w[a],w[a])) for x,w in zip(bounds,m['W'],strict=True)]
        h=[(0.,0.)]*12
        for _ in range(3):
            z=[add(p,isum(b,row,h)) for p,b,row in zip(pre,m['b'],m['R'],strict=True)]
            h=[(tanh_bound(l)[0],tanh_bound(u)[1]) for l,u in z]
        ranges.append(isum(m['o'],m['v'],h))
    top=winner(scores)
    gap=min((down(ranges[top][0]-v[1]) for i,v in enumerate(ranges) if i!=top),default=math.inf)
    # Real-network certificate plus a declared finite replay guard. The
    # independent dense replay still checks the actual selected action.
    return (top if gap>1e-10 else 255),gap if math.isfinite(gap) else None


def route(goals,options,arm,m,freq):
    ids=[MENU.index(a) for a in options];need(ids==sorted(set(ids)) and ids,'canonical menu')
    sender=receiver=0;stages=[];start=time.process_time_ns()
    tanh_bound.cache_clear()
    if arm=='frequency':
        top=winner([freq[a] for a in options]);sender=time.process_time_ns()-start
        return {'action':options[top],'bytes':1,'senderNs':sender,'receiverNs':0,'auditNs':0,'stages':[{'mode':'local_frequency','reply':top}],'denseAgreement':None}
    g=goal_vector(goals);sender=time.process_time_ns()-start
    if arm=='local_dense':
        t=time.process_time_ns();top=winner(dense(m,g,ids));sender+=time.process_time_ns()-t;cost=1;stages=[{'mode':'local_dense','reply':top}]
    else:
        modes={'shared_dense':[0],'half_exact':[7,0],'progressive':[6,7,0]}[arm];cost=0;top=255
        for mode in modes:
            t=time.process_time_ns()
            try:p=encode(g,ids,m,mode)
            except (ValueError,OverflowError,struct.error) as e:
                sender+=time.process_time_ns()-t
                need(mode!=0,'exact packet failed');stages.append({'mode':mode,'abstainBeforeSend':str(e)});continue
            sender+=time.process_time_ns()-t;t=time.process_time_ns();reply,gap=receive(p,m);receiver+=time.process_time_ns()-t
            cost+=len(p)+1;stages.append({'mode':mode,'packet':p.hex(),'reply':reply,'gap':gap,'bytes':len(p)+1})
            if reply!=255:top=reply;break
        need(top!=255,'no exact fallback')
    t=time.process_time_ns();agree=top==winner(dense(m,g,ids));audit_ns=time.process_time_ns()-t
    return {'action':options[top],'bytes':cost,'senderNs':sender,'receiverNs':receiver,'auditNs':audit_ns,'stages':stages,'denseAgreement':agree}


def pilot(archive):
    r,m,f=load(archive);out={a:Counter() for a in ARMS};cases={c['id']:c for c in r['cases']}
    for run in r['runs']:
        for e in run['events']:
            if e['kind']!='probe' or e['transport'] is None:continue
            need(cases[e['root']]['split']!='train','evaluation split')
            for arm in ARMS:
                x=route(e['before'],e['options'],arm,m,f);v=out[arm]
                need(x['denseAgreement'] is not False,'archived dense action mismatch')
                v['decisions']+=1;v['bytes']+=x['bytes'];v['senderNs']+=x['senderNs'];v['receiverNs']+=x['receiverNs'];v['auditNs']+=x['auditNs'];v['stages']+=len(x['stages'])
                v['refinements']+=sum(s.get('reply')==255 for s in x['stages']);v['exactFallbacks']+=sum(s.get('mode')==0 for s in x['stages'] if arm in ('half_exact','progressive'))
                v['senderAbstentions']+=sum('abstainBeforeSend' in s for s in x['stages'])
    need(all(v['decisions']==449 for v in out.values()),'pilot denominator')
    return {'schema':'cog13j.archive_timing.v1','archiveSha256':SHA,'evidence':'archived candidate menus, not native proof search; one sequential CPU timing pilot, no cross-decision interval cache','arms':{k:dict(v) for k,v in out.items()}}


if __name__=='__main__':
    import sys
    result=pilot(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
