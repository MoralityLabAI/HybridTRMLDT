"""COG-13I: source-pinned shared-goal packet replay (archived states only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

SHA = 'd99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea'
MENU = ['assumption','rfl','constructor','intro hnew','simp','simp_all','omega','linarith','ring','norm_num','positivity','aesop','exact h','all_goals assumption','all_goals omega','all_goals ring']


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def dot(a, b):
    check(len(a) == len(b), 'shape')
    result = 0.0
    for x, y in zip(a, b):
        result += x*y
    return result


def tokens(text):
    # Match JS /[\p{L}\p{N}_]+|[^\s]/gu on the archived goals.
    word = ''
    for c in text:
        if c == '_' or unicodedata.category(c)[0] in ('L', 'N'):
            word += c
        else:
            if word:
                yield word
                word = ''
            if not c.isspace():
                yield c
    if word:
        yield word


def goal_vector(goals):
    ts = list(tokens(' '.join(goals)))
    x = [0.0]*13
    for token in ts:
        h = 2166136261
        data = token.encode('utf-16-le')
        for i in range(0, len(data), 2):
            h = ((h ^ int.from_bytes(data[i:i+2], 'little')) * 16777619) & 0xffffffff
        x[h % 13] += 1 if h & 256 else -1
    return [f32(v/math.sqrt(max(1,len(ts)))) for v in x]+[f32(min(1,len(goals)/8)), f32(min(1,len(ts)/128)), 1.0]


def vector(g, action):
    return [float(a == action) for a in MENU]+list(g)


def forward(model, pre):
    h = [0.0]*len(model['b'])
    for _ in range(3):
        h = [math.tanh(pre[i]+dot(r,h)+model['b'][i]) for i,r in enumerate(model['R'])]
    return dot(model['v'], h)+model['o']


def order(scores):
    return sorted(range(len(scores)),key=lambda i:(-scores[i],i))


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def archive_events(r):
    rows=[]
    for run in r['training']+r['runs']:
        for e in run['events']:
            if e['kind']=='probe' and e['transport'] is not None:
                rows.append((run['tag'],run['arm']['router'],e))
    return rows


def original_packet(e, mode, model, factors, g):
    p=e['transport'];data=bytes(p['packet']);version,n,d,b=data[:4]
    check(version==1 and n==len(e['options']) and (d,b)==((32,32) if mode=='dense' else (4,4 if mode=='rank4q' else 32)), 'original header')
    check(len(data)==4+(n*d*b+7)//8 and p['bytes']==len(data)+1,'original byte count')
    expected=bytearray(data[:4]);values=[];clips=0
    for a in e['options']:
        x=vector(g,a)
        z=x if mode=='dense' else [dot(v,x) for v in factors['V']]
        if b==32:
            expected.extend(struct.pack('<'+'f'*d,*z));values.extend(f32(y) for y in z)
        else:
            for y,s in zip(z,factors['scale']):
                q=math.floor(y/s+.5);clips+=int(q < -7 or q > 7)
                values.append((max(-7,min(7,q))+7))
    if b==4:
        expected.extend(bytes(values[j]+16*values[j+1] for j in range(0,len(values),2)))
        values=[(v-7)*factors['scale'][j%d] for j,v in enumerate(values)]
        check(clips==p['clipping'],'original clipping')
    check(bytes(expected)==data,'original packet differs')
    scores=[]
    for i in range(n):
        x=values[i*d:(i+1)*d]
        pre=[dot(w,x) for w in model['W']] if mode=='dense' else [sum(u[j]*x[k] for k,u in enumerate(factors['U'])) for j in range(12)]
        scores.append(forward(model,pre))
    check(all(close(a,b) for a,b in zip(scores,p['scores'])) and order(scores)==p['order'] and e['options'][p['reply']]==e['action'],'original scores/reply')
    return scores


def factors_setup():
    # Exact double-valued installed arrays, explicit role separation. No JSON overhead.
    return {'shared_dense':8*(12*32+12*12+12+12+1),
            'shared_rank4':8*(4*16+4*16+4*12+12*12+12+12+1),
            'shared_q4':8*(4*16+4*16+4*12+12*12+12+12+1+4*2),
            'exact_action_rank4':8*(4*16+12*16+4*12+12*12+12+12+1),
            'exact_action_q4':8*(4*16+12*16+4*12+12*12+12*12+1+4*2)}


def pack_shared(g, options, mode, factors, scales):
    ids=[MENU.index(a) for a in options]
    check(ids==sorted(set(ids)) and ids,'canonical menu required')
    mask=sum(1<<i for i in ids)
    version=1;kind={'shared_dense':0,'shared_rank4':1,'shared_q4':2,'exact_action_rank4':3,'exact_action_q4':4}[mode]
    header=struct.pack('<BBH',version,kind,mask)
    z=[dot(v[16:],g) for v in factors['V']]
    if kind==0:
        return header+struct.pack('<16f',*g),0
    if kind in (1,3):
        return header+struct.pack('<4f',*z),0
    codes=[math.floor(y/s+.5) for y,s in zip(z,scales)]
    clip=sum(x < -7 or x > 7 for x in codes)
    codes=[max(-7,min(7,x))+7 for x in codes]
    return header+bytes([codes[0]|(codes[1]<<4),codes[2]|(codes[3]<<4)]),clip


def receive_shared(packet, model, factors, scales):
    ver,kind,mask=struct.unpack('<BBH',packet[:4]);check(ver==1 and mask>0,'shared header')
    ids=[i for i in range(16) if mask&(1<<i)]
    if kind==0:
        g=struct.unpack('<16f',packet[4:]);scores=[forward(model,[dot(w,vector(g,MENU[a])) for w in model['W']]) for a in ids]
    else:
        if kind in (1,3):
            z=struct.unpack('<4f',packet[4:])
        elif kind in (2,4):
            check(len(packet)==6,'shared q4 payload')
            code=[packet[4+(k//2)]>>(4*(k%2))&15 for k in range(4)]
            check(all(x<15 for x in code),'reserved code');z=[(x-7)*s for x,s in zip(code,scales)]
        else:
            raise ValueError('unknown packet mode')
        scores=[forward(model,projected_pre(model,factors,z,a,kind in (3,4))) for a in ids]
    return [MENU[i] for i in ids],scores


def projected_pre(model,factors,z,action_id,exact_action):
    if exact_action:
        return [model['W'][j][action_id]+sum(u[j]*z[k] for k,u in enumerate(factors['U'])) for j in range(12)]
    return [sum(u[j]*(z[k]+factors['V'][k][action_id]) for k,u in enumerate(factors['U'])) for j in range(12)]


METHODS=('shared_dense','shared_rank4','shared_q4','exact_action_rank4','exact_action_q4')


def interval_check(packet, model, factors, scales, g, options, dense, approximate, clipped):
    # A real-arithmetic sufficient test with a conservative finite replay guard.
    # This computes a hypothetical four-byte common goal norm packet; its
    # installation would require full receiver-side W or residual row norms.
    # No formal floating-point proof is asserted. Clipped requests abstain.
    if clipped:
        return False
    _,kind,_=struct.unpack('<BBH',packet[:4]);check(kind in (1,2,3,4),'interval mode')
    if kind in (1,3):
        z=struct.unpack('<4f',packet[4:]);qerr=[abs(x)*2**-22+1e-12 for x in z]
    else:
        z=[((packet[4+k//2]>>(4*(k%2))&15)-7)*s for k,s in enumerate(scales)];qerr=[s/2+1e-12 for s in scales]
    V,U=factors['V'],factors['U'];W=model['W'];exact_action=kind in (3,4)
    residual=[[w[i]-sum(U[k][j]*V[k][i] for k in range(4)) for i in range(32)] for j,w in enumerate(W)]
    norm=f32(math.sqrt(dot(g,g)))*(1+2**-22)+1e-12
    err0=[math.sqrt(dot(e[16:],e[16:]))*norm+sum(abs(U[k][j])*qerr[k] for k in range(4))+1e-11 for j,e in enumerate(residual)]
    intervals=[]
    for a in options:
        i=MENU.index(a);p=projected_pre(model,factors,z,i,exact_action)
        err=[e+(0 if exact_action else abs(residual[j][i])) for j,e in enumerate(err0)]
        lo,hi=[0.0]*12,[0.0]*12
        for _ in range(3):
            low=[math.tanh(p[j]-err[j]+model['b'][j]+sum(v*(lo[k] if v>=0 else hi[k]) for k,v in enumerate(r)))-1e-11 for j,r in enumerate(model['R'])]
            high=[math.tanh(p[j]+err[j]+model['b'][j]+sum(v*(hi[k] if v>=0 else lo[k]) for k,v in enumerate(r)))+1e-11 for j,r in enumerate(model['R'])]
            lo,hi=low,high
        low=model['o']+sum(v*(lo[k] if v>=0 else hi[k]) for k,v in enumerate(model['v']))-1e-10
        high=model['o']+sum(v*(hi[k] if v>=0 else lo[k]) for k,v in enumerate(model['v']))+1e-10
        intervals.append((low,high))
    check(all(lo<=s<=hi for s,(lo,hi) in zip(dense,intervals)),'interval misses observed dense score')
    top=order(approximate)[0]
    certified=all(intervals[top][0]>h for j,(_,h) in enumerate(intervals) if j!=top)
    if certified:check(top==order(dense)[0],'interval certificate disagrees')
    return certified


def audit(path):
    check(sha(path.read_bytes())==SHA,'archive pin')
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('export_manifest.json'))
        for name,digest in manifest['hashes'].items():
            check(sha(z.read(name))==digest,'member digest '+name)
        r=json.loads(z.read('result.json'))
    m,f=r['fit']['model'],r['factors'];teacher=r['teacher']['rows']
    check(len(teacher)==224 and len({x['root'] for x in teacher})==8 and all(x['split']=='train' for x in teacher),'train custody')
    train_states={tuple(x['goals']) for x in teacher}
    scales=[max(1e-6,max(abs(dot(v[16:],goal_vector(g))) for g in train_states))/7 for v in f['V']]
    tasks={c['id']:c for c in r['cases']}
    rows=[];unique={};orig=Counter()
    for tag,mode,e in archive_events(r):
        g=goal_vector(e['before']);options=e['options'];n=len(options);split=tasks[e['root']]['split']
        original_packet(e,mode,m,f,g);orig[mode]+=1
        # Dense oracle on the same public candidate menu, irrespective of arm.
        dense=[forward(m,[dot(w,vector(g,a)) for w in m['W']]) for a in options]
        dense_order=order(dense);top=dense_order[0];margin=dense[top]-dense[dense_order[1]] if n>1 else None
        row={'root':e['root'],'tag':tag,'attempt':e['attempt'],'split':split,'n':n,'denseTop':options[top],'margin':margin,'naiveDenseBytes':5+128*n,'originalBytes':e['transport']['bytes'],'originalMode':mode,'originalAgrees':e['action']==options[top],'modes':{}}
        for meth in METHODS:
            packet,clipped=pack_shared(g,options,meth,f,scales)
            decoded,scores=receive_shared(packet,m,f,scales);check(decoded==options,'action mask changed')
            ranked=order(scores);err=max(abs(a-b) for a,b in zip(scores,dense));agrees=ranked[0]==top
            if meth=='shared_dense':check(all(a==b for a,b in zip(scores,dense)) and agrees,'dense exactness')
            if n>1 and margin>2*err:check(agrees,'posterior margin bound violated')
            bound=interval_check(packet,m,f,scales,g,options,dense,scores,clipped) if meth!='shared_dense' else True
            row['modes'][meth]={'intervalSufficient':bound,'bytes':len(packet)+1,'agrees':agrees,'top':decoded[ranked[0]],'clipped':clipped,'maxScoreError':err,'posteriorMarginSufficient':n==1 or margin>2*err}
        key=json.dumps([e['root'],e['before'],options],ensure_ascii=False,separators=(',',':'))
        unique.setdefault(key,row);rows.append(row)
    check(len(rows)==493,'packet denominator')
    def summarize(items):
        out={'decisions':len(items),'naiveDenseBytes':sum(r['naiveDenseBytes'] for r in items),'originalBytes':sum(r['originalBytes'] for r in items),'originalAgrees':sum(r['originalAgrees'] for r in items)}
        for meth in METHODS:
            ms=[r['modes'][meth] for r in items];out[meth]={'bytes':sum(a['bytes'] for a in ms),'agrees':sum(a['agrees'] for a in ms),'clipped':sum(a['clipped'] for a in ms),'clippedDecisions':sum(a['clipped']>0 for a in ms),'intervalSufficient':sum(a['intervalSufficient'] for a in ms),'posteriorMarginSufficient':sum(a['posteriorMarginSufficient'] for a in ms),'maxScoreError':max((a['maxScoreError'] for a in ms),default=0)}
        return out
    result={'schema':'cog13i.shared_goal.v1','archiveSha256':SHA,'archivePacketsVerified':len(rows),'originalPacketModes':dict(orig),'trainStatesForScale':len(train_states),'setupBytes':factors_setup(),'scales':scales,'readouts':{s:summarize([x for x in rows if x['split']==s]) for s in ('train','ordinary','deeper')},'evaluation':summarize([x for x in rows if x['split']!='train']),'deduplicatedEvaluation':summarize([x for x in unique.values() if x['split']!='train']),'evidence':'archived reached candidate menus; no new native actions or proof completion'}
    return result,rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result,rows=audit(args.archive);args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(args.out/'decisions.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows));print(json.dumps(result,indent=2))
