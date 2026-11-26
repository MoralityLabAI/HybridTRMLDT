"""COG-13I-B: certified shared-preactivation transport on archived reached states."""
from __future__ import annotations

import argparse
import decimal as dec
import functools
import json
import math
import struct
import zipfile
from pathlib import Path

import audit as a

NEG, POS = -math.inf, math.inf


def down(x):
    return math.nextafter(x, NEG)


def up(x):
    return math.nextafter(x, POS)


def add(x, y):
    return down(x[0]+y[0]), up(x[1]+y[1])


def mul(k, x):
    l, h = (x if k >= 0 else x[::-1])
    return down(k*l), up(k*h)


def isum(b, weights, xs):
    v = (b, b)
    for k, x in zip(weights, xs, strict=True):
        v = add(v, mul(k, x))
    return v


@functools.lru_cache(maxsize=200000)
def tanh_bound(x):
    # Decimal.exp is correctly rounded (ROUND_HALF_EVEN). Enclose the exp
    # with its adjacent decimal values; every later decimal operation is
    # directed, then conversion back to binary64 is widened one ULP.
    if x < 0:
        l, h = tanh_bound(-x)
        return -h, -l
    with dec.localcontext() as c:
        c.prec = 48
        c.rounding = dec.ROUND_HALF_EVEN
        e = dec.Decimal.from_float(2*x).exp()
        el, eh = e.next_minus(), e.next_plus()
        one, two = dec.Decimal(1), dec.Decimal(2)
        c.rounding = dec.ROUND_FLOOR
        denom_l = one+el
        c.rounding = dec.ROUND_CEILING
        ratio_h = two/denom_l
        c.rounding = dec.ROUND_FLOOR
        lo = one-ratio_h
        c.rounding = dec.ROUND_CEILING
        denom_h = one+eh
        c.rounding = dec.ROUND_FLOOR
        ratio_l = two/denom_h
        c.rounding = dec.ROUND_CEILING
        hi = one-ratio_l
        return max(-1.0, down(float(lo))), min(1.0, up(float(hi)))


def tanh_interval(x):
    return tanh_bound(x[0])[0], tanh_bound(x[1])[1]


def pre_bounds(model, g):
    return [isum(0.0, w[16:], [(x,x) for x in g]) for w in model['W']]


def fp32_up(x):
    v = a.f32(x)
    if v < x:
        v = struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', v))[0]+1))[0]
    return v


def envelope(center, radius):
    return [(down(x-r), up(x+r)) for x,r in zip(center,radius,strict=True)]


def prove(model, bounds, ids, scores):
    ranges=[]
    for action in ids:
        pre=[add(x,(w[action],w[action])) for x,w in zip(bounds,model['W'],strict=True)]
        h=[(0.,0.)]*12
        for _ in range(3):
            h=[tanh_interval(add(p,isum(b,r,h))) for p,b,r in zip(pre,model['b'],model['R'],strict=True)]
        ranges.append(isum(model['o'],model['v'],h))
    best=a.order(scores)[0]
    gap=min((down(ranges[best][0]-r[1]) for i,r in enumerate(ranges) if i!=best),default=math.inf)
    return gap>0, gap if math.isfinite(gap) else None, ranges


def encode(g,ids,model,factors,scales,kind):
    c=[a.dot(w[16:],g) for w in model['W']]
    mask=sum(1<<i for i in ids)
    mode={'q4_envelope':5,'pre_i8':6,'pre_f16':7}[kind]
    header=struct.pack('<BBH',1,mode,mask)
    cb=pre_bounds(model,g)
    if kind=='q4_envelope':
        p,_=a.pack_shared(g,[a.MENU[i] for i in ids],'exact_action_q4',factors,scales)
        code=[p[4+k//2]>>(4*(k%2))&15 for k in range(4)]
        z=[(x-7)*s for x,s in zip(code,scales,strict=True)]
        center=[sum(u[j]*z[k] for k,u in enumerate(factors['U'])) for j in range(12)]
        radius=fp32_up(max(up(max(abs(lo-x),abs(hi-x))) for (lo,hi),x in zip(cb,center,strict=True)))
        data=header+p[4:]+struct.pack('<f',radius)
    elif kind=='pre_i8':
        scale=fp32_up(max(max(abs(x) for x in c)/127,2**-30))
        code=[max(-127,min(127,round(x/scale))) for x in c]
        data=header+struct.pack('<f12b',scale,*code)
    else:
        data=header+struct.pack('<12e',*c)
    # Sender checks all interval premises before requesting the cheap certificate.
    actual_ids,center,radius=decode(data,model,factors,scales)
    assert actual_ids==ids
    bounds=envelope(center,radius)
    assert all(l<=cl<=ch<=h for (l,h),(cl,ch) in zip(bounds,cb,strict=True)), 'sender bound violated'
    return data


def decode(packet,model,factors,scales):
    v,mode,mask=struct.unpack('<BBH',packet[:4]);assert v==1 and mask
    ids=[i for i in range(16) if mask&(1<<i)]
    if mode==5:
        assert len(packet)==10
        codes=[packet[4+k//2]>>(4*(k%2))&15 for k in range(4)]
        assert all(c<15 for c in codes)
        z=[(x-7)*s for x,s in zip(codes,scales,strict=True)]
        c=[sum(u[j]*z[k] for k,u in enumerate(factors['U'])) for j in range(12)]
        r=[struct.unpack('<f',packet[6:])[0]]*12
    elif mode==6:
        scale,*codes=struct.unpack('<f12b',packet[4:]);assert scale>0 and all(x!=-128 for x in codes)
        c=[scale*x for x in codes];r=[up(scale/2+1e-12)]*12
    elif mode==7:
        c=list(struct.unpack('<12e',packet[4:]));r=[up(abs(x)*2**-10+2**-24+1e-12) for x in c]
    else:
        raise ValueError('mode')
    assert all(math.isfinite(x) for x in c+r) and all(x>=0 for x in r)
    return ids,c,r


def audit(path):
    check=a.check
    check(a.sha(path.read_bytes())==a.SHA,'archive hash')
    # Independently verify all original packets and manifest before using the new codec.
    old,_=a.audit(path)
    with zipfile.ZipFile(path) as z:
        r=json.loads(z.read('result.json'))
    model,factors=r['fit']['model'],r['factors'];scales=old['scales']
    cases={c['id']:c for c in r['cases']}
    modes=('q4_envelope','pre_i8','pre_f16')
    cache={};rows=[]
    for tag,_,e in a.archive_events(r):
        split=cases[e['root']]['split']
        if split=='train':continue
        g=a.goal_vector(e['before']);ids=[a.MENU.index(x) for x in e['options']];key=(tuple(g),tuple(ids))
        if key not in cache:
            dense=[a.forward(model,[a.dot(w,a.vector(g,a.MENU[i])) for w in model['W']]) for i in ids]
            top=a.order(dense)[0];item={}
            for kind in modes:
                packet=encode(g,ids,model,factors,scales,kind)
                decoded,c,rad=decode(packet,model,factors,scales);check(decoded==ids,'mask')
                scores=[a.forward(model,[w[i]+x for w,x in zip(model['W'],c,strict=True)]) for i in ids]
                certified,gap,ranges=prove(model,envelope(c,rad),ids,scores)
                # Archive FP64 replay check, separate from real-arithmetic enclosure.
                check(all(lo-1e-10<=v<=hi+1e-10 for v,(lo,hi) in zip(dense,ranges,strict=True)),'observed score escaped enclosure')
                agree=a.order(scores)[0]==top
                check(not certified or agree,'certificate disagrees with dense replay')
                item[kind]={'bytes':len(packet)+1,'certified':certified,'agrees':agree,'certifiedGap':gap,'maxRadius':max(rad),'packetHex':packet.hex()}
            cache[key]=item
        rows.append({'root':e['root'],'tag':tag,'attempt':e['attempt'],'split':split,'modes':cache[key]})
    check(len(rows)==449,'evaluation denominator')
    policies={'q4_then_exact':['q4_envelope'],'i8_then_exact':['pre_i8'],'f16_then_exact':['pre_f16'],'q4_i8_f16_exact':['q4_envelope','pre_i8','pre_f16'],'i8_f16_exact':['pre_i8','pre_f16']}
    def summary(rs):
        out={'decisions':len(rs),'exactBytes':69*len(rs),'modes':{},'policies':{}}
        for m in modes:
            x=[r['modes'][m] for r in rs];out['modes'][m]={'bytes':sum(y['bytes'] for y in x),'certified':sum(y['certified'] for y in x),'agrees':sum(y['agrees'] for y in x),'maxRadius':max(y['maxRadius'] for y in x)}
        for name,stages in policies.items():
            cost=0;fallback=0
            for r in rs:
                for m in stages:
                    x=r['modes'][m];cost+=x['bytes']
                    if x['certified']:break
                else:cost+=69;fallback+=1
            out['policies'][name]={'bytes':cost,'exactFallbacks':fallback,'finalDenseAgreements':len(rs)}
        return out
    return {'schema':'cog13i.certified.v1','sourceSha256':a.SHA,'archivedPacketsVerified':old['archivePacketsVerified'],'uniqueFeatureMenus':len(cache),'evaluation':summary(rows),'ordinary':summary([r for r in rows if r['split']=='ordinary']),'deeper':summary([r for r in rows if r['split']=='deeper']),'scope':'posthoc archived states; real-arithmetic interval bounds with trusted sender, archived FP64 replay check; no native rollout'},rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result,rows=audit(args.archive);args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'certified_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'certified_decisions.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows))
    print(json.dumps(result,indent=2))
