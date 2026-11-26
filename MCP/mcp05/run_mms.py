"""Train one MMS arm on exported train replay, with optional explicit dev replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch

from .controller import VERSION, RU02
from .train import MODES, fit, grade, peak_rss, RAM_LIMIT


def load_replay(path):
    raw=path.read_bytes()
    data=json.loads(raw)
    if not isinstance(data,list) or not data:
        raise ValueError('expected nonempty episode list')
    ids=[e['case'] for e in data]
    if len(set(ids))!=len(ids) or any(not e['steps'] for e in data):
        raise ValueError('nonunique root IDs or empty episodes')
    return data,set(ids),hashlib.sha256(raw).hexdigest()


def main(a):
    if min(a.depth,a.epochs,a.threads)<1 or not math.isfinite(a.mms_mib) or a.mms_mib<0:
        raise ValueError('invalid training budget')
    if not math.isfinite(a.ram_limit_gib) or a.ram_limit_gib<=0:
        raise ValueError('invalid RAM limit')
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    train,ids,digest=load_replay(a.train_replay)
    dev=None
    dev_digest=None
    if a.development_replay:
        dev,dev_ids,dev_digest=load_replay(a.development_replay)
        if ids&dev_ids:raise ValueError('train/development roots overlap')
    options=dict(mib=a.mms_mib,mode=a.mms_mode,sites=a.mms_sites,
        schedule=a.mms_schedule,rank=a.rank,topk=a.topk,ratio=a.ratio,
        refresh=a.refresh,max_age=a.max_age,map_manifest=a.mms_map_manifest)
    receipt={}
    wall,cpu=time.perf_counter(),time.process_time()
    model,losses=fit(train,a.arm,depth=a.depth,epochs=a.epochs,seed=a.seed,
        mms_options=options,mms_receipt=receipt,ram_limit_bytes=int(a.ram_limit_gib*2**30))
    stats=dict(scope='development_only',train_replay_sha256=digest,
        development_replay_sha256=dev_digest,train_roots=len(train),
        fit_updates=len(losses),final_loss=losses[-1],process_cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.perf_counter()-wall,peak_rss_bytes=peak_rss(),mms=receipt or None,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    if dev is not None:stats['development']=grade(model,dev)
    a.out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(version=VERSION,mode=a.arm,depth=a.depth,state=model.state_dict(),
        architecture='ru02' if isinstance(model,RU02) else 'settrm'),a.out/'model.pt')
    (a.out/'receipt.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-replay',type=Path,required=True)
    p.add_argument('--development-replay',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=MODES,default='recurrent')
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=231)
    p.add_argument('--ram-limit-gib',type=float,default=RAM_LIMIT/2**30)
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--mms-mode',choices=('observe','dense','bpd','sham'),default='bpd')
    p.add_argument('--mms-sites',default='auto')
    p.add_argument('--mms-schedule',choices=('round_robin','adaptive'),default='round_robin')
    p.add_argument('--mms-map-manifest',default='')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--topk',type=int,default=4)
    p.add_argument('--ratio',type=float,default=.2)
    p.add_argument('--refresh',type=int,default=128)
    p.add_argument('--max-age',type=int,default=32768)
    main(p.parse_args())
