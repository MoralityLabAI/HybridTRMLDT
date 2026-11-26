"""Single-process MMS training lane for MCP06/07/08; development-only."""
import argparse
import json
from pathlib import Path
import time

import torch


def main(a):
    if min(a.threads,a.roots,a.epochs,a.batch,a.horizon)<1 or a.mms_mib<0:
        raise ValueError('invalid training budget')
    torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    options=dict(mib=a.mms_mib,mode=a.mms_mode,schedule=a.mms_schedule,rank=a.rank,
                 topk=a.topk,ratio=a.ratio,refresh=a.refresh,max_age=a.max_age,
                 map_manifest=a.mms_map_manifest)
    wall,cpu=time.perf_counter(),time.process_time()
    if a.gym=='06':
        from MCP.mcp06.gym import World,HIER,train,rollout
        w=World(16384,a.world_seed,'hilbert');memory={}
        model=train(w,HIER,a.seed,a.roots,a.epochs,mms_options=options,mms_receipt=memory)
        rows=[rollout(w,r,'adaptive',model,16,.1) for r in w.roots(a.dev_roots,9137)]
        stats={'mms':memory or None,'dev_success':sum(r['success'] for r in rows),'dev_roots':len(rows)}
    elif a.gym=='07':
        from MCP.mcp07.memory import SharedPhysicalMemory
        from MCP.mcp07.train import train_controller
        from MCP.mcp07.curriculum import sample_mixed_roots
        from MCP.mcp07.eval_harness import rollout_batch
        w=SharedPhysicalMemory(16384,a.world_seed)
        model,stats=train_controller(w,'recurrent_switch',seed=a.seed,num_roots=a.roots,
            epochs=a.epochs,batch_size=a.batch,horizon=a.horizon,mms_options=options)
        roots=sample_mixed_roots(w,a.dev_roots,salt=9137)
        rows,_=rollout_batch(w,roots,'hier_adaptive',model,budget=a.horizon,corruption=.1)
        stats.update(dev_roots=len(rows),dev_rows=rows)
    else:
        from MCP.mcp08.memory256k import ExtendedPhysicalMemory
        from MCP.mcp08.train import train_mcp08_controller
        w=ExtendedPhysicalMemory(16384,a.world_seed)
        model,stats=train_mcp08_controller(w,seed=a.seed,num_roots=a.roots,
            dagger_roots=max(32,a.roots//3),epochs=a.epochs,batch_size=a.batch,
            horizon=a.horizon,mms_options=options)
        stats['development']='use saved native checkpoint with original MCP08 search harness'
    a.out.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),a.out/'model.pt')
    (a.out/'receipt.json').write_text(json.dumps(dict(scope='development_only',gym=a.gym,
        wall_seconds=time.perf_counter()-wall,process_cpu_seconds=time.process_time()-cpu,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},stats=stats),
        indent=2,default=lambda v:v.item() if hasattr(v,'item') else v.tolist())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gym',choices=('06','07','08'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=107)
    p.add_argument('--world-seed',type=int,default=43210)
    p.add_argument('--roots',type=int,default=128)
    p.add_argument('--dev-roots',type=int,default=16)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--mms-mode',choices=('observe','dense','bpd','sham'),default='bpd')
    p.add_argument('--mms-schedule',choices=('round_robin','adaptive'),default='round_robin')
    p.add_argument('--mms-map-manifest',default='')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--topk',type=int,default=4)
    p.add_argument('--ratio',type=float,default=.2)
    p.add_argument('--refresh',type=int,default=128)
    p.add_argument('--max-age',type=int,default=32768)
    main(p.parse_args())
