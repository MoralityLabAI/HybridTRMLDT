"""MMS adapter for MCP-02/03 shared router trainer."""
from research_gym.neural.mms import MultiSiteMemory, read_map_manifest


def router_memory(model, options, row_length):
    c=model.config
    if model.arch=='looped':
        blocks={'block':(model.block,c['passes'])}
    elif model.arch in ('trm','trm_ldt'):
        blocks={f'net{i}':(b,c['cycles']*(c['inner']+1)) for i,b in enumerate(model.net)}
    elif model.arch=='hrm':
        blocks={'low':(model.f_low,c['high_cycles']*c['low_steps']),
                'high':(model.f_high,c['high_cycles'])}
    else:
        raise ValueError('MMS supports looped, TRM, TRM-LDT and HRM routers')
    sites={};counts={}
    for name,(block,visits) in blocks.items():
        for field in ('ff2','out','qkv'):
            site=f'{name}.{field}'
            sites[site]=getattr(block,field);counts[site]=visits
    selected=options.get('sites','auto')
    names=list(sites) if selected=='auto' else selected.split(',')
    if len(set(names))!=len(names) or not set(names).issubset(sites):
        raise ValueError('unknown or duplicate MMS router sites')
    maps=read_map_manifest(options['map_manifest']) if options.get('map_manifest') else {}
    memory=MultiSiteMemory({n:sites[n] for n in names},mib=options['mib'],
        mode=options.get('mode','bpd'),rank=options.get('rank',16),topk=options.get('topk',4),
        ratio=options.get('ratio',.2),refresh=options.get('refresh',128),
        max_age=options.get('max_age',32768),schedule=options.get('schedule','round_robin'),
        row_length=row_length,maps=maps)
    return memory,counts
