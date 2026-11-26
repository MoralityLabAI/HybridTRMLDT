"""MMS for the native MCP spatial-memory and search training loops."""
from contextlib import contextmanager
from research_gym.neural.mms import ExposedGRUCell, MultiSiteMemory, read_map_manifest


def valid_visits(mask,horizon):
    # Native loops stop at the first all-false column.
    count=0
    for t in range(horizon):
        if not bool(mask[:,t].any()):break
        count+=1
    return count


class SpatialMemory:
    def __init__(self,model,options,*,legacy=False):
        self.model=model
        attrs=('fast','slow') if legacy else ('fast_gru',)
        self.originals={a:getattr(model,a) for a in attrs}
        self.wrappers={a:ExposedGRUCell(m) for a,m in self.originals.items()}
        modules={a:wrapper.recurrent for a,wrapper in self.wrappers.items()}
        if not legacy:modules['proj_gate']=model.proj_gate
        maps=read_map_manifest(options['map_manifest']) if options.get('map_manifest') else {}
        self.memory=MultiSiteMemory(modules,mib=options['mib'],mode=options.get('mode','bpd'),
            rank=options.get('rank',16),topk=options.get('topk',4),
            ratio=options.get('ratio',.2),refresh=options.get('refresh',128),
            max_age=options.get('max_age',32768),schedule=options.get('schedule','round_robin'),
            maps=maps)
        self.step=0
        self.unavailable_fallbacks=0

    @contextmanager
    def capture(self,ids,counts,extra=None):
        chosen=self.memory.select(self.step)
        name=chosen
        if not counts.get(name,0):
            name=next(n for n in self.memory.sites if counts.get(n,0))
            self.unavailable_fallbacks+=1
        for attr,wrapper in self.wrappers.items():setattr(self.model,attr,wrapper)
        try:
            visits=counts[name]
            with self.memory.sites[name].capture(ids,self.step,visits//2,visits,
                    extra_visits=(extra or {}).get(name,0),key_stride=128):
                yield
        finally:
            for attr,original in self.originals.items():setattr(self.model,attr,original)
        self.step+=1

    def receipt(self):
        return dict(updates=self.step,unavailable_fallbacks=self.unavailable_fallbacks,**self.memory.receipt())
