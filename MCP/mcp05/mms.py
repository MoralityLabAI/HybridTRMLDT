"""MCP05's own episode-based trainer, distinct from MCP02/03."""
import torch
from research_gym.neural.mms import MultiSiteMemory, read_map_manifest
from MCP.mcp02.features import SEQ_LEN
from .controller import RU02


class EpisodeMemory:
    """One site/visit in the first slate; later episode steps keep full BPTT."""

    def __init__(self,model,episodes,depth,options):
        self.depth = depth
        self.episodes = episodes
        self.ru = isinstance(model,RU02)
        self.offsets = [0]
        sizes=[]
        for e in episodes:
            if not e['steps'] or not e['steps'][0]['observation']['slate']:
                raise ValueError('MMS requires nonempty training episodes and first slates')
            size=len(e['steps'][0]['observation']['slate'])
            sizes.append(size)
            self.offsets.append(self.offsets[-1]+size)
        block = model.router.block if self.ru else model.refine
        available = {'ff2':block.ff2,'out':block.out,'qkv':block.qkv}
        selection=options.get('sites','auto')
        names=list(available) if selection=='auto' else selection.split(',')
        if len(set(names))!=len(names) or not set(names).issubset(available):
            raise ValueError('invalid MCP05 MMS sites')
        maps=read_map_manifest(options['map_manifest']) if options.get('map_manifest') else {}
        self.memory=MultiSiteMemory({n:available[n] for n in names},mib=options['mib'],
            mode=options.get('mode','bpd'),rank=options.get('rank',16),
            topk=options.get('topk',4),ratio=options.get('ratio',.2),
            refresh=options.get('refresh',128),max_age=options.get('max_age',32768),
            schedule=options.get('schedule','round_robin'),
            row_length=SEQ_LEN if self.ru else 1+max(sizes),maps=maps)

    def capture(self,index,step):
        site=self.memory.select(step)
        keys=(torch.arange(self.offsets[index],self.offsets[index+1])
              if self.ru else torch.tensor([index]))
        return self.memory.sites[site].capture(keys,step,self.depth//2,self.depth,
            extra_visits=self.depth*(len(self.episodes[index]['steps'])-1))
