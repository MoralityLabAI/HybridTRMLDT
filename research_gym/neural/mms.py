"""CPU-only mechanistic memory scaling (MMS), development prototype.

Native BPD sites supply V[in,C], U[C,out]; ordinary Linear sites use a
SHA-pinned exported factor map or an explicitly labelled SVD seed proxy.
Store detached physical cotangents, not historical component indices.
"""
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import math
from pathlib import Path
import time

import torch
from torch import nn

MODES = ('observe', 'dense', 'bpd', 'sham')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GradientBank:
    """Bounded, lazily allocated direct-mapped BF16 tensor cache.

    Budget covers data + two int64 tags per slot. Python/allocator overhead
    and live temporaries are additional. Collisions are exact misses.
    """
    def __init__(self, mib, shape, page_rows=16):
        if not math.isfinite(mib) or mib <= 0:
            raise ValueError('positive finite MMS budget required')
        self.shape = tuple(shape)
        self.budget_bytes = int(mib * 2**20)
        self.capacity = self.budget_bytes // (2 * math.prod(shape) + 16)
        if self.capacity < 1:
            raise ValueError('MMS budget cannot hold one row')
        self.page_rows = min(page_rows, self.capacity)
        self.pages = {}
        self.pool = None
        self.site = None
        self.hits = self.queries = self.writes = self.collisions = 0

    @property
    def allocated_bytes(self):
        return sum(sum(t.numel() * t.element_size() for t in page)
                   for page in self.pages.values())

    def _indices(self, keys):
        keys = keys.detach().to(device='cpu', dtype=torch.long).reshape(-1)
        if bool((keys < 0).any()):
            raise ValueError('negative MMS key')
        slots = (keys * 6364136223846793005).remainder(self.capacity)
        return keys, slots // self.page_rows, slots % self.page_rows

    def read(self, keys, step, max_age):
        keys, pages, rows = self._indices(keys)
        result = torch.zeros((len(keys), *self.shape), dtype=torch.float32)
        valid = torch.zeros(len(keys), dtype=torch.bool)
        self.queries += len(keys)
        for p in pages.unique().tolist():
            if p not in self.pages:
                continue
            ix = (pages == p).nonzero().flatten()
            if self.pool is not None:
                self.pool.touch(self.site, p)
            values, tags, age = self.pages[p]
            r = rows[ix]
            ok = (tags[r] == keys[ix]) & (age[r] < step) & (step - age[r] <= max_age)
            result[ix[ok]] = values[r[ok]].float()
            valid[ix[ok]] = True
        self.hits += int(valid.sum())
        return result, valid

    def write(self, keys, values, step):
        keys, pages, rows = self._indices(keys)
        if tuple(values.shape) != (len(keys), *self.shape):
            raise ValueError('MMS row shape changed')
        values = values.detach().to(device='cpu', dtype=torch.bfloat16)
        for p in pages.unique().tolist():
            if p not in self.pages:
                size = min(self.page_rows, self.capacity - p * self.page_rows)
                if self.pool is not None:
                    self.pool.claim(self.site, p, size * (2 * math.prod(self.shape) + 16))
                self.pages[p] = (torch.zeros((size, *self.shape), dtype=torch.bfloat16),
                                 torch.full((size,), -1, dtype=torch.long),
                                 torch.full((size,), -1, dtype=torch.long))
            if self.pool is not None:
                self.pool.touch(self.site, p)
            data, tags, age = self.pages[p]
            ix = (pages == p).nonzero().flatten()
            r = rows[ix]
            order = torch.argsort(r, stable=True)
            ordered = r[order]
            last = torch.ones(len(order), dtype=torch.bool)
            last[:-1] = ordered[:-1] != ordered[1:]
            ix = ix[order[last]]
            r = rows[ix]
            self.collisions += int(((tags[r] >= 0) & (tags[r] != keys[ix])).sum())
            data[r], tags[r], age[r] = values[ix], keys[ix], step
            self.writes += len(ix)

    def state_dict(self):
        return dict(shape=self.shape, capacity=self.capacity, budget_bytes=self.budget_bytes,
                    page_rows=self.page_rows, pages=self.pages, hits=self.hits,
                    queries=self.queries, writes=self.writes, collisions=self.collisions)

    def load_state_dict(self, state):
        if (tuple(state['shape']), state['capacity'], state['budget_bytes'], state['page_rows']) != (self.shape, self.capacity, self.budget_bytes, self.page_rows):
            raise ValueError('MMS checkpoint budget/shape mismatch')
        self.pages = state['pages']
        for k in ('hits', 'queries', 'writes', 'collisions'):
            setattr(self, k, int(state[k]))
        if self.allocated_bytes > self.budget_bytes:
            raise ValueError('MMS checkpoint exceeds budget')

    def receipt(self):
        return dict(budget_bytes=self.budget_bytes, allocated_bytes=self.allocated_bytes,
                    capacity=self.capacity, hits=self.hits, queries=self.queries,
                    writes=self.writes, collisions=self.collisions,
                    occupied_rows=sum(int((p[1] >= 0).sum()) for p in self.pages.values()))


class MechanisticMemory:
    """One physical recurrent write site and one selected visit per backward.

    Add a bounded historical cotangent at that site. The signal is
    first-order gradient replay, not evidence of causal component discovery.
    """
    def __init__(self, module, *, mib, mode='bpd', rank=16, topk=4,
                 ratio=.2, refresh=128, max_age=32768, map_file='', map_sha='',
                 min_map_cosine=.5, row_length=None):
        if mode not in MODES or min(rank, topk, refresh, max_age) < 1 or not 0 <= ratio <= 1:
            raise ValueError('invalid MMS settings')
        self.module, self.mib, self.mode = module, mib, mode
        self.rank, self.topk, self.ratio = rank, topk, ratio
        self.refresh, self.max_age = refresh, max_age
        self.min_map_cosine = min_map_cosine
        self.row_length = row_length
        self.bank, self.v, self.u = None, None, None
        self.page_pool, self.site_name = None, None
        self.map_step, self.map_cosine = -1, None
        self.admitted = self.updates = self.map_refreshes = 0
        self.cpu_seconds = 0.
        self.native = hasattr(module, 'V') and hasattr(module, 'U')
        self.map_kind = 'native_live_bpd' if self.native else ('native_export_unverified_semantics' if map_file else 'svd_seed_proxy')
        if self.native and map_file:
            raise ValueError('native live site cannot also use an export')
        if not self.native and not isinstance(module, nn.Linear):
            raise TypeError('MMS requires native BPD or nn.Linear')
        self.export = None
        if map_file:
            if not map_sha or file_sha256(map_file) != map_sha:
                raise ValueError('MMS map file SHA-256 mismatch')
            data = torch.load(map_file, map_location='cpu', weights_only=True)
            self.export = (data['V'].float(), data['U'].float())

    @torch.no_grad()
    def _refresh(self, step):
        if self.v is not None and step - self.map_step < self.refresh:
            return
        if self.native:
            v, u = self.module.V.detach().float().cpu(), self.module.U.detach().float().cpu()
            weight = (v @ u).T
        else:
            weight = self.module.weight.detach().float().cpu()
            if self.export is not None:
                v, u = self.export
            else:
                left, s, right = torch.linalg.svd(weight, full_matrices=False)
                n = min(self.rank, len(s))
                root = s[:n].sqrt()
                v, u = right[:n].T * root, root[:, None] * left[:, :n].T
        if v.ndim != 2 or u.ndim != 2 or v.shape[1] != u.shape[0] or (u.shape[1], v.shape[0]) != tuple(weight.shape):
            raise ValueError('MMS factor shape mismatch')
        reconstructed = (v @ u).T
        cosine = torch.nn.functional.cosine_similarity(weight.flatten(), reconstructed.flatten(), dim=0)
        if not bool(torch.isfinite(cosine)) or not bool(torch.isfinite(v).all() and torch.isfinite(u).all()):
            raise ValueError('nonfinite MMS factors')
        self.map_cosine = float(cosine)
        self.v = v.clone()
        self.u = torch.nn.functional.normalize(u, dim=1).clone()
        self.map_step = step
        self.map_refreshes += 1

    @torch.no_grad()
    def _gradient(self, keys, step, x, gradient):
        started = time.process_time()
        if gradient.device.type != 'cpu':
            raise ValueError('MMS v0 is CPU-only')
        g = gradient.detach().float()
        if not bool(torch.isfinite(g).all()):
            return gradient
        shape = tuple(g.shape[1:])
        if self.row_length is not None:
            if g.ndim != 3 or g.shape[1] > self.row_length:
                raise ValueError('MMS dynamic rows exceed fixed storage length')
            shape = (self.row_length, g.shape[-1])
        if self.bank is None:
            self.bank = GradientBank(self.mib, shape)
            if self.page_pool is not None:
                self.page_pool.attach(self.site_name, self.bank)
        if shape != self.bank.shape:
            raise ValueError('MMS requires fixed output row shape')
        old, hit = self.bank.read(keys, step, self.max_age)
        if self.row_length is not None:
            old = old[:, :g.shape[1]]
        flat = g.reshape(len(g), -1)
        norm = flat.norm(dim=1)
        normalized = (flat / norm.clamp_min(1e-12)[:, None]).reshape_as(g)
        if self.row_length is not None:
            normalized = torch.nn.functional.pad(normalized,(0,0,0,self.row_length-g.shape[1]))
        self.bank.write(keys, normalized, step)
        self.updates += len(g)
        # Truncated SVD cosine is explained energy, which depends on rank.
        # Only a frozen exported map uses the absolute drift/fidelity gate.
        minimum = self.min_map_cosine if self.export is not None else 1e-8
        if self.mode == 'observe' or not bool(hit.any()) or self.map_cosine < minimum:
            self.cpu_seconds += time.process_time() - started
            return gradient
        h = old[hit].reshape(int(hit.sum()), -1, g.shape[-1])
        current = g[hit].reshape_as(h)
        inputs = x.detach().float()[hit].reshape(len(h), -1, self.v.shape[0])
        reads = (inputs @ self.v).abs().mean(dim=1)
        coeff = h @ self.u.T
        scores = reads * coeff.abs().mean(dim=1)
        indices = scores.topk(min(self.topk, scores.shape[-1]), dim=1).indices
        selected = self.u[indices]
        # Sum normalized component outer products; U need not be orthogonal.
        projected = (h @ selected.transpose(1, 2)) @ selected
        p, c = projected.flatten(1), current.flatten(1)
        admitted = (p.norm(dim=1) > 1e-12) & ((p*c).sum(1) > 0) & (c.norm(dim=1) > 1e-12)
        chosen = h if self.mode == 'dense' else (projected.roll(1, dims=-1) if self.mode == 'sham' else projected)
        z = chosen.flatten(1)
        eligible = admitted & (z.norm(dim=1) > 1e-12)
        delta = z / z.norm(dim=1).clamp_min(1e-12)[:, None] * (self.ratio * c.norm(dim=1) * eligible)[:, None]
        out = g.clone()
        out[hit] += delta.reshape_as(out[hit])
        self.admitted += int(eligible.sum())
        self.cpu_seconds += time.process_time() - started
        return out.to(gradient.dtype)

    @contextmanager
    def capture(self, keys, step, visit, visits, *, extra_visits=0):
        if not 0 <= visit < visits or step < 0 or extra_visits < 0:
            raise ValueError('invalid visit/step')
        if next(self.module.parameters()).device.type != 'cpu':
            raise ValueError('MMS v0 is CPU-only')
        keys = keys.detach().long().cpu()
        if bool((keys < 0).any()) or bool((keys > (2**63-1-visit)//visits).any()):
            raise ValueError('MMS key out of range')
        keys = keys * visits + visit
        started = time.process_time()
        self._refresh(step)
        self.cpu_seconds += time.process_time() - started
        calls = backward_calls = 0
        def hook(module, inputs, output):
            nonlocal calls
            ordinal = calls
            calls += 1
            if ordinal != visit:
                return
            if not output.requires_grad or len(output) != len(keys):
                raise ValueError('MMS capture expects differentiable batched output')
            x = inputs[0].detach()
            def backward(g):
                nonlocal backward_calls
                backward_calls += 1
                if backward_calls != 1:
                    raise ValueError('MMS capture supports one backward')
                return self._gradient(keys, step, x, g)
            output.register_hook(backward)
        handle = self.module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
        if calls != visits + extra_visits or backward_calls != 1:
            raise ValueError(f'MMS expected {visits + extra_visits} site visits and one backward, got {calls}/{backward_calls}')

    def state_dict(self):
        return dict(bank=None if self.bank is None else self.bank.state_dict(), v=self.v, u=self.u,
                    map_step=self.map_step, map_cosine=self.map_cosine, admitted=self.admitted,
                    updates=self.updates, map_refreshes=self.map_refreshes, cpu_seconds=self.cpu_seconds)

    def load_state_dict(self, state):
        for k in ('v','u','map_step','map_cosine','admitted','updates','map_refreshes','cpu_seconds'):
            setattr(self, k, state[k])
        if state['bank'] is not None:
            self.bank = GradientBank(self.mib, state['bank']['shape'])
            self.bank.load_state_dict(state['bank'])

    def receipt(self):
        return dict(mode=self.mode, map_kind=self.map_kind, map_cosine=self.map_cosine,
                    map_refreshes=self.map_refreshes, admitted=self.admitted,
                    sampled_rows=self.updates, extra_process_cpu_seconds=self.cpu_seconds,
                    bank=None if self.bank is None else self.bank.receipt())


class SharedPagePool:
    """One tensor-payload ceiling across sites; global deterministic LRU eviction.

    Each site's hash address space remains fixed even when pages are evicted.
    Maps, allocator overhead, live tensors and serialization are outside this
    cap. The selected site may evict its own or another site's old pages.
    """
    def __init__(self, mib):
        if not math.isfinite(mib) or mib <= 0:
            raise ValueError('invalid shared memory budget')
        self.budget_bytes = int(mib * 2**20)
        self.banks = {}
        self.order = OrderedDict()
        self.used = self.evictions = self.peak = 0

    def attach(self, site, bank):
        if site in self.banks and self.banks[site] is not bank:
            raise ValueError('duplicate bank site')
        self.banks[site] = bank
        bank.pool, bank.site = self, site

    def touch(self, site, page):
        self.order.move_to_end((site, page))

    def claim(self, site, page, size):
        key = (site, page)
        if key in self.order or not 0 < size <= self.budget_bytes:
            raise ValueError('invalid shared page allocation')
        while self.used + size > self.budget_bytes:
            (other, p), old = self.order.popitem(last=False)
            del self.banks[other].pages[p]
            self.used -= old
            self.evictions += 1
        self.order[key] = size
        self.used += size
        self.peak = max(self.peak, self.used)

    def state_dict(self):
        return dict(order=list(self.order.items()), used=self.used,
                    peak=self.peak, evictions=self.evictions)

    def load_state_dict(self, state):
        actual = {(site, p): sum(t.numel()*t.element_size() for t in page)
                  for site, bank in self.banks.items() for p, page in bank.pages.items()}
        order = OrderedDict((tuple(k), v) for k,v in state['order'])
        if dict(order) != actual or sum(order.values()) != state['used'] or not 0 <= state['used'] <= state['peak'] <= self.budget_bytes:
            raise ValueError('inconsistent MMS shared page checkpoint')
        self.order, self.used, self.peak, self.evictions = order, state['used'], state['peak'], state['evictions']


class MultiSiteMemory:
    """One selected physical write per optimizer step across shared RAM.

    Fixed round-robin is the matched scheduling control. Adaptive chooses
    the site with highest historical admission fraction (hit fraction in
    observe), with an exploration step every four updates. Admission is a
    cheap engineering proxy, not causal task utility. Never reads heldout.
    """
    def __init__(self, modules, *, mib, schedule='round_robin', maps=None, **kwargs):
        if schedule not in ('round_robin', 'adaptive') or not modules:
            raise ValueError('invalid MMS site schedule')
        if len({id(m) for m in modules.values()}) != len(modules):
            raise ValueError('MMS site aliases must not share a physical module')
        self.pool = SharedPagePool(mib)
        self.schedule = schedule
        self.sites = {}
        maps = maps or {}
        if not set(maps).issubset(modules):
            raise ValueError('unrecognized MMS map site')
        for name, module in modules.items():
            options = maps.get(name, {})
            mem = MechanisticMemory(module, mib=mib, **kwargs, **options)
            mem.page_pool, mem.site_name = self.pool, name
            self.sites[name] = mem
        self.last_step, self.selected = -1, None
        self.selections = dict.fromkeys(self.sites, 0)

    def select(self, step):
        if step < self.last_step:
            raise ValueError('MMS time reversed')
        if step == self.last_step:
            return self.selected
        names = tuple(self.sites)
        if self.schedule == 'round_robin':
            name = names[step % len(names)]
        elif step % 4 == 0:
            name = names[(step//4) % len(names)]
        else:
            def score(n):
                m = self.sites[n]
                if m.bank is None:
                    return 1.0
                hits = m.bank.hits if m.mode == 'observe' else m.admitted
                return (hits+1)/(m.updates+2)
            name = max(names, key=score)
        self.last_step, self.selected = step, name
        self.selections[name] += 1
        return name

    @contextmanager
    def capture(self, keys, step, visit, visits):
        name = self.select(step)
        with self.sites[name].capture(keys, step, visit, visits):
            yield

    def state_dict(self):
        return dict(sites={n:m.state_dict() for n,m in self.sites.items()},
                    pool=self.pool.state_dict(), schedule=self.schedule,
                    selections=self.selections, last_step=self.last_step, selected=self.selected)

    def load_state_dict(self, state):
        if tuple(state['sites']) != tuple(self.sites) or state['schedule'] != self.schedule:
            raise ValueError('MMS site schedule mismatch')
        for name, mem in self.sites.items():
            mem.load_state_dict(state['sites'][name])
            if mem.bank is not None:
                self.pool.attach(name, mem.bank)
        self.pool.load_state_dict(state['pool'])
        self.selections = dict(state['selections'])
        self.last_step, self.selected = state['last_step'], state['selected']

    def receipt(self):
        return dict(schedule=self.schedule, budget_bytes=self.pool.budget_bytes,
                    allocated_bytes=self.pool.used, peak_bank_bytes=self.pool.peak,
                    page_evictions=self.pool.evictions, site_selections=self.selections,
                    sites={n:m.receipt() for n,m in self.sites.items()})


class ExposedGRUCell(nn.Module):
    """Same Parameters and GRU equations, explicit recurrent affine for MMS."""
    def __init__(self, source):
        super().__init__()
        if not source.bias:
            raise ValueError('biased GRUCell required')
        self.weight_ih, self.bias_ih = source.weight_ih, source.bias_ih
        with torch.random.fork_rng(devices=[]):
            self.recurrent = nn.Linear(source.hidden_size,3*source.hidden_size)
        self.recurrent.weight, self.recurrent.bias = source.weight_hh, source.bias_hh

    def forward(self, x, h):
        ir,iz,inn=torch.nn.functional.linear(x,self.weight_ih,self.bias_ih).chunk(3,-1)
        hr,hz,hn=self.recurrent(h).chunk(3,-1)
        r,z=torch.sigmoid(ir+hr),torch.sigmoid(iz+hz)
        n=torch.tanh(inn+r*hn)
        return n+z*(h-n)


def read_map_manifest(path):
    import json
    path=Path(path)
    maps=json.loads(path.read_text(encoding='utf-8'))
    result={}
    for site,value in maps.items():
        source=(path.parent/value['path']).resolve()
        if file_sha256(source)!=value['sha256']:
            raise ValueError(f'BPD map digest mismatch for {site}')
        result[site]=dict(map_file=str(source),map_sha=value['sha256'])
    return result
