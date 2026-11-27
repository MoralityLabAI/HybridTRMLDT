"""COG-13K frozen-menu replay of a table-backed interval receiver."""
import ctypes as ct
import json
import math
import random
import statistics
import struct
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cog13j_native_progressive'))
try:
    import transport as j
except ModuleNotFoundError:
    import jtransport as j

C_SHA = 'd99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea'
J_SHA = '624fd661c820796d0db786aab6ce585d575bc0524db25fdb0b76b2d3f86fba69'
D = ct.c_double
P = ct.POINTER(D)
ARMS = ['local_dense', 'shared_dense', 'decimal_progressive', 'table_progressive']


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def arr(x):
    return (D * len(x))(*x)


class Receiver:
    def __init__(self, lib, model):
        t0 = time.process_time_ns()
        self.lib = ct.CDLL(str(lib.resolve()))
        self.lib.endpoint.argtypes = [D, P, P]
        self.lib.endpoint.restype = ct.c_int
        self.lib.certify.argtypes = [P, P, P, P, D, P, P, ct.c_uint16, P, P, P]
        self.lib.certify.restype = ct.c_int
        weights = model['W'] + model['R']
        need(all(math.isfinite(x) and abs(x) <= 64 for x in
                 [model['o'], *model['b'], *model['v'], *(v for w in weights for v in w)]), 'model range')
        self.w = arr([x for row in model['W'] for x in row])
        self.r = arr([x for row in model['R'] for x in row])
        self.b, self.v, self.o = arr(model['b']), arr(model['v']), model['o']
        j.tanh_bound.cache_clear()
        table = [v for i in range(2049) for v in j.tanh_bound(i / 256)]
        self.table = arr(table)
        self.table_bytes = struct.pack('<' + 'd' * len(table), *table)
        self.setup_ns = time.process_time_ns() - t0
        self.decimal = j.receive
        self.raw = []

    def receive(self, packet, model):
        ids, center, radius = j.decode(packet)
        if radius is None:
            return self.decimal(packet, model)
        gap, ranges = D(), (D * (2 * len(ids)))()
        code = self.lib.certify(self.w, self.r, self.b, self.v, self.o,
                                arr(center), arr(radius), sum(1 << a for a in ids),
                                self.table, ct.byref(gap), ranges)
        if code < 0:
            self.raw.append({'invalid': True})
            return 255, None
        need(code == 255 or 0 <= code < len(ids), 'receiver reply range')
        self.raw.append({'bounds': list(ranges), 'code': code})
        return code, gap.value if math.isfinite(gap.value) else None

    def test(self):
        r = random.Random(271828)
        xs = [0., -0., 8., -8., 64., -64., math.nextafter(8., 0.), math.nextafter(8., math.inf)]
        for i in range(0, 2049, 17):
            x = i / 256
            xs.extend([x, -x, math.nextafter(x, math.inf), math.nextafter(x, -math.inf)])
        xs += [r.uniform(-12, 12) for _ in range(512)]
        for x in xs:
            y = (D * 2)()
            need(self.lib.endpoint(x, self.table, y) == 0, 'endpoint status')
            lo, hi = j.tanh_bound(x)
            need(y[0] <= lo and hi <= y[1], f'endpoint containment {x}')
        y = (D * 2)()
        need(self.lib.endpoint(math.inf, self.table, y) < 0, 'infinite endpoint accepted')
        for packet in [b'', struct.pack('<BBH', 1, 6, 0),
                       struct.pack('<BBHf12b', 1, 6, 1, math.inf, *([0]*12)),
                       struct.pack('<BBHf12b', 1, 6, 1, 1., -128, *([0]*11))]:
            try:
                j.decode(packet)
            except (ValueError, struct.error):
                continue
            raise ValueError('malformed packet accepted')
        gap, ranges = D(), (D * 2)()
        need(self.lib.certify(self.w, self.r, self.b, self.v, self.o, arr([65.]*12),
                              arr([0.]*12), 1, self.table, ct.byref(gap), ranges) < 0, 'range check')
        return {'endpointChecks': len(xs), 'invalidEndpointRejected': True,
                'malformedPacketsRejected': 4, 'unsupportedRangeRejected': True}


def menus(c_path, j_path):
    old, model, freq = j.load(c_path)
    c = [(e['before'], e['options']) for run in old['runs'] for e in run['events']
         if e['kind'] == 'probe' and e['transport'] is not None]
    need(len(c) == 449, 'C roster')
    need(j.sha(j_path.read_bytes()) == J_SHA, 'J archive hash')
    new = []
    with zipfile.ZipFile(j_path) as z:
        report = json.loads(z.read('native/result.json'))
        need(report['complete'] and report['unknown'] == 0 and all(report['gates'].values()), 'J gate')
        rows = [r for r in report['rows'] if r['arm'] == 'shared_dense']
        need(len(rows) == 16 and all(r['solved'] for r in rows), 'J roots')
        for r in rows:
            folder = 'native/' + r['root'] + '_shared_dense/'
            episode = json.loads(z.read(folder + 'episode.json'))
            need(episode['solved'] and episode['arm'] == 'shared_dense', 'J episode')
            for e in episode['events']:
                need(e['reply'] is not None and e['route']['denseAgreement'], 'J reply')
                new.append((e['before'], e['options']))
    need(len(new) == 189, 'J menu roster')
    return {'C': c, 'J': new}, model, freq


def benchmark(c_path, j_path, lib):
    cohorts, model, freq = menus(c_path, j_path)
    fast = Receiver(lib, model)
    tests = fast.test()
    results = {}
    dense_checks = 0
    for cohort, rows in cohorts.items():
        runs = []
        for rep in range(3):
            metrics = {}
            for arm in ARMS[rep:] + ARMS[:rep]:
                j.receive = fast.receive if arm == 'table_progressive' else fast.decimal
                route_arm = 'progressive' if arm.endswith('progressive') else arm
                totals = Counter()
                for goals, options in rows:
                    fast.raw.clear()
                    result = j.route(goals, options, route_arm, model, freq)
                    need(result['denseAgreement'] is True, 'false certified action')
                    for k in ('bytes', 'senderNs', 'receiverNs', 'auditNs'):
                        totals[k] += result[k]
                    totals['decisions'] += 1
                    for s in result['stages']:
                        totals['refinements'] += s.get('reply') == 255
                        totals['exactFallbacks'] += s.get('mode') == 0 and route_arm == 'progressive'
                        totals['abstentionsBeforeSend'] += 'abstainBeforeSend' in s
                    if arm == 'table_progressive':
                        g = j.goal_vector(goals)
                        scores = j.dense(model, g, [j.MENU.index(a) for a in options])
                        for rec in fast.raw:
                            if rec.get('invalid'):
                                continue
                            need(all(lo-1e-10 <= s <= hi+1e-10 for lo, hi, s in
                                     zip(rec['bounds'][::2], rec['bounds'][1::2], scores, strict=True)), 'score outside table interval')
                            need(rec['code'] == 255 or rec['code'] == j.winner(scores), 'false positive')
                            dense_checks += 1
                metrics[arm] = dict(totals)
                print(cohort, rep, arm, totals['bytes'], totals['receiverNs'], flush=True, file=sys.stderr)
            runs.append(metrics)
        results[cohort] = {'menus': len(rows), 'runs': runs,
                           'median': {a: {k: statistics.median(r[a][k] for r in runs)
                                          for k in runs[0][a]} for a in ARMS}}
    j.receive = fast.decimal
    gate = {}
    for cohort, row in results.items():
        m = row['median']
        old, new = m['decimal_progressive'], m['table_progressive']
        gate[cohort] = {'bytes': new['bytes'] <= 1.05 * old['bytes'],
                        'receiver5x': 5 * new['receiverNs'] <= old['receiverNs'],
                        'receiverSpeedup': old['receiverNs'] / new['receiverNs'],
                        'routingSpeedup': (old['receiverNs'] + old['senderNs']) /
                                          (new['receiverNs'] + new['senderNs'])}
    return {'schema': 'cog13k.table_interval.v1', 'sourceArchives': {'C': C_SHA, 'J': J_SHA},
            'setup': {'tableBytes': len(fast.table_bytes), 'tableSha256': j.sha(fast.table_bytes),
                      'cpuNs': fast.setup_ns, 'modelBytes': 5960},
            'tests': tests, 'checkedTableIntervals': dense_checks, 'falseCertificates': 0,
            'cohorts': results, 'gate': gate,
            'scope': 'archived reached menus, three CPU timing repetitions, no native rollout'}


if __name__ == '__main__':
    c, old_j, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=False)
    lib = out / 'receiver.so'
    cmd = ['gcc', '-O3', '-std=c11', '-fPIC', '-shared', '-fno-fast-math',
           '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', str(HERE / 'receiver.c'), '-o', str(lib), '-lm']
    subprocess.run(cmd, check=True)
    result = benchmark(c, old_j, lib)
    result['build'] = {'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
                       'sourceSha256': j.sha((HERE / 'receiver.c').read_bytes())}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'setup': result['setup'], 'tests': result['tests'], 'gate': result['gate'],
                      'median': {k: v['median'] for k,v in result['cohorts'].items()}}, indent=2))
