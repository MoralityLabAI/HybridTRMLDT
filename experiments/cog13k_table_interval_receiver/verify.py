"""Check frozen discrete counts, source custody, and registered timing gates."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
expected = json.loads((here / 'results.json').read_text())
got = json.loads(Path(sys.argv[1]).read_text())

def need(ok, msg):
    if not ok:
        raise ValueError(msg)

for p, name in [(here / 'receiver.c', 'receiverBlob'),
                (here / 'benchmark.py', 'benchmarkBlob'),
                (here.parent / 'cog13j_native_progressive' / 'transport.py', 'transportBlob')]:
    b = p.read_bytes()
    need(hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest() == expected['build'][name],
         f'source custody {name}')
need(got['sourceArchives'] == expected['sourceArchives'], 'source archives')
need(got['build']['sourceSha256'] == expected['build']['sourceSha256'], 'receiver source')
for k in ('tableBytes', 'tableSha256', 'modelBytes'):
    need(got['setup'][k] == expected['setup'][k], 'setup ' + k)
need(got['tests'] == expected['tests'], 'endpoint/malformed tests')
need(got['checkedTableIntervals'] == expected['checkedTableIntervals'] and got['falseCertificates'] == 0,
     'interval audit')
need(set(got['cohorts']) == set(expected['cohorts']), 'cohorts')
keys = ('bytes', 'decisions', 'refinements', 'exactFallbacks', 'abstentionsBeforeSend')
for name, old in expected['cohorts'].items():
    new = got['cohorts'][name]
    need(new['menus'] == old['menus'] and len(new['runs']) == 3, 'menu/repetition roster')
    need(set(new['median']) == set(old['median']), 'arm roster')
    for run in new['runs']:
        need(set(run) == set(old['median']), 'run roster')
        for arm, values in run.items():
            need(all(values[k] == old['median'][arm][k] for k in keys), 'discrete ' + name + ' ' + arm)
    for arm, med in new['median'].items():
        for k, x in med.items():
            need(x == statistics.median(run[arm][k] for run in new['runs']), 'median arithmetic')
    d, t = new['median']['decimal_progressive'], new['median']['table_progressive']
    need(t['bytes'] <= d['bytes'] * 1.05 and 5*t['receiverNs'] <= d['receiverNs'], 'registered gate ' + name)
    need(got['gate'][name]['bytes'] and got['gate'][name]['receiver5x'], 'gate receipt')
print('PASS: source pins, all discrete counts, three repetitions, and timing gates; timings measured anew.')
