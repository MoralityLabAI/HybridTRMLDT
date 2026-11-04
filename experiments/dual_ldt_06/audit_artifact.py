"""Post-run independent Fraction replay of a Dual-LDT-06 artifact directory.

This reimplementation follows the registered synthetic environment. It does
not import the training/evaluation transition code or retrain neural models.
The frontier calculation is explicitly retrospective.
"""
from argparse import ArgumentParser
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import comb
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.dual_ldt_06.outcome_frontier import Outcome, compile_cover, check_cover

PROGRAMS = tuple(product(range(7), repeat=3))
THETA = {'trickster': (4,1,1,1), 'partisan': (1,4,1,1),
         'reconciler': (1,1,4,1), 'legalist': (1,1,1,4)}


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'))+'\n').encode()


def exact_features(w, plan):
    @lru_cache(None)
    def visit(t, credits, guard, prog):
        if t == 3:
            return (F(0),)*4
        a = plan[t]
        if a == 6:
            return visit(t+1, credits, guard, prog)
        if credits < 1:
            return None
        if a < 3:
            if not w['prepare_capability'][a]:
                return None
            return visit(t+1, credits-1, guard | (1<<a), prog)
        i = a-3
        if t == 2 and i == w['forbidden_project']:
            return None
        if w['hazards'][i] and not guard & (1<<i):
            return None
        if prog[i] >= w['requirements'][i]:
            return visit(t+1, credits-1, guard, prog)
        after = list(prog)
        after[i] += 1
        after = tuple(after)
        pay = (0,)*4
        if after[i] == w['requirements'][i]:
            pred = w['prerequisites'][i]
            enabled = pred >= 0 and prog[pred] >= w['requirements'][pred]
            pay = tuple(x+(b if enabled else 0) for x,b in zip(w['features'][i],w['bonuses'][i]))
        p = F(w['probability_quarters'][i], 4)
        success = visit(t+1, credits-1, guard, after)
        if success is None:
            return None
        failure = (F(0),)*4 if p == 1 else visit(t+1, credits-1, guard, prog)
        if failure is None:
            return None
        return tuple(p*(pay[j]+success[j])+(1-p)*failure[j] for j in range(4))
    return visit(0, w['initial_credits'], 0, (0,0,0))


def upper(w, plan, theta):
    value = F(-sum(a != 6 for a in plan))
    for i in range(3):
        n = plan.count(i+3)
        p = F(w['probability_quarters'][i], 4)
        probability = sum((comb(n,k)*p**k*(1-p)**(n-k)
                           for k in range(w['requirements'][i],n+1)), F(0))
        payout = sum(t*(f+b) for t,f,b in zip(theta,w['features'][i],w['bonuses'][i]))
        value += probability*payout
    return value


def audit(root: Path):
    read = lambda name: json.loads((root/name).read_text())
    digest = lambda name: sha256((root/name).read_bytes()).hexdigest()
    manifest = read('manifest.json')
    freeze = read('weights/FREEZE.json')
    raw, summary = read('report_raw.json'), read('report_summary.json')
    assert raw['source_sha256'] == summary['source_sha256']
    assert digest('report_raw.json') == summary['raw_sha256']
    assert set(freeze['files']) == {f'k{k}_seed{s}.json' for k in (1,4) for s in (17,23,41)}
    for name,h in freeze['files'].items():
        assert digest('weights/'+name) == h
    assert raw['weights_sha256'] == freeze['files']
    all_ids = set()
    for split, receipt in manifest['splits'].items():
        assert sha256(canonical(receipt['worlds'])).hexdigest() == receipt['sha256']
        ids = [w['id'] for w in receipt['worlds']]
        assert len(ids) == len(set(ids))
        assert all(x.startswith(split) for x in ids)
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
    assert freeze['training_manifest']['world_manifest_sha256'] == manifest['splits']['train']['sha256']
    assert raw['world_ids'] == [w['id'] for w in manifest['splits']['report']['worlds']]

    all_values, cover_rows = {}, []
    bound_checks = selected_checks = 0
    for w in manifest['splits']['report']['worlds']:
        evaluated = {}
        for i, plan in enumerate(PROGRAMS):
            f = exact_features(w, plan)
            if f is not None:
                numerators = tuple(x*64 for x in f)
                assert all(n.denominator == 1 for n in numerators)
                evaluated[i] = (tuple(int(n) for n in numerators), sum(a != 6 for a in plan))
        for persona, theta in THETA.items():
            values = {i: sum(t*x for t,x in zip(theta,f))-64*c
                      for i,(f,c) in evaluated.items()}
            all_values[(w['id'],persona)] = values
            for i,v in values.items():
                assert F(v,64) <= upper(w,PROGRAMS[i],theta)
                bound_checks += 1
        records = tuple(Outcome(i,f,64*c) for i,(f,c) in evaluated.items())
        certificate = compile_cover(records)
        assert check_cover(records, certificate)
        for theta in product((0,1,4), repeat=4):
            assert max(r.score(theta) for r in records) == max(
                r.score(theta) for r in certificate.representatives)
        cover_rows.append({
            'world': w['id'], 'certified_plans': len(records),
            'unique_exact_vectors': len({r.profile for r in records}),
            'nondominated_vectors': len(certificate.representatives),
            'representatives': [r.plan_id for r in certificate.representatives],
        })

    cases = {(x['world'],x['persona']): x for x in raw['cases']}
    assert len(cases) == 256
    for key, case in cases.items():
        assert case['oracle_numerator'] == max(all_values[key].values())
        assert case['feasible'] == len(all_values[key])
    sequences, group_rows = defaultdict(list), defaultdict(list)
    for r in raw['rows']:
        values = all_values[(r['world'],r['persona'])]
        assert r['selected'] in values
        assert values[r['selected']] == r['return_numerator']
        assert max(values.values()) == r['oracle_numerator']
        assert r['regret_numerator'] == r['oracle_numerator']-r['return_numerator']
        assert r['bound_gap_numerator'] == r['upper_numerator']-r['return_numerator']
        assert 0 <= r['regret_numerator'] <= r['bound_gap_numerator']
        assert r['primitive_expansions'] == r['issuer_expansions']+r['checker_expansions']
        assert r['exact_optimum_attained'] == (r['regret_numerator'] == 0)
        assert r['exact_optimum_certified'] == (r['bound_gap_numerator'] == 0)
        assert r['end_to_end_seconds'] >= 0
        sequences[(r['world'],r['persona'],r['method'])].append(r)
        group_rows[(r['method'],r['budget'])].append(r)
        selected_checks += 1
    for rows in sequences.values():
        rows.sort(key=lambda r: r['budget'])
        assert [r['budget'] for r in rows] == [1,4,16,64,343]
        assert [r['return_numerator'] for r in rows] == sorted(r['return_numerator'] for r in rows)
        assert [r['upper_numerator'] for r in rows] == sorted((r['upper_numerator'] for r in rows), reverse=True)
        assert rows[-1]['regret_numerator'] == rows[-1]['bound_gap_numerator'] == 0
    assert bound_checks == raw['upper_bound_checks'] == 31500
    assert raw['upper_bound_violations'] == 0
    assert selected_checks == 10240
    for s in summary['summary']:
        rows = group_rows[(s['method'],s['budget'])]
        assert abs(mean(r['regret_numerator']/64 for r in rows)-s['mean_oracle_regret']) < 1e-12
        assert abs(mean(r['return_numerator']/64 for r in rows)-s['mean_return']) < 1e-12
        assert sum(r['exact_optimum_attained'] for r in rows) == s['oracle_attained']
        assert sum(r['exact_optimum_certified'] for r in rows) == s['exact_certificates']
    return {
        'status': 'post_run_independent_fraction_replay_and_outcome_cover_audit',
        'source_commit': (root/'SOURCE_COMMIT.txt').read_text().strip(),
        'report_raw_sha256': digest('report_raw.json'),
        'report_summary_sha256': digest('report_summary.json'),
        'weights_freeze_sha256': digest('weights/FREEZE.json'),
        'distinct_report_worlds': 64, 'world_persona_cases': 256,
        'all_plan_world_evaluations': 64*343,
        'feasible_plan_worlds': sum(x['certified_plans'] for x in cover_rows),
        'exact_upper_checks': bound_checks, 'selected_action_checks': selected_checks,
        'anytime_sequences': len(sequences), 'violations': 0,
        'frontier_audit': {
            'status': 'post_hoc_representation_analysis_not_runtime_speed_comparison',
            'weight_grid_checks_per_world': 81,
            'mean_certified_plans': mean(x['certified_plans'] for x in cover_rows),
            'mean_unique_vectors': mean(x['unique_exact_vectors'] for x in cover_rows),
            'mean_nondominated_vectors': mean(x['nondominated_vectors'] for x in cover_rows),
            'min_frontier': min(x['nondominated_vectors'] for x in cover_rows),
            'max_frontier': max(x['nondominated_vectors'] for x in cover_rows),
            'full_evaluation_required': True, 'rows': cover_rows,
        },
        'limits': 'Assumes the registered synthetic semantics. Recompute covers after feasible-domain changes. No neural transfer or runtime advantage is established by compression.',
    }


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('artifact_directory', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    result = audit(args.artifact_directory)
    target = args.out or args.artifact_directory/'INDEPENDENT_AUDIT.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('AUDIT PASSED', result['selected_action_checks'], 'selected plans;',
          result['exact_upper_checks'], 'exact bounds; violations', result['violations'])
    print('MEAN COVER', result['frontier_audit']['mean_nondominated_vectors'])
    print('WROTE', target)


if __name__ == '__main__':
    main()
