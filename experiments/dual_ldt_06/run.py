"""Reproducible frozen-training and once-only report evaluation for Dual-LDT-06."""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from experiments.dual_ldt_06.environment import (
    NAMESPACE, PERSONAS, PROGRAMS, DENOMINATOR, canonical, world, world_ids,
    evaluate, input_features, count_upper_numerator,
)
from experiments.dual_ldt_06.learning import SCALE, load, train_all
from experiments.dual_ldt_06.search import search, ordered_indices


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(obj)
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def manifest():
    splits = {}
    for split in ('train', 'dev', 'report'):
        records = [world(wid).record() for wid in world_ids(split)]
        splits[split] = {'worlds': records, 'sha256': sha256(canonical(records)).hexdigest()}
    return {'namespace': NAMESPACE, 'splits': splits}


def evaluate_split(split, weights, out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    freeze = json.loads((weights/'FREEZE.json').read_text())
    models = {}
    for name, digest in freeze['files'].items():
        if sha256((weights/name).read_bytes()).hexdigest() != digest:
            raise ValueError('model changed after freeze: '+name)
        m, rec = load(weights/name)
        models[f'gru{rec["passes"]}_s{rec["seed"]}'] = m
    # Same frozen four-pass weights evaluated at other depths; explicitly secondary.
    ablations = {f'{name}_eval{k}': (m, k) for name, m in models.items()
                 if m.passes == 4 for k in (1, 2, 8)}
    source_hashes = {str(p.relative_to(REPO)): sha256(p.read_bytes()).hexdigest()
                     for p in sorted((REPO/'experiments/dual_ldt_06').glob('*.py'))}
    rows, errors, cases = [], [], []
    checked, upper_failures = 0, 0
    start = perf_counter()
    for wid in world_ids(split):
        w = world(wid)
        all_eval = [evaluate(w, p) for p in PROGRAMS]
        feasible = [i for i, e in enumerate(all_eval) if e.feasible]
        for persona, theta in PERSONAS.items():
            actual = {i: all_eval[i].numerator(theta) for i in feasible}
            optimum = max(actual.values())
            t0 = perf_counter()
            uppers = tuple(count_upper_numerator(w, p, theta) for p in PROGRAMS)
            bound_cost = perf_counter()-t0
            for i in feasible:
                checked += 1
                if actual[i] > uppers[i]:
                    upper_failures += 1
                    raise AssertionError('unsound relaxation')
            cases.append({'world': wid, 'persona': persona,
                          'oracle_numerator': optimum, 'feasible': len(feasible)})
            methods = ['fixed', 'symbolic_upper', *models]
            # Stable per-case order avoids systematically favoring a warm method.
            methods.sort(key=lambda n: sha256(f'{wid}|{persona}|{n}'.encode()).digest())
            for method in methods:
                t0 = perf_counter()
                if method == 'fixed':
                    scores = [-float(i) for i in range(len(PROGRAMS))]
                elif method == 'symbolic_upper':
                    scores = list(uppers)
                else:
                    x = torch.tensor([input_features(w, theta, p) for p in PROGRAMS],
                                     dtype=torch.float32)
                    with torch.no_grad():
                        scores = (models[method](x) * SCALE).tolist()
                    errs = [abs(scores[i]-actual[i]/DENOMINATOR) for i in feasible]
                    errors.append({'world': wid, 'persona': persona, 'method': method,
                                   'mae': mean(errs), 'max_error': max(errs)})
                order = ordered_indices(scores)
                ordering_cost = perf_counter()-t0
                record = search(w, theta, order, uppers, optimum, ordering_cost, bound_cost)
                for row in record:
                    rows.append({'world': wid, 'persona': persona, 'method': method,
                                 'split': split, **row})
            # Pass-depth audit evaluates predictive error only, never chooses a checkpoint.
            x = torch.tensor([input_features(w, theta, p) for p in PROGRAMS], dtype=torch.float32)
            for name, (m, k) in ablations.items():
                with torch.no_grad():
                    predicted = (m(x, passes=k) * SCALE).tolist()
                errs = [abs(predicted[i]-actual[i]/DENOMINATOR) for i in feasible]
                errors.append({'world': wid, 'persona': persona, 'method': name,
                               'mae': mean(errs), 'max_error': max(errs), 'secondary': True})
        print('EVAL', split, wid, 'seconds', round(perf_counter()-start, 2), flush=True)
    raw = {'split': split, 'status': 'new_synthetic_worlds_new_value_students',
           'namespace': NAMESPACE, 'world_ids': list(world_ids(split)),
           'weights_sha256': freeze['files'], 'source_sha256': source_hashes,
           'cases': cases, 'rows': rows, 'prediction_errors': errors,
           'upper_bound_checks': checked, 'upper_bound_violations': upper_failures,
           'evaluation_seconds': perf_counter()-start}
    raw_sha = save(out/f'{split}_raw.json', raw)
    by_group, by_error = defaultdict(list), defaultdict(list)
    for row in rows:
        by_group[(row['method'], row['budget'])].append(row)
    for e in errors:
        by_error[e['method']].append(e)
    summary = []
    for (method, budget), group in sorted(by_group.items()):
        summary.append({'method': method, 'budget': budget, 'cases': len(group),
                        'mean_return': mean(r['return_numerator']/DENOMINATOR for r in group),
                        'mean_oracle_regret': mean(r['regret_numerator']/DENOMINATOR for r in group),
                        'oracle_attained': sum(r['exact_optimum_attained'] for r in group),
                        'exact_certificates': sum(r['exact_optimum_certified'] for r in group),
                        'mean_invalid_queries': mean(r['infeasible_proposals'] for r in group),
                        'mean_primitive_expansions': mean(r['primitive_expansions'] for r in group),
                        'mean_end_to_end_ms': 1000*mean(r['end_to_end_seconds'] for r in group),
                        'mean_ordering_ms': 1000*mean(r['ordering_seconds'] for r in group),
                        'mean_bound_ms': 1000*mean(r['bound_seconds'] for r in group)})
    error_summary = {name: {'world_persona_mae': mean(e['mae'] for e in group),
                            'max_absolute_error': max(e['max_error'] for e in group)}
                     for name, group in sorted(by_error.items())}
    result = {'split': split, 'world_count': len(world_ids(split)),
              'world_persona_cases': len(cases), 'raw_sha256': raw_sha,
              'source_sha256': source_hashes, 'weights_sha256': freeze['files'],
              'summary': summary, 'errors': error_summary,
              'upper_bound_checks': checked, 'evaluation_seconds': raw['evaluation_seconds']}
    save(out/f'{split}_summary.json', result)
    for r in summary:
        if r['budget'] in (4, 16, 64):
            print('RESULT', split, r['method'], 'K', r['budget'],
                  'regret', round(r['mean_oracle_regret'], 6), 'exact', r['oracle_attained'],
                  'ms', round(r['mean_end_to_end_ms'], 3), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('manifest', 'train', 'dev', 'report'))
    ap.add_argument('--out', type=Path, default=Path('experiments/dual_ldt_06/outputs'))
    ap.add_argument('--weights', type=Path, default=None)
    args = ap.parse_args()
    weights = args.weights or args.out/'weights'
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out/'manifest.json', manifest())
    if args.mode == 'train':
        train_all(weights)
    elif args.mode in ('dev', 'report'):
        evaluate_split(args.mode, weights, args.out)


if __name__ == '__main__':
    main()
