"""Recompute HUG-CAL-02 calibration acceptance from raw factor/control/state arrays.

The worker must first freeze source-hash manifests, instrumentation and control
selection. This program reads both splits only when evaluating the sealed gate;
it never tunes candidates or updates matrices. Save its PASS/STOP output with
the source NPZ hashes. Actual data provenance still needs independent review.

State names: {core}__{selection|verification}__{early|late}
Target names: {core}_factor{id}__u and __v
Controls:    {core}_factor{id}__{family}__{0|1|2|3}__u/__v/__scale
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

try:
    from .control_repair import RankOneControl, window_report
    from .confirm_eval import PANEL
except ImportError:
    from control_repair import RankOneControl, window_report
    from confirm_eval import PANEL

MODEL = "tomg-group-umd/huginn-0125"
REV = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
FACTORS_SHA = "86f90a1c9403240e4f69fdfad1f9f88a2410a85edfb2cfef2302fc74959f4a50"
SHA_HEX = re.compile(r"^[0-9a-f]{64}$")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def _audit_sources(manifest: dict) -> tuple[bool,list[str]]:
    problems = []
    groups = {
        'selection_source_hashes': 120,
        'verification_source_hashes': 120,
        'confirmation_source_hashes': 480,
    }
    seen = set()
    for field, count in groups.items():
        hashes = manifest.get(field)
        if not isinstance(hashes,list) or len(hashes)!=count:
            problems.append(f'{field}: expected {count} hashes')
            continue
        if any(not isinstance(s,str) or not SHA_HEX.fullmatch(s) for s in hashes):
            problems.append(f'{field}: bad SHA-256 encoding')
        if len(set(hashes))!=count:
            problems.append(f'{field}: internal duplicate source')
        if seen.intersection(hashes):
            problems.append(f'{field}: cross-split source reuse')
        seen.update(hashes)
    old_hashes = manifest.get('prior_source_hashes')
    if not isinstance(old_hashes, list) or not old_hashes:
        problems.append('prior_source_hashes: old source-hash inventory required')
    elif seen.intersection(old_hashes):
        problems.append('new split overlaps prior sources')
    for field in ('capture_split_precommitted','verification_frozen','same_batch_and_seed',
                  'zero_edit_bitwise','original_weight_forward','no_confirmation_inspection'):
        if manifest.get(field) is not True:
            problems.append(f'{field}: required true instrument/provenance receipt')
    if manifest.get('model')!=MODEL or manifest.get('model_revision')!=REV:
        problems.append('model revision mismatch')
    if manifest.get('factor_sha256')!=FACTORS_SHA:
        problems.append('factor hash mismatch')
    return not problems, problems


def verify_gate(manifest: dict, states_path: Path, controls_path: Path,
                original_factors_path: Path, family: str,
                *, expected_factors_sha: str = FACTORS_SHA) -> dict:
    if family not in ('write_rotation','read_rotation'):
        raise ValueError('unknown family')
    passed_sources, problems = _audit_sources(manifest)
    if _sha(original_factors_path) != expected_factors_sha:
        problems.append('original factors.npz byte checksum disagrees with pinned source')
    results = {}
    with np.load(states_path, allow_pickle=False) as states, np.load(controls_path, allow_pickle=False) as ctrls:
        for key in PANEL:
            core = key.split('_')[0]
            selected = {w: states[f'{core}__selection__{w}'] for w in ('early','late')}
            verified = {w: states[f'{core}__verification__{w}'] for w in ('early','late')}
            target = RankOneControl(ctrls[key+'__u'], ctrls[key+'__v'], 'factor', 0)
            candidates=[]
            for i in range(4):
                prefix = f'{key}__{family}__{i}__'
                candidates.append(RankOneControl(ctrls[prefix+'u'],ctrls[prefix+'v'],family,i,
                                                 float(ctrls[prefix+'scale'])))
            family_problems=[]
            normu = np.linalg.norm(target.u)
            v_unit = target.v/np.linalg.norm(target.v)
            for c in candidates:
                u_cos = float(c.u@target.u/(np.linalg.norm(c.u)*normu))
                v_cos = float(c.v@v_unit/np.linalg.norm(c.v))
                if family=='write_rotation':
                    if abs(u_cos)>1e-6 or not np.allclose(c.v,target.v,rtol=1e-10,atol=1e-12):
                        family_problems.append(f'control {c.index} breaks same-input-gate orthogonal-write contract')
                    if not np.isclose(np.linalg.norm(c.u)*abs(c.scale),normu,rtol=1e-8):
                        family_problems.append(f'control {c.index} has different immediate write norm')
                elif abs(v_cos)>.95+1e-8:
                    family_problems.append(f'control {c.index} too close to original input gate')
            for i,a in enumerate(candidates):
                for b in candidates[i+1:]:
                    if family=='write_rotation':
                        cos = abs(float(a.u@b.u/(np.linalg.norm(a.u)*np.linalg.norm(b.u))))
                        if cos>1e-6:
                            family_problems.append(f'write controls {a.index}/{b.index} not orthogonal')
                    else:
                        cos = abs(float(a.v@b.v/(np.linalg.norm(a.v)*np.linalg.norm(b.v))))
                        if cos>.99:
                            family_problems.append(f'read controls {a.index}/{b.index} too similar')
            selected_reports=[window_report(target,c,selected) for c in candidates]
            verified_reports=[window_report(target,c,verified) for c in candidates]
            eligible = (not family_problems and
                        all(z['accepted_both_windows'] for z in selected_reports) and
                        all(z['accepted_both_windows'] for z in verified_reports))
            if not eligible:
                problems.append(key+': failed control gate')
            results[key]={
                'eligible':eligible,
                'construction_errors':family_problems,
                'selected_joint_passed':sum(r['accepted_both_windows'] for r in selected_reports),
                'verification_joint_passed':sum(r['accepted_both_windows'] for r in verified_reports),
                'selection':selected_reports,
                'verification':verified_reports,
            }
    eligible=[k for k,v in results.items() if v['eligible']]
    return {
        'status':'PASS' if not problems and len(eligible)==len(PANEL) else 'STOP',
        'control_family':family,
        'eligible_factors':eligible,
        'source_split_disjoint':passed_sources,
        'verification_frozen':manifest.get('verification_frozen') is True,
        'panel_n':len(PANEL),
        'source_manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        'states_npz_sha256':_sha(states_path),
        'controls_npz_sha256':_sha(controls_path),
        'original_factors_sha256':_sha(original_factors_path),
        'problems':sorted(set(problems)),
        'factor_results':results,
        'warning':'CPU gate certifies saved arrays/manifest assertions, not whether GPU hooks captured authentic model activations.',
    }


def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--original-factors', type=Path, required=True)
    parser.add_argument('--family', choices=('write_rotation','read_rotation'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args(argv)
    result=verify_gate(json.loads(args.manifest.read_text()),args.states,args.controls,
                       args.original_factors,args.family)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['status'], len(result['eligible_factors']), '/ 8 matched factors')
    return 0 if result['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
