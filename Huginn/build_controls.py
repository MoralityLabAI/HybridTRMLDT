"""Selection-only HUG-CAL-02 control builder. Optional CUDA projections; no model loading.

Inputs: an independently exported factor vector NPZ, with keys
`core1_factor19__u`, `core1_factor19__v`, etc., and a state NPZ holding
`core1__selection__early/late` (same for core3). Verification states must
NOT be present in the input to this process. Outputs are immutable proposal
matrices and selection-only diagnostics. This script does not issue PASS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .confirm_eval import PANEL
    from .control_repair import (RankOneControl, read_rotation_pool,
                                 rotated_write_controls, select_joint_controls,
                                 window_report)
except ImportError:
    from confirm_eval import PANEL
    from control_repair import (RankOneControl, read_rotation_pool,
                                rotated_write_controls, select_joint_controls,
                                window_report)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(vector_path: Path, states_path: Path, out_path: Path, report_path: Path,
          *, n_candidates: int = 512, seed: int = 17, backend: str = 'auto') -> dict:
    if n_candidates < 4:
        raise ValueError('at least four proposed read controls required')
    output: dict[str, np.ndarray] = {}
    ledger: dict[str, dict] = {}
    with np.load(vector_path, allow_pickle=False) as vectors, np.load(states_path, allow_pickle=False) as states:
        if any('__verification__' in name or '__confirmation__' in name for name in states.files):
            raise ValueError('seal breach: selection-only builder received verification/confirmation arrays')
        for name in PANEL:
            core = name.split('_')[0]
            xs = {w: np.asarray(states[f'{core}__selection__{w}']) for w in ('early', 'late')}
            u, v = np.asarray(vectors[name+'__u']), np.asarray(vectors[name+'__v'])
            target = RankOneControl(u, v, 'factor', 0)
            output[name+'__u'] = u
            output[name+'__v'] = v
            write_seed = seed + 100 * list(PANEL).index(name)
            read_seed = seed + 1000 + 100 * list(PANEL).index(name)
            w_controls = rotated_write_controls(u, v, seed=write_seed)
            trace: list[dict] = []
            r_pool = read_rotation_pool(u, v, xs, count=n_candidates,
                                        seed=read_seed,
                                        backend=backend, candidate_trace=trace)
            r_controls, r_reports = select_joint_controls(target, r_pool, xs,
                                                           backend=backend, candidate_trace=trace)
            ledger[name] = {'read_candidates_generated': len(r_pool),
                            'read_candidate_seed': read_seed,
                            'write_control_seed': write_seed,
                            'read_candidate_diagnostics': trace,
                            'read_selection_passed':len(r_controls),
                            'read_pool_success':len(r_controls)==4,
                            'read_selected_candidate_ids':[c.index for c in r_controls],
                            'read_selected_reports':r_reports,
                            'write_selection_reports':[window_report(target,c,xs) for c in w_controls]}
            for family, selected in [('write_rotation',w_controls),('read_rotation',r_controls)]:
                for i,c in enumerate(selected):
                    prefix=f'{name}__{family}__{i}__'
                    output[prefix+'u'] = np.asarray(c.u)
                    output[prefix+'v'] = np.asarray(c.v)
                    output[prefix+'scale'] = np.asarray(c.scale)
    # This output can contain fewer than four R controls: the external gate must STOP.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,**output)
    report={'status':'SELECTION_ONLY', 'selection_states_sha256':digest(states_path),
            'factor_vectors_sha256':digest(vector_path), 'controls_sha256':digest(out_path),
            'candidate_count_per_factor':n_candidates,'seed':seed,
            'read_all_eight_ready':all(v['read_pool_success'] for v in ledger.values()),
            'factors':ledger,
            'notice':'No verification states accessed; no acceptance or causal verdict.'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--factor-vectors',type=Path,required=True)
    p.add_argument('--selection-states',type=Path,required=True)
    p.add_argument('--controls-out',type=Path,required=True)
    p.add_argument('--report-out',type=Path,required=True)
    p.add_argument('--candidates',type=int,default=512)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--backend',choices=('auto','numpy','cuda'),default='auto')
    a=p.parse_args(argv)
    report=build(a.factor_vectors,a.selection_states,a.controls_out,a.report_out,
                 n_candidates=a.candidates,seed=a.seed,backend=a.backend)
    print('Selection complete; read controls available for all factors:',report['read_all_eight_ready'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
