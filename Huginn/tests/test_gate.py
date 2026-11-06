from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np

from Huginn.gate_evals import PANEL,MODEL,REV,FACTORS_SHA,verify_gate,_sha
from Huginn.control_repair import rotated_write_controls


def hashes(offset,n):
    return [hashlib.sha256(str(i).encode()).hexdigest() for i in range(offset,offset+n)]


class GateReplayTests(unittest.TestCase):
    def _make(self,p:Path):
        rng=np.random.default_rng(101)
        state={}
        ctrls={}
        for core in ('core1','core3'):
            for s in ('selection','verification'):
                for window in ('early','late'):
                    state[f'{core}__{s}__{window}']=rng.normal(size=(256,7))
        for key in PANEL:
            u=rng.normal(size=9)
            v=rng.normal(size=7)
            ctrls[key+'__u']=u
            ctrls[key+'__v']=v
            for c in rotated_write_controls(u,v,4,seed=77):
                prefix=f'{key}__write_rotation__{c.index}__'
                ctrls[prefix+'u']=c.u
                ctrls[prefix+'v']=c.v
                ctrls[prefix+'scale']=np.array(c.scale)
        np.savez(p/'states.npz', **state)
        np.savez(p/'controls.npz',**ctrls)
        np.savez(p/'factors.npz', dummy=np.arange(5))
        manifest={
            'model':MODEL,'model_revision':REV,'factor_sha256':FACTORS_SHA,
            'selection_source_hashes':hashes(0,120),
            'verification_source_hashes':hashes(120,120),
            'confirmation_source_hashes':hashes(240,480),
            'prior_source_hashes':hashes(720,7),
            'capture_split_precommitted':True,'verification_frozen':True,
            'same_batch_and_seed':True,'zero_edit_bitwise':True,
            'original_weight_forward':True,'no_confirmation_inspection':True,
        }
        return manifest

    def test_synthetic_algebraic_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); manifest=self._make(p)
            receipt=verify_gate(manifest,p/'states.npz',p/'controls.npz',p/'factors.npz',
                                'write_rotation',expected_factors_sha=_sha(p/'factors.npz'))
            self.assertEqual(receipt['status'],'PASS')
            self.assertEqual(len(receipt['eligible_factors']),8)
            self.assertTrue(all(v['verification_joint_passed']==4 for v in receipt['factor_results'].values()))

    def test_split_collision_and_instrument_failure_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); manifest=self._make(p)
            manifest['confirmation_source_hashes'][0] = manifest['selection_source_hashes'][0]
            manifest['zero_edit_bitwise']=False
            receipt=verify_gate(manifest,p/'states.npz',p/'controls.npz',p/'factors.npz',
                                'write_rotation',expected_factors_sha=_sha(p/'factors.npz'))
            self.assertEqual(receipt['status'],'STOP')
            self.assertTrue(any('cross-split' in s for s in receipt['problems']))
            self.assertTrue(any('zero_edit' in s for s in receipt['problems']))

    def test_source_bytes_mismatch_stops_even_when_norms_match(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); manifest=self._make(p)
            receipt=verify_gate(manifest,p/'states.npz',p/'controls.npz',p/'factors.npz',
                                'write_rotation',expected_factors_sha='bad')
            self.assertEqual(receipt['status'],'STOP')
            self.assertTrue(any('checksum' in s for s in receipt['problems']))

if __name__ == '__main__':
    unittest.main()
