from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Huginn.confirm_eval import PANEL, evaluate


class ConfirmationReceiptTests(unittest.TestCase):
    def _create_data(self, folder: Path):
        n, s = 42, 3
        arrays = {}
        for key in PANEL:
            prefix = key + "__write_rotation__"
            clean = np.full((n,s,2), -1.2)
            factor = clean.copy(); factor[:,:,0] -= .18
            ctrls = np.broadcast_to(clean, (4,n,s,2)).copy()
            arrays[prefix+'clean'] = clean
            arrays[prefix+'factor'] = factor
            arrays[prefix+'controls'] = ctrls
        np.savez(folder/'outcomes.npz', **arrays)
        (folder/'gate.json').write_text(json.dumps({
            'status':'PASS', 'control_family':'write_rotation',
            'eligible_factors':list(PANEL), 'source_split_disjoint':True,
            'verification_frozen':True,
        }))

    def test_gated_eight_factor_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); self._create_data(p)
            result = evaluate(p/'outcomes.npz', p/'gate.json', 'write_rotation',
                              n_boot=200, n_perm=999)
            self.assertEqual(result['n_hypotheses'], 8)
            self.assertTrue(all(z['interaction_holm_rejects_zero'] for z in result['results'].values()))
            self.assertTrue(result['results']['core1_factor19']['active_harmful_and_95pct_excludes_zero'])
            self.assertTrue(result['results']['core1_factor19']['inactive_90pct_equivalent_to_zero'])

    def test_rejects_missing_or_incomplete_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); self._create_data(p)
            receipt=json.loads((p/'gate.json').read_text()); receipt['status']='STOP'
            (p/'gate.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'PASS'):
                evaluate(p/'outcomes.npz',p/'gate.json','write_rotation',n_boot=100,n_perm=99)
            receipt['status']='PASS'; receipt['eligible_factors']=list(PANEL)[:-1]
            (p/'gate.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'all eight'):
                evaluate(p/'outcomes.npz',p/'gate.json','write_rotation',n_boot=100,n_perm=99)

if __name__ == '__main__':
    unittest.main()
