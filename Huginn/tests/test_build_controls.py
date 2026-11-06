import tempfile
import unittest
from pathlib import Path
import numpy as np
from Huginn.build_controls import build
from Huginn.confirm_eval import PANEL

class ControlBuilderTests(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory()
        self.root=Path(self.dir.name)
        vec={}
        for name in PANEL:
            vec[name+'__u']=np.array([1.,0.,0.,0.,0.,0.])
            vec[name+'__v']=np.array([1.,0.,0.,0.,0.,0.])
        self.vectors=self.root/'vectors.npz'
        np.savez(self.vectors,**vec)
        rng=np.random.default_rng(0)
        self.states=self.root/'states.npz'
        np.savez(self.states,**{f'{core}__selection__{w}':rng.standard_normal((240,6))
                    for core in ('core1','core3') for w in ('early','late')})
    def tearDown(self): self.dir.cleanup()
    def test_deterministic_and_write_controls(self):
        kw=dict(n_candidates=24,seed=21)
        a=build(self.vectors,self.states,self.root/'a.npz',self.root/'a.json',**kw)
        b=build(self.vectors,self.states,self.root/'b.npz',self.root/'b.json',**kw)
        with np.load(self.root/'a.npz') as x, np.load(self.root/'b.npz') as y:
            self.assertEqual(set(x.files),set(y.files))
            for key in x.files: np.testing.assert_array_equal(x[key],y[key])
            for name in PANEL:
                self.assertIn(name+'__write_rotation__3__u',x.files)
        self.assertEqual(a['status'],'SELECTION_ONLY')
        self.assertEqual(a['read_all_eight_ready'],b['read_all_eight_ready'])
    def test_rejects_verification_data(self):
        np.savez(self.states,core1__verification__early=np.ones((4,6)))
        with self.assertRaisesRegex(ValueError,'seal breach'):
            build(self.vectors,self.states,self.root/'c.npz',self.root/'c.json',n_candidates=5)
    def test_rejects_too_small_pool(self):
        with self.assertRaises(ValueError):
            build(self.vectors,self.states,self.root/'c.npz',self.root/'c.json',n_candidates=3)

if __name__=='__main__': unittest.main()
