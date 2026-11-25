import unittest
import numpy as np
import torch
from .gym import *

class GymTests(unittest.TestCase):
    def test_permutations(self):
        for n in CAPACITIES:
            for mode in ('row','random','hilbert'):
                self.assertTrue(np.array_equal(np.sort(order(n,mode,19)),np.arange(n)))
    def test_hilbert_neighbors(self):
        p=[hilbert_xy(16,i) for i in range(256)]
        self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(p,p[1:])))
    def test_matched_roots(self):
        worlds=[World(16384,42,m) for m in ('row','random','hilbert')]
        self.assertEqual(list(worlds[0].roots(10,9)),list(worlds[2].roots(10,9)))
        self.assertTrue(np.array_equal(worlds[0].values,worlds[1].values))
        self.assertTrue(np.array_equal(worlds[0].stale,worlds[1].stale))
    def test_corruption(self):
        w=World(16384,42,'hilbert'); r=next(w.roots(1,9)); p=int(w.inv[2*r.target]); key=w.keys[r.target]
        self.assertEqual(w.probe(p,key,0)[0],w.values[r.target]); self.assertIsNone(w.probe(p,key,1)[0])
        w.stale[2*r.target:2*r.target+2]=[0.,1.]; p=int(w.inv[2*r.target+1]); self.assertEqual(w.probe(p,key,.5)[0],w.values[r.target])
    def test_training_and_budgets(self):
        torch.set_num_threads(1); w=World(16384,42,'hilbert'); model=train(w,LOCAL,7,8,1)
        self.assertTrue(16000<=sum(p.numel() for p in model.parameters())<=128000)
        for arm in ('scan','hilbert','adaptive'):
            for r in w.roots(4,99):
                result=rollout(w,r,arm,None if arm=='scan' else model,4,0)
                self.assertLessEqual(result['probes'],4); self.assertEqual(len(set(result['pages'])),result['probes']); self.assertEqual(result['records_read'],32*result['probes'])
                if result['success']: self.assertEqual(result['value'],w.values[r.target])
    def test_split(self):
        a=World(16384,10107,'hilbert'); b=World(16384,2000000+10700+16384,'hilbert'); self.assertFalse(np.array_equal(a.keys,b.keys))

if __name__=='__main__': unittest.main()
