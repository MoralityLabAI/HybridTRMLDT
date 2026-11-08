import math,random,unittest
from clock import Step,gae,shape,discounted_return,spr
class ClockTests(unittest.TestCase):
    def assert_vector(self,a,b):
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):self.assertAlmostEqual(x,y,places=11)
    def test_monte_carlo_unit_clock(self):
        self.assert_vector(gae([Step(0,.2),Step(.5,.3),Step(1,.7,terminal=True)]),[1.3,1.2,.3])
    def test_true_terminal_masks_both_continuations(self):
        self.assert_vector(gae([Step(1,.2,terminal=True)],next_value=99,next_advantage=77),[.8])
    def test_empty_chain(self):self.assertEqual(gae([]),[])
    def test_chunk_continuation_200_cases(self):
        rng=random.Random(901)
        for _ in range(200):
            n=rng.randrange(2,40);cut=rng.randrange(1,n)
            xs=[Step(rng.random(),rng.random(),rng.uniform(.6,1),rng.random(),i==n-1) for i in range(n)]
            suffix=gae(xs[cut:])
            prefix=gae(xs[:cut],next_value=xs[cut].value,next_advantage=suffix[0])
            self.assert_vector(gae(xs),prefix+suffix)
    def test_naive_chunk_trace_reset_loses_credit(self):
        xs=[Step(0,0,.9,.8),Step(0,0,.9,.8),Step(1,0,terminal=True)]
        full=gae(xs);cut=gae(xs[:2],next_value=0)
        self.assertAlmostEqual(full[0]-cut[0],(.9*.8)**2)
    def test_stutter_refinement_200_cases(self):
        rng=random.Random(902)
        for _ in range(200):
            n=rng.randrange(1,25)
            old=[Step(rng.random(),rng.random(),.95,.8,i==n-1) for i in range(n)]
            new=[];anchors=[]
            for s in old:
                anchors.append(len(new));new.append(Step(0,s.value))
                for _ in range(rng.randrange(0,5)):new.append(Step(0,rng.random()))
                new.append(Step(s.reward,rng.random(),s.gamma,s.lam,s.terminal))
            full=gae(new);self.assert_vector(gae(old),[full[i] for i in anchors])
    def test_stutter_with_real_cost_changes_value(self):
        a=gae([Step(1,0,terminal=True)])[0]
        b=gae([Step(-.1,0),Step(1,0,terminal=True)])[0]
        self.assertAlmostEqual(a-b,.1)
    def test_bad_boundary_value_breaks_invariance(self):
        a=gae([Step(0,0,.9,.5),Step(1,.2,terminal=True)])[0]
        b=gae([Step(0,0,.9,.5),Step(0,.8),Step(1,.2,terminal=True)])[0]
        self.assertAlmostEqual(b-a,.9*(1-.5)*(.8-.2))
    def test_potential_telescope_200_cases(self):
        rng=random.Random(903)
        for _ in range(200):
            n=rng.randrange(1,30);r=[rng.random() for _ in range(n)]
            p=[rng.uniform(-1,1) for _ in range(n)]+[0.0]
            g=[rng.uniform(.8,1) for _ in range(n)]
            self.assertAlmostEqual(discounted_return(shape(r,p,g),g),discounted_return(r,g)-p[0],places=11)
    def test_spr_policy_reversal(self):
        a=.8*spr(True,100,100)+.2*spr(False,0,100)
        b=.75*spr(True,100,100)+.25*spr(False,99,100)
        self.assertGreater(b,a)
        self.assertGreater(spr(True,100,100),spr(False,99,100))
    def test_invalid_inputs(self):
        for kwargs in ({'reward':float('nan'),'value':0},{'reward':0,'value':0,'lam':1.1}):
            with self.assertRaises(ValueError):Step(**kwargs)
        with self.assertRaises(TypeError):Step(0,0,terminal='false')
        with self.assertRaises(ValueError):shape([1],[.2,.1],[1])
        with self.assertRaises(ValueError):spr(True,-1,1)
        with self.assertRaises(ValueError):spr(True,1,0)
if __name__=='__main__':unittest.main()
