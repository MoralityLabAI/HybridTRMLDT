import itertools
import pytest
import torch
from research_gym.trm_skill_gym_v2.hybrid import exact_graph, run_arm
from research_gym.trm_skill_gym_v2.runner import hard_graph
from research_gym.trm_skill_gym import tasks as T


def test_exact_graph_baseline():
    for n in range(1,6):
        edges=list(itertools.combinations(range(n),2))
        for bits in itertools.product((0,1),repeat=len(edges)):
            p=[[] for _ in range(n)]
            for (i,j),v in zip(edges,bits):
                if v:p[j].append(i)
            k=[]
            for ps in p:k.append(-1 if any(k[v]==1 for v in ps) else 1)
            assert exact_graph("deduction",p,None)[0]==k
            for q in range(n):
                r={q}
                for _ in range(n): r|={j for i in r for j in p[i]}
                assert exact_graph("retrieval",p,q)[0]==[1 if i in r and i!=q else -1 for i in range(n)]
    with pytest.raises(ValueError):
        exact_graph("deduction",[[1],[0]],None)


@pytest.mark.parametrize("skill",["deduction","retrieval"])
def test_bypass(skill):
    b=T.stack([hard_graph(skill,14,r) for r in range(8)])
    # No model object: any accidental neural computation will raise.
    for mode,rounds in [("exact",0),("hybrid",16)]:
        a,m=run_arm(None,None,skill,b,mode,rounds)
        assert a.mean()==1
        assert m["nn_instances"]==0 and m["model_flops_per_instance"]==0
        assert m["certificate_violations"]==0
