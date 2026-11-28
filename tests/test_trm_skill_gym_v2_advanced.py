import numpy as np
import torch
from research_gym.trm_skill_gym_v2 import subspace, student
from research_gym.trm_skill_gym_v2.core import distill_rows


def test_projector():
    q=subspace.orthogonal_random(12,4,3)
    z=torch.randn(2,3,12)
    c=torch.randn(12)
    out=subspace.remove_subspace(z,c,q)
    assert torch.allclose(q.T@q,torch.eye(4),atol=1e-6)
    assert torch.allclose((out-c)@q,torch.zeros(2,3,4),atol=2e-6)
    assert torch.allclose(out,subspace.remove_subspace(out,c,q),atol=2e-6)
    assert torch.allclose(subspace.remove_subspace(z,c,torch.eye(12)),c.expand_as(z),atol=1e-6)


def test_tied_matrix():
    layer=torch.nn.Linear(8,4)
    p=subspace.TiedWriteProbe(layer)
    assert torch.allclose((p.u*p.s)@p.v.T,layer.weight,atol=1e-6)
    x=torch.randn(2,3,8)
    m=torch.ones(2,3)
    clean=layer(x)
    p.begin(m)
    p.fitting=True
    with p.installed():
        assert torch.equal(layer(x),clean)
        assert torch.equal(layer(x),clean)
    p.fitting=False
    p.begin(m)
    p.selected=tuple(range(4))
    with p.installed():
        edited=layer(x)
        assert torch.allclose(edited,layer.bias+(p.means[0]/p.n[0])@layer.weight.T,atol=1e-6)
    assert torch.equal(layer(x),clean)
    assert p.call==1


def test_student_future_blind():
    a={"arm":"a","round":0,"proposal_index":0,"skill":"repair","incumbent":{},"config":{},
       "incumbent_dev_exact":.7,"rung0":{"dev_exact":.9},"decision":"promoted"}
    b=dict(a,rung0={"dev_exact":.1},decision="failed")
    r,s=list(distill_rows([a]))[0],list(distill_rows([b]))[0]
    assert np.array_equal(student.features(r),student.features(s))
    assert student.observed(r,"rung0")==.9
    assert student.observed(s,"rung1") is None
