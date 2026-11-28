import itertools
import json
import pytest
import torch
from research_gym.trm_skill_gym_v2.core import paired_rank, strict_selected, graph_certificate, research_values, distill_rows
from research_gym.trm_skill_gym_v2.runner import hard_graph, cert_report, pools, CAL, TEST
from research_gym.trm_skill_gym import tasks as T


def test_pairing():
    with pytest.raises(ValueError):
        paired_rank({"easy":{"exact":1,"flops":1}},["easy","hard"])
    assert paired_rank({"easy":{"exact":.9,"flops":4},"hard":{"exact":.9,"flops":4}},["easy","hard"]) > paired_rank({"easy":{"exact":1,"flops":1},"hard":{"exact":.4,"flops":1}},["easy","hard"])


@pytest.mark.parametrize("text",['{"selected":[true]}','{"selected":[1,1]}','{"selected":[2]}','{"selected":[]}', '{"selected":[1],"selected":[0]}','x {"selected":[1]}','{"selected":[1],"extra":0}'])
def test_parser(text):
    with pytest.raises(ValueError):
        strict_selected(text,{0,1},True)


def test_parser_valid():
    assert strict_selected('{"selected":[1]}',{0,1},True)==[1]


def test_all_small_dags():
    for n in range(1,6):
        edges=list(itertools.combinations(range(n),2))
        for bits in itertools.product((0,1),repeat=len(edges)):
            parents=[[] for _ in range(n)]
            for (a,b),v in zip(edges,bits):
                if v: parents[b].append(a)
            lab=[]
            for ps in parents: lab.append(not any(lab[j] for j in ps))
            for depth in (1,n):
                cert,_=graph_certificate("deduction",parents,rounds=depth)
                assert all(c==0 or (c==1)==v for c,v in zip(cert,lab))
                if depth==n: assert all(cert)
            for q in range(n):
                reach={q}
                for _ in range(n): reach|={j for i in reach for j in parents[i]}
                for depth in (1,n+1):
                    cert,_=graph_certificate("retrieval",parents,q,depth)
                    assert all(c==0 or (c==1)==(i in reach and i!=q) for i,c in enumerate(cert))
                    if depth==n+1: assert all(cert)


@pytest.mark.parametrize("skill",["retrieval","deduction"])
def test_hard_graph(skill):
    for depth in (4,8,12,14):
        b=T.stack([hard_graph(skill,depth,r) for r in range(8)])
        assert torch.all(T.exact(skill,b,b["label"])==1)
        report=cert_report(skill,{"x":b})["x"]
        assert report["16"]["violations"]==0
        assert report["16"]["symbolic_exact"]==1
        assert report["1"]["coverage"]<1


def test_research():
    p=[.5,.5]
    l=[[.9,.1],[.2,.8]]
    q1=research_values(p,l,[.05,.05],1)
    q2=research_values(p,l,[.05,.05],2)
    assert all(q2[k]>=q1[k]-1e-12 for k in q1)
    assert max(q2.values())<=1


def test_distill():
    a={"arm":"a","round":1,"proposal_index":0,"skill":"research","rung1":{"dev_exact":.9},"decision":"promoted"}
    row=list(distill_rows([a]))[0]
    assert row["split"]=="test" and "rung1" not in row["input"]
    assert row["target"]["rung1"]["dev_exact"]==.9
    with pytest.raises(ValueError):
        list(distill_rows([a,a]))
