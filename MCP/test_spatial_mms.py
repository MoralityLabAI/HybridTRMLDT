import copy
import pytest
torch=pytest.importorskip('torch')
from MCP.spatial_mms import SpatialMemory, valid_visits
from MCP.mcp06.gym import Controller, World, HIER, train
from MCP.mcp07.models import AdaptiveGeometryController
from MCP.mcp08.models import MCP08RecursiveController


def test_stable_keys_for_variable_visit_counts():
    from research_gym.neural.mms import MechanisticMemory
    l=torch.nn.Linear(4,4)
    m=MechanisticMemory(l,mib=.1,mode='observe',rank=2)
    x=torch.randn(1,4)
    with m.capture(torch.tensor([3]),0,1,2,key_stride=128):
        l(l(x)).sum().backward()
    l.zero_grad()
    with m.capture(torch.tensor([3]),1,1,3,key_stride=128):
        l(l(l(x))).sum().backward()
    assert m.bank.hits==1


def test_mcp06_training_receipt_and_checkpoint_shape():
    torch.set_num_threads(1)
    w=World(16384,31,'row')
    report={}
    model=train(w,HIER,17,8,3,mms_options=dict(mib=.1,mode='observe'),mms_receipt=report)
    assert report['updates']==3
    assert isinstance(model.fast,torch.nn.GRUCell)
    assert isinstance(model.slow,torch.nn.GRUCell)
    assert report['allocated_bytes']<=report['budget_bytes']
    assert set(model.state_dict())==set(Controller().state_dict())


@pytest.mark.parametrize('model_class',[AdaptiveGeometryController,MCP08RecursiveController])
def test_spatial_native_state_rollout(model_class):
    torch.manual_seed(17);torch.set_num_threads(1)
    model=model_class()
    base=copy.deepcopy(model)
    m=SpatialMemory(model,dict(mib=.1,mode='observe'))
    keys=torch.arange(4);obs=torch.zeros(4,model.cfg.d_obs);proj=torch.zeros(4,dtype=torch.long)
    for step in range(4):
        model.zero_grad();base.zero_grad()
        def run(net):
            state=net.initial_state(4);out=net.step_recurrent(obs,state,t=0)
            out=net.step_recurrent(obs,out[-1],t=1,active_proj=proj)
            return sum(v.square().mean() for v in out[:-1])
        with m.capture(keys,{'fast_gru':2,'proj_gate':1}):
            run(model).backward()
        run(base).backward()
        for a,b in zip(model.parameters(),base.parameters(),strict=True):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
    assert isinstance(model.fast_gru,torch.nn.GRUCell)
    assert m.memory.pool.used<=m.memory.pool.budget_bytes
    assert valid_visits(torch.tensor([[True,False,True]]),3)==1
