import copy
import pytest
torch=pytest.importorskip('torch')
from research_gym.neural.mms import ExposedGRUCell,MechanisticMemory
from MCP.mcp02.models import Router
from MCP.mcp02 import features as FE
from MCP.mcp02.mms import router_memory


def test_exposed_gru_parity():
    torch.manual_seed(31)
    src=torch.nn.GRUCell(14,32);ex=ExposedGRUCell(copy.deepcopy(src))
    x,h=torch.randn(5,14),torch.randn(5,32)
    a,b=src(x,h),ex(x,h)
    torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(src.weight_hh.grad,ex.recurrent.weight.grad,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('config',[dict(arch='looped',passes=3),dict(arch='trm',cycles=2,inner=2),
      dict(arch='trm_ldt',cycles=2,inner=2),dict(arch='hrm',high_cycles=2,low_steps=2)])
def test_router_dynamic_rows_and_native_gradient(config):
    torch.manual_seed(42);torch.set_num_threads(1)
    model=Router(dict(d=16,heads=4,**config));base=copy.deepcopy(model)
    memory,counts=router_memory(model,dict(mib=.2,mode='observe'),FE.SEQ_LEN)
    name=next(iter(memory.sites));mem=memory.sites[name];visits=counts[name]
    for step,width in enumerate((8,10)):
        ids=torch.ones((4,width),dtype=torch.long)*FE.OFFSET
        seg=torch.zeros_like(ids);struct=torch.zeros(4,FE.STRUCT_DIM)
        model.zero_grad();base.zero_grad()
        def objective(net):
            a,b=net(ids,seg,struct);return a.square().mean()+(b.square().mean() if b is not None else 0)
        with mem.capture(torch.arange(4),step,visits//2,visits,extra_visits=visits):
            (objective(model)+objective(model)).backward()
        (objective(base)+objective(base)).backward()
        for a,b in zip(model.parameters(),base.parameters(),strict=True):
            if a.grad is not None: assert torch.equal(a.grad,b.grad)
    assert mem.bank.hits==4
    assert memory.pool.used<=memory.pool.budget_bytes
