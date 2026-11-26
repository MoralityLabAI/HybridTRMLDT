import pytest
torch=pytest.importorskip('torch')
from MCP.mcp05.train import fit


def episode(name,n):
    context=dict(task='repair test',subgoal='verify',verifier_status='',
        calls_left=12,tokens_left=512,dirty=False,failed=False,expanded=False)
    def step(n):
        slate=[{'action':'read_symbol','uri':f'mod::{i}','score':1.} for i in range(n)]
        slate.append({'action':'submit_to_solver'})
        return dict(observation=dict(context=context,slate=slate),
                    utility=[0.2]*n+[1.0])
    return dict(case=name,steps=[step(n),step(n+1)])


@pytest.mark.parametrize('arm',['recurrent','dual'])
def test_native_episode_mms(arm):
    torch.set_num_threads(1)
    data=[episode('a',2),episode('b',3)]
    original,l0=fit(data,arm,depth=2,epochs=3,seed=7)
    receipt={}
    model,l1=fit(data,arm,depth=2,epochs=3,seed=7,
        mms_options=dict(mib=1,mode='observe',sites='ff2',rank=4),
        mms_receipt=receipt)
    assert l0==l1
    for k,v in original.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    assert receipt['sites']['ff2']['bank']['hits']>0
    assert receipt['allocated_bytes']<=receipt['budget_bytes']
