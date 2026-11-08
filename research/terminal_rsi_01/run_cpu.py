import json
from clock import Step,gae,spr
factor=.99*.95
rows=[{'observation_padding_tokens':p,'event_clock_credit':factor**9,
       'token_clock_credit':factor**(9+p)} for p in (0,100,1000,10000)]
result={'status':'finite_input_reference_only','training_run':False,'model_loaded':False,
        'observation_padding':rows,
        'spr_counterexample':{
            'divisor':100,'success_pass_count':100,
            'policy_a':{'strict_success_probability':.8,'failure_pass_count':0,'expected_reward':.8*spr(True,100,100)},
            'policy_b':{'strict_success_probability':.75,'failure_pass_count':99,'expected_reward':.75*spr(True,100,100)+.25*spr(False,99,100)}}}
print(json.dumps(result,indent=2,allow_nan=False))
