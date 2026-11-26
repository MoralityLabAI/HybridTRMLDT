"""Development-only storyworld TRM MMS run, separate from frozen benchmark."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv
from research_gym.core.typed_soundness import SoundnessType
from research_gym.neural.trm import TRMProposer,TARGETS
from research_gym.neural.rollout import StoryExample,oracle_action,action_environment_sound,train_proposer,action_candidate_state


def main(a):
    if min(a.steps,a.width,a.visits,a.states)<1 or a.mms_mib<0:raise ValueError('invalid budget')
    torch.manual_seed(a.seed)
    env=CoupledStoryworldEnv()
    states=[s for s in env.all_states() if not env.terminal(s) and not env.target(s)]
    train=[s for s in states if s.trust<=0];dev=[s for s in states if s.trust==1]
    rng=random.Random(a.seed)
    rng.shuffle(train);rng.shuffle(dev)
    def examples(rows,split):
        return [StoryExample(f'{split}-{i}-{target}',target,s,6,split,split)
                for i,s in enumerate(rows[:a.states]) for target in TARGETS]
    train,dev=examples(train,'train'),examples(dev,'development')
    assert not ({r.group_id for r in train}&{r.group_id for r in dev})
    model=TRMProposer(latent_dim=a.width,recurrence_steps=a.visits)
    actions=[oracle_action(env,r) for r in train]
    provenance=[SoundnessType.ENV_SOUND_DEAD if action_environment_sound(env,r,act)
                else SoundnessType.UNKNOWN for r,act in zip(train,actions)]
    memory={}
    wall,cpu=time.perf_counter(),time.process_time()
    losses=train_proposer(model,train,actions,provenance,steps=a.steps,seed=a.seed,
          learning_rate=.003,mms_options=dict(mib=a.mms_mib,mode=a.mms_mode),mms_receipt=memory)
    sec,core=time.perf_counter()-wall,time.process_time()-cpu
    candidates=action_candidate_state(model.action_vocab)
    predictions=[model.propose(r.scenario,candidates,r.state).metadata['selected_action'] for r in dev]
    correct=sum(p==oracle_action(env,r) for p,r in zip(predictions,dev))
    a.out.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),a.out/'model.pt')
    (a.out/'receipt.json').write_text(json.dumps(dict(scope='development_only',seed=a.seed,
        train_rows=len(train),dev_rows=len(dev),dev_correct=correct,steps=len(losses),
        wall_seconds=sec,process_cpu_seconds=core,mms=memory or None,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--states',type=int,default=32)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--visits',type=int,default=6)
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--mms-mode',choices=('observe','dense','bpd','sham'),default='bpd')
    main(p.parse_args())
