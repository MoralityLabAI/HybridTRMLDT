# MeTTa Frame Scaffolding

The starter repo does not parse or execute real MeTTa. It provides local frame types that a later interpreter bridge can emit.

## Frame classes

- `ExecutionFrame`: local before/after transition from code or a mechanic.
- `DeductionFrame`: monotone candidate pruning or conflict under a named rule.
- `RepairFrame`: diagnostic plus buggy and repaired code snippets.
- `RoutingFrame`: task signature, candidate skills, selected skill.

These frames do not require solved rollouts. They are intended to create dense training and evaluation signals before end-to-end success exists.

## Synthetic directive format

For bootstrapping, `research_gym.envs.metta_synthetic` reads small directive lines:

```text
!exec rule=investigate before=trust:1,evidence:0,heat:1,scene:2 after=trust:1,evidence:1,heat:2,scene:3 invariants=bounds,scene_progress
!exec rule=arith_add_heat before=heat:1,delta:2 after=heat:3,delta:2 invariants=arithmetic,bounds
!exec rule=precondition_fail_defuse before=heat:1,scene:2 after=heat:1,scene:2 invariants=precondition_failed,no_state_change passed=false
!deduce rule=ending_requires_low_heat before=secret:{possible,dead};heat:{3,4} after=secret:{dead};heat:{3,4} eliminated=secret:possible conflict=true soundness=env_sound_dead
!deduce rule=type_check_action before=action:{befriend,defuse,42};type:{Action,Number} after=action:{befriend,defuse};type:{Action} eliminated=action:42,type:Number conflict=false soundness=env_sound_dead
!route task=secret_ending candidates=befriend,investigate,defuse,rush chosen=defuse reason=heat_too_high
!repair diagnostic=unknown_symbol buggy="(call rush-fast state)" repaired="(call rush state)" passed=true
```

The expanded synthetic corpus includes:

- arithmetic and comparison execution frames
- type checking deduction frames
- precondition failure execution frames
- postcondition update execution frames
- routing among named skills
- repair of malformed rules
- deduction closure
- conflict and bottom detection

These are still interpreter-free examples. The goal is to stress the frame schema and evaluation targets before integrating a real MeTTa runtime.

Run:

```bash
python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta
```

## Role in the larger design

Metta mechanics produce frames. TRM-style models can train on execution, repair, and routing frames. LDT-style heads can train on deduction frames. VPD remains an instrumentation layer that can later label which weights, adapters, or modules implement these frame families.
