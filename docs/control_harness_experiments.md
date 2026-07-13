# Cross-Project Control Harness Experiments

## Purpose

This benchmark asks where typed LDT/TRM control can help game-playing, control harnesses, and skill routing.
It first scans local projects for relevant interfaces, then evaluates deterministic source-inspired task cards.
The task cards are proxies. They preserve control distinctions from the source repositories but do not claim
native game or model performance.

## Decision contract

Each task exposes:

- latent TRM action scores;
- explicit LDT action scores;
- an environment-derived allowed set;
- a model-, experience-, or unknown-provenance candidate set;
- held-out utility used only for evaluation.

For action set $A$, let $A_{env} \subseteq A$ contain actions admitted by exact mechanics. Typed policies may
hard-eliminate only actions outside $A_{env}$. A model- or experience-derived set remains soft.

Confidence arbitration uses the TRM top-two margin

```text
margin = score_trm(a_1) - score_trm(a_2)
```

and defers to LDT when the margin is below a train-selected threshold. Typed confidence first restricts to
$A_{env}$, then applies the margin rule. This prevents a confident illegal proposal while retaining cheap TRM
decisions on clear preference tasks.

Counterfactual beam keeps the top $k$ TRM proposals, removes environment-invalid branches, and ranks survivors
by

```text
0.65 * score_trm(action) + 0.35 * score_ldt(action)
```

The beam width is selected on the train split. The skill router separately chooses an architecture for each
skill using train utility, exact-choice accuracy, unsafe rate, and deliberation cost.

## Applications

The current task cards are inspired by:

- GPTStoryworld secret routes and moral choices;
- AI_Diplomacy order legality and coalition forecasts;
- SmallControlHarness evidence attestation and semantic re-anchoring;
- BitVMArena settlement control profiles;
- StoryForge prisoner-dilemma choice gates;
- TheySing treaty channels and campaign-clock pressure;
- Blighted Galaxy strategic simulation ticks;
- CodexGameStudio agent scopes and skill delegation.

See `reports/hybrid_application_map.md` for the broader local shortlist and native adapter order.

## Current result

With 48 train and 48 held-out cases per application, seed 23 selects confidence margin 0.10 and beam width 2.
Typed confidence reaches 0.928 exact-choice accuracy with zero unsafe actions. The skill router reaches 0.935
accuracy, zero unsafe actions, and lower mean deliberation cost than typed confidence. Raw TRM remains selected
for morality and opponent-modeling tasks; typed confidence is selected for provenance, reachability, treaty,
and strategic-tick control.

These results support an architecture-selection hypothesis, not a universal hybrid: use exact certification
where the environment exposes a real invariant, and avoid forcing lattice proxy scores onto preference surfaces.

## Reproduce

```powershell
python -m research_gym.scripts.bench_control_harnesses `
  --projects-root C:\projects `
  --n-train 48 `
  --n-eval 48 `
  --seed 23
```

Outputs:

- `data/benchmarks/local_game_env_inventory.json`
- `data/benchmarks/control_harness_results.json`
- `reports/local_game_env_inventory.md`
- `reports/hybrid_application_map.md`
- `reports/control_harness_bench.md`
- `experiments/control_harness_structures/results.json`
- `experiments/control_harness_structures/training_notes.md`
