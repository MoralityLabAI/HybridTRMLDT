# MOLT-GOBLIN-03 preregistration

Registered 2026-11-28, before the discovery campaign. Builds on MOLT-GOBLIN-02 at `adf4e17`.
Status: development study. CPU only, free Inkling-2-Mini only, campaign ≤ 6 h. RAM: hard
3 GiB of resident memory summed over each driver and its worker processes. A watchdog
samples every 0.25 s and terminates the whole tree on breach, writing `ram_breach.json`.
A Windows Job Object also caps each process at 1.5 GiB of committed memory. A job-wide
commit limit was tried first and dropped: torch commits ~0.8 GiB of address space per
process on import while staying ~0.2 GiB resident, so four processes exceeded 3 GiB of
commit at ~1.3 GiB of RAM.

## Question

MOLT-GOBLIN-02 found that a soft four-module HRM mixture (B) beat single-block prefix
recurrence on dependency and selection tasks. Its router did not specialize, and top-1
routing at evaluation collapsed accuracy. Which computational ingredient produces the
advantage: parameters, FLOPs, module visits, the manager, separate module weights,
parallel composition, or loop forcing? Can a sparse or distilled single-path model keep it
at a quarter of the module visits?

## Tournament arms

The request referenced arms "A–K specified above", but no such list was supplied, so they
are defined here. All arms use width 32, 2 heads, 4 loops, prefix memory, loop forcing
(`loss=all`), 300 AdamW updates of 32 sequences, unless stated otherwise.

| Arm | Definition | Isolates |
| --- | --- | --- |
| A_single | one block | reference |
| B_hrm | 4 modules, slow/fast GRU manager, soft mixture (MOLT-02 f2) | replication |
| C_mix | 4 modules, uniform mixture, no manager | manager |
| D_tied | B with one shared block behind 4 type embeddings | separate module weights |
| E_param | one block, width matched to B's parameters (76) | parameters |
| F_visit | one block, 16 loops (= B's 16 module visits) | module visits / depth |
| G_flop | one block, 8 loops, width matched to B's measured forward FLOPs | FLOPs |
| H_sparse_st | B's manager, Gumbel straight-through top-1, load balance 0.01 | native sparse routing |
| I_anneal | B's manager, soft-to-hard temperature anneal, top-1 at evaluation | native sparse routing |
| J_sequential | 4 modules applied in series within each loop | parallel vs sequential |
| K_distill | A's architecture trained with labels + per-loop KL to B (frozen teacher, same family and seed) | distillation at 4 visits |

Controls: `A_cpu`, A with updates scaled by the measured development B/A train-CPU ratio
(process-CPU-matched). Loop-forcing factorial: `A_single_last`, `B_hrm_last`, `C_mix_last`
and `H_sparse_st_last` (final-loop loss) on the four principal families. If the best
development mixture is not B, `K_win` distills that mixture as well. Up to two Inky
finalists are added at freeze.

Accounting (measured with `FlopCounterMode` on proof_resolve): A 10.5 k params / 1.03 MFLOP
per example; B 51.7 k / 5.50 MFLOP, of which the HRM manager is 15.9 k params and
1.39 MFLOP; C 35.8 k / 4.11 MFLOP; E 51.6 k / 5.45 MFLOP; F 10.5 k / 4.13 MFLOP. Native
sparse arms are also measured on the hard-dispatch path (each module computed only on its
routed positions).

## Families

Development (campaign, rungs): proof_resolve (AND/OR dependency resolution),
experiment_portfolio (selection with prerequisites), revision_retract (revision with
retraction), cex_repair (counterexample-directed repair of a linear rule mod 5). proof_deps
from MOLT-02 enters finals only as a replication anchor. Repaired controls: local_map3
(window-3 lookup over digits 0..3, solvable in one pass) replaces shallow_copy and
local_map, which sat at the 1.000 ceiling. A first local_map3 version over digits 0..6
landed at the floor (A 0.016, B 0.051), so the alphabet was reduced before registration;
the registered version calibrates at A 0.965 / B 0.984 (length 12) and 0.20 / 0.25 at length 24.
Development calibration of the other families, A vs B exact: proof_resolve 0.414 / 0.504,
experiment_portfolio 0.418 / 0.504, revision_retract 0.719 / 0.789, cex_repair 0.418 / 0.691,
proof_deps 0.516 / 0.758, cumsum3 0.699 / 0.734. cumsum3 replaces the floor-bound cumsum (A: 0.699 at calibration).
**Finals-only family:** research_plan (prerequisite DAG with durations; label = earliest
finish time). It is never trained or shown to the proposer before the freeze. Calibration:
`receipts/calibration/calib_panel.jsonl` (seed 11, development roots).

Roots: TRAIN3 21 M, DEV3 25 M, HIDDEN3 29 M, all disjoint from the MOLT-02 ranges. Finals use
fresh HIDDEN3 roots, 256 per evaluation.

## Campaign

Inky Mini proposes over the grammar in `factory3.describe_grammar`: routing, sparsity
schedule, functional module objectives, auxiliary losses, composition, sharing,
distillation mode, capacity and loop forcing. Each proposal needs a discriminating control.
It sees development receipts, specialization measurements (module CKA, single-module
ablation, permutation, routing interventions, effective rank) and the list of tested
canonical configurations; it never sees data, labels or hidden roots. Duplicates are
rejected by canonical digest before training. Successive halving: rung 0 = seed 11,
rung 1 = 11, 23, rung 2 = 11, 23, 37. Promotion score = quality − 0.03·log2(inference
visits / 4), with quality = mean over development families of (exact − A) + 0.5·(OOD − A OOD).
Inky's task variants are used only after `verify_spec`. Training runs execute in one process
with 2 torch threads at below-normal priority. The operator asked to reduce CPU use at
12:20, so the planned 3 workers were cut to 1 before any campaign run completed.

## Freeze and finals

`finals3.freeze` writes `freeze.json` with its SHA-256 before any hidden run. Seeds: 11, 23,
37, 41, 53, 67, 79, 97 (eight paired training seeds; with 256 paired roots each). Finals run
as one below-normal-priority process, as the operator requested. Tier 1 runs first for
all eight seeds: arms A, B, C, D, E, F, H, I, K, the Inky finalists and any K_win/teacher,
on the four principal families plus research_plan. Tier 2 then runs everything else (G, J,
A_cpu, the loop-forcing factorial, proof_deps, local_map3, cumsum3). Both tiers are
seed-major, so a deadline-truncated run stays balanced across arms within a tier, and the
seed count actually completed is reported per contrast. Specialization
measurements, including gradient similarity, are taken on seeds 11 and 23 for mixture arms
on the principal families and proof_deps. Task-conditional routing comes from one model
per arm and seed (B, H, I; seeds 11, 23, 37), trained jointly on the four principal
families.

## Registered contrasts

Seed-level paired mean differences with t 95% intervals (n = number of completed seeds),
per family:

* P1 source decomposition: B, C, D, H, I, J minus A.
* P2 matched cost: B minus E (params), G (FLOPs), F (visits), A_cpu (process CPU).
* P3 structure: B minus C (manager), D (separate weights), J (parallel vs sequential).
* P4 native sparse routing: H and I minus B and minus A.
* P5 loop forcing: X minus X_last for A, B, C, H.
* P6 distillation: K minus A and B. Retained fraction (K − A) / (B − A) per family and
  pooled over families other than local_map3, bootstrap over seeds.
* P7 Inky finalists minus A and B.

Effect floor 0.05 exact. Interpretation rules, fixed in advance:

* Within P2, a matched control within the floor of B means the advantage is attributed to
  that resource.
* A P3 arm within the floor of B means that ingredient is not required.
* A sparse arm within the floor of B at 4 visits counts as an efficient replacement.
* Distillation counts as successful if pooled retention ≥ 0.5 with a CI lower bound > 0.

No multiplicity correction. This is a development tournament that informs the next
preregistration; promotion is not authorized.
