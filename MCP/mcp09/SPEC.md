# MCP-09 — Prospective Recursive Planning (development draft)

Parent: MCP-08 sealed evidence commit 8b90fc78eabd15a4c526aa5b7713fe921507c875. Preserve MCP-07/08 and all their results byte-for-byte. MCP-08 Gate 2 remains FAIL. This branch is NOT a frozen protocol and no registered MCP-09 claims are authorized.

## A. Integrity audit and paired control

MCP-08 can execute randomly selected unretrieved options via execute_ground_truth_action; its state key omits steps and route history although terminal counterfactual validity depends on action order; its final verifier does not independently enforce act milestones. Audit these effects on fresh development worlds and, separately, on the old sealed worlds as retrospective diagnostics only. Replaying an archived route must check each local verifier and the final five conditions. Audit generated witness solvability separately from policy success. Do not infer unreachability from search exhaustion. Store actual routes and lock decisions, which old episode receipts do not retain.

Implemented here: a development-only, fixed-medium frozen-retriever adapter and bounded precommit planner (depth 0 vs depth 2), strict route replay, unknown-vs-dead handling, branch lock enforcement, charged retrieval and verification, paired smoke runner, dependency-free tests. The adapter intentionally resets retrieval state for each query so both reference arms isolate planning; it is not an MCP-08 full reproduction. The reference planner uses route-specific recursive search without dead-state memoization. Cache only retrieved records, charge hits, log lookahead and actual commits separately. Its recursive stack differs from MCP-08's eight-frame stack: account its actual memory and match restoration policy before registered comparisons. Do not use hidden world nodes, reachability, option IDs/records or witnesses to predict actions, except public option queries and a verifier accepting a proposed route. The strict verifier may read native truth only for the proposed route. Missing retrieval is unknown; finite lookahead with a live prefix is not terminal viability.

Commands from repository root:
    python -m unittest MCP.mcp09.test_planner
    python -m MCP.mcp09.smoke --seed 107 --capacity 32768
    python -m MCP.mcp09.smoke --seed 107 --capacity 262144 --out MCP/results/mcp09/smoke/paired_256k.json

## B. Learned prospective operator (next implementation)

Keep the frozen MCP-08 controller as reference. For each retrieved candidate, encode the observable action and current fast/slow state, use a shared action-conditioned recurrent transition with latent width 24–32 and an explicitly charged unroll of 1, 2 or 4 steps. Predict local legality, irreversible risk and conditional eventual success, plus an uncertainty/abstention head. Train on fresh 16K worlds and only on prefixes accessible to the policy. Train labels may use the isolated exact verifier or bounded offline search, with PROVEN_DEAD / VERIFIED_SOLVED / UNKNOWN outcomes; mask unknown terminal labels, separately model censoring. Audit label leakage and train/eval family overlap. Compare an equal-parameter static risk MLP, shuffled action head, depth-0 head, deterministic charged lookahead, hardened MCP-08, fixed-medium scratch, MCP-07, beam and symbolic controls. Charge every speculative read, verifier call, transition update, cache entry and stack frame. Separate terminal verifier charge and count wall time; no oracle calls hidden in policy. Match total resource caps and report paired Pareto frontiers. On new test worlds require the learned arm to outperform the strongest hardened deterministic control at matched cost, not just historic MCP-08.

## C. Retrieval router (separate factorial)

Train a lightweight dispatch head using only cheap observed geometry/probe features and verification failure history: fast MCP-07-like path, escalated MCP-08 retrieval. No ground-truth family label at inference. Count both arms' active parameters/storage if keeping both resident; a dual 54K model cannot claim a 65K model budget. Include deterministic escalation, always fast, always deep, MCP-08, and the frozen original MCP-07. Per-family paired outcomes and reads; preserve easy-family ceiling. Train exclusively at 16K. Use the same corruption and memory/layout per root across arms.

## D. Draft gates (freeze only after independent smoke)

Fresh namespaces, disjoint from MCP-08 and development. Three model seeds, 144 paired 32K worlds and 144 paired 256K worlds with 10% corruption and balanced horizon buckets; clean control; 1,152 paired 256K synthetic roots across eight families. No tuning on registered data. Hierarchical bootstrap by training seed and paired world/root with all per-seed effects, CIs and failures preserved. Historical absolute targets: 32K >=70%; 256K Storyworld >=55%; long bucket >=50%; 256K retrieval >=70%; irreversible lock failures <= half of hardened control at matched budgets; learned arm >=5pp over deterministic lookahead, positive paired interval; active learned parameters <=65,536; peak RSS <2GiB CPU only. Absolute targets are secondary to direct paired comparisons on the fresh benchmark. Track the fraction of worlds with a strictly verified route offline and report intent-to-treat over all generated worlds; do not discard failures. Never treat unknown as dead. Report strict and legacy completion separately.

## E. Mechanistic follow-up

Probe slow state for progress beyond elapsed steps, remaining budget, clue/projector deficit and halt logit. Fit nuisance regressors on train only, test cross-domain residual prediction, swap slow state between matched progress/budget states, use norm-preserving random donors and shuffled actions. Report conditional terminal success, mediation by predicted viability, and all null results. Lean-style simulation is not native Lean proof search; verify external Lean traces in a separate track.

## F. Data Goblin handoff

Expose a narrow observation/action/receipt interface for the 1B textual model. Export verifier-backed traces, failed proposals, UNKNOWN labels and exact cost provenance. Keep synthetic controller evidence distinct from language-model results. No training on sealed eval receipts. See HANDOFF.md for implementation sequence.
