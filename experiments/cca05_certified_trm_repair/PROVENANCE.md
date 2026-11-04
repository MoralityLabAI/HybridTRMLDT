# PROVENANCE: cca05_certified_trm_repair

## Checkpoints

The six TRM checkpoints are the originals of `vpd_control_modules_v1` and `complete_accounts_v1` in the private `MoralityLabAI/jinn-or-beast-paper` repository. They are stored outside Git at `C:\Users\patri\Documents\Codex\jinn-vpd\part_b_real\`. They were found there at the registered paths, so no recovery or retraining was needed. Each was hashed on 2026-11-04 before any logits were read:

| Tag | File | Bytes | File SHA-256 | State-dict SHA-256 (registered) | Match |
|---|---|---:|---|---|---|
| X211 | `trm_X_s211.pt` | 67,381 | `346ce7efc8cdaf2560e7bfe2ad2055ee5fd161dd1e10c8f517491a27af34d00f` | `3b852c9aa570ec5d3903854790a3c7163c11dc59f5c1d5ca6218b59e560b3e49` | yes |
| X223 | `trm_X_s223.pt` | 67,381 | `e69c07e53aeb7c61394921ab6592f08913c48abf68094cd18a18cecaea9cd6e3` | `9fdff79e1f65ae95263aa6c556f3d14a7e70019cbd56db32bdf69d17797ee573` | yes |
| X227 | `trm_X_s227.pt` | 67,381 | `557615aa5aed1145e40d5caff8c968abf7fe5ad868929a8ffb83b6315f4c5824` | `16fad49a3f7c9f7c51f801442d18ef8fb7e3c97e3c26c6078e0e921643513408` | yes |
| R211 | `trm_R_s211.pt` | 67,381 | `c2a6c1db6f626f102ab61d7e0b2c6ab5be0e0cea7f4d31f8b05e500fb6781842` | `26982b0927cb2a6d3e60a52781414025c58b686f2a6367cf0ce7046e2a1e653f` | yes |
| R223 | `trm_R_s223.pt` | 67,381 | `6386b0df0444a06eaddb692f417d83a91dd6d1fb4588fe9eb745b8d4fc24ff07` | `642977fc31f974062504aefd5bcd9a1c397ae37de3a17a122bb5700fac328c60` | yes |
| R227 | `trm_R_s227.pt` | 67,381 | `912aadad9ad6bb0696514e0f45a03f3026c4ea6f4bde2a1015184f8e7a87b07f` | `4dd5a7a283acad295916cc5ab68f02e2468724c7d2569cec0a327db98bdc7a02` | yes |

The state-dict hash is SHA-256 over `key || tensor bytes`, in sorted key order. This is the procedure of `complete_accounts_v1/src/part_b.py`, and the registered values are copied from `complete_accounts_v1/registration.json` (`part_b.checkpoint_state_dict_sha256`). `src/trm_live.load_checkpoint` refuses to load on any mismatch of either hash. Each file holds 15,521 parameters: `TRMProposer(latent_dim=48, recurrence_steps=4)`.

**Training.** These checkpoints were produced by `vpd_control_modules_v1/src/part_b_trm.py` (`build_checkpoint`) at HybridTRMLDT `f162153` with the `gaming_vs_improvement_v2` settings (320 steps, learning rate 0.008, seeds 211, 223, 227):

* **R** is the benchmark's `_undertrained_base`. Its action head is trained toward the unconstrained oracle, its mode head toward the constant `deduce`, and its conflict head receives no gradient.
* **X** adds conflict and mode targets.

The environment, the TRM module and the rollout code are byte-identical between `f162153` and the baseline `59a2d50` (`git diff --stat f162153 59a2d50 -- research_gym/envs research_gym/neural research_gym/benchmarks configs/gaming_vs_improvement_v2.json` is empty).

**Unavailable, reconstructed.** R's calibrated provenance threshold (`model.claim_threshold`) is a Python attribute and is not stored in the state dict. `trm_live.reconstructed_claim_threshold` recomputes it with the `_undertrained_base` rule: the 0.9 quantile of `env_claim_score` over the seed's proposer-train proposals that fail `Verify`. X uses the default 0.5. The provenance-claim counts for R depend on this reconstruction.

**Behavioural cross-check.** The live run compares each checkpoint's pass-4 action on all 2,100 inputs with the exact final-pass decision tree published in `complete_accounts_v1/outputs/part_b/rules/{tag}_final_pass_trees.metta` (paper commit `35fa070b`). That tree reproduces the published module on `D_full` by construction. The agreement and blob hashes are in `outputs/arms/{tag}.json` (`published_tree_parity`).

## Environment

* Code: `research_gym/envs/coupled_storyworld.py`, with the verifier, utility and fallback of `research_gym/neural/rollout.py` restated torch-free in `src/semantics.py`. The restatement equals `rollout.py` on every input (`tests/test_semantics_parity.py`).
* Behaviour digest: `ENV_DIGEST` (`outputs/environment.json`, `env_digest`). It is the SHA-256 of the canonical table of the following, over all 1,050 states:
  * transitions under every (action, rival action) pair;
  * the rival policy;
  * target, terminal and moral score;
  * horizon and threshold.

  Every certificate binds it.

## Neural evidence used

`outputs/tapes/{tag}.json.gz` holds every logit any arm consumed:

* the full-mask action logits at passes 1 to 8 for all 2,100 inputs;
* the pass-4 logits for every refined mask A4 queried;
* the provenance claim score.

These are float32 values from single-row forwards, the way `TRMProposer.propose` runs them, and they are stored exactly. A replay that asks for an unrecorded query raises `KeyError`. Weights and hidden states are not published; the checkpoints stay outside Git.

## What is not available

* **Machine-checked proofs.** No Lean toolchain is installed.
* **The neural internals behind each logit.** The tape records outputs. Mechanistic accounts of these checkpoints are in `complete_accounts_v1` and `vpd_control_modules_v1`.
