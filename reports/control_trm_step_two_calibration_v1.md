# ControlTRM step-two intervention: calibration-only implementation

Status: **bounded infrastructure check**, not a latent-steerability or alignment result.

## Frozen source and inputs

Parent: `feat/rlm-trm-ldt-hybrid-neighborhood-v1` at `dcc7a0e93a73b0f96f2c2ccbfee603029b116d6a`.
Original model and training scripts are unchanged. The three CPU checkpoints have 17,203 parameters each, 48-dimensional hidden state, four tied GRU visits, and final training step 200. Their checkpoint SHA-256 pins, the 152-task input suite, and the saved proposal-table hash are inherited from `configs/rlm_trm_ldt_hybrid_neighborhood_v1_1.json` and its registration.

## Read-only protocol

From the repository root, in an existing CPU-only environment with PyTorch installed:

```sh
python -m research_gym.scripts.verify_control_trm_latent_calibration
```

The runner verifies the registered configuration, suite, proposal table, manifest, and each checkpoint SHA-256 before loading weights with `torch.load(..., weights_only=True, map_location="cpu")`. It uses only the 32 pre-existing *calibration* tasks, eight in each of four families, for each checkpoint seed (211, 223, 227). It reconstructs step-two hidden state, resumes steps three/four, and compares original action logits with the sealed proposal scores and rankings. Each recipient receives a deterministic same-family donor insertion, then its original state is restored; only sham/rescue identities and finite donor outputs are assessed. No donor-effect score, conditional intervention claim, new training, holdout evaluation, external API call, or output file is generated.

Fail closed on hash, shape, identity, seed, nonfinite tensor, reconstruction, sham, rescue, ranking, or in-memory weight mutation mismatches. Model parameters and production `forward()` remain unchanged. The helper `research_gym/neural/control_trm_intervention.py` is confined to CPU tensor access and the action head.

## Evidence and claim limits

An independent read-only examination retrieved all three original 73,746-byte checkpoint blobs and matched their SHA-256 hashes to the manifest. A separate numerically faithful GRU reconstruction matched all 408 archived *calibration* candidate-action logits to within `1e-6` and all 96 saved rankings exactly. In that reconstruction, self-state sham and restore-after-donor produced zero logit delta across the 96 calibration/checkpoint cells. These checks were computed without executing the new PyTorch runner against a repository checkout; the committed runner should reproduce them on an authorized checkout.

Synthetic local unit tests cover exact logit reconstruction, sham, donor insertion, rescue, parameter immutability, invalid state rejection, fail-closed pinned hashes, and exclusion of malformed held-out rows. Calibration case reuse does not constitute independent evidence of a causal effect. Hidden state initializes at zero on every call; this is per-call recurrence, not event-persistent memory. An external LDT gate remains responsible for admissible execution, and future matched-information baselines would be needed for a comparative architectural claim.
