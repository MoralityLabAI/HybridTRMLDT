# Proposed registration

## Scope and existing assets

Extend `research_gym/core/hybrid.py` through an observation/receipt adapter; preserve existing verifier ownership. Reuse held-out storyworld and gaming-versus-oversight fixtures already indexed in the root README. Existing negative controller results remain in the baseline ledger. Do not duplicate the sovereign-learning codec or Kimilings protocols. Integration must first inspect the current branch and repository instructions.

## TRSI-CLOCK-01: fixed-input gate, then learned effect

The executed gate covers exact continuation, stutter refinements, terminal masks and counterexamples. Its float64 tolerance is 1e-11. These checks are necessary implementation contracts.

Next, create host-owned event records with task-family/root identity, turn ID, reward location, generated-token mask, true-terminal flag, timeout cause, observation bytes, compute units, and critic provenance. Freeze JSONL bytes and their SHA-256 before fitting. Deduplicate common task roots across splits; a padding variant inherits its parent's split.

For an unrun learned screen, use five paired seeds [11,23,37,53,71] and four estimator arms: raw-token trace, observation-skipping token trace, event trace, and complete Monte Carlo. Use gamma=1 throughout, lambda=.95 where applicable, the same actor initialization, fixed reward, task schedule, optimizer updates and training transition cap. The cap is 32,768 host decisions per arm/seed; stop if the existing environment cannot support the prescribed disjoint splits. A categorical or MLP controller is the co-primary baseline for any recurrent variant.

Begin with 256 training roots, 128 validation roots and 256 sealed test roots per available task family; this is a screening plan, not a power claim. Root identity and family boundaries must come from the existing task generator. Families with insufficient support are reported as blocked, never synthesized by duplicating rows.

Add observation padding of 0/100/1,000/10,000 tokens and permitted internal-depth budgets 1/2/4/8 in separate diagnostics. Keep useful information and external decisions fixed in the arithmetic test. Behavioral padding runs allow the policy to change and are analyzed separately. Record compute at every depth. Learned halting decisions stay in the policy loss. Define both episode-normalized and action-level loss aggregation before running, since token averaging can reintroduce length dependence.

Primary endpoint: equal-family strict completion at matched host-decision and measured-compute budgets. Secondary endpoints include gradient variance across independent episodes, critic error, progress, latency and unsafe attempts. Never treat tokens as independent trials. Use paired seed effects and root/family-clustered intervals; report seed-level values. The preregistered continuation gate requires positive paired improvement against the strongest simple baseline with a confidence interval excluding zero, at least +0.02 absolute completion, and no authority-contract failure. Five seeds provide a screen; confirm on a newly sealed family split before scaling.

## TRSI-REWARD-01

Hold the chosen clock and training schedule fixed. Compare protected strict-success reward, the paper's SPR, and strict success plus a frozen potential. Include a task family with many passable checks and an independent final gate. Log both check count and strict outcome. The synthetic reversal in `CPU_RESULTS.json` is a positive control for objective mismatch.

Promotion requires improvement in strict completion. Changes in average shaped reward do not satisfy this gate. Freeze potential parameters using training data only. Use the same computation budget across all reward arms and keep protected verifier internals out of both prompts and training features.

## Execution authority

Native integration and training are UNRUN. No model downloads, paid calls, GPU jobs, funded actions, workflow edits or automatic merges are authorized by this packet.
