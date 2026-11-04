# CCA-09: Full retrospective six-checkpoint anytime search

Status: **CPU replay completed; exploratory/descriptive.** The six checkpoint tapes and the underlying storyworld labels were published in CCA-05 before this follow-up. No new model training, held-out seal, or inferential significance claim is made.

Run: [GitHub Actions 41813708333](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41813708333) (both Lean and CPU jobs passed). Full machine-readable `cca09-full` artifact ID `13314295050`, containing `metrics.json` (SHA-256 `a06b5a8ca7981e6ccf3ebe23c6fe18a6a80a4b74a0503163fd105dd7393d312b`).

The replay consumed the original pass-4 logits of X211, X223, X227, R211, R223 and R227. Each checkpoint has 1,750 nonterminal decisions: 875 secret-ending states with 208 viable, and 875 moral-optimization states with 585 viable. All six use the same exact CCA-05 environment-owned verifier, certificate checker, and utility evaluator. Candidate order is either recorded TRM scores or the fixed priority `defuse,befriend,investigate,rush,wait`. This is a fixed-priority comparison, not a reconstruction of the entire prior L0 controller.

## Utility retained on viable states

Mean executed utility includes a value of **zero for abstentions**. Ranges are over six checkpoint rankings; fixed priority is identical for every checkpoint. At each budget K, at most K distinct actions are evaluated.

| Scenario | K | TRM order | Fixed priority | TRM minus fixed | Viable abstentions, TRM / fixed |
|---|---:|---:|---:|---:|---|
| Moral (585 viable) | 1 | 0.7596–0.7686 | 0.6353 | +0.1243 to +0.1333 | 17–23 / 108 |
| Moral | 2 | 0.7868–0.7892 | 0.7339 | +0.0529 to +0.0553 | 2–3 / 43 |
| Moral | 3 | 0.7892–0.7920 | 0.7395 | +0.0498 to +0.0526 | 0–2 / 39 |
| Moral | 4 | 0.7920 | 0.7587 | +0.0333 | 0 / 24 |
| Moral | 5 | 0.7920 | 0.7920 | 0 | 0 / 0 |
| Secret ending (208 viable) | 1 | 0.8654–0.9135 | 0.6683 | +0.1971 to +0.2452 | 18–28 / 69 |
| Secret ending | 2 | 0.9279–0.9760 | 0.9087 | +0.0192 to +0.0673 | 5–15 / 19 |
| Secret ending | 3 | 0.9712–0.9952 | 0.9856 | −0.0144 to +0.0096 | 1–6 / 3 |
| Secret ending | 4 | 0.9904–1.0000 | 0.9856 | +0.0048 to +0.0144 | 0–2 / 3 |
| Secret ending | 5 | 1.0000 | 1.0000 | 0 | 0 / 0 |

Across both scenarios (793 viable states), the TRM ranking improves utility-with-abstentions at K=1 by 0.1497–0.1617, at K=3 by 0.0350–0.0413, and at K=4 by 0.0258–0.0284. At K=5 both orderings exhaust the same action set and reach identical certified oracle utility of approximately 0.84659.

## Certificate strength versus achieved utility

The **stronger empirical choice** and **stronger mathematical guarantee** are distinct:

- In secret ending, every certified action has utility 1. At K=1 the TRM already certifies exact optimality on 180–190 of 875 states, against 139 for fixed priority. All 208 viable states are certified by K=5. The TRM needs a mean 4.082–4.103 evaluator calls per secret-ending state at K=5, against 4.157 for fixed priority.
- In moral optimization, all six TRM rankings reach the exact oracle utility on every viable state by K=4. Yet **none** can certify epsilon=0 optimality at K=4 with the fixed global utility upper bound of 1. Every moral state requires all five evaluations for an exact certificate.
- At K=1 on executed moral actions, TRM mean regret is only 0.0020–0.0044, against 0.0328 for fixed priority; its nonzero certified upper gap can be as large as 0.32558. Actual low regret is measured using the hindsight oracle for the experiment, while the controller itself only knows the conservative global bound.
- At K=3, TRM mean executed regret is at most 0.00024 in moral tasks, versus 0.00532 for fixed priority. Secret-ending TRM is sometimes worse than fixed priority at K=3, showing that ranking improvement is empirical rather than a theorem.

## Verified invariants and boundaries

The full replay raised zero authorization or regret-bound assertions. Exhaustive small-domain tests check every feasible subset, evaluated budget, utility combination and proposer ordering. Every selected action is independently checked, including its bound context and the negative status of rejected candidates.

The Lean lemma establishes a sound upper envelope under a transitive utility comparison relation and independently correct judgments on checked and unchecked actions. Lean does **not** prove the specific environment implements those assumptions.

The measured evaluation counts exclude the cost of producing TRM logits because the runner replays frozen tapes; the extra checker's work is also distinct from a cold verifier evaluation. Actual deployment economics require a separate measurement of neural forward time and proof issuance/checking under identical runtime conditions.

This experiment assumes the CCA-05 modeled rival and exact symbolic action semantics. It makes no claim about adversarial natural-language tag extraction or robust viability against a nondeterministic rival.

## Source hashes and reproduction

Environment digest: `575c3a100b21278948aae75c20a659acf4da076654c2323af90384881be7ffcd`. SHA-256 hashes of the six source tapes are stored in the attached full metrics JSON.

```bash
python -m pytest -q tests/test_cca_anytime_search.py
python experiments/cca09_anytime_certified_search/src/run.py --out /tmp/cca09-full
```

The key result is **candidate-order efficiency under proof-checked execution**. The next mathematical target is to construct tighter cheap, independently verified action-value upper bounds so that finding a near-optimal action can produce a near-optimality certificate without exhausting every action.
