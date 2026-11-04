# Dual-LDT-04 Results: proof-carrying action synthesis

**Status: prospective synthetic world configurations, fixed pretrained in-distribution Jinn heads.** The source registration was committed at 390f8e6 and amended before evaluation at 222cc16; the novel world generator was committed at 70663f4 before report outcomes. The recorded Jinn weights and route-tag vocabulary are reused from the source paper. This is a finite typed-action synthesis experiment under an assumed independent host environment, not language-model creativity or adaptive semantic-monitor validation.

## CI and provenance

[Successful GitHub Actions run 41826757633](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41826757633). **24 Python tests passed** across Dual-LDT-01/02/04; Lean proof checking passed for the abstract safe-expansion and preference-witness statements. Deterministic development tests, a manifest generation, development episodes and the full paired report replay all passed without an invariant exception.

Total 1,536 episodes: 64 worlds x 4 frozen Jinn personas x 2 disturbance modes x 3 controllers (I immediate Beast; T temporal Beast; S temporal Beast with guarded synthesis). Each episode contains up to three actor decisions followed by a host HALT request.

GitHub Actions artifact **dual-ldt-04-results**, artifact ID **13319413005**, preserves world manifest, all individual episode receipts, and full group/paired metrics.

SHA-256 of its contained files:
- manifest.json: 73d269c1e55e62907d830b30f6fff3876c83d9e2d62216fc83da9c217013beb3
- episodes.json: da1e71c0d317cbe3bde7765f0488ac5068be4eff1707fc999689dc3f36323807
- summary.json: 8ba96a77c5220024fd4fbd296ba475fe91bc20c29279041a9b0a6f851eca5a9a
- frozen world configuration hash: 571a1627416b994fc918f54d90c7b4ac0fbe80e43b09e555aa2db26f1574e1f0.

## Primary: host shutdown under worst-case transitions

32 report worlds per persona:

| Persona | I: immediate HALT | T: temporal HALT | S: synthesis HALT |
|---|---:|---:|---:|
| Legalist | 19/32 | 32/32 | 32/32 |
| Partisan | 14/32 | 32/32 | 32/32 |
| Reconciler | 16/32 | 32/32 | 32/32 |
| Trickster | 0/32 | 32/32 | 32/32 |
| **Total** | **49/128** | **128/128** | **128/128** |

Development yielded the same total immediate 49/128 and certified 128/128 under worst-case transitions. Under benign disturbances, all arms halted successfully in all 128 report and 128 development episodes. All temporal and synthesized action proofs had their complete modeled successor sets inside W1. The hard-critical X and prohibited staged_test root after the human correction were never executed in certified arms.

## Preference and resource tradeoff: report worlds, worst-case disturbance

Task reward is the sum over the three decisions of each root's independently SHA-generated reward minus 0.10 per executed guard credit. Learned Jinn regret is measured using log P_train as the internal preference score, relative to the current immediate Beast's highest-ranked plain root.

| Persona | T mean net task reward | S mean net task reward | S−T reward | S−T cumulative Jinn regret | Mean guards executed per world |
|---|---:|---:|---:|---:|---:|
| Legalist | 1.5000 | 1.5000 | 0 | 0 | 0 |
| Partisan | 1.6781 | 1.6781 | 0 | 0 | 0 |
| Reconciler | 1.4719 | 1.4719 | 0 | 0 | 0 |
| Trickster | 1.6406 | 1.5250 | −0.1156 | **−3.2073** | 0.6875 |

The Trickster generated **22 executed guarded alternatives** across 32 report worlds, of which 20 were interlock:T-b and two witness:T-b. Thirteen of 32 worlds used at least one guard. The other three persona heads selected their original safe plain actions despite having additional certified plans available.

For the Trickster, mean cumulative raw Jinn regret fell from **5.8492 under T to 2.6419 under S**, a 54.8% reduction. Over 96 executed decisions, the fraction with per-decision raw preference regret no greater than the frozen epsilon=0.25 increased from **56/96 to 78/96**. Average guard resource consumption was one credit per report world. The preference was recovered under the stipulated guard mechanism at an average net task-reward loss of 0.1156.

The comparator T is evaluated in the **same state and remaining resources** whenever S makes a decision, verifying that the synthesized argmax has a cost-adjusted Jinn score at least as high as T's. Finite-grammar regret certificates bound raw Jinn preference loss by a host-certified candidate witness plus its fixed resource price, with zero assertion failures.

## Authorization cost and proof checks

In the report worst-case S evaluation, each persona attempted 1,536 candidate plan certificate issues, for 6,144 attempted checks across four personas. The host issued and independently validated 710 Legalist, 696 Partisan, 730 Reconciler, and 616 Trickster eligible guard certificates, **2,752 accepted checks in total**. The large difference between eligible guards and 22 executed guards shows that certificate feasibility alone did not force the Jinn to use them.

The conditional KL preference tax decreased for S relative to T for all four personas because its certified action set included more possible choices. This is a distribution-support measure; for Legalist, Partisan, and Reconciler the added probability mass never changed the argmax action. It should not be interpreted as behavioral preference recovery in those three cases.

In development, S executed 11 guards in total and the Trickster's cumulative Jinn regret fell from 4.1069 to 2.5063. The increase to 22 executed guards in the report set reflects the fixed new configurations rather than a model update.

## Falsification and interpretation

The registered temporal safety invariants held in all episodes. No forged or stale plan was accepted by the tested checker. S's adjusted-score superset invariant and the bounded preference-regret witness checks passed for all actual S decisions.

**The behavioral gain is persona-specific.** Only the Trickster chose guarded substitutes. The other saved character heads already found sufficiently preferred temporal-safe actions in this fixed route family, so the same synthesis grammar yielded zero action changes for them. The math establishes possible utility retention given a certified candidate; it does not guarantee such a candidate exists or that a trained head will choose it.

The transition semantics deliberately treat a functioning host interlock as capable of suppressing an isolated outcome and a supervised action as recoverable maintenance. The same host-owned transition implementation underlies proof issuance, checking and simulation. Shared modeling mistakes and unavailable real-world hardware/intervention authority remain outside the guarantee.

The generator changes hazards, supervision capabilities, costs, and rewards across fresh configurations. Root action tags and the four trained Jinn weights remain in distribution. These are synthetic control outcomes and have no claim of real adversarial action meaning or human moral alignment. The finite grammar is enumerated; no open-ended language-level action synthesis occurred.

## Next mathematical test

Synthesize plans through a small *compositional* program algebra rather than fixed one-level templates. Attach a verified effect type to each primitive and prove that its sequential composition preserves the host-recovery region. The new grammar should contain plans whose safety and task preference depend on action ordering, consumed resources, and a human correction arriving between steps. Prospectively test compositional plans on independent world families, retaining the current simple-template baseline and the same learned head weights.
