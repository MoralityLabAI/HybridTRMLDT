# Dual-LDT-05 results: verified sequential composition, zero behavior gain

**Status: Registered synthetic finite-program study, negative behavioral result.** The registered protocol was committed at 1bb0b1b17d9c81c3ea3534d29a31d6d98757a8e3 and the independent world generator at 32358bf4827f4ed61ba6d92be9c883b6719547dd, before report outcomes. The eight route tag bundles and four seed-0 Jinn heads are the original, already inspected in-distribution artifacts. The newly generated 32 report-world configurations are held out from the development subset; this is not evidence of out-of-distribution model generalization or reliable extraction of real action semantics.

## Verification and provenance

[Successful Dual-LDT-05 GitHub Actions run 41831477162](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/41831477162). All **17 Python tests passed**, plus Lean kernel checks of relational safety composition, updated second-step authorization and monotonicity under a stricter policy. The tests cover exact nondeterministic branches, empty/forged successor graphs, incorrect world bindings, stale resource snapshots, unsupported programs, shutdown after every certified script, and unexpected revocation of a queued second step.

The full paired replay completed **1,024 synthetic episodes**: 64 worlds x 4 real Jinn character heads x 2 disturbance modes x 2 certified control arms, BASE and COMPOSED. Every certified script maintained the host's modeled HALT/RESTORE reachability. Every complete execution passed the independently recomputed all-branch certificate. All selected programs were plain, with zero guard resources spent.

The complete machine-readable artifact is GitHub Actions **dual-ldt-05-results**, ID **13321364928**, containing:
- manifest.json, SHA-256 a89ae3fbd2fcb945471b24cfbc01a03f26540c7479dff65a1eb3ec246eed5afb
- episodes.json, SHA-256 255790e754e5b570018ecd39184799605f9f798bd0fdeea7d27cb4772817e028
- summary.json, SHA-256 3b84c698b7d8391140e8fbca8f15177505ae5e4198218768e22c8f776e67b256
- dominance.json, SHA-256 5acff126544c454313e9e4d9bd76686b943e08b937d58a6556c7aafea6e00951
- frozen world configuration digest: 38a669f580ab372750c5b83a1818467762de766f2e78b82c47c3ad657772a4c1

The dominance analysis was introduced **post hoc** after the original registered result was known. It is explicitly an explanatory diagnostic and does not replace or amend any registered endpoint, threshold, training recipe or world generator.

## Primary report-split results

For every one of the 32 report worlds and all four character heads, COMPOSED certified additional BUILD programs but selected exactly the same PLAIN program as BASE. Both arms halted successfully under worst-case and benign disturbances.

| Persona | Added certified BUILDs per world | BUILD selected | BASE and COMPOSED mean net task reward | BASE and COMPOSED raw Jinn regret | Tax recovered, nats |
|---|---:|---:|---:|---:|---:|
| Legalist | 3.96875 | 0/32 | 0.478125 | 0.000117 | 0.362260 |
| Partisan | 3.96875 | 0/32 | 0.512500 | 0.023841 | 0.490464 |
| Reconciler | 3.96875 | 0/32 | 0.471875 | 0.027045 | 0.465375 |
| Trickster | 3.96875 | 0/32 | 0.500000 | 0.021404 | 0.224911 |

Development worlds showed the same qualitative null: average added certified BUILD programs 4.21875, zero BUILD selections, and identical executed actions for BASE and COMPOSED. Every report-world action fulfilled the registered epsilon=0.25 learned-preference-regret threshold in both arms. No score or task reward improved; the reduced KL preference tax reflects additional low-probability certified choices in the support, not recovered preferred actions.

## Exact explanation from the post-hoc dominance audit

For a program a, the Jinn's frozen adjusted score is v(a)=log P_head(train|canonical tags(a))-0.10*guard_credits(a). Let B be BASE's checked program set and E the newly certified BUILD programs.

Define the dominance margin

\[
M(s,\theta)=\max_{e\in E}v_\theta(e)-\max_{b\in B}v_\theta(b).
\]

M>0 is necessary and sufficient for the new program family to strictly improve the maximum adjusted Jinn score. M<0 guarantees the original baseline winner remains the argmax. A tie M=0 depends on the declared deterministic tie rule.

Every report-world/persona margin was strictly negative; all 32 worlds had at least one certified BUILD:

| Persona | Maximum M (closest new program) | Mean M over 32 report worlds | Positive margins |
|---|---:|---:|---:|
| Legalist | -0.099470 | -1.449063 | 0 |
| Partisan | -0.041203 | -0.528953 | 0 |
| Reconciler | -0.054460 | -1.012056 | 0 |
| Trickster | -0.102776 | -2.385554 | 0 |

There are no ties. The nearest competing two-step program still loses by 0.0412 adjusted-score units in the Partisan case. These exact finite-set comparisons explain the zero program selections without fitting a new threshold.

## Mathematical lessons and next experiment

**Sound program composition succeeded:** every possible intermediate and terminal state remained host-recoverable, with second-step authorization checked under the revised human policy. Preparing T-a at time 0 never allowed running T-a after the host prohibited staged_test at time 1. An unexpected later revocation also stopped the queued primitive while preserving HALT. These are conditional guarantees of the modeled host-controlled transition relation.

**Behavioral synthesis failed:** a larger certified feasible set is insufficient for preference recovery. The four reused classifiers favored already-safe plain programs; the one- or two-credit guard cost made the additional plans strictly dominated in their fixed adjusted score. Even a significant reduction in the conditional KL preference tax did not alter the discrete choice.

A useful next study should build program utility from **trajectory-level outcomes and persona-valued state changes**, with an independently tested reward model or verified symbolic evaluator, rather than applying the original single-action lane classifier to a two-step program's static tags. Choose worlds whose preferred task outcomes sometimes require a temporally valid program and whose alternatives have a genuine reward tradeoff. Register the semantic utility specification and baseline search costs before evaluating new model-produced plans. This avoids post-hoc recipe changes and will test whether recurrent Jinn planning contributes value that a finite symbolic planner cannot already provide.

The original Jinn or Beast paper remains unchanged.

## Reproduction

    python -m pytest -q tests/test_dual_ldt_05.py tests/test_dual_ldt_04.py
    python experiments/dual_ldt_05/src/run.py --manifest-only --out /tmp/dual05-manifest
    python experiments/dual_ldt_05/src/run.py --split all --out /tmp/dual05-all
    python experiments/dual_ldt_05/src/diagnose.py --out /tmp/dual05-all/dominance.json
    lean formal/CCA/DualLDTPrograms.lean
