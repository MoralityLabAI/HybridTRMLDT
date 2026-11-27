# COG-13M: Goal-shape prior after native modular failures

Protocol registered at fd3b7213b85482053302e3611f52a70ddf3362e7. Source: [COG-13L native CI 45920577858](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/45920577858), artifact 15165670472, SHA-256 d9a8884acacfa8250466e1a85262ddd6e60723b66980effdcae8f94d186b79f9. The L result is green with all six engineering gates passing: 120 episodes, zero unknowns/errors, identical Decimal/table trajectories. Each of the five policies solves 20/24. All modular roots fail. All other families solve.

| L arm | Certified roots | Probes | Routing CPU s | Charged proof CPU s |
|---|---:|---:|---:|---:|
| Frequency | 20 | 388 | .005424 | 217.778979 |
| Local dense | 20 | 338 | .227078 | 219.840426 |
| Shared dense | 20 | 338 | .229052 | 219.569815 |
| Decimal progressive | 20 | 338 | 7.014534 | 225.721294 |
| Table progressive | 20 | 338 | .158328 | 219.952444 |

Table/Decimal receiver: 0.069420411 vs 6.873279522 seconds (99.01x); same 9,423 recurring bytes vs 23,322 shared dense, 65 refinements, 11 exact fallbacks. Installed data makes the table's setup-plus-traffic cost 48,167B vs 27,746B shared dense on this small run. Total proof CPU saving vs Decimal about 2.56%; frequency is still competitive.

## Executed retrospective diagnosis

Inspected all 20 failed modular episodes in the downloaded pinned L archive: 960 attempted queries, each episode uses the full 48-probe cap. For each dense-derived arm, 192 queries across four failed roots include 30 accepted constructors, 27 printed atomic-strengthening events (target contains modulus and an atomic inequality, resulting target displays .le), 16 omega/all_goals omega attempts, six root-level attempts. Frequency: 192, 30, 24, 16, six respectively. On ordinary roots aesop opens four atomic goals; constructor then strengthens the first bound (e.g. remainder <=6 to .le 5), and aesop changes the display back to <=5. On both deeper roots aesop opens nine goals, exceeding the narrow eight-goal admission limit; constructor branches then fall into the same strengthening pattern. These observations motivate a separate shape and width test. They do not establish a causal repair or show that every constructor application is counterproductive.

The audit_l.py script binds all failed probes to original request/response files and checks these exact tallies; its hosted execution is pending. Prior manual inspection used the archive's episode JSON. Retain the spent roots as diagnosis only.

## New native matrix

24 distinct generated source strings (12 modular, 12 control; balanced ordinary/deeper), six arms, 144 paired episodes. Independent reference preflight plus invalid identifier sentinel. Compare original dense and frequency, dense with 16-goal cap, shape-aware frequency and dense with eight-goal cap, and shape-aware dense with 16-goal cap. All other budgets fixed: 48 queries, 45s wall, 12 advancing actions, 16 exact in-process snapshots. The shape rule uses displayed target syntax only: favor omega for atomic natural inequalities, all_goals omega for multiple such atomic goals; defer constructor on those atoms. Prefer appropriate logical decomposition or ring at recognized shapes. All untried tactics remain available. Local policy choices and changed base rank are separately recorded. The heuristic is deliberately imperfect and makes no soundness or progress guarantee. Proof credit requires the same two independently compiled certificates as J.

Frozen gates in PROTOCOL.md. Record exact paired gains/losses, all failures, per-arm CPU, attempted/restored commands and accepted atomic constructors. The new cohort uses different modulus/offsets and structural variants; generated similar families are not independent theorem-distribution evidence. No new model fitting, new grammar or communication improvement claim. Native server and proof audit inherited unchanged; the width and routing hooks are explicit. The new fixture and native evaluation remain pending CI. No promotion before the fresh gates pass.

Reproduce using the archive pins in the workflow:
```
python experiments/cog13m_goal_shape/audit_l.py L.zip /tmp/l-diagnosis.json
python experiments/cog13m_goal_shape/run.py C.zip --old-l L.zip --fixture --out /tmp/m-fixture
```
Native run uses the pinned J snapshot server, C/H/J/L ZIPs and Lean 4.35.0 + pinned Mathlib. See workflow for exact invocation. Output directory must be new.
