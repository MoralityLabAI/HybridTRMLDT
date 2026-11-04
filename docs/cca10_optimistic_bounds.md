# CCA-10: Cheap optimistic utility bounds by action-count abstraction

**Goal:** give the CCA-09 anytime controller a stronger proof of optimality using independently checked, cheaper upper bounds on unevaluated actions. This is a new exploratory analysis on the already inspected CCA-05 storyworld.

For a state x=(trust t, evidence e, heat h, scene s) and at most k future actions, enumerate nonnegative counts (b,i,d,r,w) of befriend, investigate, defuse, rush, wait, summing to l <= k.

Every trajectory using these counts obeys:

- t_final <= min(3, t + b + l), since befriend adds at most 1 and each rival action can increase trust by at most 1.
- e_final <= min(4, e + i), since only investigate can increase evidence.
- h_final >= max(0, h - b - 2d - w), ignoring heat-increasing actions and rivals.
- s_final <= min(5, s + b + i + d + 2r), since rush advances scene at most twice and every other non-wait move at most once.

The terminal moral score is 3t+2e-3h+s, clipped to [(score+21)/43] between zero and one. Therefore for each count vector its score on the coordinate envelope is at least the score of every reachable terminal state with those counts. Take the maximum over count vectors and apply the clipped score conversion. This is a sound optimistic bound under the documented action alphabet and state clamps, including any sequence of the modeled rival's possible moves.

The number of count vectors is binomial(k+5,5), so at horizon k=5 the reference calculation checks 252 combinations, without enumerating 5^k continuation paths. After caching per (state,k), the same upper values are reused across actions and checkpoint rankings. A host-owned verifier recomputes the count envelope before accepting the certificate.

**CCA-N (sound optimistic bound)**: for every target state and remaining horizon k in the registered domain, U_k(x) <= optimistic_moral_upper(x,k). The inequality follows from the coordinate bounds and monotonicity of the score in t,e,s and antitonicity in h, with a maximum over every possible count vector. The exact CCA-05 dynamic-programming implementation is used for an exhaustive 1,050-state x 6-horizon regression test (6,300 comparisons).

**Limitations:** the bound may remain loose because it combines coordinate extrema from mutually incompatible action sequences. Its runtime cost must be included in end-to-end comparisons. This is a hand-derived abstraction tested after the CCA-09 result and is labeled retrospective exploratory. It does not establish semantic-reader coverage or robustness to unknown action types.

Implementation: research_gym/core/cca_optimistic_bound.py; tests/test_cca_optimistic_bound.py. Run the unchanged six-tape CCA-09 replay with --upper-mode optimistic to measure certificate improvements.
