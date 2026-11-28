# FG-06: elastic drafting, RLT clocks and prefix-bank transport

Prospective development protocol, committed before the new implementation.
Uses the GSF-10 frozen synthetic decoder; see [first results](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/46096326247).
That one-root pilot favored first-loop rather than final-loop prefix memory.
FG-05 is a separate adapter-initialization investigation. This study fixes
ordinary rank-four adapters and asks whether inference-only extra passes
improve accepted-prefix length enough to repay their work.

## Computation

Train two fresh four-loop targets, final-only supervision, using GSF-10's
existing target fitter and a new prefix-root offset of 20000. Fit ordinary
one-pass rank-four adapters separately on the first and final prefix caches,
then freeze them. Every experimental inference policy for a target/cache
pair uses the identical target and adapter weights. Every speculative round
starts with the same anchor and masks, and executes the ordinary adapted
first block. Further passes use the frozen pretrained shared block, with
no new adapter calls or updates. The native verifier always reads its own
per-loop caches and runs the unchanged full four-pass target.

For an additional draft visit with proposed next state F(h; K_r, V_r), use
h' = h + alpha * (F(h; K_r, V_r) - h). All schedules begin with the same
ordinary first visit, alpha=1. For four draft passes the RLT-style three
refinement coefficients are 1/3, 1/2, 1. The constant damping control uses
5/12, 5/12, 1, matching the mean of the two nonterminal coefficients and
the terminal replacement. This is a forward interpolation experiment;
it does not implement or test RLT gradient identities.

Let b be the original adapter's cache depth (0 or 3). In a bank cycle,
visit depths b, (b+1)%4, (b+2)%4, (b+3)%4. Reverse cycle uses decrements.
Both read precisely the same multiset of existing prefix caches. This is
a fixed MoLT-inspired read-only transport schedule; it does not reproduce
upstream MoLT's learned depth attention or its training objective.

| Policy | Draft passes | Prefix depth schedule | Interpolation |
| --- | ---: | --- | --- |
| single | 1 | b | ordinary |
| plain2 | 2 | b,b | ordinary |
| plain4 | 4 | b,b,b,b | ordinary |
| rlt4 | 4 | b,b,b,b | RLT-style |
| damp4 | 4 | b,b,b,b | mean-matched damping |
| cycle4 | 4 | forward bank cycle | ordinary |
| reverse4 | 4 | reverse bank cycle | ordinary |
| cycle_rlt4 | 4 | forward bank cycle | RLT-style |

All draft policies propose four slots, shortened only at the final generation
boundary. No development-dependent selection, adaptation, learned gate or
oracle-derived stop. The first pass must match GSF-10's one-pass predictions;
the native greedy verifier and physical cache rollback remain unchanged.
Full target and adapter hashes stay fixed. Every output ID must equal cold-
checked serial target decoding; if a mismatch occurs, fail the run.
Record each pass's draft IDs, prefix depth, coefficient, and work.

## Costs and outcomes

Primary descriptive comparison: paired root-level difference in total
serialized shared-block calls versus single-pass drafting. For a trajectory,
this is the sum of actually executed draft passes and four times the number
of target-verification calls. Ordinary serial decoding uses four blocks per
generated token, excluding the common prefill.

Report separately target calls, emitted tokens per verification, draft and
target token-block visits, logical prefix KV bytes consumed by draft reads,
accepted-prefix survival, rejected-cache positions, true target accuracy and
peak provisional verifier KV tensor bytes. Logical byte counts are not
memory-traffic measurements. Extra parallel slot work and cache copies can
dominate latency; block-call counts are not a speedup claim. Diagnostic
elapsed times for speculative decoding omit prefill and correctness audits.
A reduction in serialized calls with increased token-block work is a
tradeoff, not an unqualified improvement. Include pairwise deltas versus
plain4, rlt4 versus damp4, cycle4 versus reverse4, and cycle_rlt4 versus
cycle4. These are development contrasts; two weight seeds cannot establish
architecture superiority. Retain unfavorable outcomes.

## Resource and custody contract

Two seeds 137,163. Width 32, vocab 11, four target loops, 240 target updates,
180 adapter updates, batch 16, 256 target-fit roots, 96 separate adapter-fit
roots and 24 each CAL/dev/shift/trace-train. Fresh offset 20000 in the original
generator, distinct from FG-01 and FG-05 default root ranges. All model seeds
and policy variants are paired views. This yields two target fits, four
one-pass adapter fits, 32 policy cells, 768 trace trajectories on 24 distinct
trace prefix roots. CAL remains diagnostic; no controller is trained here.
FG-07 will fit budget choices from separate data.

Save model/adapter artifacts and source hashes. Audit checks complete
coverage, actual weight identities, regeneration of public worlds, greedy
parity, per-pass order/coefficient, work ledger and exact receipt replay
on the dedicated trace split. Recompute trace summaries and matched
contrasts. All records remain training_ready=false. No pretrained LM,
accelerator, sampling preservation or concurrent serving claim.
