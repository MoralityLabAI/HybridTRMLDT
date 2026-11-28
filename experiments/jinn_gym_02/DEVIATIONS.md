# JINN-GYM-02 deviations from the preregistration

## D1: Python verifier timeout raised during BUILD (10 s to 60 s, plus one retry); build workers 4 to 2

The first launch stopped during BUILD, after 830 of 840 tasks were built. A reference-oracle subprocess exceeded the 10 s timeout.

Scanning the bank found 20 Python receipts in 15 tasks that had timed out. These were candidates whose code finishes in well under a second: the median receipt took 0.28 s, but under build load (4 parallel verifier subprocesses plus Lean) some runs took over 10 s. A load-induced timeout scores correct code as failing, so those utilities were invalid.

Changes:

* `verifiers.PY_TIMEOUT` is now 60 s.
* `run_python` retries once on timeout.
* The reference oracle `_expected` retries once.
* `python_build_workers` dropped from 4 to 2.
* The 15 affected tasks were removed from the bank and rebuilt on resume:
  * dev: python_repair-f0-dev02-0016
  * selection: python_repair-f2-sel02-0125, 0137, 0138, 0139, 0140
  * held-out: python_repair-f1-held02-0004, 0011, 0013, 0025, 0029, 0039, 0045, 0046, 0054

No selection, evaluation or held-out decision had run when this happened: the phase was BUILD. The cached neutral Inky responses for those tasks stay valid, because the task text is deterministic and does not depend on verifier outcomes. The deadline (launch + 8 h) is unchanged, and `manifest.json` keeps the launch configuration and code hashes.

## D1a: CPU cap at the user's request (build workers 2 to 1, single-threaded BLAS and torch)

Before the resume, `python_build_workers` was set to 1 and the router's `torch.set_num_threads` to 1, and the campaign was relaunched with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1. Peak use is about three cores: the main thread, one Python verifier subprocess and one Lean subprocess. This slows BUILD and ROUTER but changes no verifier outcome, selection rule or metric.

## D2: numeric bounds stated in the Inky patch prompt; partial-round recovery (after round 0)

In round 0, 6 of 8 Inky patches were rejected as invalid, almost all because a feature weight above 10 was proposed (11 to 12). The GYM-02 patch prompt listed the operations but not the numeric bounds that the validator enforces. The GYM-01 roots already sit near the weight cap. The other four arms respect the bounds by construction, so the omission biased the registered inky − random contrast against Inky.

Changes:

* The patch prompt now states the bounds: weight in (0, 10], threshold in [0, 0.95], `min_evidence_items` an integer 0–5, at most 10 features, each tag in at most one feature.
* The cache key is now `patch02v2`, so no old response is reused.

The run was stopped with the STOP file during round 1, with 3 of 40 candidates recorded. The new `drop_partial_round` moves a partial round's records to `mutations_aborted.jsonl` and rebuilds every lineage from the completed rounds' accepted children, so round 1 reruns in full for every arm. Round 0, which used the old prompt for the Inky arm, is kept as recorded.
