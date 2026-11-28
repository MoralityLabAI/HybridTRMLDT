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
