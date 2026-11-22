# HUG-CAL-02 minimal worker handoff

Source of truth: [PR #8](https://github.com/MoralityLabAI/HybridTRMLDT/pull/8), especially `PROTOCOL.md`. Prior inconclusive experiment: [Silico #2](https://u-4772be8c71d9--moralitylab.prod1-usw2.runtime.silico.run/workspace/ws_01kxhf0ex5f46v2j085vc4m236/experiment/exp_01m6qn9g9sevsrc0mzszsna4gq).

The CPU harness is complete. No reasoning from scratch required: use the checked-in Python modules and existing frozen factor panel. Supply only authentic Silico assets, original-weight capture, and runtime receipts.

1. Verify durable `factors.npz` equals registered SHA256 `86f90a1c9403240e4f69fdfad1f9f88a2410a85edfb2cfef2302fc74959f4a50`; verify registered instrument source hash. Read `PROTOCOL.md` and run `python -m unittest discover -s Huginn/tests -v` (expected 17 tests).
2. Export factor vectors with keys `{core}_factor{id}__u/__v` and selection-only pre-projection states with keys `{core}__selection__{early|late}` in NPZ; preserve 16-step original-weight hook, source hashes, and three seeds. Verify shapes and zero-edit traces before proceeding.
3. Run `python Huginn/build_controls.py --factor-vectors factors-export.npz --selection-states selection-only.npz --controls-out control-matrices.npz --report-out selection-report.json --candidates 512`. This emits four fixed write controls and up to four read controls per factor, chosen only on selection states. `read_all_eight_ready=false` is STOP for read-family confirmation; retain receipt.
4. Freeze and check independent verification sources, hashes and instrument manifest. Use `python Huginn/gate_evals.py --manifest manifest.json --states state-captures.npz --controls control-matrices.npz --original-factors factors.npz --family write_rotation --out write-gate.json` (repeat for read_rotation only if four chosen per factor). Gate PASS requires independent provenance review.
5. Only after approved PASS, execute conditional causal confirmation and score through `confirm_eval.py`, preserving original three seeds, fixed control IDs across windows, clustered uncertainty, and Holm correction. Enforce 0.5/1.0/4.5 H100-hour stage caps (6 total).

Save source hashes, exact commands, GPU runtime, split ledger, selection/gate/confirmation receipts, and any failures. Keep original closeout inconclusive and Nanbeige out of scope. A reviewable patch is acceptable. Report new findings as post-registration only.
