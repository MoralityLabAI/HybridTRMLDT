# LSPG-v0 Stage A Report

## Outcome

Stage A timed out at the registered 1,800-second wall limit. Three of twelve proposals produced receipts; nine remain pending. This is an incomplete cost-calibration result, not an architecture ranking.

## Observed proposals

| Proposal | Status | Stop | Exposures | Max gradient | Loss ratio | Mean step seconds |
|---|---|---|---:|---:|---:|---:|
| LSPG-S0-01 | stopped | gradient_norm_above_100 | 0 | 235.40384419897052 | None | None |
| LSPG-S0-02 | stopped | gradient_norm_above_100 | 0 | 278.8234641012909 | None | None |
| LSPG-S0-06 | completed | none | 8192 | 81.43500420073573 | 0.31100859772379325 | 10.5925422843751 |

The two lower-p grid extensions stopped on their first backward pass because gradient norms exceeded 100. The untied R=4 control completed 8,192 state-visit exposures. The next proposal reached a local 10% checkpoint but has no final receipt and is not included in the posterior.

## Promotion

No promotion is allowed. Stage A is incomplete, the original boundary remains left-censored, the run timed out, and measured throughput invalidates the nominal local compute calibration.

The cleanup audit observed two unrelated Research_Engine processes and 100% GPU utilization both before and after LSPG cleanup. No LSPG-owned process lingered. The measured 27,034x forecast ratio is therefore retained as confounded calibration evidence, not treated as a clean device-throughput estimate.

## Integrity

- Raw partial SHA-256: `d17382a9e9871374b14559f0924cb1e03ededa544e70a534ee0a068d0d5250eb`.
- Normalized records SHA-256: `8f36945413a613ce6be1817624dc525c59add3477c9dd77400d059a169ccfaea`.
- Resource receipt SHA-256: `69f0d6cebe0f84e5a208a93bb8e9d36f8a2881dd0d635189e52068cac2d86460`.
- Cleanup receipt SHA-256: `60e2565c973765629a829d4786f3c3e53c6fce56d3af603a6bb64a0d5e640e64`.
- Non-finite values in immediate-stop raw rows are preserved in the raw trace and normalized to null in the strict JSON receipt.

No prediction about task accuracy, sample efficiency, or reasoning quality is made. LSPG-v0 remains scoped to trainability boundaries, cost, and experiment selection.
