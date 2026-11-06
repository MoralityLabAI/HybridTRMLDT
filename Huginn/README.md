# Huginn HUG-CAL-02: control repair and temporal specialization

Status: **CPU-prepared, prospective GPU calibration not run** (2026-11-06).

The [Silico Huginn closeout](https://u-4772be8c71d9--moralitylab.prod1-usw2.runtime.silico.run/workspace/ws_01kxhf0ex5f46v2j085vc4m236/experiment/exp_01m6qn9g9sevsrc0mzszsna4gq) stopped because only **81/144 controls** and **10/36 complete sets** passed the frozen activation-quantile gate. The original label remains **inconclusive**. The archive audit in `archive_audit.json` reproduces its exact counts from 414,720 saved candidate verification norm samples and reports retrospective feasibility over *all* saved candidate metadata.

New diagnostics from the raw archive:

- In **11/36 cases**, fewer than four candidates from the entire old pool satisfy the saved verification gate. This oracle calculation **uses verification** and is therefore a retrospective impossibility witness for *re-ranking those old pools*, not an acceptance criterion.
- **17 factor controls** fail on p10 alone. Fitting only the median is inadequate.
- **0/18 early/late case pairs** share the same four `(seed,index,scale)` control IDs. HUG-CAL-02 constructs **fixed matrices jointly fitted for both time windows**.

## Contents

| File | Purpose |
| --- | --- |
| `audit_archive.py` | SHA-checked, read-only replay of saved closeout controls, quantiles and candidate pool feasibility |
| `archive_audit.json` | Observed CPU audit receipt, retaining its retrospective label |
| `control_repair.py` | CPU reference algorithms for joint-window rank-one controls, selection-only scoring, paired effect, program-clustered bootstrap, Holm/sign-flip inference |
| `gate_evals.py` | Recompute independent new-split control acceptance from raw factor/control/state arrays, plus source manifest checks |
| `confirm_eval.py` | Strict post-gate runner for frozen eight-hypothesis factor panel, requiring an external calibration PASS receipt |
| `PROTOCOL.md` | GPU task, model version, data split and intervention gates |
| `WORKER_PROMPT.md` | Self-contained Silico/Claude workorder |
| `tests/` | Deterministic synthetic tests, no model inference |

## CPU validation

From the directory **above** `Huginn/`:

```bash
python -m pip install numpy
python -m unittest discover -s Huginn/tests -v
python Huginn/audit_archive.py /path/to/closeout-unzipped/data --out /tmp/huginn_audit.json
```

For an independently captured prospective calibration, run `python Huginn/gate_evals.py --manifest manifest.json --states state_captures.npz --controls control_matrices.npz --original-factors factors.npz --family write_rotation --out gate_write.json`. Gate PASS also requires external review of GPU traces.

`audit_archive.py` validates the source calibration NPZ checksum and the pinned factor/model/program hashes before reconstructing all 144 selected-control quantile verdicts. The program-generator sources, model checkpoint, factor `factors.npz`, and original `x_t` activation tensors must be recovered from the worker experiment workspace. These are **not included** in this small CPU package.

## External evidence and model

- [Silico experiment #2](https://u-4772be8c71d9--moralitylab.prod1-usw2.runtime.silico.run/workspace/ws_01kxhf0ex5f46v2j085vc4m236/experiment/exp_01m6qn9g9sevsrc0mzszsna4gq)
- [Silico closeout thread](https://u-4772be8c71d9--moralitylab.prod1-usw2.runtime.silico.run/workspace/ws_01kxhf0ex5f46v2j085vc4m236/thread/th_01m6aw5gaqfa3tgqhp4w33yt3d?exp=exp_01m6qn9g9sevsrc0mzszsna4gq)
- [Huginn-0125 pinned model](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c)
- [HybridTRMLDT research repo](https://github.com/MoralityLabAI/HybridTRMLDT)

The existing Nanbeige audit is a distinct experiment. Preserve that separation.
