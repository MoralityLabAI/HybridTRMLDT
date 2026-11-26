# MMS straggler audit: HybridTRMLDT master

MCP-02 and MCP-03 share MCP.mcp02.train.fit. MMS now targets looped, TRM,
TRM-LDT, and HRM routers. Each physical block exposes ff2, attention out and
QKV sites. One is chosen per optimizer step; for HRM the low/high clocks use
their own invocation counts. Listwise slates use stable train-candidate IDs
assigned before shuffling. TRM-LDT's exclusion-row forward still participates
in the same loss/backward; its site invocations are counted but not cached.
Variable sequence lengths are padded only in storage to FE.SEQ_LEN; live
model padding and arithmetic stay unchanged. Shared page budget covers
BF16 gradients and tags; all model/data/allocator memory is additional.
Optional JSON map manifest entries are {site:{path,sha256}}; paths are
relative to the manifest. Maps must correspond to exact physical matrix and
model lineage. Unspecified sites use SVD proxies.

Use original fit command with:
```sh
--device cpu --mms-mib 512 --mms-mode bpd --mms-sites auto --mms-schedule adaptive
```
Site examples: block.ff2, net0.out, low.qkv, high.ff2.
The adaptive scheduler uses admitted rows, not measured downstream utility.
Start with fixed round robin and matched observe/dense/sham; no test worlds
enter the bank. Existing zero-budget fit calls remain unchanged.

The storyworld TRMProposer training helper now accepts mms_options and
mms_receipt, exposing the fused GRU affine temporarily with the same
Parameters; it restores the original native checkpoint format. Positive
budget arms use explicit cell arithmetic, so use observe as the closest
matched no-feedback control. The companion development runner uses
train/probe trust partitions and native oracle action targets, outside
frozen gaming-vs-improvement receipts.
```sh
python -m research_gym.scripts.run_storyworld_mms --mms-mib 4 --mms-mode bpd --out runs/story_mms
```

Audit: lexical routing, Sudoku exact checker, typed LDT membranes, AIRIS,
CCA and DCF structural search use symbolic rules/finite verification or
frozen inference; they need bounded search or feature caches rather than
a gradient hook. Kimilings is a measured activation/weight audit; no
ordinary optimizer loop in its router. TheySing adapter invokes external
native policies; separate training owner. Recent MCP-06–09 are on stacked
open branches, audited and patched separately where compatible.
No large-memory or causal learning result is claimed.
