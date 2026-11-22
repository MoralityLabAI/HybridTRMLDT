# MCP-05 research receipt

## Real 16K repair

The fixed solver was the existing variable-loop Inkling checkpoint, TML-d24-varloop-research IQ4_NL. Weight revision `618c48f1e3c54094333e3e652a92b1d99c52e34d` and SHA-256 `e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2`. The matched trials used llama.cpp b12602, one model loop, greedy sampling, a 16,384-token slot and a 512-token completion cap on an RTX 5080 Laptop GPU. Goose GDK remained on `0.1.0-alpha.14`. The recurrent controller used depth three across tool calls.

On the first completed matched platformer task, both variants corrected the held-out health function and passed the independent regression rerun. The recurrent run authenticated the source digest, applied the model's proposed replacement, invalidated the old handle, and verified the resulting patch. The final patch SHA-256 was `0553ee40e21d6bd240c5e9134753b0be5959895fcd6b6bc189c8ce754a717449`.

| Platformer health case 523-001 | Verified repairs | Provider prompt | Provider completion | Total model tokens | Inference rounds |
|---|---:|---:|---:|---:|---:|
| A / raw | 1/1 | 5,926 | 210 | 6,136 | 4 |
| D / recurrent typed | 1/1 | 11,845 | 301 | 12,146 | 6 |

All ten calls in this pair matched the upstream exact preflight count. D used one repeated read, while A used zero. Extra lookup and format-recovery rounds made D about 1.98 times as expensive on this case despite prefetching the right source. An earlier three-loop compatibility call returned thinking without an executable action. The retained matched protocol normalizes tool markers and rejects unresolved malformed arguments through the same Goose authorization boundary.

Both arms also passed the held-out two-file regression with changes to the jump and health modules. Across the two matched platformer cases, D used 43,022 model tokens and A used 30,006. The second case stayed within the available prompt window in every round, so the raw-first compaction policy never invoked a typed fallback for that repair.

## GPTStoryworld building

The targeted building exercise used the Tiny Diplomacy example at GPTStoryworld revision `c8d46591f983473023daf0232e9b413baaa02194`. D added the requested terminal council-accord node and connected the new outcome. After its first edit, the host rejected a proposal using the old source handle. A fresh read supplied the current handle for the second edit, and GPTStoryworld's validator plus the independent outcome check passed. The final patch SHA-256 was `934d31f5c3ae16824ea1109b6abd24d4d3449d533d8e58402b4c488c0bf6cae9`. D used 39,966 model tokens.

A used 33,582 model tokens and produced no working-tree patch. The host rejected five A edit proposals because the supplied old string did not occur exactly once in the source span. The largest exact prompt was 5,996 tokens for D and 5,133 for A. This single building exercise supports measuring anchored edit success separately from retrieval and token cost.

## CPU controller

Four threads, 18 disjoint training roots, six held-out roots and 108 optimizer updates per learned condition. The matched replay occupied 774,068 serialized bytes and had SHA-256 `3b88c779665bdcf3121bdc9ad5e2aacd78217ae60c5b18c8c0a3f2eb4437761c`. Peak process RSS was 425,750,528 bytes against the 3 GiB cap. The learned runtime in the real repair recorded 587,140 bytes of SetTRM parameters and a 128-byte recurrent state. The allocation log reported 10,316 MiB of model buffers, 356 MiB of KV cache and 557 MiB of compute buffers on Vulkan, with about 11.0 GiB of total device use sampled during the run. Host-mapped model buffers totaled 6,484 MiB.

| Model | Top-target selections on recorded holdout slates | Mean utility regret |
|---|---:|---:|
| RU02 shared | 39/46 | 0.1243 |
| RU02 dual | 44/46 | 0.0070 |
| Recurrent SetTRM | 44/46 | 0.0070 |
| SetTRM reset each step | 44/46 | 0.0070 |
| Shuffled utility targets | 24/46 | 0.3800 |

These controller scores are from recorded localization and repair states. The shuffle control supports a useful target signal on those slates. The tied recurrent and reset scores leave the value of persistent recurrence unresolved. The health repair shows the end-to-end loop works and provides an example where adding the controller cost tokens at the same repair outcome.

## Real tokenizer stress test

The relay replayed a synthetic history of 20 closed tool interactions through the running Inkling server's template and tokenizer endpoints. Raw history required 36,297 prompt tokens against a 15,808-token prompt allowance. The typed packet and two original recent tool pairs required 1,878 tokens after exact remeasurement. This replay exercised the overflow path without requesting a model completion.

## Research lookup failure and handoff

The recurrent arm read the RU02 results for its RAM-training preregistration, then repeated the same lookup five times. It exhausted all ten rounds without an answer and used 57,650 model tokens. Goose now hands a read-only session back for a final answer when an exact search repeats after an accepted source read. A deterministic replay confirmed that the repeated query receives a typed error and the next turn can answer from the previously retrieved source. The original benchmark uses the unchanged executable. The longer Vulkan runs were stopped to evaluate the cached CUDA packs.

## CUDA quant replay and tool-call fix

The two cached CUDA packs ran on the same RTX 5080 Laptop GPU using Inkling CUDA commit `f8acdaea8230181e9f7772a0c1f3d165a49b17cc`, greedy decoding, 48 logical visits per token, and 16,384-token sessions. Each matched replay provided the original task and source locations from the successful recurrent Goose rollout. The original patch was withheld. Each edit had a 768-token limit, and the same independent verifier graded both packs.

| Pack | Matched platformer passes | Accepted edits | Peak tracked CUDA memory |
|---|---:|---:|---:|
| 5.4 GB | 0/2 | 1/3 | 6.48 GiB |
| 8.9 GB | 2/2 | 3/3 | 9.73 GiB |

The 5.4 GB pack also failed the Storyworld replay. The 8.9 GB run reached the device memory limit with another workload active. Its first Storyworld edit reached the token cap, and the second was stopped to free the GPU for the fix. The original variable-loop Q4 recurrent Goose episodes passed all three tasks under a full retrieval and repair protocol.

On the matched health edit, the 5.4 GB pack reasoned out the correct clamp and repeated its thinking until it reached the completion cap. CUDA runtime commit `52dc6ecb642f65744017dd9e52287a6bb28bc49d` adds a local provider with explicit thinking-channel suppression and native tool-call serialization. The provider continued past a prose preamble to accept the following native tool frame. With that mode, the 5.4 GB pack passed the same exact-source replay in 84 completion tokens and reproduced the original Q4 patch SHA-256. The provider matched all 305 preflight prompt tokens. All 14 CUDA runtime tests passed.

On a fresh end-to-end 5.4 GB Goose D health rollout, the native repair prompt directed the model to reuse a current source handle when the required code was already visible. Goose applied the model's edit and the independent regression passed. Five provider calls used 9,756 prompt tokens and 468 completion tokens with exact preflight parity throughout. This episode used 10,224 total tokens versus 12,146 in the earlier variable-loop Q4 Goose D episode. The two runs used different runtimes, model depths, and tool protocols.
