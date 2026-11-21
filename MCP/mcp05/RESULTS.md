# MCP-05 research receipt

## Real 16K repair

The fixed solver was the existing variable-loop Inkling checkpoint, TML-d24-varloop-research IQ4_NL. Weight revision `618c48f1e3c54094333e3e652a92b1d99c52e34d` and SHA-256 `e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2`. The matched trials used llama.cpp b12602, one model loop, greedy sampling, a 16,384-token slot and a 512-token completion cap on an RTX 5080 Laptop GPU. Goose GDK remained on `0.1.0-alpha.14`. The recurrent controller used depth three across tool calls.

On the first completed matched platformer task, both variants corrected the held-out health function and passed the independent regression rerun. The recurrent run authenticated the source digest, applied the model's proposed replacement, invalidated the old handle, and verified the resulting patch. The final patch SHA-256 was `0553ee40e21d6bd240c5e9134753b0be5959895fcd6b6bc189c8ce754a717449`.

| Platformer health case 523-001 | Verified repairs | Provider prompt | Provider completion | Total model tokens | Inference rounds |
|---|---:|---:|---:|---:|---:|
| A / raw | 1/1 | 5,926 | 210 | 6,136 | 4 |
| D / recurrent typed | 1/1 | 11,845 | 301 | 12,146 | 6 |

All ten calls in this pair matched the upstream exact preflight count. D used one repeated read, while A used zero. Extra lookup and format-recovery rounds made D about 1.98 times as expensive on this case despite prefetching the right source. An earlier three-loop compatibility call returned thinking without an executable action. The retained matched protocol normalizes tool markers and rejects unresolved malformed arguments through the same Goose authorization boundary.

## CPU controller

Four threads, 18 disjoint training roots, six held-out roots and 108 optimizer updates per learned condition. The matched replay occupied 774,068 serialized bytes and had SHA-256 `3b88c779665bdcf3121bdc9ad5e2aacd78217ae60c5b18c8c0a3f2eb4437761c`. Peak process RSS was 425,750,528 bytes against the 3 GiB cap. The learned runtime in the real repair recorded 587,140 bytes of SetTRM parameters and a 128-byte recurrent state. The Vulkan allocation log reported 10,316 MiB of model buffers, 356 MiB of KV cache and 557 MiB of compute buffers for the one-loop configuration, with about 11.0 GiB of total device use sampled during the run. CPU-offloaded weights use mapped host memory.

| Model | Top-target selections on recorded holdout slates | Mean utility regret |
|---|---:|---:|
| RU02 shared | 39/46 | 0.1243 |
| RU02 dual | 44/46 | 0.0070 |
| Recurrent SetTRM | 44/46 | 0.0070 |
| SetTRM reset each step | 44/46 | 0.0070 |
| Shuffled utility targets | 24/46 | 0.3800 |

These controller scores are from recorded localization and repair states. The shuffle control supports a useful target signal on those slates. The tied recurrent and reset scores leave the value of persistent recurrence unresolved. The health repair shows the end-to-end loop works and provides an example where adding the controller cost tokens at the same repair outcome.
