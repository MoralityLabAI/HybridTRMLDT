# Kappa Data-Order Intervention v0.2.1 Artifact Audit

## Provenance

1. `e5a094d`: seed-decoupled protocol, branch rules, and implementation frozen before outcomes.
2. `f3d5845`: no-outcome capped preflight sealed.
3. `7ef610a`: all 20 records plus the post-completion result-assembly failure sealed before correction.
4. `54c8d0a`: no-training recovery and corrected parent-subset replay rule registered before aggregate classification.
5. `745c3c9`: recovery admission preflight sealed.
6. `726f954`: recovered aggregate result committed before finalization.

## Sealed Identity

- recovery config: `21717c2f67e32f35424c60e53c7bd86c7afadc23c018a01a27ecdc49b9b38de4`
- source records: `84419b04bbb59ceba0a87f91b67f8e143bcf3f2378b71ce9f8a47b86cb3518b3`
- recovered result: `b6010c5171ce6d9c552bbd0d5e089b7622397f02d21067b17c7071dd22b93d2b`
- recovery run resource: `e30a8eb86d300cc44b0308212216e9663597e1fa9ede6b9a1d43b8b178b98ced`
- result receipt: `c9b3ff9037b7d9dfcd9e74c14cfeb64be541dcd0964a84ac979a40335fae6fff`
- finalizer resource: `f3ae03a89af338abfa1eeae26d84ee797c6a864ea94a143a6cb473b56815c462`
- report figure: `e22d54eb1f99e4f0a6aa599d09dd2d1af314199f4fe5a148338890a1a8abbdbf`

The local and `data/benchmarks` receipts are byte-identical. The result receipt re-verifies the original source JSONL rather than a recovery copy. No training was rerun during recovery.

## Integrity Results

- baseline replay: exact at `E={0,1024,2048,4096}`;
- source records: `20/20` unique, four complete five-point order traces;
- non-order seed channels: fixed at 103 in every record;
- training cells: finite and complete at 4096 state-visit exposures;
- registered fresh-order persistence: `2/3`;
- destructive sensitivity interference at `E=2048`: `4/4` orders;
- cleanup: passed for training, recovery, and finalization;
- v0.1 alignment-note artifacts: untouched.

Full repository verification after packaging: `326 passed in 31.08s`.
