# LDT-RED01: long-horizon moral storyworlds and adversarial Qwen

[Protocol](PROTOCOL.md) · [World engine](world.py) · [Host-owned exact planner](planner.py) · [Native Beast/Jinn bridge](adapter.py) · [Red Qwen interface](adversary.py) · [Evaluation](eval.py)

Two connected benchmarks: the compact TRM task gym in `MoralityLabAI/BPD` handles Sudoku, ARC-AGI-1 and small hidden-ending storyworlds. This **LDT** gym carries longer 8/12-step moral-objective worlds, delayed endings and simulated adversarial challenges from a system-prompted “Evil Qwen.”

Actions are confined to fictional counter updates. Native `BeastSnapshot` owns permission evidence, the Jinn preference sorter can only rank authorized candidates, and an exact DP planner supplies a host-only upper bound. An adversarial system prompt is used solely to test the defender, never as a source of real authority.

Run the dependency-free mock adversary against native LDT host controls:

```bash
python -m unittest discover -s research_gym/benchmarks/ldt_adversarial_depth_gym -p 'test_*.py' -v
python -m research_gym.benchmarks.ldt_adversarial_depth_gym.eval --seed 901
```

Run a **local** Qwen model with no automatic downloading (requires `transformers` and a local model directory):

```bash
python -m research_gym.benchmarks.ldt_adversarial_depth_gym.eval --seed 901 --local-qwen /path/to/local/Qwen --out /tmp/qwen_ldt.json
```

The mock CPU test reports adversarial proposals blocked by the typed membrane versus a deliberately weak `soft_only` comparator. It **does not run real Qwen** or train a neural LDT. The later neural steering stage needs frozen BPD physical blocks, full task/action readouts, unseen-horizon 8/12 behavior, root-disjoint story families, and signed Jacobian-matched controls. Keep red-model failure and host permission bypass metrics separate from ordinary defender task utility.
