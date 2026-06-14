# Bootstrap Workorder

Goal: verify the repo runs and produce first frames.

Commands:

```bash
python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7
python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl
python -m research_gym.scripts.write_agent_tasks --out tasks/generated
```

Deliverable: `handoff.md` with label distribution and any errors.
