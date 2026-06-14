.PHONY: frames metta eval tasks test clean all

frames:
	python -m research_gym.scripts.generate_frames --out data/frames.jsonl --n 128 --horizon 6 --seed 7

metta:
	python -m research_gym.scripts.generate_metta_frames --out data/metta_frames.jsonl --source examples/metta_rules/toy_skills.metta

all: frames metta eval tasks test

eval:
	python -m research_gym.scripts.eval_symbolic --frames data/frames.jsonl

tasks:
	python -m research_gym.scripts.write_agent_tasks --out tasks/generated

test:
	python -m pytest tests

clean:
	rm -f data/*.jsonl data/*.md
	rm -rf tasks/generated
