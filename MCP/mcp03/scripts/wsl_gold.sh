#!/bin/bash
mkdir -p ~/swebench-runs && cd ~/swebench-runs
IDS=$(cat /mnt/c/Users/patri/Documents/Codex/mcp03-work/e2e_ids.txt)
~/swebench-venv/bin/python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Lite --split test \
  --predictions_path gold --instance_ids $IDS --max_workers 4 --run_id goblin_gold_check --timeout 1800 2>&1 | tail -25
