#!/bin/bash
# Score the three prediction files with the official SWE-bench harness (images already cached).
cd ~/swebench-runs
OUT=/mnt/c/Users/patri/Documents/Codex/mcp03-work/e2e
IDS=$(cat /mnt/c/Users/patri/Documents/Codex/mcp03-work/e2e_ids.txt)
for c in none bm25_13k goblin; do
  ~/swebench-venv/bin/python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Lite --split test \
    --predictions_path $OUT/predictions_$c.jsonl --instance_ids $IDS --max_workers 4 --run_id goblin_e2e_$c --timeout 1800 \
    > $OUT/harness_$c.log 2>&1 &
done
wait
for c in none bm25_13k goblin; do cp ~/swebench-runs/goblin-gdk__qwen3-4b-instruct-2507__$c.goblin_e2e_$c.json $OUT/report_$c.json 2>/dev/null || ls ~/swebench-runs/*.json; done
echo "E2E SCORING DONE"
