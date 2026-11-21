#!/usr/bin/env bash
# Overnight queue: storyworlds (raw), agentic A/B pairs (raw and goblin side by side), raw-only agentic sets.
# Resumable: a job whose runs/<tag>/DONE exists is skipped.
set -u
S=/mnt/c/Users/patri/Documents/Codex/hub-overnight
H=/root/hub-overnight
Q=$H/logs/queue.log
mkdir -p $H/runs

run_job() {  # run_job <tag> <command...>
  local tag=$1; shift
  if [ -f $H/runs/$tag/DONE ]; then echo "$(date +%FT%T) SKIP $tag" >> $Q; return; fi
  mkdir -p $H/runs/$tag
  echo "$(date +%FT%T) START $tag" >> $Q
  "$@" > $H/runs/$tag/job.log 2>&1
  local rc=$?
  echo "$(date +%FT%T) END $tag rc=$rc" >> $Q
  if grep -q "^EXIT 0 $tag" $H/runs/$tag/job.log; then date +%FT%T > $H/runs/$tag/DONE; fi
}

vfe() { bash $S/scripts/vfe.sh "$@"; }
v1e() { bash $S/scripts/v1e.sh "$@"; }
pair() {  # pair <tag-stem> <runner> <args with ARM and TAG placeholders...>
  local stem=$1; shift
  local runner=$1; shift
  local raw=() gob=()
  for a in "$@"; do raw+=("${a//@ARM@/raw}"); gob+=("${a//@ARM@/goblin}"); done
  for i in "${!raw[@]}"; do raw[$i]="${raw[$i]//@TAG@/$stem-raw}"; gob[$i]="${gob[$i]//@TAG@/$stem-goblin}"; done
  run_job $stem-raw $runner "${raw[@]}" &
  run_job $stem-goblin $runner "${gob[@]}" &
  wait
}

ER_COMMON=(--env.taskset.task.tools.runtime.type subprocess --env.agent.runtime.type subprocess
           --env.taskset.task.max-turns 40 --env.timeout.episode 3600)

echo "$(date +%FT%T) QUEUE BEGIN" >> $Q

# Storyworlds: held-out eval and development lanes only.
run_job sw-constitution-eval vfe storyworld-constitution storyworld-constitution raw sw-constitution-eval 15 2 $S/args/sw_eval.json 1800
run_job jbm-cross-dev80 vfe jinn-beast-metta jinn-beast-metta raw jbm-cross-dev80 80 2 $S/args/jbm_cross.json 1800
run_job jbm-constructs-dev vfe jinn-beast-metta jinn-beast-metta raw jbm-constructs-dev 4 2 $S/args/jbm_constructs.json 1800
run_job jbm-reasoner-dev vfe jinn-beast-metta jinn-beast-metta raw jbm-reasoner-dev 16 2 $S/args/jbm_reasoner.json 1800
run_job jbm-mesh2-dev vfe jinn-beast-metta jinn-beast-metta raw jbm-mesh2-dev 48 2 $S/args/jbm_mesh2.json 1800

# Agentic A/B where the packet engages (peak prompts above the 6K threshold in smoke runs).
pair er-t1 v1e event-recon eval event-recon @ARM@ @TAG@ 5 1 --env.taskset.tier 1 "${ER_COMMON[@]}"
pair docedit vfe doc-edit doc-edit @ARM@ @TAG@ 8 1 $S/args/docedit.json 2400
run_job sqlite-all v1e mcp-sqlite eval mcp-sqlite raw sqlite-all 57 2 --env.agent.max-turns 20 --env.agent.runtime.type subprocess --env.taskset.task.judges '[]'
pair feishu-hard vfe feishu-office-v1 feishu-office-v1 @ARM@ @TAG@ 7 1 $S/args/feishu_hard.json 2400 --shuffle --shuffle-seed 0
pair er-t2 v1e event-recon eval event-recon @ARM@ @TAG@ 3 1 --env.taskset.tier 2 "${ER_COMMON[@]}"
run_job notebook-hardened vfe durable-notebook durable-notebook raw notebook-hardened 8 2 $S/args/notebook.json 2400
pair feishu-expert vfe feishu-office-v1 feishu-office-v1 @ARM@ @TAG@ 7 1 $S/args/feishu_expert.json 2400 --shuffle --shuffle-seed 0

echo "$(date +%FT%T) QUEUE END" >> $Q
