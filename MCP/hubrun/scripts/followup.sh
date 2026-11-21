#!/usr/bin/env bash
# After the main queue ends: event-recon tier 1 through the late-compaction relay (port 18131), tag er-t1-late.
set -u
H=/root/hub-overnight
Q=$H/logs/queue.log
until grep -q "QUEUE END" $Q; do sleep 30; done
tag=er-t1-late
if [ -f $H/runs/$tag/DONE ]; then exit 0; fi
mkdir -p $H/runs/$tag
echo "$(date +%FT%T) START $tag" >> $Q
export LOCAL_API_KEY=local
cd $H/runs/$tag
$H/venvs/event-recon/bin/eval event-recon -m bonsai-27b-ptq1 \
  --client.base-url http://127.0.0.1:18131/goblin/$tag/v1 --client.api-key-var LOCAL_API_KEY \
  -n 5 -r 1 -c 2 -o $H/runs/$tag --sampling.temperature 0.6 --env.taskset.tier 1 \
  --env.taskset.task.tools.runtime.type subprocess --env.agent.runtime.type subprocess \
  --env.taskset.task.max-turns 40 --env.timeout.episode 3600 > $H/runs/$tag/job.log 2>&1
rc=$?
echo "$(date +%FT%T) END $tag rc=$rc" >> $Q
[ $rc -eq 0 ] && date +%FT%T > $H/runs/$tag/DONE
echo "$(date +%FT%T) FOLLOWUP END" >> $Q
