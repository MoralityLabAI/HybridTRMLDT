#!/usr/bin/env bash
# Native paired replay for raw, MCP-04 and MCP-05 on pinned tasksets.
# Usage: bash MCP/hubrun/scripts/mcp05_replay.sh smoke|full
# Requires the pinned Bonsai llama-server and existing Prime Hub venvs in WSL.
set -euo pipefail

MODE=${1:-}
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 smoke|full" >&2
  exit 2
fi

REPO=${MCP05_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
H=${MCP05_HUB:-/root/hub-overnight}
S=${MCP05_HUB_SOURCE:-/mnt/c/Users/patri/Documents/Codex/hub-overnight}
UPSTREAM=${MCP05_UPSTREAM:-http://127.0.0.1:18127}
VIA_CURL=${MCP05_VIA_CURL:-/mnt/c/Windows/System32/curl.exe}
PORT=${MCP05_PORT:-18134}
LOG="$H/logs/calls_mcp05_$MODE.jsonl"
REPORT="$H/reports/mcp05_$MODE.json"
cd "$REPO"

[[ -f "$S/args/docedit.json" ]] || { echo "Missing pinned doc-edit args" >&2; exit 3; }
[[ -x "$H/venvs/doc-edit/bin/vf-eval" ]] || { echo "Missing pinned doc-edit vf-eval venv" >&2; exit 3; }
[[ -x "$H/venvs/event-recon/bin/eval" ]] || { echo "Missing pinned event-recon eval venv" >&2; exit 3; }
[[ -f "MCP/hubrun/paired_replay.py" ]] || { echo "Missing paired scorer" >&2; exit 3; }
[[ -x "$VIA_CURL" ]] || { echo "Missing WSL Windows curl bridge" >&2; exit 3; }
command -v python3 >/dev/null
mkdir -p "$H/logs" "$H/runs" "$H/reports"
[[ ! -e "$LOG" ]] || { echo "Refusing to append to prior replay log $LOG" >&2; exit 3; }
if (echo >"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "Replay port $PORT busy" >&2
  exit 3
fi

# Exact upstream schema/template contract. No inference until this succeeds.
python3 -m MCP.hubrun.preflight_probe --upstream "$UPSTREAM" --via-curl "$VIA_CURL" \
  --context-window 32768 --completion-reserve 4096 --safety-margin 1024 \
  --out "$H/logs/mcp05_$MODE"_"preflight.json"

python3 -u -m MCP.hubrun.proxy --port "$PORT" --upstream "$UPSTREAM" \
  --via-curl "$VIA_CURL" --log "$LOG" --max-tokens-cap 4096 \
  --context-window 32768 --completion-reserve 4096 --safety-margin 1024 \
  --threshold 6000 --budget 1800 --keep-recent 2 --per-message-cap 6000 \
  >"$H/logs/mcp05_$MODE"_"relay.log" 2>&1 &
RELAY_PID=$!
trap 'kill "$RELAY_PID" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$RELAY_PID" || { cat "$H/logs/mcp05_$MODE"_"relay.log" >&2; exit 4; }

# Do not turn tool-run failures into DONE receipts. Full and smoke have distinct tags.
run_verified() {
  local tag=$1
  shift
  local directory="$H/runs/$tag"
  [[ ! -e "$directory/DONE" ]] || { echo "Refusing prior DONE marker: $tag" >&2; exit 3; }
  mkdir -p "$directory"
  echo "START $tag"
  local status=0
  "$@" >"$directory/job.log" 2>&1 || status=$?
  if [[ "$status" -ne 0 ]] || ! grep -qx "EXIT 0 $tag" "$directory/job.log"; then
    tail -40 "$directory/job.log" >&2
    echo "Native evaluation failed: $tag" >&2
    return 5
  fi
  date -u +%FT%TZ >"$directory/DONE"
  echo "DONE $tag"
}
DOCEDIT_N=8
EVENT_N=5
if [[ "$MODE" == "smoke" ]]; then DOCEDIT_N=1; EVENT_N=1; fi

# Sequential arms prevent server-slot contention. The scorer requires
# identical task IDs in all arms before allowing a paired claim.
for arm in raw mcp04 mcp05; do
  tag="mcp05-$MODE-docedit-$arm"
  MCP_HUB_RELAY_PORT="$PORT" run_verified "$tag" \
    bash "$REPO/MCP/hubrun/scripts/vfe.sh" doc-edit doc-edit "$arm" "$tag" \
    "$DOCEDIT_N" 1 "$S/args/docedit.json" 2400

  tag="mcp05-$MODE-er-t1-$arm"
  MCP_HUB_RELAY_PORT="$PORT" run_verified "$tag" \
    bash "$REPO/MCP/hubrun/scripts/v1e.sh" event-recon eval event-recon \
    "$arm" "$tag" "$EVENT_N" 1 \
    --env.taskset.tier 1 \
    --env.taskset.task.tools.runtime.type subprocess \
    --env.agent.runtime.type subprocess \
    --env.taskset.task.max-turns 40 \
    --env.timeout.episode 3600
done

python3 -m MCP.hubrun.paired_replay \
  --runs "$H/runs" --calls "$LOG" --mode "$MODE" --out "$REPORT"
echo "Report: $REPORT"
echo "Review task solves and edit-site traces before promotion."
