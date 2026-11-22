#!/usr/bin/env bash
# Run the paired native Bonsai MCP-04 doc-edit and event-recon replay.
# Usage: bash MCP/hubrun/scripts/mcp04_replay.sh smoke|full
# Expects pinned Bonsai llama-server running on the Windows host.
set -euo pipefail

MODE=${1:-}
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 smoke|full" >&2
  exit 2
fi

REPO=${MCP04_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
H=${MCP04_HUB:-/root/hub-overnight}
S=${MCP04_HUB_SOURCE:-/mnt/c/Users/patri/Documents/Codex/hub-overnight}
UPSTREAM=${MCP04_UPSTREAM:-http://127.0.0.1:18127}
VIA_CURL=${MCP04_VIA_CURL:-/mnt/c/Windows/System32/curl.exe}
EARLY_PORT=${MCP04_EARLY_PORT:-18132}
LATE_PORT=${MCP04_LATE_PORT:-18133}

cd "$REPO"
mkdir -p "$H/logs" "$H/runs"
[[ -f "$S/args/docedit.json" ]] || { echo "Missing pinned doc-edit args: $S/args/docedit.json" >&2; exit 3; }
[[ -x "$H/venvs/doc-edit/bin/vf-eval" ]] || { echo "Missing pinned doc-edit vf-eval venv" >&2; exit 3; }
[[ -x "$H/venvs/event-recon/bin/eval" ]] || { echo "Missing pinned event-recon eval venv" >&2; exit 3; }
command -v python3 >/dev/null
command -v curl >/dev/null

# Check exact tool schema/template handling without any model inference.
# The script will refuse full sampling if the server fails this gate.
python3 -m MCP.hubrun.preflight_probe --upstream "$UPSTREAM" \
  --via-curl "$VIA_CURL" --context-window 32768 \
  --completion-reserve 4096 --safety-margin 1024 \
  --out "$H/logs/mcp04_preflight.json"

for port in "$EARLY_PORT" "$LATE_PORT"; do
  if (echo >"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Relay port $port already in use; choose free MCP04_EARLY_PORT/MCP04_LATE_PORT" >&2
    exit 3
  fi
done

EARLY_PID=""
LATE_PID=""
cleanup() {
  [[ -z "$EARLY_PID" ]] || kill "$EARLY_PID" 2>/dev/null || true
  [[ -z "$LATE_PID" ]] || kill "$LATE_PID" 2>/dev/null || true
}
trap cleanup EXIT

python3 -u -m MCP.hubrun.proxy --port "$EARLY_PORT" --upstream "$UPSTREAM" \
  --via-curl "$VIA_CURL" --log "$H/logs/calls_mcp04.jsonl" \
  --max-tokens-cap 4096 --context-window 32768 --completion-reserve 4096 \
  --safety-margin 1024 --threshold 6000 --budget 1800 \
  --keep-recent 2 --per-message-cap 6000 \
  >"$H/logs/mcp04_relay.log" 2>&1 &
EARLY_PID=$!

python3 -u -m MCP.hubrun.proxy --port "$LATE_PORT" --upstream "$UPSTREAM" \
  --via-curl "$VIA_CURL" --log "$H/logs/calls_mcp04_late.jsonl" \
  --max-tokens-cap 4096 --threshold 18000 --budget 6000 \
  --keep-recent 4 --per-message-cap 6000 \
  >"$H/logs/mcp04_late_relay.log" 2>&1 &
LATE_PID=$!
sleep 1
kill -0 "$EARLY_PID" || { cat "$H/logs/mcp04_relay.log"; exit 4; }
kill -0 "$LATE_PID" || { cat "$H/logs/mcp04_late_relay.log"; exit 4; }

run_verified() {
  local tag=$1
  shift
  local output="$H/runs/$tag"
  mkdir -p "$output"
  if [[ -f "$output/DONE" ]]; then
    echo "SKIP completed $tag"
    return 0
  fi
  echo "START $tag"
  "$@" >"$output/job.log" 2>&1 || true
  if grep -qx "EXIT 0 $tag" "$output/job.log"; then
    date -u +%FT%TZ > "$output/DONE"
    echo "DONE $tag"
  else
    tail -40 "$output/job.log"
    echo "Native evaluation $tag did not complete successfully" >&2
    return 5
  fi
}

N=8
ER_N=5
if [[ "$MODE" == "smoke" ]]; then N=1; ER_N=1; fi
for lane in raw goblin mcp04 late; do
  port="$EARLY_PORT"
  arm="$lane"
  if [[ "$lane" == "late" ]]; then
    port="$LATE_PORT"
    arm="goblin"
  fi
  tag="mcp04-${MODE}-docedit-${lane}"
  MCP_HUB_RELAY_PORT="$port" run_verified "$tag" \
    bash "$REPO/MCP/hubrun/scripts/vfe.sh" doc-edit doc-edit "$arm" "$tag" \
    "$N" 1 "$S/args/docedit.json" 2400

  tag="mcp04-${MODE}-er-t1-${lane}"
  MCP_HUB_RELAY_PORT="$port" run_verified "$tag" \
    bash "$REPO/MCP/hubrun/scripts/v1e.sh" event-recon eval event-recon \
    "$arm" "$tag" "$ER_N" 1 \
    --env.taskset.tier 1 \
    --env.taskset.task.tools.runtime.type subprocess \
    --env.agent.runtime.type subprocess \
    --env.taskset.task.max-turns 40 \
    --env.timeout.episode 3600
done

# A provider/template mismatch must be corrected before promotion.
python3 - "$H/logs/calls_mcp04.jsonl" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
rows = [row for row in rows if row.get("arm") == "mcp04"]
if not rows:
    raise SystemExit("No MCP-04 calls were recorded")
bad = [row for row in rows if row.get("preflight_matches_provider") is False]
missing = [row for row in rows if row.get("preflight_matches_provider") is None
           and row.get("status") == 200]
print(json.dumps({"mcp04_calls": len(rows), "provider_mismatches": len(bad),
                  "missing_provider_usage": len(missing)}))
if bad or missing:
    raise SystemExit("MCP-04 template token count differs from provider; gate failed")
PY

echo "Complete. Preserve logs/runs and compare the native grader outcomes, including failed assigned tasks."
