#!/bin/bash
# Inky (Inkling-2-Mini on the squid5080 engine) over the Receipt Goblin benchmark, through the
# receipt-goblin GDK app with its 16,384-token context limit. Uses this host's paths: Git Bash on Windows, the server
# and the app inside WSL Ubuntu-24.04, the app built in /root/gdk-build/goblin-gdk.
#
#   bash MCP/receipt_goblin/scripts/inky_16k.sh [WORK_DIR]
#
# Without a running server it queues one through with_squid.sh, so this waits for the card. Arms:
#   inky_packet       values shown, default packet (40 lexical + 8 neighbours, 3,000-token budget)
#   inky_blind        values withheld, same packet
#   inky_packet_nonb  values shown, no neighbour expansion, 1,500-token budget
# A 6-claim pilot runs first and the run stops if 4 or more of them end without a certified verdict.
set -u
export MSYS_NO_PATHCONV=1
WORK=${1:-C:/Users/patri/Documents/Codex/receipt-goblin-work/inky16k}
REPO=C:/Users/patri/OneDrive/Documents/GitHub/HybridTRMLDT
REPO_WSL=/mnt/c/Users/patri/OneDrive/Documents/GitHub/HybridTRMLDT
WORK_WSL=/mnt/$(echo "${WORK:0:1}" | tr 'A-Z' 'a-z')${WORK:2}
APP=/root/gdk-build/goblin-gdk/target/release/receipt-goblin
PY=C:/Users/patri/AppData/Local/Programs/Python/Python312/python.exe
PORT=18090
mkdir -p "$WORK"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$WORK/run.log"; }

# Without a server on $PORT, re-run this script under with_squid.sh, which queues one for the card and
# stops it afterwards.
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  exec bash "$(dirname "$0")/with_squid.sh" "$WORK" -- bash "$0" "$WORK"
fi
log "server answers on $PORT"

run_arm() {
  local arm=$1; shift
  log "arm $arm $*"
  wsl.exe -d Ubuntu-24.04 -u root --cd "$REPO_WSL" -- "$APP" --bench MCP/receipt_goblin/bench.jsonl \
    --server-cwd . --python python3 --base-url "http://127.0.0.1:$PORT/v1/" --model inkling-2-mini \
    --max-tokens 2048 --timeout-secs 300 --arm "$arm" --out "$WORK_WSL/$arm.jsonl" "$@" 2>>"$WORK/$arm.log"
  log "arm $arm exit $? rows $(wc -l <"$WORK/$arm.jsonl" 2>/dev/null)"
}

run_arm inky_packet --limit 6
bad=$("$PY" -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r['verdict'] in ('error','unparsed','no_answer') for r in rows))" "$WORK/inky_packet.jsonl")
bad=${bad:-6}
log "pilot: $bad of 6 without a certified verdict"
if [ "$bad" -ge 4 ]; then log "pilot failed; stopping"; exit 2; fi

run_arm inky_packet
run_arm inky_blind --blind
run_arm inky_packet_nonb --k-neighbours 0 --budget 1500

cd "$REPO" && "$PY" -m MCP.receipt_goblin.score "$WORK/inky_packet.jsonl" "$WORK/inky_blind.jsonl" \
  "$WORK/inky_packet_nonb.jsonl" >"$WORK/summary.json"
log "done; summary in $WORK/summary.json"
