#!/bin/bash
# Inky (Inkling-2-Mini on the squid5080 engine, context capped at 16,384 tokens) over the Receipt Goblin
# benchmark, through the receipt-goblin GDK app. Uses this host's paths: Git Bash on Windows, the server
# and the app inside WSL Ubuntu-24.04, the app built in /root/gdk-build/goblin-gdk.
#
#   bash MCP/receipt_goblin/scripts/inky_16k.sh [WORK_DIR]
#
# The server queues for a 15,000 MiB vramorama lease, so this waits for the card. Arms:
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

V="$LOCALAPPDATA/Programs/vramorama/vramorama.exe"
OWNER=squid5080-serve
ours=0
cleanup() { [ "$ours" = 1 ] && bash C:/Users/patri/tools/squid5080/stop.sh >>"$WORK/run.log" 2>&1; }
trap cleanup EXIT

# The same lease, owner, guard and WSL tagging as tools/squid5080/serve.sh, without its --guard-idle
# (not an option of the installed vramorama 0.16.1); the EXIT trap stops the server instead.
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  log "a server is already answering on $PORT; using it and leaving it running"
else
  ours=1
  log "start; queueing squid5080 (ctx 16384) for a 15000M lease"
  "$V" run --vram 15000M --owner "$OWNER" --guard-power 120 -- \
    wsl.exe -d Ubuntu-24.04 -u root -- /opt/squid5080/bin/squid5090 serve /opt/squid5080/model \
    --host 127.0.0.1 --port "$PORT" --ctx 16384 >"$WORK/serve.log" 2>&1 &
  SERVE=$!
  lease=""
  i=0
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVE" 2>/dev/null; then log "server exited before serving"; tail -20 "$WORK/serve.log"; exit 1; fi
    [ -z "$lease" ] && lease=$("$V" leases 2>/dev/null | awk -v o="$OWNER" '$7==o && $2=="admitted" {print $1; exit}')
    if [ -n "$lease" ] && [ $((i % 10)) -eq 0 ]; then
      pid=$("$V" ps 2>/dev/null | awk '/vmwp.exe/ {print $1; exit}')
      [ -n "$pid" ] && "$V" tag "$lease" "$pid" >/dev/null 2>&1
      "$V" renew "$lease" >/dev/null 2>&1
    fi
    i=$((i + 1))
    sleep 30
  done
  # Keep the lease tagged and renewed while the arms run.
  ( while kill -0 "$SERVE" 2>/dev/null; do
      [ -z "$lease" ] && lease=$("$V" leases 2>/dev/null | awk -v o="$OWNER" '$7==o && $2=="admitted" {print $1; exit}')
      if [ -n "$lease" ]; then
        pid=$("$V" ps 2>/dev/null | awk '/vmwp.exe/ {print $1; exit}')
        [ -n "$pid" ] && "$V" tag "$lease" "$pid" >/dev/null 2>&1
        "$V" renew "$lease" >/dev/null 2>&1
      fi
      sleep 300
    done ) &
fi
curl -s "http://127.0.0.1:$PORT/v1/models" >"$WORK/server_models.json"
log "server ready (started by this run: $ours)"

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
log "pilot: $bad of 6 without a certified verdict"
if [ "$bad" -ge 4 ]; then log "pilot failed; stopping"; exit 2; fi

run_arm inky_packet
run_arm inky_blind --blind
run_arm inky_packet_nonb --k-neighbours 0 --budget 1500

cd "$REPO" && "$PY" -m MCP.receipt_goblin.score "$WORK/inky_packet.jsonl" "$WORK/inky_blind.jsonl" \
  "$WORK/inky_packet_nonb.jsonl" >"$WORK/summary.json"
log "done; summary in $WORK/summary.json"
