#!/bin/bash
# Run a command while the squid5080 Inkling-2-Mini server answers on 127.0.0.1:18090. Reuses a server that
# already answers; otherwise queues one for a 15,000 MiB vramorama lease (owner squid5080-serve,
# --guard-power 120, WSL vmwp tagged and renewed) and stops it when the command ends.
#
# The server sizes its own context to the VRAM left after the weights (the [vram] lines of serve.log are
# copied to run.log); the 16,384-token window is enforced by the clients. An explicit --ctx 16384 makes the
# server exit when another process takes VRAM during its 7-minute load (2026-11-26: an untagged 337 MiB
# llama-server left 0.90 GiB free for a 0.90 GiB runtime). Host paths: Git Bash on Windows, the server in
# WSL Ubuntu-24.04.
#
#   bash MCP/receipt_goblin/scripts/with_squid.sh WORK_DIR -- COMMAND [ARGS...]
set -u
export MSYS_NO_PATHCONV=1
WORK=$1; shift
[ "${1:-}" = "--" ] && shift
PORT=18090
V="$LOCALAPPDATA/Programs/vramorama/vramorama.exe"
OWNER=squid5080-serve
mkdir -p "$WORK"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$WORK/run.log"; }
ours=0
cleanup() { [ "$ours" = 1 ] && bash C:/Users/patri/tools/squid5080/stop.sh >>"$WORK/run.log" 2>&1; }
trap cleanup EXIT

lease=""
tag_and_renew() {
  [ -z "$lease" ] && lease=$("$V" leases 2>/dev/null | awk -v o="$OWNER" '$7==o && $2=="admitted" {print $1; exit}')
  if [ -n "$lease" ]; then
    pid=$("$V" ps 2>/dev/null | awk '/vmwp.exe/ {print $1; exit}')
    [ -n "$pid" ] && "$V" tag "$lease" "$pid" >/dev/null 2>&1
    "$V" renew "$lease" >/dev/null 2>&1
  fi
}

if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  log "a server already answers on $PORT; using it and leaving it running"
else
  ours=1
  log "queueing squid5080 for a 15000M lease"
  "$V" run --vram 15000M --owner "$OWNER" --guard-power 120 -- \
    wsl.exe -d Ubuntu-24.04 -u root -- /opt/squid5080/bin/squid5090 serve /opt/squid5080/model \
    --host 127.0.0.1 --port "$PORT" >"$WORK/serve.log" 2>&1 &
  SERVE=$!
  i=0
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVE" 2>/dev/null; then log "server exited before serving"; tail -20 "$WORK/serve.log"; exit 1; fi
    [ $((i % 10)) -eq 0 ] && tag_and_renew
    i=$((i + 1))
    sleep 30
  done
  ( while kill -0 "$SERVE" 2>/dev/null; do tag_and_renew; sleep 300; done ) &
  grep -E '^\[(vram|weights)\]' "$WORK/serve.log" >>"$WORK/run.log"
fi
curl -s "http://127.0.0.1:$PORT/v1/models" >"$WORK/server_models.json"
log "server ready (started here: $ours); running: $*"
"$@"
status=$?
log "command exit $status"
exit $status
