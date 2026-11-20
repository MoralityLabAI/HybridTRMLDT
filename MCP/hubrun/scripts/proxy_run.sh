#!/usr/bin/env bash
# Stop any running hub-run proxy, then run a fresh one in the foreground.
for pid in $(pgrep -f "MCP.hubrun.proxy"); do
  if [ "$pid" != "$$" ]; then kill $pid 2>/dev/null; fi
done
sleep 1
mkdir -p /root/hub-overnight/logs
cd /mnt/c/Users/patri/Documents/Codex/HybridTRMLDT-mcp02
exec python3 -u -m MCP.hubrun.proxy --port 18130 --upstream http://127.0.0.1:18127 \
  --via-curl /mnt/c/Windows/System32/curl.exe --log /root/hub-overnight/logs/calls.jsonl \
  --max-tokens-cap 4096 --threshold 6000 --budget 1800 --keep-recent 2 --per-message-cap 6000 \
  >> /root/hub-overnight/logs/proxy.log 2>&1
