#!/usr/bin/env bash
# Second relay for the late-compaction arm: packet only past 18K estimated tokens (port 18131).
mkdir -p /root/hub-overnight/logs
cd /mnt/c/Users/patri/Documents/Codex/HybridTRMLDT-mcp02
exec python3 -u -m MCP.hubrun.proxy --port 18131 --upstream http://127.0.0.1:18127 \
  --via-curl /mnt/c/Windows/System32/curl.exe --log /root/hub-overnight/logs/calls_late.jsonl \
  --max-tokens-cap 4096 --threshold 18000 --budget 6000 --keep-recent 4 --per-message-cap 6000 \
  >> /root/hub-overnight/logs/proxy_late.log 2>&1
