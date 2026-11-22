#!/bin/bash
# End-to-end SWE-bench Lite run inside WSL: Qwen3-4B-Instruct-2507 on CUDA behind goblin-gdk, three context conditions.
set -e
REPO=/mnt/c/Users/patri/Documents/Codex/HybridTRMLDT-mcp02
SNAP=/mnt/d/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
OUT=/mnt/c/Users/patri/Documents/Codex/mcp03-work/e2e
mkdir -p $OUT ~/swebench-trees
cd $REPO
export PYTHONPATH=$REPO
export MCP03_TOKENIZER=/mnt/d/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json
export MCP03_TOKENIZER_NAME=Qwen2.5-0.5B-Instruct@7ae5576
~/goblin-cuda/bin/python -m MCP.mcp02.hf_server --model-dir $SNAP --served-name qwen3-4b-instruct-2507 --port 8095 \
  --device cuda --dtype bfloat16 --cuda-memory-fraction 0.9 > $OUT/server.log 2>&1 &
for i in $(seq 1 120); do curl -s http://127.0.0.1:8095/v1/models > /dev/null 2>&1 && break; sleep 5; done
curl -s http://127.0.0.1:8095/v1/models > $OUT/server_identity.json
~/goblin-cuda/bin/python MCP/mcp03/swebench_e2e.py --binary ~/gdk-build/goblin-gdk/target/release/goblin-gdk \
  --work /mnt/c/Users/patri/Documents/Codex/mcp03-work --trees ~/swebench-trees --out $OUT \
  --ids /mnt/c/Users/patri/Documents/Codex/mcp03-work/e2e_ids.txt --base-url http://127.0.0.1:8095/v1/ \
  --model qwen3-4b-instruct-2507 --max-tokens 2048 --conditions none bm25_13k goblin > $OUT/run.log 2>&1 || echo "runner failed"
curl -s http://127.0.0.1:8095/v1/models > $OUT/server_identity_final.json || true
curl -s -X POST http://127.0.0.1:8095/shutdown > /dev/null 2>&1 || true
echo "E2E GENERATION DONE"
