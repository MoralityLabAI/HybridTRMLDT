#!/usr/bin/env bash
# vfe.sh <venv> <env-id> <arm> <tag> <n> <concurrency> <env-args-file> <timeout> [extra vf-eval args...]
set -u
H=/root/hub-overnight
venv=$1; env=$2; arm=$3; tag=$4; n=$5; c=$6; argsfile=$7; timeout=$8; shift 8
out=$H/runs/$tag
mkdir -p $out
export LOCAL_API_KEY=local
cd $out
help=$($H/venvs/$venv/bin/vf-eval --help 2>&1)
opt=()
if grep -q -- "--output-dir" <<<"$help"; then opt+=(-o "$out"); fi
if grep -q -- "--timeout" <<<"$help"; then opt+=(--timeout "$timeout"); fi
if grep -q -- "--disable-tui" <<<"$help"; then opt+=(-d); fi
$H/venvs/$venv/bin/vf-eval $env -m bonsai-27b-ptq1 -b http://127.0.0.1:${MCP_HUB_RELAY_PORT:-18130}/$arm/$tag/v1 -k LOCAL_API_KEY \
  -n $n -r 1 -c $c -T 0.6 -a "$(cat $argsfile)" -s "${opt[@]}" "$@"
echo "EXIT $? $tag"
