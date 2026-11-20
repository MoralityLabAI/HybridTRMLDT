#!/usr/bin/env bash
# v1e.sh <venv> <cli> <taskset> <arm> <tag> <n> <concurrency> [extra args...]
# Runs a verifiers v1 taskset (eval / vf-run) against the relay, writing to runs/<tag>.
set -u
H=/root/hub-overnight
venv=$1; cli=$2; taskset=$3; arm=$4; tag=$5; n=$6; c=$7; shift 7
out=$H/runs/$tag
mkdir -p $out
export LOCAL_API_KEY=local
cd $out
$H/venvs/$venv/bin/$cli $taskset -m bonsai-27b-ptq1 \
  --client.base-url http://127.0.0.1:18130/$arm/$tag/v1 --client.api-key-var LOCAL_API_KEY \
  -n $n -r 1 -c $c -o $out --sampling.temperature 0.6 "$@"
echo "EXIT $? $tag"
