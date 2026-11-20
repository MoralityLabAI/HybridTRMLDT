#!/usr/bin/env bash
# pin.sh <verifiers-version> <env>...: reinstall verifiers at one version in each env's venv
set -u
H=/root/hub-overnight
ver=$1; shift
for env in "$@"; do
  if VIRTUAL_ENV=$H/venvs/$env uv pip install "verifiers==$ver" >> $H/logs/install_$env.log 2>&1; then
    echo "OK $env $($H/venvs/$env/bin/python -c 'import verifiers; print(verifiers.__version__)' 2>&1 | tail -1) vf-eval=$(test -x $H/venvs/$env/bin/vf-eval && echo yes || echo no)"
  else
    echo "FAIL $env"; tail -5 $H/logs/install_$env.log
  fi
done
