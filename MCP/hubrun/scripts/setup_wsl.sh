#!/usr/bin/env bash
# Copy pulled hub environments into WSL and give each its own uv venv.
set -u
H=/root/hub-overnight
SRC=/mnt/c/Users/patri/Documents/Codex/hub-overnight/envs
mkdir -p $H/envs $H/venvs $H/logs
for env in "$@"; do
  rm -rf $H/envs/$env
  cp -r $SRC/$env $H/envs/$env
  rm -rf $H/venvs/$env
  uv venv -q $H/venvs/$env --python 3.12 > $H/logs/install_$env.log 2>&1
  if VIRTUAL_ENV=$H/venvs/$env uv pip install -e $H/envs/$env >> $H/logs/install_$env.log 2>&1; then
    v=$($H/venvs/$env/bin/python -c "import verifiers; print(verifiers.__version__)" 2>&1 | tail -1)
    echo "OK $env verifiers=$v"
  else
    echo "FAIL $env"; tail -5 $H/logs/install_$env.log
  fi
done
