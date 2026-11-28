"""Process-isolated job supervisor with a hard process-tree memory limit.

Each job runs as a child process.  The parent samples the RSS of the child and all
of its descendants every ``period`` seconds and kills the whole tree when the
total exceeds ``limit_bytes`` (default 3 GiB) or the wall cap expires.  Before
starting a job it waits (bounded) while system available RAM is below
``min_available``, which keeps the run from competing with a loaded desktop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil

LIMIT = 3 * 1024**3
MIN_AVAILABLE = int(1.5 * 1024**3)


def tree_rss(proc: psutil.Process) -> int:
    total = 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def kill_tree(proc: psutil.Process) -> None:
    try:
        procs = proc.children(recursive=True) + [proc]
    except psutil.Error:
        procs = [proc]
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)


def wait_for_memory(min_available: int = MIN_AVAILABLE, max_wait: float = 600.0, log=None) -> Dict:
    t0 = time.time()
    waited = False
    while psutil.virtual_memory().available < min_available and time.time() - t0 < max_wait:
        if not waited and log:
            log(f"available RAM {psutil.virtual_memory().available / 2**30:.2f} GiB below floor; waiting")
        waited = True
        time.sleep(15)
    return {"waited_s": time.time() - t0, "available_at_start": psutil.virtual_memory().available}


def run(cmd: List[str], wall_s: float, limit_bytes: int = LIMIT, period: float = 0.5,
        log_path: Optional[Path] = None, env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None,
        log=None) -> Dict:
    """Run ``cmd``; return status ok|failed|memory_killed|wall_killed with peak tree RSS,
    wall seconds and child CPU seconds (user+system of the process tree while alive)."""
    mem = wait_for_memory(log=log)
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    t0 = time.time()
    child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    proc = psutil.Process(child.pid)
    peak, cpu, status = 0, {}, None
    while True:
        rc = child.poll()
        if rc is not None:
            status = "ok" if rc == 0 else "failed"
            break
        rss = tree_rss(proc)
        peak = max(peak, rss)
        try:
            for p in [proc] + proc.children(recursive=True):
                t = p.cpu_times()
                cpu[p.pid] = t.user + t.system
        except psutil.Error:
            pass
        if rss > limit_bytes:
            kill_tree(proc)
            status = "memory_killed"
            break
        if time.time() - t0 > wall_s:
            kill_tree(proc)
            status = "wall_killed"
            break
        time.sleep(period)
    child.wait()
    if log_path:
        out.close()
    return {"status": status, "returncode": child.returncode, "peak_tree_rss_bytes": int(peak),
            "wall_s": time.time() - t0, "tree_cpu_s_observed": float(sum(cpu.values())),
            "limit_bytes": limit_bytes, **mem}


def python_module(module: str, args: List[str]) -> List[str]:
    return [sys.executable, "-u", "-m", module, *args]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
