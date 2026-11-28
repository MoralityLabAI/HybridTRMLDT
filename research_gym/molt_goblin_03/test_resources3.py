import json
import pathlib
import subprocess
import sys

import pytest

CHILD = r"""
import json, multiprocessing as mp, sys
sys.path.insert(0, ROOT)
from research_gym.molt_goblin_03 import resources3 as R
from research_gym.molt_goblin_03.test_resources3 import child_probe
if __name__ == "__main__":
    cap = R.install_job_cap(1.0)
    with mp.get_context("spawn").Pool(1) as p:
        res = p.map(child_probe, [int(1.2 * 2 ** 30)])
    print(json.dumps({"cap": cap["windows_job_object"], "res": res}))
"""


def child_probe(n):
    """Runs in a spawned worker: is it inside the job, and can it exceed the job limit alone?"""
    import ctypes
    from ctypes import wintypes
    inside = wintypes.BOOL()
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.IsProcessInJob(wintypes.HANDLE(k32.GetCurrentProcess()), None, ctypes.byref(inside))
    try:
        b = bytearray(n)
        del b
        alloc = "ok"
    except MemoryError:
        alloc = "MemoryError"
    return {"in_job": bool(inside.value), "alloc": alloc}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object")
def test_job_cap_covers_child_workers(tmp_path):
    root = str(pathlib.Path(__file__).resolve().parents[2])
    script = tmp_path / "child.py"
    script.write_text(CHILD.replace("ROOT", repr(root)))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["cap"] is True
    assert r["res"][0] == {"in_job": True, "alloc": "MemoryError"}, r
