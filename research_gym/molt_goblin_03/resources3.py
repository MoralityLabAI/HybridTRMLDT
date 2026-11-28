"""Hard total RAM budget for a process tree: a Windows Job Object with a job-wide memory limit.

Child worker processes inherit the job, so the limit bounds the sum of committed memory
across the campaign or finals driver and all of its workers.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import psutil  # noqa: E402

GIB = 1 << 30
_KEEP: dict = {}


def tree_rss() -> int:
    me = psutil.Process()
    total = me.memory_info().rss
    for c in me.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total


def install_tree_guard(total_gib: float = 3.0, per_process_commit_gib: float = 1.5, receipt_path=None,
                       interval_s: float = 0.25) -> dict:
    """Hard RAM budget for the driver and its workers.

    A watchdog sums resident memory over the process tree every interval and terminates the
    whole tree if it exceeds total_gib, writing a breach receipt first. A Job Object caps each
    process's committed memory at per_process_commit_gib as a backstop (torch commits about
    0.8 GiB of address space per process on import while keeping ~0.2 GiB resident, so a job-wide
    commit limit would bound reservations rather than RAM).
    """
    import json
    import threading
    import time
    receipt = install_job_cap(per_process_commit_gib, scope="per_process")
    limit = int(total_gib * GIB)
    state = {"peak_tree_rss": 0}

    def guard():
        while True:
            t = tree_rss()
            state["peak_tree_rss"] = max(state["peak_tree_rss"], t)
            if t > limit:
                if receipt_path:
                    with open(receipt_path, "w", encoding="utf-8") as f:
                        json.dump({"breach_tree_rss_bytes": t, "limit_bytes": limit, "t": time.time()}, f)
                for c in psutil.Process().children(recursive=True):
                    try:
                        c.kill()
                    except psutil.Error:
                        pass
                os._exit(97)
            time.sleep(interval_s)

    threading.Thread(target=guard, daemon=True).start()
    _KEEP["tree_state"] = state
    receipt.update({"tree_rss_limit_bytes": limit, "tree_rss_watchdog_interval_s": interval_s,
                    "per_process_commit_limit_bytes": int(per_process_commit_gib * GIB)})
    return receipt


def peak_tree_rss() -> int | None:
    return _KEEP.get("tree_state", {}).get("peak_tree_rss")


def install_job_cap(total_gib: float = 3.0, scope: str = "job") -> dict:
    limit = int(total_gib * GIB)
    ok = False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("a", "b", "c", "d", "e", "f")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.windll.kernel32
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        if scope == "job":
            info.Basic.LimitFlags = 0x00000200      # JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = limit
        else:
            info.Basic.LimitFlags = 0x00000100      # JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = limit
        if job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            ok = bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
        _KEEP["job"] = job
        _KEEP["k32"], _KEEP["EXT"] = k32, EXT
    return {"job_memory_limit_bytes": limit, "windows_job_object": ok,
            "scope": "sum over driver and child workers" if scope == "job" else "each process in the job",
            "baseline_rss_bytes": psutil.Process().memory_info().rss, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}


def peak_job_bytes() -> int | None:
    if "job" not in _KEEP or not _KEEP["job"]:
        return None
    import ctypes
    info = _KEEP["EXT"]()
    if _KEEP["k32"].QueryInformationJobObject(_KEEP["job"], 9, ctypes.byref(info), ctypes.sizeof(info), None):
        return int(info.PeakJobMemoryUsed)
    return None
