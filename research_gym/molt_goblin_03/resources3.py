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


def install_job_cap(total_gib: float = 3.0) -> dict:
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
        info.Basic.LimitFlags = 0x00000200          # JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = limit
        if job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            ok = bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
        _KEEP["job"] = job
        _KEEP["k32"], _KEEP["EXT"] = k32, EXT
    return {"job_memory_limit_bytes": limit, "windows_job_object": ok, "scope": "driver and all child workers",
            "baseline_rss_bytes": psutil.Process().memory_info().rss, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}


def peak_job_bytes() -> int | None:
    if "job" not in _KEEP or not _KEEP["job"]:
        return None
    import ctypes
    info = _KEEP["EXT"]()
    if _KEEP["k32"].QueryInformationJobObject(_KEEP["job"], 9, ctypes.byref(info), ctypes.sizeof(info), None):
        return int(info.PeakJobMemoryUsed)
    return None
