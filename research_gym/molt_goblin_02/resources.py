"""CPU-only execution and a hard incremental RAM cap (Windows Job Object, psutil fallback)."""
from __future__ import annotations

import os
import sys
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import psutil  # noqa: E402

GIB = 1 << 30
_STATE: dict = {}


def rss() -> int:
    return psutil.Process().memory_info().rss


def peak_rss() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", 0) or getattr(info, "rss", 0))


def _job_object_cap(limit_bytes: int) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x00000100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = limit_bytes
    if not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        return False
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    ok = bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
    _STATE["job"] = job
    return ok


def install_cap(incremental_gib: float = 3.0, on_breach=None) -> dict:
    """Cap committed memory at baseline + incremental_gib. Returns a receipt."""
    base = psutil.Process().memory_info()
    committed = int(getattr(base, "private", 0) or base.rss)
    limit = committed + int(incremental_gib * GIB)
    hard = _job_object_cap(limit)

    def watch():
        while True:
            if rss() > base.rss + incremental_gib * GIB:
                if on_breach:
                    on_breach()
                os._exit(97)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()
    receipt = {"baseline_rss_bytes": base.rss, "baseline_private_bytes": committed,
               "incremental_cap_bytes": int(incremental_gib * GIB), "process_limit_bytes": limit,
               "windows_job_object": hard, "rss_watchdog": True, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    _STATE["receipt"] = receipt
    return receipt
