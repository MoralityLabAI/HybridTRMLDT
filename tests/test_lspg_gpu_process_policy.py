from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "scripts" / "lspg_gpu_process_policy.ps1"


def _classify(process_name: str, command_line: str) -> bool:
    command = (
        f". '{POLICY}'; "
        f"Test-LspgCpuOnlyNvidiaRegistration -ProcessName '{process_name}' "
        f"-CommandLine '{command_line}'"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() == "True"


def test_explicit_cpu_only_llama_server_is_not_gpu_contention() -> None:
    assert _classify("llama-server.exe", "llama-server.exe --n-gpu-layers 0 --port 1")


def test_gpu_backed_llama_server_remains_foreign_contention() -> None:
    assert not _classify("llama-server.exe", "llama-server.exe --n-gpu-layers 1")


def test_other_process_cannot_claim_cpu_only_llama_exception() -> None:
    assert not _classify("python.exe", "python.exe --n-gpu-layers 0")


def test_non_literal_argument_pair_is_not_exempt() -> None:
    assert not _classify("llama-server.exe", "llama-server.exe --n-gpu-layers=0")
