from __future__ import annotations

import json
import gzip
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PLAYABLE_FACTIONS = (
    "HEGEMON",
    "STATE",
    "INFILTRATOR",
    "BROKER",
    "ARCHIVIST",
    "CONVENOR",
    "CANTOR",
)


@dataclass(frozen=True)
class TheySingRunSpec:
    run_id: str
    scenario: str
    scenario_path: str
    enforcement_mode: str
    seed: int
    turns: int
    max_turns: int = 40
    proposal_source: str = "heuristic"


@dataclass(frozen=True)
class TheySingNativeRun:
    spec: TheySingRunSpec
    session_id: str
    snapshot: dict[str, Any]
    trace_path: Path
    trace: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


class TheySingHarnessClient:
    def __init__(
        self,
        repo_root: Path,
        artifact_root: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout: float = 30.0,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.host = host
        self.port = port or _free_port(host)
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen[str] | None = None
        self._stdout = None
        self._stderr = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> TheySingHarnessClient:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def start(self) -> None:
        server = self.repo_root / "dist-harness" / "harness" / "server.js"
        if not server.exists():
            raise FileNotFoundError(
                f"TheySing compiled harness not found at {server}. Run `npm run build:harness` in {self.repo_root}."
            )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._stdout = (self.artifact_root / "server.stdout.log").open("w", encoding="utf-8")
        self._stderr = (self.artifact_root / "server.stderr.log").open("w", encoding="utf-8")
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            ["node", str(server), "--host", self.host, "--port", str(self.port)],
            cwd=self.repo_root,
            stdout=self._stdout,
            stderr=self._stderr,
            text=True,
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"TheySing harness exited during startup with code {self._process.returncode}; "
                    f"see {self.artifact_root / 'server.stderr.log'}"
                )
            try:
                health = self.request("GET", "/health")
                if health.get("ok") is True:
                    return
            except RuntimeError:
                time.sleep(0.1)
        raise TimeoutError(f"TheySing harness did not become ready at {self.base_url}")

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None
        for handle in (self._stdout, self._stderr):
            if handle is not None:
                handle.close()
        self._stdout = None
        self._stderr = None

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"content-type": "application/json"} if body is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"TheySing harness request failed: {method} {path}: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"TheySing harness returned non-object payload for {method} {path}")
        return result

    def run(self, spec: TheySingRunSpec) -> TheySingNativeRun:
        if spec.enforcement_mode not in {"hard", "soft", "graduated"}:
            raise ValueError(f"Unknown TheySing enforcement mode: {spec.enforcement_mode}")
        session_id, log_dir = self._create_session(spec)
        snapshot = self.request(
            "POST",
            f"/sessions/{urllib.parse.quote(session_id)}/run",
            {"turns": spec.turns},
        )
        return self._collect_run(spec, session_id, log_dir, snapshot)

    def run_manual(self, spec: TheySingRunSpec, turn_plan: dict[str, Any]) -> TheySingNativeRun:
        if spec.enforcement_mode not in {"hard", "soft", "graduated"}:
            raise ValueError(f"Unknown TheySing enforcement mode: {spec.enforcement_mode}")
        session_id, log_dir = self._create_session(spec)
        snapshot = self.request(
            "POST",
            f"/sessions/{urllib.parse.quote(session_id)}/run-manual-turn",
            turn_plan,
        )
        return self._collect_run(spec, session_id, log_dir, snapshot)

    def _create_session(self, spec: TheySingRunSpec) -> tuple[str, Path]:
        log_dir = self.artifact_root / "native_logs" / spec.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        agents = {
            faction: {"type": "heuristic", "profile": faction}
            for faction in PLAYABLE_FACTIONS
        }
        initial = self.request(
            "POST",
            "/sessions",
            {
                "name": spec.run_id,
                "maxTurns": spec.max_turns,
                "seed": spec.seed,
                "enforcementMode": spec.enforcement_mode,
                "logDir": str(log_dir),
                "scenarioPath": spec.scenario_path,
                "agents": agents,
            },
        )
        return str(initial["sessionId"]), log_dir

    def _collect_run(
        self,
        spec: TheySingRunSpec,
        session_id: str,
        log_dir: Path,
        snapshot: dict[str, Any],
    ) -> TheySingNativeRun:
        trace_path = log_dir / f"{session_id}.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(f"TheySing session did not produce trace {trace_path}")
        trace = read_jsonl(trace_path)
        compressed_trace_path = trace_path.with_suffix(".jsonl.gz")
        with trace_path.open("rb") as source, gzip.open(compressed_trace_path, "wb", compresslevel=9) as target:
            shutil.copyfileobj(source, target)
        trace_path.unlink()
        return TheySingNativeRun(
            spec=spec,
            session_id=session_id,
            snapshot=snapshot,
            trace_path=compressed_trace_path,
            trace=trace,
        )


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
