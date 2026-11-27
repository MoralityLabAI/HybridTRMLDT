"""Scripted reference trainers (no LLM) that give scores their calibration points.

``default``: fit one fixed mesh (``space.default_candidate``) and submit it.
``random``: draw meshes uniformly from the search space, keep those whose cost bound fits the remaining
budget, fit and dev-evaluate until the budget is gone, submit the best dev mesh. Seeded.

Both talk to the episode only through the tool surface, either in process or over the stdio server, so a
reference run exercises the same contract an LLM runner does.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from . import space
from .episode import Episode


class InProcessClient:
    def __init__(self, config: dict, mcp_seed: int, threads: int = 4):
        self.episode = Episode(config, mcp_seed, threads=threads)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        return self.episode.call(name, arguments or {})

    def receipt(self) -> dict:
        return self.episode.receipt()

    def close(self) -> None:
        pass


class StdioToolClient:
    """The same interface over ``MCP.pocketgoblin_bench.server``."""

    def __init__(self, config_path: Path, mcp_seed: int, threads: int = 4, receipt_path: Path | None = None,
                 timeout: float = 900.0):
        from MCP.transport import StdioClient
        command = [sys.executable, "-u", "-m", "MCP.pocketgoblin_bench.server", "--config", str(config_path),
                   "--mcp-seed", str(mcp_seed), "--threads", str(threads)]
        if receipt_path is not None:
            command += ["--receipt", str(receipt_path)]
        self.client = StdioClient(command=command, timeout=timeout, max_message_bytes=4 * 1024 * 1024)
        self.client.__enter__()
        self.client.initialize()
        self.receipt_path = receipt_path

    def tools(self) -> list[dict]:
        return self.client.tools()

    def call(self, name: str, arguments: dict | None = None) -> dict:
        return self.client.call(name, arguments or {})["structuredContent"]

    def receipt(self) -> dict | None:
        """Close the server (which writes the receipt) and read it back."""
        self.client.close()
        if self.receipt_path and self.receipt_path.is_file():
            return json.loads(self.receipt_path.read_text(encoding="utf-8"))
        return None

    def close(self) -> None:
        self.client.close()


def _caps(client: Any) -> tuple[int, int]:
    r = client.call("list_mesh_families")["ranges"]
    return r["train"]["worlds"]["max"], r["filters"]["max_tool_calls"]["max"]


def run_default(client: Any) -> dict:
    worlds, calls = _caps(client)
    fitted = client.call("fit", {"candidate": space.default_candidate(worlds, calls)})
    if "error" in fitted:
        raise RuntimeError(f"default fit refused: {fitted['error']}")
    return client.call("submit", {"candidate_id": fitted["candidate_id"]})


def run_random(client: Any, seed: int, max_draws: int = 60) -> dict:
    worlds, calls = _caps(client)
    rng = random.Random(f"pgb-random/{seed}")
    best: tuple[float, float, str] | None = None
    for _ in range(max_draws):
        candidate = space.sample(rng, worlds, calls)
        estimate = client.call("estimate_cost", {"candidate": candidate})
        if estimate["upper_bound_units"] > estimate["remaining_units"]:
            continue
        fitted = client.call("fit", {"candidate": candidate})
        if "error" in fitted:
            if fitted["code"] == "fit_limit":
                break
            if fitted["code"] == "budget_exceeded":
                continue
            raise RuntimeError(f"fit refused: {fitted['error']}")
        dev = client.call("evaluate", {"candidate_id": fitted["candidate_id"]})
        if "error" in dev:
            if dev["code"] == "eval_limit":
                break
            if dev["code"] == "budget_exceeded":
                continue
            raise RuntimeError(f"evaluate refused: {dev['error']}")
        key = (dev["metrics"]["packet_recall"], -fitted["charged_units"], fitted["candidate_id"])
        if best is None or key[:2] > best[:2]:
            best = key
    if best is None:
        raise RuntimeError("random search fitted no mesh within the budget")
    return client.call("submit", {"candidate_id": best[2]})
