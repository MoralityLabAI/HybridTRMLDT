"""Workspace-bound MCP process for Goose's authorized actions."""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from MCP.mcp03.goblin_server import GoblinService
from MCP.mcp03.repo import path_flags
from MCP.serve import serve
from .context import MAX_SOURCE_BYTES, WorkingSet, sha


def envelope(payload: dict, error: bool = False) -> dict:
    return {"structuredContent": payload, "isError": error,
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}


class RepairWorkspace:
    def __init__(self, working: WorkingSet, manifest: dict | None):
        self.working = working
        self.root = working.root
        self.manifest = manifest
        self.creatable = ()
        self.edits: list[dict] = []
        self.verifications: list[dict] = []
        if manifest:
            argv, editable = manifest["argv"], manifest["editable"]
            if not argv or not all(isinstance(s, str) and s for s in argv):
                raise ValueError("verifier requires host-supplied argv")
            if not editable or not all(isinstance(s, str) and s for s in editable):
                raise ValueError("verifier requires editable path prefixes")
            creatable = manifest.get("creatable", [])
            if not isinstance(creatable, list) or not all(isinstance(rel, str) and rel for rel in creatable):
                raise ValueError("verifier requires a list of exact creatable paths")
            self.creatable = tuple(creatable)
            check_create = manifest.get("validate_create_argv", [])
            if not isinstance(check_create, list) or not all(isinstance(arg, str) and arg for arg in check_create):
                raise ValueError("creation validator requires host-supplied argv")
            for rel in self.creatable:
                path = self.root / rel
                if (path.relative_to(self.root).as_posix() != rel or path.resolve() != path
                        or any(part.startswith(".") for part in Path(rel).parts)
                        or not self.allowed(rel)):
                    raise ValueError("creatable paths must be exact editable workspace paths")
            if self.git("rev-parse", "--show-prefix").strip() or self.git("status", "--porcelain").strip():
                raise ValueError("repair verifier requires a clean standalone worktree at the workspace root")
            self.base = self.git("rev-parse", "HEAD").strip()

    def git(self, *args: str) -> str:
        run = subprocess.run(["git", *args], cwd=self.root, timeout=20, text=True,
                             encoding="utf-8", errors="replace", capture_output=True, check=True)
        return run.stdout

    def allowed(self, path: str) -> bool:
        flags = path_flags(path)
        is_swmd = path in self.creatable and path.endswith(".swmd.min.md")
        return ((is_swmd or not flags["is_doc"]) and not flags["is_test"] and
                any(path == item.rstrip("/") or (item.endswith("/") and path.startswith(item))
                    for item in self.manifest["editable"]))

    def create(self, path: str, content: str) -> dict:
        if path not in self.creatable or not isinstance(content, str):
            raise ValueError("provide source for a host-authorized new path")
        raw = content.encode("utf-8")
        if not 0 < len(raw) <= MAX_SOURCE_BYTES:
            raise ValueError("new source must fit the source-size limit")
        if argv := self.manifest.get("validate_create_argv"):
            checked = subprocess.run([*argv, path], cwd=self.root, input=raw, capture_output=True, timeout=20)
            if checked.returncode:
                raise ValueError(checked.stderr[-1100:].decode("utf-8", "replace").strip() or "creation validator rejected source")
        target = self.root / path
        if target.resolve() != target:
            raise ValueError("workspace path is outside the creatable source set")
        stream = target.open("xb")
        try:
            with stream:
                stream.write(raw)
            self.git("add", "--intent-to-add", "--", path)
        except (subprocess.SubprocessError, OSError):
            target.unlink()
            raise
        self.working.refresh()
        receipt = {"path": path, "before": None, "after": sha(raw), "revision": self.working.revision}
        if len(self.creatable) > 1:
            receipt["remaining_to_create"] = [name for name in self.creatable if not (self.root / name).is_file()]
        self.edits.append(receipt)
        return receipt

    def apply(self, handle: str, old: str, new: str) -> dict:
        if not self.manifest:
            raise ValueError("the host did not enable workspace edits")
        if not isinstance(old, str) or not isinstance(new, str) or not old or old == new:
            raise ValueError("provide a nonempty old string and a different replacement")
        if len(old) + len(new) > 24_000:
            raise ValueError("edit exceeds the per-action limit")
        body = self.working.read(handle=handle, record=False)
        rel = body["path"]
        path = self.root / rel
        if not self.allowed(rel) or path.resolve(strict=True) != path or path.is_symlink():
            raise ValueError("workspace path is outside the editable source set")
        if body["source"].count(old) != 1:
            raise ValueError("old string must occur exactly once inside the read anchor")
        raw = path.read_bytes()
        if sha(raw) != body["source_digest"]:
            raise ValueError("source changed since the read anchor was issued")
        encoding = raw.decode("utf-8")
        if "\r\n" in encoding:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        if encoding.count(old) != 1:
            raise ValueError("old string must occur exactly once in the file")
        replacement = encoding.replace(old, new, 1).encode("utf-8")
        if len(replacement) > MAX_SOURCE_BYTES:
            raise ValueError("replacement would exceed the source-size limit")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(replacement)
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            if path.resolve(strict=True) != path or path.read_bytes() != raw:
                raise ValueError("source changed while preparing the edit")
            os.replace(temporary, path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        self.working.refresh()
        receipt = {"path": rel, "handle": handle, "before": sha(raw), "after": sha(replacement),
                   "start": body["start"], "end": body["end"], "revision": self.working.revision}
        self.edits.append(receipt)
        return receipt

    def patch(self) -> dict:
        if not self.manifest:
            return {"patch": "", "patch_sha256": sha(b""), "changed_paths": [], "approved": False}
        diff = self.git("diff", "--no-color", "--no-ext-diff", "--no-textconv", self.base, "--")
        paths = [path for path in self.git("diff", "--name-only", "-z", self.base, "--").split("\0") if path]
        clean_metadata = (self.git("rev-parse", "HEAD").strip() == self.base
                          and not self.git("ls-files", "--other", "--exclude-standard").strip())
        return {"patch": diff, "patch_sha256": sha(diff.encode()), "changed_paths": paths,
                "approved": bool(paths) and clean_metadata and all(self.allowed(p) for p in paths)}

    def verify(self) -> dict:
        if not self.manifest:
            raise ValueError("the host did not configure a native verifier")
        before = self.patch()
        allowed_env = ("PATH", "SystemRoot", "WINDIR", "PATHEXT", "TEMP", "TMP", "LANG", "LC_ALL")
        env = {key: os.environ[key] for key in allowed_env if key in os.environ}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        started = time.monotonic()
        try:
            proc = subprocess.run(self.manifest["argv"], cwd=self.root, env=env, capture_output=True,
                                  timeout=min(int(self.manifest.get("timeout_seconds", 90)), 3600))
            status, output = proc.returncode, (proc.stdout + proc.stderr)[-8000:].decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            status, output = None, "Native verifier timed out"
        after = self.patch()
        receipt = {"exit_code": status, "output": output, "patch_sha256": after["patch_sha256"],
                   "passed": status == 0 and before["approved"] and after["approved"]
                   and before["patch_sha256"] == after["patch_sha256"],
                   "seconds": time.monotonic() - started}
        self.verifications.append(receipt)
        return receipt

    def state(self, include_patch: bool = True) -> dict:
        patch = self.patch()
        latest = self.verifications[-1] if self.verifications else {}
        verified = bool(patch["approved"] and latest.get("passed")
                        and latest["patch_sha256"] == patch["patch_sha256"])
        if not include_patch:
            patch.pop("patch")
        return {**patch, "verified": verified, "edits": self.edits,
                "verification": latest, "verifier_calls": len(self.verifications)}


class Service(GoblinService):
    def __init__(self, args):
        super().__init__()
        self.root = str(Path(args.repo).resolve(strict=True))
        self.working = self.working_set(self.root)
        self.workspace = RepairWorkspace(self.working, json.loads(Path(args.verifier).read_text(encoding="utf-8"))
                                         if args.verifier else None)
        self.controller = None
        self.controller_calls = 0
        self.controller_seconds = 0.
        self.retrieval_seconds = 0.
        if args.controller:
            from .controller import Controller
            self.controller = Controller(args.controller, args.depth, args.reset, args.threads)
        self.federation = None
        if args.federation:
            from .federation import Federation
            router = None
            if args.index_router:
                from .index_router import IndexRouter
                router = IndexRouter(args.index_router, args.threads)
            self.federation = Federation(Path(args.federation), router)
        elif args.index_router:
            raise ValueError("an index router requires a federation")
        self.expanded = False

    def tools(self) -> list[dict]:
        tools = super().tools()
        for tool in tools:
            schema = tool["inputSchema"]
            schema["properties"].pop("repo_path", None)
            if "repo_path" in schema["required"]:
                schema["required"].remove("repo_path")
            if self.federation and tool["name"] == "context.query":
                tool["description"] = ("Rank source across repository indexes. Prefix the query with @index-id "
                                       "to scope a repository or @workspace to find editable sources.")
        text = {"type": "string"}
        integer = {"type": "integer", "minimum": 0}
        schemas = {
            "host.edit": {"handle": text, "old": text, "new": text},
            "host.inspect": {}, "host.verify": {}, "host.final": {},
            "host.propose": {"task": text, "subgoal": text, "calls_left": integer, "tokens_left": integer},
        }
        if self.workspace.creatable:
            schemas["host.create"] = {"path": text, "content": text}
        return tools + [
            {"name": name, "description": "Host-authorized workspace operation.",
             "inputSchema": {"type": "object", "properties": props,
                             "required": list(props), "additionalProperties": False}}
            for name, props in schemas.items()
        ]

    def call(self, name: str, arguments: dict) -> dict:
        try:
            if name == "host.create":
                return envelope(self.workspace.create(**arguments))
            if name == "host.edit":
                return envelope(self.workspace.apply(**arguments))
            if name == "host.verify":
                return envelope(self.workspace.verify())
            if name == "host.inspect":
                summary = self.workspace.state()
                summary["patch"] = summary["patch"][-8000:]
                return envelope(summary)
            if name == "host.final":
                return envelope({**self.working.state(), **self.workspace.state(),
                                 **({"federation": self.federation.state()} if self.federation else {}),
                                 "controller_calls": self.controller_calls,
                                 "controller_seconds": self.controller_seconds,
                                 "retrieval_seconds": self.retrieval_seconds})
            if name == "host.propose":
                from .controller import observation
                state = self.workspace.state(include_patch=False)
                verification = state["verification"]
                started = time.monotonic()
                verifier_status = (verification.get("output", "").splitlines()[-5:]
                                   if verification and not verification.get("passed") else [])
                slate = observation(self.working, **arguments, expanded=self.expanded,
                                    dirty=bool(state["changed_paths"]), failed=bool(verifier_status),
                                    verifier_status="\n".join(verifier_status))
                inference_started = time.monotonic()
                self.retrieval_seconds += inference_started - started
                result = self.controller.propose(slate)
                self.controller_calls += 1
                self.controller_seconds += time.monotonic() - inference_started
                return envelope(result)
            if name == "context.expand":
                self.expanded = True
            started = time.monotonic()
            if self.federation:
                indexed = self.federated_call(name, arguments)
                if indexed is not None:
                    self.retrieval_seconds += time.monotonic() - started
                    return envelope(indexed)
            result = super().call(name, {**arguments, "repo_path": self.root})
            if name in {"context.query", "context.expand", "context.read", "compact_context"}:
                self.retrieval_seconds += time.monotonic() - started
            return result
        except (OSError, TypeError, ValueError, KeyError, UnicodeError, subprocess.SubprocessError) as exc:
            return envelope({"error": str(exc)}, error=True)

    def federated_call(self, name: str, arguments: dict) -> dict | None:
        if name == "context.query":
            query = arguments["query"]
            if query.startswith("@workspace "):
                payload = super().call(name, {**arguments, "query": query.removeprefix("@workspace "),
                                              "repo_path": self.root})
                if payload.get("isError"):
                    raise ValueError(payload["structuredContent"]["error"])
                return payload["structuredContent"]
            return self.federation.query(**arguments)
        if name == "context.read":
            reference = arguments.get("uri") or arguments.get("handle") or ""
            if reference.startswith(("index:", "indexed:")):
                return self.federation.read(**arguments)
        if name == "context.state":
            return {**self.working.state(), "federation": self.federation.state()}
        if name == "context.expand" and arguments.get("node_id") == "indexes":
            if arguments.get("predicate") or arguments.get("exclude"):
                raise ValueError("index exclusions are computed from the exact query")
            return self.federation.state()
        if name == "compact_context":
            found = self.federation.query(arguments["task"], limit=4)
            budget = int(arguments.get("budget_tokens", 1024))
            if not 128 <= budget <= 32768:
                raise ValueError("invalid packet budget")
            entries = [f"{row['uri']} {row['path']} (line {row['start']})" for row in found["candidates"]]
            packet = "\n".join(entries).encode("utf-8")[:budget * 4].decode("utf-8", errors="ignore")
            return {"packet": packet, "packet_tokens": (len(packet.encode("utf-8")) + 3) // 4,
                    "tokenizer": "estimate:utf8_bytes_div_4", "tool_calls": 1, "routing": found["routing"]}
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--verifier")
    parser.add_argument("--controller")
    parser.add_argument("--federation", default=os.environ.get("MCP05_FEDERATION"))
    parser.add_argument("--index-router", default=os.environ.get("MCP05_INDEX_ROUTER"))
    parser.add_argument("--depth", type=int)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    wire_log = os.environ.get("MCP05_SERVER_WIRE_LOG")
    serve(Service(parser.parse_args()), sys.stdin.buffer, sys.stdout.buffer,
          max_message_bytes=8 * 1024 * 1024, wire_log=Path(wire_log) if wire_log else None)


if __name__ == "__main__":
    main()
