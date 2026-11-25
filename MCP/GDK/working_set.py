"""GDK-02 durable, revision-bound capture store.

Store a *complete MCP-03 compact packet and metadata*, separate from the
provider-visible view. Content-addressed immutable blobs survive process
restarts; the latest-session index moves atomically under a single-writer lock.
This is not yet a full Goose tool-event journal or fact verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = "gdk.working_set.v1"
MAX_CAPTURE_BYTES = 8 * 1024 * 1024


class WorkingSetError(ValueError):
    pass


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gdk.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600) if hasattr(os, "fchmod") else None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as error:
        raise WorkingSetError(f"unreadable working set: {path.name}") from error
    if not isinstance(value, dict):
        raise WorkingSetError("working-set object must be JSON mapping")
    return value


class WorkingSetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _paths(self, session_id: str):
        if not isinstance(session_id, str) or not 0 < len(session_id) <= 512:
            raise WorkingSetError("session id must be nonempty (maximum 512)")
        token = _digest(session_id.encode("utf-8"))
        session_dir = self.root / "sessions" / token
        return session_dir, session_dir / "current.json"

    @contextmanager
    def _locked(self, session_dir: Path):
        session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = session_dir / ".write.lock"
        try:
            os.mkdir(lock, mode=0o700)
        except FileExistsError as error:
            raise WorkingSetError("session already has a writer") from error
        try:
            yield
        finally:
            os.rmdir(lock)

    def capture(
        self,
        *, session_id: str, catalog_epoch: str, packet: str,
        source: str = "mcp03", metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(catalog_epoch, str) or not catalog_epoch.strip():
            raise WorkingSetError("catalog/worktree revision required")
        if not isinstance(packet, str) or not packet:
            raise WorkingSetError("capture packet must be nonempty text")
        if not isinstance(source, str) or not source.strip():
            raise WorkingSetError("source identifier required")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise WorkingSetError("capture metadata must be JSON mapping")
        obj = {
            "version": SCHEMA, "session_id": session_id,
            "catalog_epoch": catalog_epoch, "source": source,
            "packet": packet, "metadata": metadata,
        }
        raw = _canonical(obj)
        if len(raw) > MAX_CAPTURE_BYTES:
            raise WorkingSetError("capture exceeds maximum 8 MiB")
        sha = _digest(raw)
        handle = f"sha256:{sha}"
        session_dir, head_path = self._paths(session_id)
        with self._locked(session_dir):
            current: dict[str, Any] | None = None
            if head_path.exists():
                current = _read_json(head_path)
                if current.get("version") != SCHEMA or current.get("session_id") != session_id:
                    raise WorkingSetError("session index version/identity mismatch")
            # Immutable, content-addressed records are never modified.
            blob_path = self.root / "blobs" / f"{sha}.json"
            if blob_path.exists():
                if _digest(blob_path.read_bytes()) != sha:
                    raise WorkingSetError("existing evidence blob checksum mismatch")
            else:
                _atomic_write(blob_path, raw)
            generation = int(current.get("generation", 0)) + 1 if current else 1
            # Worktree edits invalidate the active view, but preserve old blobs.
            refs = list(current.get("active_handles", [])) if (
                current and current.get("catalog_epoch") == catalog_epoch
            ) else []
            if handle not in refs:
                refs.append(handle)
            refs = refs[-32:]
            head = {
                "version": SCHEMA,
                "session_id": session_id,
                "catalog_epoch": catalog_epoch,
                "generation": generation,
                "active_handles": refs,
                "last_handle": handle,
            }
            _atomic_write(head_path, _canonical(head))
            return {
                "handle": handle, "catalog_epoch": catalog_epoch,
                "generation": generation, "active_count": len(refs),
                "packet_bytes": len(packet.encode("utf-8")),
            }

    def load(self, *, session_id: str, catalog_epoch: str) -> dict[str, Any]:
        _session_dir, head_path = self._paths(session_id)
        head = _read_json(head_path)
        if head.get("version") != SCHEMA or head.get("session_id") != session_id:
            raise WorkingSetError("stored session identity/version mismatch")
        if not catalog_epoch or head.get("catalog_epoch") != catalog_epoch:
            raise WorkingSetError("stale working set: catalog epoch changed")
        handle = head.get("last_handle")
        if not isinstance(handle, str) or not handle.startswith("sha256:"):
            raise WorkingSetError("session head has invalid evidence handle")
        sha = handle.removeprefix("sha256:")
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise WorkingSetError("invalid content digest in handle")
        blob_path = self.root / "blobs" / f"{sha}.json"
        try:
            raw = blob_path.read_bytes()
        except OSError as error:
            raise WorkingSetError("evidence blob unavailable") from error
        if _digest(raw) != sha:
            raise WorkingSetError("evidence blob checksum mismatch")
        obj = json.loads(raw)
        if not isinstance(obj, dict) or (
            obj.get("version") != SCHEMA
            or obj.get("session_id") != session_id
            or obj.get("catalog_epoch") != catalog_epoch
        ):
            raise WorkingSetError("evidence blob scope/version mismatch")
        return {"index": head, "capture": obj}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("capture", "load"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--catalog-epoch", required=True)
    parser.add_argument("--source", default="mcp03")
    args = parser.parse_args()
    try:
        store = WorkingSetStore(args.root)
        if args.operation == "capture":
            raw = sys.stdin.buffer.read(MAX_CAPTURE_BYTES + 1)
            if len(raw) > MAX_CAPTURE_BYTES:
                raise WorkingSetError("capture stdin too large")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise WorkingSetError("capture stdin must be JSON object")
            result = store.capture(
                session_id=args.session_id, catalog_epoch=args.catalog_epoch,
                packet=data["packet"], source=args.source,
                metadata=data.get("metadata", {}),
            )
        else:
            result = store.load(session_id=args.session_id, catalog_epoch=args.catalog_epoch)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (WorkingSetError, OSError, KeyError, ValueError, TypeError) as error:
        print(f"GDK working set rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
