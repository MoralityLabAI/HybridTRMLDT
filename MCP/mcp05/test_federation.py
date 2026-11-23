"""Contracts for pinned snapshots and cross-index exclusions."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .federation import Federation, build
from .service import Service


class FederationContracts(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.first = self.repository("first", {"source.metta": "(specialbeacon current)", "sealed/key.md": "sentinel"})
        self.second = self.repository("second", {"rules.lean": "theorem theorembeacon : True := trivial"})
        spec = self.root / "sources.json"
        spec.write_text(json.dumps({"repositories": [
            {"id": "a", "path": str(self.first)},
            {"id": "b", "path": str(self.second)},
        ]}), encoding="utf-8")
        self.indexes = self.root / "indexes"
        build(spec, self.indexes)

    def repository(self, name, files):
        target = self.root / name
        target.mkdir()
        for relative, text in files.items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        for args in (("init", "-q"), ("add", "--", "."), ("commit", "-qm", "fixture")):
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "-C", str(target), *args], check=True)
        return target

    def test_exact_exclusion_and_frozen_read(self):
        federation = Federation(self.indexes)
        result = federation.query("specialbeacon")
        self.assertEqual(result["routing"]["searched"], ["a"])
        self.assertEqual(result["routing"]["excluded"], ["b"])
        self.assertEqual(result["routing"]["revision"], federation.revision)
        uri = result["candidates"][0]["uri"]
        (self.first / "source.metta").write_text("uncommitted content", encoding="utf-8")
        read = federation.read(uri=uri)
        self.assertEqual(read["source"], "(specialbeacon current)")
        self.assertEqual(federation.read(handle=read["handle"]), read)

    def test_all_empty_and_scoped_query(self):
        federation = Federation(self.indexes)
        blocked = federation.query("sentinel")
        self.assertEqual(blocked["routing"]["excluded"], ["a", "b"])
        self.assertEqual(blocked["routing"]["searched"], [])
        self.assertEqual(blocked["candidates"], [])
        with self.assertRaises(ValueError):
            federation.query("@missing theorembeacon")
        scoped = federation.query("@b theorembeacon")
        self.assertEqual(scoped["routing"]["searched"], ["b"])
        self.assertEqual(scoped["candidates"][0]["path"], "b/rules.lean")

    def test_federated_pagination_has_no_duplicate_or_missing_candidate(self):
        federation = Federation(self.indexes)
        query = "specialbeacon theorembeacon"
        full = federation.query(query, limit=2)["candidates"]
        first = federation.query(query, limit=1)
        second = federation.query(query, limit=1, cursor=first["next_cursor"])
        self.assertEqual(first["candidates"] + second["candidates"], full)
        self.assertIsNone(second["next_cursor"])

    def test_service_routes_workspace_and_federated_sources(self):
        service = Service(SimpleNamespace(repo=self.first, controller=None, verifier=None,
                                          federation=self.indexes, index_router=None))
        indexed = service.call("context.query", {"query": "specialbeacon"})["structuredContent"]
        found = service.call("context.read", {"uri": indexed["candidates"][0]["uri"]})["structuredContent"]
        self.assertEqual(found["source"], "(specialbeacon current)")
        workspace = service.call("context.query", {"query": "@workspace specialbeacon"})["structuredContent"]
        self.assertEqual(workspace["candidates"][0]["uri"], "source.metta")
        self.assertTrue(service.call("host.edit", {"handle": found["handle"], "old": "current",
                                                   "new": "changed"})["isError"])


if __name__ == "__main__":
    unittest.main()
