"""Prepare train-only COG-13H inputs from the exact audited COG-13C archive."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

SHA = "d99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea"
MENU = ["assumption", "rfl", "constructor", "intro hnew", "simp", "simp_all", "omega", "linarith", "ring", "norm_num", "positivity", "aesop", "exact h", "all_goals assumption", "all_goals omega", "all_goals ring"]


def prepare(archive):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SHA
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("export_manifest.json"))
        for name, digest in manifest["hashes"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        r = json.loads(z.read("result.json"))
        assert r["cases"] == json.loads(z.read("task_registry.json"))
    cases = {c["id"]: c for c in r["cases"]}
    roots = {c["id"] for c in cases.values() if c["split"] == "train"}
    assert len(cases) == 24 and len(roots) == 8
    train, fit, observed = [], {}, {}
    for x in r["teacher"]["rows"]:
        assert x["split"] == "train" and x["root"] in roots and x["action"] in MENU
        y = int(x["receipt"]["accepted"] and x["receipt"]["goals"] != x["goals"])
        assert y == x["target"]
        train.append({k: x[k] for k in ("root", "split", "goals", "action", "target")})
        key = (x["root"], tuple(x["goals"]), tuple(x["prefix"]))
        g = fit.setdefault(key, {})
        assert x["action"] not in g
        g[x["action"]] = y
    assert len(train) == 224 and sum(x["target"] for x in train) == 82
    assert len(fit) == 14 and all(set(g) == set(MENU) for g in fit.values())
    for run in r["runs"]:
        for root in run["roots"]:
            c = root["c"]
            assert c == cases[c["id"]] and c["id"] not in roots
            for e in root["events"]:
                if e["reply"].get("unknown"):
                    continue
                key = (c["id"], tuple(e["before"]), tuple(e["prefix"][:-1]))
                g = observed.setdefault(key, {})
                y = int(e["reply"]["accepted"] and e["reply"]["goals"] != e["before"])
                assert e["action"] in MENU and (e["action"] not in g or g[e["action"]] == y)
                g[e["action"]] = y
    def groups(data, diagnostic=False):
        return [{"root": k[0], "split": "fit_diagnostic" if diagnostic else cases[k[0]]["split"],
                 "goals": list(k[1]), "actions": [a for a in MENU if a in g],
                 "labels": [g[a] for a in MENU if a in g]} for k, g in data.items()]
    retrospective = groups(observed)
    assert len(retrospective) == 121 and sum(len(x["labels"]) for x in retrospective) == 370
    return {"archiveSha256": SHA, "train": train, "fitDiagnostic": groups(fit, True),
            "retrospective": retrospective, "oldStatements": [c["statement"] for c in cases.values()]}


if __name__ == "__main__":
    value = prepare(Path(sys.argv[1]))
    with Path(sys.argv[2]).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({"train": len(value["train"]), "fitStates": len(value["fitDiagnostic"]),
                      "retrospectiveStates": len(value["retrospective"]),
                      "retrospectiveLabels": sum(len(x["labels"]) for x in value["retrospective"])}))
