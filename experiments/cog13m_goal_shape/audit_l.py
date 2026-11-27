"""Retrospective diagnosis of L's failed modular trajectories."""
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

SHA = "d9a8884acacfa8250466e1a85262ddd6e60723b66980effdcae8f94d186b79f9"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def audit(path):
    need(hashlib.sha256(path.read_bytes()).hexdigest() == SHA, "archive pin")
    with zipfile.ZipFile(path) as z:
        result = json.loads(z.read("native/result.json"))
        need(result["complete"] and not result["unknowns"] and not result["errors"]
             and all(result["gates"].values()), "L run custody")
        summary, roots = {}, []
        for arm in result["totals"]:
            counts = Counter()
            rows = [x for x in result["rows"] if x["family"] == "modular" and x["arm"] == arm]
            need(len(rows) == 4 and not any(x["solved"] for x in rows), "modular roster")
            for x in rows:
                folder = f"native/{x['root']}_{arm}/"
                e = json.loads(z.read(folder+"episode.json"))
                need(e["reason"] == x["reason"] == "probe_budget" and not e["solved"]
                     and e["closed"] is None and len(e["events"]) == 48, "failed tape")
                ct = Counter()
                first = None
                for v in e["events"]:
                    i = v["id"]
                    req = json.loads(z.read(folder+f"session/request-{i}.json"))
                    reply = json.loads(z.read(folder+f"session/response-{i}.json"))
                    need(req == {"id": i, "actions": v["actions"], "stop": False}
                         and reply == v["reply"], "raw receipt")
                    t, r = v["actions"][-1], reply["result"]
                    ct["probes"] += 1
                    ct["root_attempts"] += len(v["actions"]) == 1
                    ct["omega_attempts"] += t in ("omega", "all_goals omega")
                    if r["accepted"] and len(r["goals"]) > 8:
                        ct["wide_result"] += 1
                    if r["accepted"] and t == "constructor":
                        ct["constructor_accepted"] += 1
                        before = v["before"][0].rsplit("⊢",1)[-1].strip()
                        after = r["goals"][0].rsplit("⊢",1)[-1].strip() if r["goals"] else ""
                        if "%" in before and "∧" not in before and "≤" in before and ".le" in after:
                            ct["atomic_strengthening"] += 1
                            if first is None:
                                first = {"before": before, "after": after, "prefix": v["actions"]}
                counts.update(ct)
                roots.append({"root": x["root"], "arm": arm, "counts": dict(ct), "example": first})
            summary[arm] = dict(counts)
    need(sum(v["probes"] for v in summary.values()) == 960, "receipt count")
    need(summary["frequency"]["atomic_strengthening"] == 24 and
         all(summary[a]["atomic_strengthening"] == 27 for a in summary if a != "frequency"),
         "diagnosis tally")
    return {"schema": "cog13m.l_failure_audit.v1", "archiveSha256": SHA,
            "summary": summary, "roots": roots,
            "scope": "observed display changes and admitted goal counts on spent roots; not causal repair evidence"}


if __name__ == "__main__":
    result = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["summary"], indent=2))
