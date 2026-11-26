"""Independent archive-only replay of the original COG-13D native artifacts."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

EXPECTED = {
    2048: "dff129389f1877141795b2c9a65663e4bfe40df7c8c25241277db6a839a14a45",
    4096: "c610eba2575855087e4efc21cb92ae6f399a2fff091a7a549531d0f623c25bbd",
}

def audit(cap, archive):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED[cap]
    with zipfile.ZipFile(archive) as z:
        read = lambda n: json.loads(z.read(n))
        reg, res = read("registry.json"), read("result.json")
        assert int(res["cgroupBefore"]["memory.max"]) == cap * 1048576
        assert int(res["cgroupAfter"]["memory.max"]) == cap * 1048576
        assert res["cgroupBefore"]["memory.swap.max"] == res["cgroupAfter"]["memory.swap.max"] == "0"
        totals, certs = {}, []
        for case in reg["cases"]:
            base = None
            for arm in reg["arms"]:
                name = f'{case["id"]}_{arm["name"]}'
                r = read(name + "/receipt.json")
                cache = []
                counts = dict(requests=0, executed=0, restored=0, hits=0, insertions=0, evictions=0, peak=0)
                assert r["error"] is None and r["process"]["exitCode"] == 0
                assert len(r["events"]) == len(case["requests"])
                if arm["cap"] == 0:
                    base = r
                else:
                    assert base["rootResponse"]["goals"] == r["rootResponse"]["goals"]
                for j, (q, e) in enumerate(zip(case["requests"], r["events"])):
                    assert e["id"] == q["id"]
                    ok, goals = e["result"]["accepted"], e["result"].get("goals")
                    assert ok == (q["expect"] != "reject")
                    if ok:
                        assert bool(goals) == (q["expect"] == "open")
                    b = base["events"][j]
                    assert b["result"]["accepted"] == ok and (not ok or b["result"]["goals"] == goals)
                    hits = [x for x in cache if q["actions"][:len(x["actions"])] == x["actions"]]
                    hit = max(hits, key=lambda x: len(x["actions"]), default=None)
                    restored = len(hit["actions"]) if hit else 0
                    if hit:
                        cache.remove(hit)
                        cache.append({**hit, "hits": hit["hits"] + 1, "touched": q["id"]})
                    ins = bool(arm["cap"] and ok and goals and not any(x["actions"] == q["actions"] for x in cache))
                    evicted = None
                    if ins:
                        cache.append(dict(actions=q["actions"], hits=0, touched=q["id"]))
                        if len(cache) > arm["cap"]:
                            victim = min(cache, key=lambda x: (
                                len(x["actions"]) * (x["hits"] + 1) if arm["policy"] == "value" else x["touched"],
                                x["touched"]))
                            evicted = victim["actions"]
                            cache.remove(victim)
                    # Dict equality is deliberately independent of serialization order.
                    assert (e["restored"], e["inserted"], e["evicted"], e["cache"]) == (restored, ins, evicted, cache)
                    assert e["executed"] == len(q["actions"]) - restored
                    assert e["executed"] + restored == b["executed"]
                    for k, v in dict(requests=1, executed=e["executed"], restored=restored,
                                     hits=int(restored > 0), insertions=int(ins), evictions=int(evicted is not None)).items():
                        counts[k] += v
                    counts["peak"] = max(counts["peak"], len(cache))
                    assert len(cache) <= arm["cap"]
                a = totals.setdefault(arm["name"], dict(validRoots=0, requests=0, executed=0, restored=0,
                                                         hits=0, insertions=0, evictions=0, peak=0))
                a["validRoots"] += 1
                for k, v in counts.items():
                    a[k] = max(a[k], v) if k == "peak" else a[k] + v
        for cert in res["certificates"]:
            root, s = cert["root"], cert["script"]
            hashes = []
            for j in range(2):
                d = f"{root}_certificate_{s}_{j}/"
                rec, log, source = read(d+"receipt.json"), z.read(d+"stdout.log"), z.read(d+"Certificate.lean")
                assert hashlib.sha256(source).hexdigest() == rec["sourceSha256"]
                assert hashlib.sha256(log).hexdigest() == rec["process"]["logSha256"]
                assert rec["passed"] and rec["process"]["exitCode"] == 0
                assert set(rec["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
                assert b"sorryAx" not in log
                hashes.append(rec["sourceSha256"])
            assert hashes[0] == hashes[1]
            certs.append(root + ":" + str(s))
        neg, log = read("negative_certificate/receipt.json"), z.read("negative_certificate/stdout.log")
        assert neg["process"]["exitCode"] != 0 and b"error(lean.unknownIdentifier)" in log
        assert b"cog13d_missing_identifier" in log
        assert len(certs) == 16 and sum(a["requests"] for a in totals.values()) == 1680
        return dict(totals=totals, certificates=len(certs), negativeRejected=True,
                    cgroupPeakBytes=int(res["cgroupAfter"]["memory.peak"]),
                    timing={k:{key:v[key] for key in ("queryWallMs","userSeconds","systemSeconds","processWallMs")}
                            for k,v in res["summary"].items()})

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    for cap in (2048,4096):
        print(cap, json.dumps(audit(cap, folder / f"cog13d-native-{cap}-mib.zip")))
