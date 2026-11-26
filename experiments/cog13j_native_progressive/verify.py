"""Check reproducible discrete archive/fixture fields; timings are observations."""
import hashlib
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out = Path(sys.argv[1])
frozen = json.loads((here / "archive_timing.json").read_text())
got = json.loads((out / "archive_timing.json").read_text())
fixture = json.loads((out / "fixture.json").read_text())

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

for name, expected in frozen["sourceBlobs"].items():
    data = (here / (name + ".py")).read_bytes()
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    require(actual == expected, "executed source blob differs: " + name)
require(got["archiveSha256"] == frozen["archiveSha256"], "archive custody")
require(set(got["arms"]) == set(frozen["arms"]), "arm roster")
keys = ("decisions", "bytes", "stages", "refinements", "exactFallbacks", "senderAbstentions")
for arm, expected in frozen["arms"].items():
    actual = got["arms"][arm]
    require(all(actual[k] == expected[k] for k in keys), f"{arm}: discrete replay mismatch")
    require(all(type(actual[k]) is int and actual[k] >= 0 for k in
                ("senderNs", "receiverNs", "auditNs")), f"{arm}: invalid timing")
require(fixture == frozen["fixture"], "fixture mismatch")
result = {
    "schema": "cog13j.archive_replay.v1",
    "passed": True,
    "archive": got["archiveSha256"],
    "arms": len(got["arms"]),
    "menus": 449,
    "decisions": sum(a["decisions"] for a in got["arms"].values()),
    "fixture": fixture,
    "timingScope": "CPU timings recorded anew; not compared for equality",
}
(out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
