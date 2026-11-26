"""Compare exact discrete evidence fields across replay environments."""
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
frozen = json.loads(Path(__file__).with_name("results.json").read_text())
old = json.loads((folder / "shared" / "summary.json").read_text())
new = json.loads((folder / "certified" / "certified_summary.json").read_text())
a, b = old["evaluation"], new["evaluation"]
assert old["archiveSha256"] == new["sourceSha256"] == frozen["source"]["archiveSha256"]
assert old["archivePacketsVerified"] == new["archivedPacketsVerified"] == frozen["packets"]
assert a["decisions"] == b["decisions"] == frozen["menus"]
assert new["uniqueFeatureMenus"] == frozen["uniqueFeatureMenus"]
assert a["naiveDenseBytes"] == frozen["naiveDenseBytes"]
assert a["originalBytes"] == frozen["originalMixedBytes"]
for name, expected in frozen["firstProtocol"].items():
    assert {k: a[name][k] for k in expected} == expected, name
for name, expected in frozen["certified"].items():
    assert {k: b["modes"][name][k] for k in expected} == expected, name
assert b["policies"] == frozen["progressive"]
d = new["deeper"]
assert d["decisions"] == frozen["deeper"]["menus"]
assert old["readouts"]["deeper"]["exact_action_q4"]["agrees"] == frozen["deeper"]["exactActionQ4Agrees"]
assert d["modes"]["pre_i8"]["certified"] == frozen["deeper"]["int8Certified"]
assert d["modes"]["pre_f16"]["certified"] == frozen["deeper"]["halfCertified"]
assert d["policies"]["i8_f16_exact"]["bytes"] == frozen["deeper"]["progressiveBytes"]
print("Frozen discrete receipt reproduced: 493 original packets, 449 evaluation decisions.")
