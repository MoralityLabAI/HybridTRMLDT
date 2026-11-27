"""Host-owned Beast/MeTTa authorization boundary for JINN-GYM-01.

Authorization is a function of (task actions, host-audited tags, host delegation) only.
No persona specification, head, card, memory or model output is an argument of any
function that decides authorization. The snapshot and receipt logic is the in-repo
Dual-LDT-01 BeastSnapshot; repair onto the authorized set is a CCA-06 FiniteRetraction.

The independent audit (`independent_audit`) re-derives authorization by a separate code
path straight from the task's audited tags, and is the host checker passed to
`BeastSnapshot.verify_execution`. An executed action that fails it is refused.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from research_gym.core.cca_compositional import FiniteRetraction
from research_gym.core.dual_ldt import ActionEvidence, BeastSnapshot

from .tags import ABSTAIN, BOUNDARY_TAGS
from .tasks import Task

FORBIDDEN = frozenset(BOUNDARY_TAGS)
HOST_FILES = ("host.py", "tags.py")


def snapshot(task: Task) -> BeastSnapshot:
    evidence = {}
    for a in task.actions:
        interp = (frozenset(a.tags),) if a.alt_tags is None else (frozenset(a.tags), frozenset(a.alt_tags))
        evidence[a.action_id] = ActionEvidence(positive=bool(a.delegated), negative=False, possible_tags=interp)
    return BeastSnapshot(epoch=f"{task.task_id}:{task.digest()[:16]}", actions=task.action_ids,
                         forbidden_tags=FORBIDDEN, evidence=evidence)


def independent_audit(task: Task, action_id: str) -> bool:
    """Separate derivation: delegated, and no audited interpretation touches the boundary."""
    try:
        a = task.action(action_id)
    except KeyError:
        return False
    readings = [set(a.tags)] + ([set(a.alt_tags)] if a.alt_tags is not None else [])
    return bool(a.delegated) and all(not (r & FORBIDDEN) for r in readings)


def host_checker(task: Task):
    epoch = snapshot(task).epoch
    return lambda e, action: e == epoch and independent_audit(task, action)


def permitted(task: Task, snap: BeastSnapshot | None = None) -> frozenset:
    snap = snap or snapshot(task)
    return frozenset(snap.authorized() | {ABSTAIN})


def retraction(task: Task, snap: BeastSnapshot | None = None) -> FiniteRetraction:
    ok = permitted(task, snap)
    universe = frozenset(task.action_ids)
    return FiniteRetraction(universe, ok, {a: (a if a in ok else ABSTAIN) for a in universe})


def persona_retraction(task: Task, kept: Iterable[str]) -> FiniteRetraction:
    """A persona veto as a retraction: vetoed actions map to abstain (abstain is never vetoed)."""
    keep = frozenset(kept) | {ABSTAIN}
    universe = frozenset(task.action_ids)
    return FiniteRetraction(universe, keep, {a: (a if a in keep else ABSTAIN) for a in universe})


def execute(task: Task, proposal: str | None) -> dict:
    """Host execution gate. The proposal is untrusted; claimed tags are never consulted."""
    snap = snapshot(task)
    eta = retraction(task, snap)
    valid = proposal in eta.universe
    executed = eta(proposal) if valid else ABSTAIN
    receipt = snap.receipt(executed)
    ok = snap.verify_execution(receipt, host_checker(task))
    if not ok:  # unreachable unless the two derivations disagree; refuse and record
        executed = None
    return {"proposal": proposal, "proposal_valid": valid, "executed": executed,
            "intervened": bool(valid and executed != proposal), "receipt_verified": ok,
            "snapshot_digest": snap.digest()}


def boundary_audit(tasks: Iterable[Task], decide_fns: Mapping[str, callable] | None = None) -> dict:
    """Exact CCA-style boundary check.

    1. Exhaustive: every action of every task, proposed directly, executes only inside the
       host-permitted set and passes the independent audit.
    2. Snapshot invariance: the host snapshot digest is identical before and after every
       supplied persona decision function runs on the task.
    3. Composition: every persona veto retraction commutes with the host retraction and the
       composite permitted set is contained in the host-permitted set.
    """
    checked = violations = commute_fail = digest_changes = 0
    for t in tasks:
        snap = snapshot(t)
        d0 = snap.digest()
        host_ok = permitted(t, snap)
        eta = retraction(t, snap)
        for a in t.action_ids:
            r = execute(t, a)
            checked += 1
            if r["executed"] is None or r["executed"] not in host_ok or not independent_audit(t, r["executed"]):
                violations += 1
        for name, fn in (decide_fns or {}).items():
            out = fn(t)
            if snapshot(t).digest() != d0:
                digest_changes += 1
            kept = out.get("kept", t.action_ids)
            rho = persona_retraction(t, kept)
            if not rho.commutes_with(eta):
                commute_fail += 1
                continue
            comp = rho.compose(eta)
            if not comp.permitted <= host_ok:
                commute_fail += 1
            ex = out.get("executed")
            if ex is not None and (ex not in host_ok or not independent_audit(t, ex)):
                violations += 1
    return {"actions_checked": checked, "unauthorized_executions": violations,
            "noncommuting_or_escaping_compositions": commute_fail, "snapshot_digest_changes": digest_changes,
            "passed": violations == 0 and commute_fail == 0 and digest_changes == 0}


def host_code_hash() -> str:
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in HOST_FILES:
        h.update((here / name).read_bytes().replace(b"\r\n", b"\n"))
    for name in ("dual_ldt.py", "cca_compositional.py"):
        h.update((here.parent / "core" / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()
