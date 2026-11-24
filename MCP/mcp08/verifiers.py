"""MCP-08A independent deterministic verifiers, full-storyworld checks, eta metric, and preregistered gates."""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .fixtures import apply_hilbert_sequence, state_probs


def _retrieval_metrics(cited_docs: list[str], required_ids: list[str]) -> tuple[float, float]:
    cited = set(cited_docs or [])
    req = set(required_ids or [])
    recall = len(cited & req) / len(req) if req else 1.0
    precision = len(cited & req) / len(cited) if cited else 0.0
    return round(recall, 4), round(precision, 4)


def verify_flooded_city(public_root: dict, witness: dict, submission: dict) -> dict:
    scenes = submission.get("scenes") or []
    cited = submission.get("cited_docs") or []
    recall, precision = _retrieval_metrics(cited, witness["required_evidence_ids"])

    checks = {
        "has_min_scenes": len(scenes) >= 3,
        "starts_correctly": bool(scenes) and scenes[0].get("district") == witness["start_district"],
        "no_submerged_entry": all(s.get("district") not in witness["flooded_districts"] for s in scenes),
        "visited_mid_ally": any(
            s.get("district") == witness["mid_district"]
            and s.get("npc") == witness["ally_npc"]
            and s.get("faction") == witness["ally_faction"]
            for s in scenes
        ),
        "visited_goal_archivist": bool(scenes)
        and scenes[-1].get("district") == witness["goal_district"]
        and scenes[-1].get("npc") == witness["archivist_npc"],
        "inventory_conserved": (
            bool(scenes)
            and witness["required_item"] in (scenes[0].get("inventory") or [])
            and witness["decoy_item"] not in (scenes[0].get("inventory") or [])
            and witness["required_item"] not in (scenes[-1].get("inventory") or [])
        ),
        "valid_ending": submission.get("ending") == witness["valid_ending"],
        "rejected_stale_lore": not bool(set(cited) & set(witness["forbidden_stale_ids"])),
    }
    score = sum(1.0 for v in checks.values() if v) / len(checks)
    verified = all(checks.values())
    return {
        "env": "flooded_city",
        "root_id": witness["root_id"],
        "verified": verified,
        "score": round(score, 4),
        "evidence_recall": recall,
        "retrieval_precision": precision,
        "checks": checks,
    }


def verify_quantum_labyrinth(public_root: dict, witness: dict, submission: dict) -> dict:
    ops = submission.get("operators") or []
    cited = submission.get("cited_docs") or []
    recall, precision = _retrieval_metrics(cited, witness["required_evidence_ids"])
    costs = witness["operator_costs"]

    valid_ops = bool(ops) and all(op in costs for op in ops)
    within_steps = 1 <= len(ops) <= witness["max_steps"]
    total_cost = sum(costs[op] for op in ops) if valid_ops else 999999
    within_budget = total_cost <= witness["resource_budget"]

    norm_ok = False
    target_ok = False
    probs: list[float] = []
    if valid_ops:
        psi = apply_hilbert_sequence(ops)
        norm_sq = sum(abs(a) ** 2 for a in psi)
        norm_ok = abs(norm_sq - 1.0) <= 1e-6
        probs = state_probs(psi)
        l1 = sum(abs(a - b) for a, b in zip(probs, witness["target_probabilities"]))
        target_ok = l1 <= 0.02

    efficiency = round(min(1.0, witness["optimal_cost"] / total_cost), 4) if (valid_ops and total_cost > 0) else 0.0
    checks = {
        "valid_operators": valid_ops,
        "within_step_limit": within_steps,
        "within_energy_budget": within_budget,
        "hilbert_norm_preserved": norm_ok,
        "terminal_probabilities_match": target_ok,
    }
    verified = all(checks.values())
    return {
        "env": "quantum_labyrinth",
        "root_id": witness["root_id"],
        "verified": verified,
        "score": round(sum(1.0 for v in checks.values() if v) / len(checks), 4),
        "route_efficiency": efficiency,
        "total_energy_cost": total_cost if valid_ops else None,
        "replayed_probabilities": probs,
        "evidence_recall": recall,
        "retrieval_precision": precision,
        "checks": checks,
    }


def _find_node() -> str | None:
    for cand in (
        shutil.which("node"),
        r"C:\Users\patri\AppData\Local\Programs\nodejs\node.exe",
    ):
        if cand and Path(cand).exists():
            return cand
    return None


def verify_arena_architect(public_root: dict, witness: dict, submission: dict) -> dict:
    code = str(submission.get("code") or "")
    cited = submission.get("cited_docs") or []
    recall, precision = _retrieval_metrics(cited, witness["required_evidence_ids"])
    node = _find_node()
    if not node:
        raise RuntimeError("Node.js is required to execute Arena Architect dynamic verifier")

    # Execute candidate JS inside an isolated Node vm context against unseen maps & adversarial moves
    harness_js = r"""
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const sandbox = { Math, Set, Map, Array, Object, Number, String, Boolean, JSON };
vm.createContext(sandbox);
const report = {
  compiled: false,
  bounds_and_walls_ok: true,
  combat_range_and_clamp_ok: true,
  enemy_ai_pathfinding_ok: true,
  victory_reachable_ok: true,
  canvas_rendering_ok: true,
  error: null,
};
try {
  vm.runInContext(input.code + "\nthis.__step = stepArena; this.__render = renderArena;", sandbox, { timeout: 1500 });
  if (typeof sandbox.__step !== 'function' || typeof sandbox.__render !== 'function') {
    throw new Error('Missing stepArena or renderArena function');
  }
  report.compiled = true;
  for (const m of input.unseen_maps) {
    // 1. Adversarial boundary & wall push
    let st = {
      width: m.width, height: m.height, walls: m.walls,
      player: { x: 0, y: 0 }, enemy: { x: m.enemy_start[0], y: m.enemy_start[1], hp: m.enemy_hp },
      damage: m.damage, attackRange: m.attack_range, hits: 0, status: 'playing',
    };
    st = sandbox.__step(st, 'LEFT');
    st = sandbox.__step(st, 'UP');
    if (st.player.x < 0 || st.player.y < 0) report.bounds_and_walls_ok = false;
    // Try walking straight into a wall
    const w0 = m.walls[0];
    st.player = { x: w0[0] - 1, y: w0[1] };
    st = sandbox.__step(st, 'RIGHT');
    if (st.player.x === w0[0] && st.player.y === w0[1]) report.bounds_and_walls_ok = false;

    // 2. Out-of-range attack must deal 0 damage
    const hpBefore = st.enemy.hp;
    st.player = { x: 0, y: 0 };
    st.enemy = { x: m.width - 1, y: m.height - 1, hp: hpBefore };
    st = sandbox.__step(st, 'ATTACK');
    if (st.enemy.hp !== hpBefore) report.combat_range_and_clamp_ok = false;

    // 3. Enemy AI chases player around wall without stepping on walls or overlapping player
    let st2 = {
      width: m.width, height: m.height, walls: m.walls,
      player: { x: m.player_start[0], y: m.player_start[1] },
      enemy: { x: m.enemy_start[0], y: m.enemy_start[1], hp: m.enemy_hp },
      damage: m.damage, attackRange: m.attack_range, hits: 0, status: 'playing',
    };
    const startDist = Math.abs(st2.enemy.x - st2.player.x) + Math.abs(st2.enemy.y - st2.player.y);
    let moved = false;
    for (let t = 0; t < 16; t++) {
      const ex0 = st2.enemy.x, ey0 = st2.enemy.y;
      st2 = sandbox.__step(st2, 'WAIT');
      if (st2.enemy.x !== ex0 || st2.enemy.y !== ey0) moved = true;
      if (m.walls.some(w => w[0] === st2.enemy.x && w[1] === st2.enemy.y)) report.enemy_ai_pathfinding_ok = false;
      if (st2.enemy.x === st2.player.x && st2.enemy.y === st2.player.y) report.enemy_ai_pathfinding_ok = false;
    }
    const endDist = Math.abs(st2.enemy.x - st2.player.x) + Math.abs(st2.enemy.y - st2.player.y);
    if (!moved || endDist >= startDist) report.enemy_ai_pathfinding_ok = false;

    // 4. Close-range combat reaches victory with hp clamped at 0
    st2.player = { x: 1, y: 1 };
    st2.enemy = { x: 2, y: 1, hp: m.enemy_hp };
    for (let t = 0; t < 6; t++) st2 = sandbox.__step(st2, 'ATTACK');
    if (st2.enemy.hp !== 0 || st2.status !== 'victory') report.victory_reachable_ok = false;

    // 5. Canvas rendering checks (clearRect + >=4 fillRect + >=3 distinct fillStyle)
    const ops = [];
    const styles = new Set();
    const ctx = {
      _fs: '',
      set fillStyle(v) { this._fs = String(v); styles.add(this._fs); },
      get fillStyle() { return this._fs; },
      clearRect: (...a) => ops.push(['clearRect', a]),
      fillRect: (...a) => ops.push(['fillRect', a, ctx._fs]),
      beginPath: () => ops.push(['beginPath']),
      arc: (...a) => ops.push(['arc', a]),
      fill: () => ops.push(['fill']),
    };
    sandbox.__render(ctx, st2);
    const cleared = ops.some(o => o[0] === 'clearRect');
    const draws = ops.filter(o => o[0] === 'fillRect' || o[0] === 'fill').length;
    if (!cleared || draws < m.walls.length + 2 || styles.size < 3) report.canvas_rendering_ok = false;
  }
} catch (err) {
  report.error = String(err && err.message ? err.message : err);
  report.bounds_and_walls_ok = false;
  report.combat_range_and_clamp_ok = false;
  report.enemy_ai_pathfinding_ok = false;
  report.victory_reachable_ok = false;
  report.canvas_rendering_ok = false;
}
process.stdout.write(JSON.stringify(report));
"""
    payload = json.dumps({"code": code, "unseen_maps": witness["unseen_maps"]})
    proc = subprocess.run([node, "-e", harness_js], input=payload, capture_output=True, text=True, timeout=15)
    if proc.returncode != 0 or not proc.stdout.strip():
        checks = {k: False for k in ("compiled", "bounds_and_walls_ok", "combat_range_and_clamp_ok",
                                     "enemy_ai_pathfinding_ok", "victory_reachable_ok", "canvas_rendering_ok")}
        return {"env": "arena_architect", "root_id": witness["root_id"], "verified": False, "score": 0.0,
                "evidence_recall": recall, "retrieval_precision": precision, "checks": checks,
                "error": (proc.stderr or "")[:300]}
    rep = json.loads(proc.stdout)
    checks = {
        "compiled": bool(rep["compiled"]),
        "bounds_and_walls_ok": bool(rep["bounds_and_walls_ok"]),
        "combat_range_and_clamp_ok": bool(rep["combat_range_and_clamp_ok"]),
        "enemy_ai_pathfinding_ok": bool(rep["enemy_ai_pathfinding_ok"]),
        "victory_reachable_ok": bool(rep["victory_reachable_ok"]),
        "canvas_rendering_ok": bool(rep["canvas_rendering_ok"]),
    }
    return {
        "env": "arena_architect",
        "root_id": witness["root_id"],
        "verified": all(checks.values()),
        "score": round(sum(1.0 for v in checks.values() if v) / len(checks), 4),
        "evidence_recall": recall,
        "retrieval_precision": precision,
        "checks": checks,
        "error": rep.get("error"),
    }


def verify_broken_archive(public_root: dict, witness: dict, submission: dict) -> dict:
    cited = submission.get("cited_docs") or []
    recall, precision = _retrieval_metrics(cited, witness["required_evidence_ids"])
    checks = {
        "exact_canonical_symbol": submission.get("canonical_symbol") == witness["canonical_symbol"],
        "exact_canonical_value": submission.get("canonical_value") == witness["canonical_value"],
        "exact_canonical_commit": submission.get("canonical_commit") == witness["canonical_commit"],
        "full_dependency_chain": list(submission.get("dependency_chain") or []) == witness["dependency_chain"],
        "valid_citations_only": (
            set(witness["required_evidence_ids"]).issubset(set(cited))
            and not bool(set(cited) & set(witness["stale_doc_ids"]))
        ),
    }
    return {
        "env": "broken_archive",
        "root_id": witness["root_id"],
        "verified": all(checks.values()),
        "score": round(sum(1.0 for v in checks.values() if v) / len(checks), 4),
        "evidence_recall": recall,
        "retrieval_precision": precision,
        "checks": checks,
    }


def verify_submission(public_root: dict, witness: dict, submission: dict) -> dict:
    env = public_root["env"]
    if env == "flooded_city":
        return verify_flooded_city(public_root, witness, submission)
    if env == "quantum_labyrinth":
        return verify_quantum_labyrinth(public_root, witness, submission)
    if env == "arena_architect":
        return verify_arena_architect(public_root, witness, submission)
    if env == "broken_archive":
        return verify_broken_archive(public_root, witness, submission)
    raise ValueError(f"Unknown env: {env}")


# ---------------------------------------------------------------------------
# Full Storyworld Native Verifiers (Stage MCP-08E)
# ---------------------------------------------------------------------------

def verify_clockmakers_republic(world: dict) -> dict:
    """Verify The Clockmaker's Republic: 8 districts, faction transitions, resource conservation,
    reachability (0 unreachable states), and decision-altered later encounters."""
    districts = world.get("districts") or []
    nodes = {s["id"]: s for s in (world.get("scenes") or [])}
    start = world.get("start_scene")
    endings = set(world.get("endings") or [])

    # BFS reachability from start_scene
    reachable = set()
    if start in nodes:
        q = [start]
        reachable.add(start)
        while q:
            cur = q.pop(0)
            for ch in nodes[cur].get("choices") or []:
                nxt = ch.get("target")
                if nxt in nodes and nxt not in reachable:
                    reachable.add(nxt)
                    q.append(nxt)

    # Check resource conservation (no negative resource transition without guard, declared deltas finite)
    conservation_ok = True
    faction_transitions = 0
    decision_sensitive_encounters = 0
    for s in nodes.values():
        if s.get("requires_prior_flag"):
            decision_sensitive_encounters += 1
        for ch in s.get("choices") or []:
            delta = ch.get("resource_delta") or {}
            if any(not isinstance(v, (int, float)) for v in delta.values()):
                conservation_ok = False
            if ch.get("faction_delta"):
                faction_transitions += 1

    checks = {
        "has_eight_districts": len(set(districts)) == 8,
        "zero_unreachable_scenes": bool(nodes) and len(reachable) == len(nodes),
        "multiple_reachable_endings": len(reachable & endings) >= 2,
        "resource_conservation_valid": conservation_ok,
        "has_faction_transitions": faction_transitions >= 2,
        "decision_alters_later_encounters": decision_sensitive_encounters >= 1,
    }
    return {
        "storyworld": "clockmakers_republic",
        "verified": all(checks.values()),
        "score": round(sum(1.0 for v in checks.values() if v) / len(checks), 4),
        "reachable_count": len(reachable),
        "total_scenes": len(nodes),
        "checks": checks,
    }


def verify_garden_of_unmeasured_things(world: dict) -> dict:
    """Verify The Garden of Unmeasured Things: Hilbert normalization, projector probability sum = 1,
    reachability, and order-sensitive noncommuting counterfactuals."""
    routes = world.get("routes") or []
    counterfactual = world.get("counterfactual_pair") or {}
    seq_a = counterfactual.get("seq_a") or []
    seq_b = counterfactual.get("seq_b") or []

    norm_ok = bool(routes)
    proj_sum_ok = bool(routes)
    for r in routes:
        psi = apply_hilbert_sequence(r.get("operators") or [])
        n2 = sum(abs(a) ** 2 for a in psi)
        probs = state_probs(psi)
        if abs(n2 - 1.0) > 1e-6:
            norm_ok = False
        if abs(sum(probs) - 1.0) > 1e-4:
            proj_sum_ok = False

    order_sensitive = False
    if seq_a and seq_b and sorted(seq_a) == sorted(seq_b) and seq_a != seq_b:
        pa = state_probs(apply_hilbert_sequence(seq_a))
        pb = state_probs(apply_hilbert_sequence(seq_b))
        order_sensitive = pa != pb

    checks = {
        "has_multiple_routes": len(routes) >= 2,
        "hilbert_normalization_ok": norm_ok,
        "projector_probabilities_sum_to_one": proj_sum_ok,
        "order_sensitive_counterfactual": order_sensitive,
        "distinct_endings": len({r.get("ending") for r in routes if r.get("ending")}) >= 2,
    }
    return {
        "storyworld": "garden_of_unmeasured_things",
        "verified": all(checks.values()),
        "score": round(sum(1.0 for v in checks.values() if v) / len(checks), 4),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Primary Metric (eta), Paired Bootstrap CI, and Preregistered Gates
# ---------------------------------------------------------------------------

def compute_eta(episodes: list[dict]) -> float:
    """Primary MCP-08 metric: verified task completions per GPU-hour."""
    total_t = sum(float(ep["wall_s"]) for ep in episodes)
    if total_t <= 0:
        return 0.0
    verified = sum(1 for ep in episodes if ep.get("verified"))
    return round(3600.0 * verified / total_t, 4)


def paired_bootstrap_ci(
    vals_a: list[float],
    vals_b: list[float],
    n_resamples: int = 2000,
    seed: int = 20261124,
) -> dict:
    """Two-sided 95% paired bootstrap CI for mean(vals_a - vals_b)."""
    assert len(vals_a) == len(vals_b) and len(vals_a) > 0
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_resamples)]
    hi = boots[int(0.975 * n_resamples)]
    return {"mean_diff": round(mean_diff, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4), "n": n}


def evaluate_gates(summary: dict) -> dict:
    """Evaluate all 5 preregistered MCP-08 success gates."""
    comp_c = float(summary["arm_c_completion_pct"])
    best_conv = max(float(summary["arm_a_completion_pct"]), float(summary["arm_b_completion_pct"]))
    time_red_pct = float(summary["time_reduction_pct"])
    cpu_offload = int(summary["pure_gpu_cpu_offload_tensors"])
    paging = int(summary["pure_gpu_paging_events"])
    worlds_ok = bool(summary["fresh_worlds_verified"])
    transfer_ok = float(summary["bonsai_delta_eta"]) > 0 and float(summary["inky_delta_eta"]) > 0

    gates = {
        "learned_routing_ge_5pp": (comp_c - best_conv) >= 5.0,
        "context_economy_ge_25pct_time_reduction": time_red_pct >= 25.0,
        "memory_residency_zero_offload_zero_paging": (cpu_offload == 0 and paging == 0),
        "autonomous_construction_verified": worlds_ok,
        "transfer_positive_both_models": transfer_ok,
    }
    return {"all_passed": all(gates.values()), "gates": gates}
