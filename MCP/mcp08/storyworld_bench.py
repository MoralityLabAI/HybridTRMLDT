"""Independent long-horizon native Storyworld benchmark (5-50 decisions) for MCP-08.

Built directly on MCP-05's exact two-qubit Hilbert algebra (`MCP.mcp05.hilbert_storyworld`):
- Independent procedurally synthesized worlds across 4 horizon buckets:
  `5-10`, `12-24`, `25-38`, `40-50` sequential decisions.
- Branching narrative DAGs with deceptive distractor scenes (1-2 step traps),
  convergent sub-DAGs (testing transposition caching), and irreversible transitions
  (both necessary act gates and irreversible trap doors).
- Every world is algebraically verified at generation to possess a secret-ending route
  with a native noncommuting adjacent-operation counterfactual, and ALL witnesses are
  mechanically stripped (`strip_world_witnesses`) prior to evaluation.
- Scene-option transition records are stored in external structured physical memory
  (`ExtendedPhysicalMemory` at 32K..256K) across diverse spatial geometries.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from MCP.mcp05.hilbert_storyworld import (
    EPSILON,
    START,
    apply,
    counterfactual,
    order_changes_operator,
    pos_conditions,
    probability,
    replay_path,
    signature,
)
from MCP.mcp07.curriculum import HopSpec, coarse_scale
from .memory256k import ExtendedPhysicalMemory

HORIZON_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("short_5_10", 5, 10),
    ("medium_12_24", 12, 24),
    ("long_25_38", 25, 38),
    ("ultra_40_50", 40, 50),
)

# Eight distinct non-commuting two-qubit gate sequences from MCP-05's GATES
OPERATOR_POOL: Dict[str, Tuple[str, ...]] = {
    "op_0": ("H0", "CX01"),
    "op_1": ("S0", "H0"),
    "op_2": ("H1", "CX10"),
    "op_3": ("S1", "H1"),
    "op_4": ("H0", "H1"),
    "op_5": ("CX01", "CX10"),
    "op_6": ("S0", "S1"),
    "op_7": ("H0", "S1"),
}
OPERATOR_NAMES: Tuple[str, ...] = tuple(sorted(OPERATOR_POOL.keys()))
CLUE_POOL: Tuple[str, ...] = ("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7")
PHASE_PROJECTORS: Tuple[str, ...] = ("X0+", "X0-", "X1+", "X1-", "Y0+", "Y0-", "Y1+", "Y1-")
BASIS_PROJECTORS: Tuple[str, ...] = ("00", "01", "10", "11")


def _sign(v: int) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def make_spatial_hop(
    rng: np.random.Generator,
    w: int,
    h: int,
    geom_kind: int,
    used_records: set,
) -> HopSpec:
    """Create a spatial memory query across 6 challenging MCP-07/08 geometries."""
    # 0: horizontal (row), 1: vertical (col), 2: diagonal (morton), 3: isotropic (hilbert),
    # 4: regime_shift (row/col -> hilbert), 5: corrupted_replica (stale decoy -> orthogonal mirror)
    for _ in range(32):
        mode = geom_kind % 6
        decoy_slot = -1
        mhx = mhy = 0
        msx = msy = 0.0

        if mode == 0:  # horizontal
            ax = int(rng.integers(0, w))
            ay = int(rng.integers(0, h))
            dx = int(rng.choice([-1, 1])) * int(rng.integers(12, 42))
            tx = int(np.clip(ax + dx, 0, w - 1))
            if (tx // 2) == (ax // 2):
                tx = (ax + 20) % w
            ty = ay
        elif mode == 1:  # vertical
            ax = int(rng.integers(0, w))
            ay = int(rng.integers(0, h))
            dy = int(rng.choice([-1, 1])) * int(rng.integers(12, 42))
            ty = int(np.clip(ay + dy, 0, h - 1))
            if ty == ay:
                ty = (ay + 20) % h
            tx = ax
        elif mode == 2:  # Morton anti-diagonal Z-stroke
            sz = int(rng.choice([16, 32]))
            half = sz // 2
            bx = int(rng.integers(0, w // sz)) * sz
            by = int(rng.integers(0, h // sz)) * sz
            q1x = bx + half + int(rng.integers(half - 4, half))
            q1y = by + int(rng.integers(half - 4, half))
            q2x = bx + int(rng.integers(0, 4))
            q2y = by + half + int(rng.integers(0, 4))
            if int(rng.integers(0, 2)) == 0:
                ax, ay, tx, ty = q1x, q1y, q2x, q2y
            else:
                ax, ay, tx, ty = q2x, q2y, q1x, q1y
        elif mode == 3:  # Hilbert 2D isotropic cluster
            ax = int(rng.integers(12, w - 12))
            ay = int(rng.integers(12, h - 12))
            dx = int(rng.choice([-1, 1])) * int(rng.integers(3, 9))
            dy = int(rng.choice([-1, 1])) * int(rng.integers(3, 9))
            tx, ty = ax + dx, ay + dy
        elif mode == 4:  # regime_shift (far 1D + local 2D)
            if int(rng.integers(0, 2)) == 0:
                ax = int(rng.integers(8, w - 72)) if w <= 140 else int(rng.integers(8, w - 8))
                ay = int(rng.integers(12, h - 12))
                base = 64 if ax + 64 < w - 6 else -64
                tx = int(np.clip(ax + base + int(rng.integers(-4, 5)), 0, w - 1))
                ty = int(np.clip(ay + int(rng.choice([-1, 1])) * int(rng.integers(4, 10)), 0, h - 1))
            else:
                ax = int(rng.integers(12, w - 12))
                ay = int(rng.integers(8, h - 72)) if h <= 140 else int(rng.integers(8, h - 8))
                base = 64 if ay + 64 < h - 6 else -64
                tx = int(np.clip(ax + int(rng.choice([-1, 1])) * int(rng.integers(4, 10)), 0, w - 1))
                ty = int(np.clip(ay + base + int(rng.integers(-4, 5)), 0, h - 1))
        else:  # corrupted_replica (stale primary -> orthogonal mirror)
            ax = int(rng.integers(12, w - 12))
            ay = int(rng.integers(12, h - 12))
            if int(rng.integers(0, 2)) == 0:
                dx = int(rng.choice([-1, 1])) * int(rng.integers(16, 44))
                sx = int(np.clip(ax + dx, 0, w - 1))
                sy = ay
                base_y = 64 if sy + 64 < h - 6 else (-64 if sy - 64 >= 6 else 32)
                tx, ty = sx, int(np.clip(sy + base_y + int(rng.integers(-4, 5)), 0, h - 1))
            else:
                dy = int(rng.choice([-1, 1])) * int(rng.integers(16, 44))
                sx = ax
                sy = int(np.clip(ay + dy, 0, h - 1))
                base_x = 64 if sx + 64 < w - 6 else (-64 if sx - 64 >= 6 else 32)
                tx, ty = int(np.clip(sx + base_x + int(rng.integers(-4, 5)), 0, w - 1)), sy
            decoy_slot = 2 * ((sy * w + sx) // 2)
            mhx = _sign(tx - sx)
            mhy = _sign(ty - sy)
            msx = coarse_scale(tx - sx)
            msy = coarse_scale(ty - sy)
            rec = (ty * w + tx) // 2
            if rec not in used_records and rec != (decoy_slot // 2):
                used_records.add(rec)
                return HopSpec(
                    anchor=ay * w + ax,
                    target_record=rec,
                    hint_x=_sign(sx - ax),
                    hint_y=_sign(sy - ay),
                    scale_x=coarse_scale(sx - ax),
                    scale_y=coarse_scale(sy - ay),
                    decoy_slot=decoy_slot,
                    mirror_hint_x=mhx,
                    mirror_hint_y=mhy,
                    mirror_scale_x=msx,
                    mirror_scale_y=msy,
                )
            continue

        rec = (ty * w + tx) // 2
        if rec not in used_records:
            used_records.add(rec)
            return HopSpec(
                anchor=ay * w + ax,
                target_record=rec,
                hint_x=_sign(tx - ax),
                hint_y=_sign(ty - ay),
                scale_x=coarse_scale(tx - ax),
                scale_y=coarse_scale(ty - ay),
            )

    # Fallback fresh slot
    rec = int(rng.integers(0, (w * h) // 2))
    while rec in used_records:
        rec = (rec + 1) % ((w * h) // 2)
    used_records.add(rec)
    tx, ty = (2 * rec) % w, (2 * rec) // w
    ax = int(np.clip(tx + (16 if tx < w - 20 else -16), 0, w - 1))
    return HopSpec(
        anchor=ty * w + ax,
        target_record=rec,
        hint_x=_sign(tx - ax),
        hint_y=0,
        scale_x=coarse_scale(tx - ax),
        scale_y=0.0,
    )


def strip_world_witnesses(world: Dict[str, Any]) -> Dict[str, Any]:
    """Mechanically strip ALL archived/private witnesses from a Storyworld spec."""
    cleaned = copy.deepcopy(world)
    cleaned.pop("_private_witness", None)
    cleaned.pop("exact", None)
    if "balance" in cleaned and isinstance(cleaned["balance"], dict):
        cleaned["balance"].pop("witnesses", None)
        cleaned["balance"].pop("counterfactuals", None)
    return cleaned


def _verify_storyworld_counterfactual(
    nodes: Sequence[Dict[str, Any]],
    operators: Dict[str, Tuple[str, ...]],
    rule: Dict[str, Any],
    route: Sequence[Tuple[int, int]],
) -> Optional[Dict[str, Any]]:
    """Verify native noncommuting adjacent-operation swap counterfactual on DAG route."""
    gates: List[Tuple[str, ...]] = []
    clues: set = set()
    for nid, j in route:
        opt = nodes[nid]["options"][j]
        gates.append(operators[opt["operator"]])
        if opt["clue"] is not None:
            clues.add(opt["clue"])

    base_st = START
    for seq in gates:
        base_st = apply(base_st, seq)
    if not pos_conditions(rule, base_st, clues):
        return None

    for idx in range(len(gates) - 1):
        swapped_gates = list(gates)
        swapped_gates[idx], swapped_gates[idx + 1] = swapped_gates[idx + 1], swapped_gates[idx]
        st = START
        for seq in swapped_gates:
            st = apply(st, seq)
        if not pos_conditions(rule, st, clues):
            return {
                "swap_decisions": [idx, idx + 1],
                "clues": sorted(clues),
                "probabilities": {
                    p: [float(probability(base_st, p)), float(probability(st, p))]
                    for p, _, _ in rule["projectors"]
                },
            }
    return None


def generate_independent_storyworld(
    world_seed: int,
    target_horizon: int,
    bucket_name: str = "custom",
) -> Dict[str, Any]:
    """Synthesize an algebraically verified branching Storyworld requiring `target_horizon` decisions.

    Graph topology:
      - Main spine nodes `0 .. L-1` (plus terminal node `L`).
      - At ~35% of spine nodes `d`, a 1-to-2-step deceptive distractor branch `trap(d)`
        (marked `is_distractor=True`) leads to either a dead-end terminal trap (`dead_end=True`)
        or a convergent trap node shared by adjacent spine nodes (testing transposition cache!).
      - Irreversible transitions:
        * Necessary act boundaries along the spine (~15% of true transitions) lock earlier checkpoints.
        * Irreversible trap doors (~20% of distractor entries) permanently trap solvers that
          step into them without verifying option metadata first!
      - Required clues (3 to 5 distinct clues) appear ONLY along valid spine transitions,
        so skipping them or entering a distractor branch makes required clues unreachable.
      - Tight phase-sensitive Pauli + computational projector gate verified with a genuine
        noncommuting adjacent-operation counterfactual!
    """
    if not (5 <= target_horizon <= 50):
        raise ValueError(f"target_horizon must be in [5, 50], got {target_horizon}")

    rng = np.random.default_rng(world_seed)
    L = int(target_horizon)
    n_req_clues = 3 if L <= 10 else (4 if L <= 28 else 5)
    req_clues = list(CLUE_POOL[:n_req_clues])
    decoy_clues = list(CLUE_POOL[n_req_clues : n_req_clues + 2])

    # Choose spine depths where each required clue sits (spread evenly across the narrative)
    clue_depths = sorted(
        set(int(round((k + 0.5) * L / n_req_clues)) for k in range(n_req_clues))
    )
    while len(clue_depths) < n_req_clues:
        cand = int(rng.integers(0, L))
        if cand not in clue_depths:
            clue_depths.append(cand)
    clue_depths.sort()
    depth_to_req_clue = {d: req_clues[k] for k, d in enumerate(clue_depths)}

    # Build witness operator sequence ensuring adjacent noncommuting pair in the final act
    # Early/mid acts (d < L - 4) use verifiable clue/branch/act-phase structure; final 4 steps
    # execute the noncommuting two-qubit phase sequence checked by tight terminal projectors + counterfactual!
    final_act_start = max(0, L - 4)
    for attempt in range(40):
        arng = np.random.default_rng(world_seed * 17 + attempt * 1009)
        witness_ops = [OPERATOR_NAMES[int(arng.integers(0, len(OPERATOR_NAMES)))] for _ in range(L)]
        # Guarantee at least one noncommuting adjacent pair inside the final act
        swap_pos = max(final_act_start, L - 3)
        if swap_pos + 1 < L and not order_changes_operator(
            OPERATOR_POOL[witness_ops[swap_pos]], OPERATOR_POOL[witness_ops[swap_pos + 1]]
        ):
            witness_ops[swap_pos] = "op_0"      # H0, CX01
            witness_ops[swap_pos + 1] = "op_1"  # S0, H0

        st = START
        for op_n in witness_ops:
            st = apply(st, OPERATOR_POOL[op_n])

        sw_ops = list(witness_ops)
        if swap_pos + 1 < L:
            sw_ops[swap_pos], sw_ops[swap_pos + 1] = sw_ops[swap_pos + 1], sw_ops[swap_pos]
        st_sw = START
        for op_n in sw_ops:
            st_sw = apply(st_sw, OPERATOR_POOL[op_n])

        phase_diffs = sorted(
            PHASE_PROJECTORS,
            key=lambda p: abs(probability(st, p) - probability(st_sw, p)),
            reverse=True,
        )
        basis_diffs = sorted(
            BASIS_PROJECTORS,
            key=lambda p: abs(probability(st, p) - probability(st_sw, p)),
            reverse=True,
        )
        best_p1 = phase_diffs[0]
        best_p2 = phase_diffs[1]
        best_b = basis_diffs[0]
        max_diff = max(
            abs(probability(st, best_p1) - probability(st_sw, best_p1)),
            abs(probability(st, best_b) - probability(st_sw, best_b)),
        )
        if max_diff < 0.12:
            continue

        projectors = []
        for pname in (best_p1, best_b, best_p2):
            pv = float(probability(st, pname))
            hw = 0.11 if abs(pv - float(probability(st_sw, pname))) >= 0.12 else 0.16
            lo = round(max(0.0, pv - hw), 4)
            hi = round(min(1.0, pv + hw), 4)
            if hi - lo >= 0.99:
                hi = round(min(1.0, lo + 0.5), 4)
            projectors.append([pname, lo, hi])

        rule = {"requires": req_clues, "projectors": projectors}
        if pos_conditions(rule, st, set(req_clues)) and not pos_conditions(rule, st_sw, set(req_clues)):
            break
    else:
        raise RuntimeError(f"Failed to synthesize noncommuting storyworld for seed {world_seed}, L={L}")

    # Precompute native act-boundary phase & clue checkpoints every 5th spine scene before final act
    # so early/mid-act off-ray perturbations or skipped clues are verifiable locally within <=5 steps!
    act_milestones: Dict[int, Dict[str, Any]] = {}
    st_walk = START
    for d in range(L):
        st_walk = apply(st_walk, OPERATOR_POOL[witness_ops[d]])
        if (d + 1) < final_act_start and ((d + 1) % 5 == 0):
            ms_projs = []
            for p_chk in ("X0+", "Y0+", "X1+", "Y1+", "00", "01"):
                pv = float(probability(st_walk, p_chk))
                ms_projs.append([p_chk, round(max(0.0, pv - 0.12), 4), round(min(1.0, pv + 0.12), 4)])
            req_so_far = sorted(c for cd, c in depth_to_req_clue.items() if cd < (d + 1))
            act_milestones[d + 1] = {"projectors": ms_projs, "requires": req_so_far}

    # Construct DAG nodes: 0..L-1 are spine scenes, L is terminal success-check node,
    # L+1 .. are distractor scenes + dead-end sink node
    nodes: List[Dict[str, Any]] = []
    # Reserve node L as TERMINAL, node L+1 as SHARED_DEAD_END
    terminal_id = L
    dead_sink_id = L + 1
    next_extra_id = L + 2

    # Pre-allocate distractor scenes for ~40% of spine depths (with shared convergent traps!)
    trap_node_for_depth: Dict[int, int] = {}
    shared_trap_nodes: List[int] = []
    for d in range(L):
        if d < L - 1 and float(rng.random()) < 0.45:
            # 40% of traps converge onto a previously created trap node (tests transposition cache!)
            if shared_trap_nodes and float(rng.random()) < 0.40:
                trap_node_for_depth[d] = int(shared_trap_nodes[-1])
            else:
                tid = next_extra_id
                next_extra_id += 1
                trap_node_for_depth[d] = tid
                shared_trap_nodes.append(tid)

    # Create placeholder nodes list
    total_nodes = next_extra_id
    for nid in range(total_nodes):
        nodes.append(
            {
                "id": nid,
                "kind": (
                    "terminal"
                    if nid == terminal_id
                    else ("dead_sink" if nid == dead_sink_id else ("spine" if nid < L else "distractor"))
                ),
                "depth": nid if nid <= L else -1,
                "options": [],
            }
        )

    # Fill spine nodes 0..L-1 with 3 options each (randomizing option index so witness isn't always j=0!)
    witness_route: List[Tuple[int, int]] = []
    for d in range(L):
        true_j = int(rng.integers(0, 3))
        witness_route.append((d, true_j))
        # Mandatory act-boundary irreversible gate every ~10 scenes on the spine
        spine_irrev = (d > 0) and (d % 10 == 0) and (d < L - 2)
        true_opt = {
            "operator": witness_ops[d],
            "clue": depth_to_req_clue.get(d, None),
            "next_scene": d + 1,
            "irreversible": spine_irrev,
            "is_trap": False,
        }

        # Build 2 competing alternatives (deceptive distractor traps, bad skips, or milestone/final-act wrong ops)
        alts: List[Dict[str, Any]] = []
        # Alt 1: distractor trap or bad skip / milestone-adjacent wrong phase
        if d in trap_node_for_depth:
            # Distractor branch: carries decoy clue or attractive phase operator.
            # Never put an irreversible trap on the same scene as a mandatory spine irreversible act gate!
            trap_irrev = (not spine_irrev) and (float(rng.random()) < 0.20)
            alt_op = OPERATOR_NAMES[(OPERATOR_NAMES.index(witness_ops[d]) + 1 + int(rng.integers(0, 3))) % len(OPERATOR_NAMES)]
            alts.append(
                {
                    "operator": alt_op,
                    "clue": decoy_clues[d % len(decoy_clues)] if decoy_clues else None,
                    "next_scene": trap_node_for_depth[d],
                    "irreversible": trap_irrev,
                    "is_trap": True,
                }
            )
        else:
            # Bad skip over a required clue, or wrong-phase branch immediately before an act milestone / final act
            near_check = (d >= final_act_start) or ((d + 1) in act_milestones) or ((d + 2) in act_milestones)
            skip_next = min(L, d + 2) if (d + 2 <= L and d in depth_to_req_clue) else (d + 1)
            alt_op = (
                OPERATOR_NAMES[(OPERATOR_NAMES.index(witness_ops[d]) + 2) % len(OPERATOR_NAMES)]
                if near_check
                else witness_ops[d]
            )
            # On non-check mid-act nodes without a trap, create a deceptive trap if one exists nearby
            if (not near_check) and (skip_next == d + 1) and shared_trap_nodes:
                alts.append(
                    {
                        "operator": OPERATOR_NAMES[(OPERATOR_NAMES.index(witness_ops[d]) + 2) % len(OPERATOR_NAMES)],
                        "clue": decoy_clues[d % len(decoy_clues)] if decoy_clues else None,
                        "next_scene": shared_trap_nodes[-1],
                        "irreversible": False,
                        "is_trap": True,
                    }
                )
            else:
                alts.append(
                    {
                        "operator": alt_op,
                        "clue": None,
                        "next_scene": skip_next,
                        "irreversible": False,
                        "is_trap": skip_next > d + 1,
                    }
                )

        # Alt 2: In early/mid acts, deceptive distractor trap or safe phase-preserving branch (without clue);
        # in the final act (d >= final_act_start), competing noncommuting operator verified at terminal!
        if d < final_act_start:
            alt2_trap = (d in trap_node_for_depth) and (float(rng.random()) < 0.70)
            alt2_op = witness_ops[d] if not alt2_trap else OPERATOR_NAMES[(OPERATOR_NAMES.index(witness_ops[d]) + 3) % len(OPERATOR_NAMES)]
            alts.append(
                {
                    "operator": alt2_op,
                    "clue": None if (d in depth_to_req_clue or alt2_trap) else (decoy_clues[0] if decoy_clues and d % 3 == 0 else None),
                    "next_scene": trap_node_for_depth[d] if alt2_trap else (d + 1),
                    "irreversible": False,
                    "is_trap": alt2_trap,
                }
            )
        else:
            alt2_op = OPERATOR_NAMES[(OPERATOR_NAMES.index(witness_ops[d]) + 4) % len(OPERATOR_NAMES)]
            alts.append(
                {
                    "operator": alt2_op,
                    "clue": None,
                    "next_scene": d + 1,
                    "irreversible": False,
                    "is_trap": False,
                }
            )

        opts: List[Dict[str, Any]] = [{}] * 3
        opts[true_j] = true_opt
        alt_iter = iter(alts)
        for j in range(3):
            if j != true_j:
                opts[j] = next(alt_iter)
        nodes[d]["options"] = opts

    # Fill distractor nodes (all 3 options lead into dead_sink_id after 1 step — deceptive 2-step trap!)
    for tid in shared_trap_nodes:
        nodes[tid]["depth"] = min(L - 1, next(d for d, t in trap_node_for_depth.items() if t == tid) + 1)
        t_opts = []
        for j in range(3):
            op_n = OPERATOR_NAMES[int(rng.integers(0, len(OPERATOR_NAMES)))]
            t_opts.append(
                {
                    "operator": op_n,
                    "clue": None,
                    "next_scene": dead_sink_id,
                    "irreversible": False,
                    "is_trap": True,
                }
            )
        nodes[tid]["options"] = t_opts

    # Precompute which required clues are STILL reachable from every node (for the native verifier's
    # structural dead-end check — strictly public graph reachability, ZERO witness leakage!)
    reachable_clues: Dict[int, FrozenSet[str]] = {terminal_id: frozenset(), dead_sink_id: frozenset()}
    # Reverse topological order on DAG
    for nid in list(reversed(shared_trap_nodes)) + list(range(L - 1, -1, -1)):
        acc = set()
        for opt in nodes[nid]["options"]:
            if opt["clue"] is not None:
                acc.add(opt["clue"])
            acc |= reachable_clues.get(opt["next_scene"], frozenset())
        reachable_clues[nid] = frozenset(acc)

    # Verify witness route + noncommuting counterfactual strictly before returning
    cf = _verify_storyworld_counterfactual(nodes, OPERATOR_POOL, rule, witness_route)
    if cf is None:
        raise RuntimeError(f"Generated storyworld seed={world_seed} failed counterfactual verification!")

    return {
        "world_id": f"sw_{bucket_name}_s{world_seed}_L{L}",
        "domain": "storyworld",
        "bucket": bucket_name,
        "world_seed": int(world_seed),
        "target_horizon": L,
        "terminal_id": terminal_id,
        "dead_sink_id": dead_sink_id,
        "operators": {k: list(v) for k, v in OPERATOR_POOL.items()},
        "rule": rule,
        "act_milestones": {str(k): v for k, v in act_milestones.items()},
        "nodes": nodes,
        "reachable_clues": {str(k): sorted(v) for k, v in reachable_clues.items()},
        "_private_witness": {
            "route": [[int(a), int(b)] for a, b in witness_route],
            "counterfactual": cf,
        },
    }


def encode_story_option(node_id: int, opt_j: int, opt: Dict[str, Any]) -> int:
    """Pack a Storyworld option record into a 30-bit integer payload with node/opt checksum."""
    op_id = OPERATOR_NAMES.index(opt["operator"])           # 0..7 (3 bits)
    cl_id = 0 if opt["clue"] is None else (CLUE_POOL.index(opt["clue"]) + 1)  # 0..8 (4 bits)
    nxt = int(opt["next_scene"])                            # 0..255 (8 bits)
    irrev = 1 if opt["irreversible"] else 0                 # 1 bit
    tag = ((node_id * 3 + opt_j) & 0x3FF)                   # 10-bit verifiable scene-option tag
    return (((((tag << 1) | irrev) << 8) | nxt) << 4 | cl_id) << 3 | op_id


def decode_story_option(
    node_id: int, opt_j: int, payload: int
) -> Optional[Dict[str, Any]]:
    """Decode & verify checksum tag of a retrieved Storyworld option payload."""
    op_id = payload & 0x7
    payload >>= 3
    cl_id = payload & 0xF
    payload >>= 4
    nxt = payload & 0xFF
    payload >>= 8
    irrev = bool(payload & 0x1)
    tag = payload >> 1
    expected_tag = (node_id * 3 + opt_j) & 0x3FF
    if tag != expected_tag or op_id >= len(OPERATOR_NAMES) or cl_id > len(CLUE_POOL):
        return None
    return {
        "operator": OPERATOR_NAMES[op_id],
        "clue": None if cl_id == 0 else CLUE_POOL[cl_id - 1],
        "next_scene": nxt,
        "irreversible": irrev,
    }


@dataclass(frozen=True)
class StoryState:
    node_id: int
    qstate: Tuple[complex, complex, complex, complex]
    clues: FrozenSet[str]
    steps: int


class StoryworldSearchEnv:
    """Witness-withheld native Storyworld search environment backed by external physical memory."""

    domain: str = "storyworld"

    def __init__(self, stripped_world: Dict[str, Any], mem: ExtendedPhysicalMemory, layout_seed: int):
        if "_private_witness" in stripped_world or "exact" in stripped_world:
            raise ValueError("Witnesses MUST be stripped before instantiating StoryworldSearchEnv!")
        self.world = stripped_world
        self.world_id: str = stripped_world["world_id"]
        self.bucket: str = stripped_world["bucket"]
        self.target_horizon: int = int(stripped_world["target_horizon"])
        self.terminal_id: int = int(stripped_world["terminal_id"])
        self.dead_sink_id: int = int(stripped_world["dead_sink_id"])
        self.rule: Dict[str, Any] = stripped_world["rule"]
        self.act_milestones: Dict[int, Dict[str, Any]] = {
            int(k): {
                "projectors": [(str(p[0]), float(p[1]), float(p[2])) for p in v["projectors"]],
                "requires": frozenset(v["requires"]),
            }
            for k, v in stripped_world.get("act_milestones", {}).items()
        }
        self.req_clues: FrozenSet[str] = frozenset(self.rule["requires"])
        self.operators: Dict[str, Tuple[str, ...]] = {
            k: tuple(v) for k, v in stripped_world["operators"].items()
        }
        self.nodes: List[Dict[str, Any]] = stripped_world["nodes"]
        self.reachable_clues: Dict[int, FrozenSet[str]] = {
            int(k): frozenset(v) for k, v in stripped_world["reachable_clues"].items()
        }
        self.mem = mem.clone_shallow()
        self.verifier_calls: int = 0

        # Encode all (node_id, opt_j) records into external physical memory across all 4 addressable
        # structured geometries (horizontal/row, vertical/col, diagonal/morton, isotropic/hilbert)
        geom_mix = (0, 1, 2, 3)
        rng = np.random.default_rng(layout_seed)
        used_recs: set = set()
        self.hop_map: Dict[Tuple[int, int], HopSpec] = {}
        geom_counter = 0
        for node in self.nodes:
            nid = int(node["id"])
            for j, opt in enumerate(node["options"]):
                geom = geom_mix[geom_counter % len(geom_mix)]
                geom_counter += 1
                hop = make_spatial_hop(rng, self.mem.width, self.mem.height, geom, used_recs)
                self.mem.values[hop.target_record] = encode_story_option(nid, j, opt)
                self.hop_map[(nid, j)] = hop

    def initial_state(self) -> StoryState:
        return StoryState(node_id=0, qstate=START, clues=frozenset(), steps=0)

    def state_key(self, st: StoryState) -> Tuple[Any, ...]:
        return (st.node_id, signature(st.qstate), st.clues)

    def option_queries(self, node_id: int) -> List[Tuple[int, HopSpec]]:
        return [(j, self.hop_map[(node_id, j)]) for j in range(3) if (node_id, j) in self.hop_map]

    def decode_record(self, node_id: int, opt_j: int, payload: int) -> Optional[Dict[str, Any]]:
        self.verifier_calls += 1
        dec = decode_story_option(node_id, opt_j, payload)
        if dec is not None and not (0 <= int(dec["next_scene"]) < len(self.nodes)):
            return None
        return dec

    def progress_metrics(self, st: StoryState) -> Tuple[float, float]:
        """Return (observable_progress in [0,1], goal_deficit in [0,1])."""
        step_prog = min(1.0, float(st.steps) / float(max(1, self.target_horizon)))
        clue_prog = float(len(self.req_clues & st.clues)) / float(max(1, len(self.req_clues)))
        proj_ok = 0.0
        for pname, lo, hi in self.rule["projectors"]:
            p = float(probability(st.qstate, pname))
            if lo - EPSILON <= p <= hi + EPSILON:
                proj_ok += 1.0
            else:
                d = (lo - p) if p < lo else (p - hi)
                proj_ok += max(0.0, 1.0 - 2.0 * d)
        proj_prog = proj_ok / float(len(self.rule["projectors"]))
        progress = 0.45 * step_prog + 0.35 * clue_prog + 0.20 * proj_prog
        deficit = 1.0 - (0.55 * clue_prog + 0.45 * proj_prog)
        return float(np.clip(progress, 0.0, 1.0)), float(np.clip(deficit, 0.0, 1.0))

    def step_and_verify(
        self, st: StoryState, opt: Dict[str, Any]
    ) -> Tuple[StoryState, Dict[str, Any]]:
        """Execute one native transition and run verifier invariants (increments verifier_calls)."""
        self.verifier_calls += 1
        nxt_id = int(opt["next_scene"])
        nxt_q = apply(st.qstate, self.operators[opt["operator"]])
        nxt_cl = st.clues | ({opt["clue"]} if opt["clue"] is not None else frozenset())
        nxt_st = StoryState(node_id=nxt_id, qstate=nxt_q, clues=nxt_cl, steps=st.steps + 1)

        if nxt_id == self.dead_sink_id:
            return nxt_st, {
                "legal": True,
                "terminal": True,
                "solved": False,
                "dead_end": True,
                "irreversible": bool(opt["irreversible"]),
                "score": -100.0,
            }

        if nxt_id == self.terminal_id:
            solved = bool(pos_conditions(self.rule, nxt_q, nxt_cl))
            return nxt_st, {
                "legal": True,
                "terminal": True,
                "solved": solved,
                "dead_end": not solved,
                "irreversible": bool(opt["irreversible"]),
                "score": 100.0 if solved else -50.0,
            }

        # Local act-boundary verifier check (no global transitive-closure peek into unvisited traps!)
        milestone_ok = True
        if nxt_id in self.act_milestones:
            ms = self.act_milestones[nxt_id]
            clues_ms_ok = ms["requires"] <= nxt_cl
            projs_ms_ok = all(
                mlo - EPSILON <= float(probability(nxt_q, mp)) <= mhi + EPSILON
                for mp, mlo, mhi in ms["projectors"]
            )
            milestone_ok = clues_ms_ok and projs_ms_ok

        dead_end = (not milestone_ok) or (nxt_st.steps > self.target_horizon + 4)

        prog, deficit = self.progress_metrics(nxt_st)
        clue_bonus = 4.0 if (opt["clue"] is not None and opt["clue"] in (self.req_clues - st.clues)) else 0.0
        # Penalize unforced irreversible transitions when reversible options are retrieved
        irrev_pen = 1.25 if opt["irreversible"] else 0.0
        ms_bonus = 1.2 if (nxt_id in self.act_milestones and milestone_ok) else 0.0
        score = 2.0 * (1.0 - deficit) + clue_bonus + ms_bonus + 0.5 * prog - irrev_pen
        if dead_end:
            score = -80.0

        return nxt_st, {
            "legal": True,
            "terminal": False,
            "solved": False,
            "dead_end": dead_end,
            "irreversible": bool(opt["irreversible"]),
            "score": float(score),
        }

    def execute_ground_truth_action(self, st: StoryState, opt_j: int) -> Tuple[StoryState, Dict[str, Any]]:
        """Execute action `opt_j` directly against native world physics (used when fallback is forced)."""
        opt = self.nodes[st.node_id]["options"][opt_j]
        return self.step_and_verify(st, opt)

    def verify_final_route(
        self, route: Sequence[Tuple[int, int]], archived_witness: Optional[Sequence[Sequence[int]]] = None
    ) -> Dict[str, Any]:
        """Full 5-point native verification on the final executed route."""
        self.verifier_calls += 1
        st = self.initial_state()
        legal = True
        for nid, j in route:
            if nid != st.node_id or not (0 <= j < 3) or st.node_id in (self.terminal_id, self.dead_sink_id):
                legal = False
                break
            opt = self.nodes[nid]["options"][j]
            st = StoryState(
                node_id=int(opt["next_scene"]),
                qstate=apply(st.qstate, self.operators[opt["operator"]]),
                clues=st.clues | ({opt["clue"]} if opt["clue"] is not None else frozenset()),
                steps=st.steps + 1,
            )
        if st.node_id != self.terminal_id:
            legal = False

        clues_ok = self.req_clues <= st.clues
        proj_ok = all(
            lo - EPSILON <= float(probability(st.qstate, p)) <= hi + EPSILON
            for p, lo, hi in self.rule["projectors"]
        )
        eligible = legal and clues_ok and proj_ok
        cf = _verify_storyworld_counterfactual(self.nodes, self.operators, self.rule, route) if eligible else None
        cf_ok = cf is not None
        differs = (
            [list(x) for x in route] != [list(x) for x in archived_witness]
            if archived_witness is not None
            else None
        )
        return {
            "legal_transitions": legal,
            "required_clues_met": clues_ok,
            "terminal_projectors_met": proj_ok,
            "secret_ending_eligible": eligible,
            "noncommuting_counterfactual_verified": cf_ok,
            "all_five_verified": bool(legal and clues_ok and proj_ok and eligible and cf_ok),
            "route_len": len(route),
            "differs_from_archived_witness": differs,
        }
