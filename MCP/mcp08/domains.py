"""Cross-domain verifier-backed search environments: Canvas Search and Lean-Style Proof Search.

Both environments share the exact same external physical memory (`ExtendedPhysicalMemory`),
witness-withholding discipline, irreversible transitions, distractor traps, transposition-cache
state signatures, and observable `(progress, deficit)` interface as `StoryworldSearchEnv`,
enabling genuine cross-domain linear probing and activation patching of the slow recurrent state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from MCP.mcp07.curriculum import HopSpec
from .memory256k import ExtendedPhysicalMemory
from .storyworld_bench import make_spatial_hop


# =============================================================================
# Domain 2: Canvas Scene-Graph Layout Verification Search
# =============================================================================

CANVAS_ROLES: Tuple[str, ...] = (
    "backdrop",
    "architecture",
    "hero",
    "lighting",
    "foreground",
    "hud",
)


@dataclass(frozen=True)
class CanvasState:
    node_id: int
    roles_placed: FrozenSet[str]
    luminance_sum: int      # integer units (0..1000); mean must stay in target contrast band
    balance_sum: int        # integer units (-500..500); center-of-mass balance
    collisions: int
    steps: int


def generate_canvas_world(world_seed: int, target_horizon: int = 12) -> Dict[str, Any]:
    """Synthesize a verified Canvas composition DAG with collision/contrast traps and flatten locks."""
    rng = np.random.default_rng(world_seed + 41000)
    L = int(target_horizon)
    n_req = 4 if L <= 12 else 5
    req_roles = list(CANVAS_ROLES[:n_req])
    role_depths = sorted(set(int(round((k + 0.5) * L / n_req)) for k in range(n_req)))
    while len(role_depths) < n_req:
        c = int(rng.integers(0, L))
        if c not in role_depths:
            role_depths.append(c)
    role_depths.sort()
    depth_to_role = {d: req_roles[k] for k, d in enumerate(role_depths)}

    terminal_id = L
    dead_sink_id = L + 1
    next_id = L + 2

    trap_for_depth: Dict[int, int] = {}
    shared_traps: List[int] = []
    for d in range(L - 1):
        if float(rng.random()) < 0.45:
            if shared_traps and float(rng.random()) < 0.40:
                trap_for_depth[d] = shared_traps[-1]
            else:
                trap_for_depth[d] = next_id
                shared_traps.append(next_id)
                next_id += 1

    nodes: List[Dict[str, Any]] = [
        {
            "id": nid,
            "kind": (
                "terminal"
                if nid == terminal_id
                else ("dead_sink" if nid == dead_sink_id else ("spine" if nid < L else "distractor"))
            ),
            "options": [],
        }
        for nid in range(next_id)
    ]

    # Witness deltas stay smoothly inside target luminance [40*L..60*L] and balance [-15*L..15*L]
    witness_route: List[Tuple[int, int]] = []
    lum_acc = 0
    bal_acc = 0
    for d in range(L):
        true_j = int(rng.integers(0, 3))
        witness_route.append((d, true_j))
        dl = int(rng.integers(44, 57))
        db = int(rng.integers(-6, 7))
        lum_acc += dl
        bal_acc += db
        # Irreversible raster-flatten checkpoint every 6 layers
        flatten_irrev = (d > 0) and (d % 6 == 0) and (d < L - 2)
        true_opt = {
            "role": depth_to_role.get(d, None),
            "dlum": dl,
            "dbal": db,
            "collide": False,
            "next_node": d + 1,
            "irreversible": flatten_irrev,
        }

        alts = []
        if d in trap_for_depth:
            # Deceptive trap asset: looks clean locally on entry, collapses to collision/dead_sink inside
            alts.append(
                {
                    "role": None,
                    "dlum": 52,
                    "dbal": 2,
                    "collide": False,
                    "next_node": trap_for_depth[d],
                    "irreversible": float(rng.random()) < 0.20,
                }
            )
        else:
            alts.append(
                {
                    "role": None,
                    "dlum": 12,
                    "dbal": -24,
                    "collide": True,
                    "next_node": d + 1,
                    "irreversible": False,
                }
            )
        alts.append(
            {
                "role": None,
                "dlum": 88,
                "dbal": 26,
                "collide": float(rng.random()) < 0.5,
                "next_node": d + 1,
                "irreversible": False,
            }
        )

        opts: List[Dict[str, Any]] = [{}] * 3
        opts[true_j] = true_opt
        it = iter(alts)
        for j in range(3):
            if j != true_j:
                opts[j] = next(it)
        nodes[d]["options"] = opts

    for tid in shared_traps:
        nodes[tid]["options"] = [
            {"role": None, "dlum": 95, "dbal": 30, "collide": True, "next_node": dead_sink_id, "irreversible": False}
            for _ in range(3)
        ]

    reachable_roles: Dict[int, FrozenSet[str]] = {terminal_id: frozenset(), dead_sink_id: frozenset()}
    for nid in list(reversed(shared_traps)) + list(range(L - 1, -1, -1)):
        acc = set()
        for opt in nodes[nid]["options"]:
            if opt["role"] is not None:
                acc.add(opt["role"])
            acc |= reachable_roles.get(opt["next_node"], frozenset())
        reachable_roles[nid] = frozenset(acc)

    # Local layer-act milestone: verify expected roles for earlier depths without peeking into traps
    act_milestones = {
        m: sorted(r for rd, r in depth_to_role.items() if rd < m)
        for m in range(4, L, 4)
    }
    return {
        "world_id": f"canvas_s{world_seed}_L{L}",
        "domain": "canvas",
        "world_seed": int(world_seed),
        "target_horizon": L,
        "terminal_id": terminal_id,
        "dead_sink_id": dead_sink_id,
        "required_roles": req_roles,
        "lum_bounds": [int(lum_acc - 10 * L), int(lum_acc + 10 * L)],
        "bal_bounds": [int(bal_acc - 8 * L), int(bal_acc + 8 * L)],
        "max_collisions": 0,
        "act_milestones": {str(k): v for k, v in act_milestones.items()},
        "nodes": nodes,
        "reachable_roles": {str(k): sorted(v) for k, v in reachable_roles.items()},
        "_private_witness": {"route": [[int(a), int(b)] for a, b in witness_route]},
    }


def _encode_canvas_opt(node_id: int, opt_j: int, opt: Dict[str, Any]) -> int:
    role_id = 0 if opt["role"] is None else (CANVAS_ROLES.index(opt["role"]) + 1)  # 3 bits
    dlum = int(np.clip(opt["dlum"], 0, 127))           # 7 bits
    dbal = int(np.clip(opt["dbal"] + 32, 0, 63))       # 6 bits
    col = 1 if opt["collide"] else 0                   # 1 bit
    irrev = 1 if opt["irreversible"] else 0            # 1 bit
    nxt = int(opt["next_node"]) & 0x7F                 # 7 bits
    tag = (node_id * 3 + opt_j) & 0x1F                 # 5 bits
    return ((((((tag << 7) | nxt) << 1 | irrev) << 1 | col) << 6 | dbal) << 7 | dlum) << 3 | role_id


def _decode_canvas_opt(node_id: int, opt_j: int, payload: int) -> Optional[Dict[str, Any]]:
    role_id = payload & 0x7
    payload >>= 3
    dlum = payload & 0x7F
    payload >>= 7
    dbal = (payload & 0x3F) - 32
    payload >>= 6
    col = bool(payload & 0x1)
    payload >>= 1
    irrev = bool(payload & 0x1)
    payload >>= 1
    nxt = payload & 0x7F
    tag = payload >> 7
    if tag != ((node_id * 3 + opt_j) & 0x1F) or role_id > len(CANVAS_ROLES):
        return None
    return {
        "role": None if role_id == 0 else CANVAS_ROLES[role_id - 1],
        "dlum": int(dlum),
        "dbal": int(dbal),
        "collide": col,
        "next_node": int(nxt),
        "irreversible": irrev,
    }


class CanvasSearchEnv:
    """Verifier-backed 2D Canvas composition search environment backed by external physical memory."""

    domain: str = "canvas"

    def __init__(self, stripped_world: Dict[str, Any], mem: ExtendedPhysicalMemory, layout_seed: int):
        if "_private_witness" in stripped_world:
            raise ValueError("Witnesses MUST be stripped before instantiating CanvasSearchEnv!")
        self.world = stripped_world
        self.world_id: str = stripped_world["world_id"]
        self.bucket: str = f"canvas_{stripped_world['target_horizon']}"
        self.target_horizon: int = int(stripped_world["target_horizon"])
        self.terminal_id: int = int(stripped_world["terminal_id"])
        self.dead_sink_id: int = int(stripped_world["dead_sink_id"])
        self.req_roles: FrozenSet[str] = frozenset(stripped_world["required_roles"])
        self.lum_bounds: Tuple[int, int] = tuple(stripped_world["lum_bounds"])
        self.bal_bounds: Tuple[int, int] = tuple(stripped_world["bal_bounds"])
        self.nodes = stripped_world["nodes"]
        self.act_milestones = {int(k): frozenset(v) for k, v in stripped_world.get("act_milestones", {}).items()}
        self.reachable_roles = {int(k): frozenset(v) for k, v in stripped_world["reachable_roles"].items()}
        self.mem = mem.clone_shallow()
        self.verifier_calls: int = 0

        geom_mix = (0, 1, 2, 3)
        rng = np.random.default_rng(layout_seed + 13)
        used: set = set()
        self.hop_map: Dict[Tuple[int, int], HopSpec] = {}
        gc = 0
        for node in self.nodes:
            nid = int(node["id"])
            for j, opt in enumerate(node["options"]):
                hop = make_spatial_hop(rng, self.mem.width, self.mem.height, geom_mix[gc % len(geom_mix)], used)
                gc += 1
                self.mem.values[hop.target_record] = _encode_canvas_opt(nid, j, opt)
                self.hop_map[(nid, j)] = hop

    def initial_state(self) -> CanvasState:
        return CanvasState(node_id=0, roles_placed=frozenset(), luminance_sum=0, balance_sum=0, collisions=0, steps=0)

    def state_key(self, st: CanvasState) -> Tuple[Any, ...]:
        # Quantize sums to keep transposition keys compact
        return (st.node_id, st.roles_placed, st.collisions > 0, st.luminance_sum // 25, st.balance_sum // 16)

    def option_queries(self, node_id: int) -> List[Tuple[int, HopSpec]]:
        return [(j, self.hop_map[(node_id, j)]) for j in range(3) if (node_id, j) in self.hop_map]

    def decode_record(self, node_id: int, opt_j: int, payload: int) -> Optional[Dict[str, Any]]:
        self.verifier_calls += 1
        return _decode_canvas_opt(node_id, opt_j, payload)

    def progress_metrics(self, st: CanvasState) -> Tuple[float, float]:
        step_p = min(1.0, float(st.steps) / float(max(1, self.target_horizon)))
        role_p = float(len(self.req_roles & st.roles_placed)) / float(max(1, len(self.req_roles)))
        col_ok = 1.0 if st.collisions == 0 else 0.0
        prog = 0.50 * step_p + 0.40 * role_p + 0.10 * col_ok
        deficit = 1.0 - (0.60 * role_p + 0.40 * col_ok)
        return float(np.clip(prog, 0.0, 1.0)), float(np.clip(deficit, 0.0, 1.0))

    def step_and_verify(self, st: CanvasState, opt: Dict[str, Any]) -> Tuple[CanvasState, Dict[str, Any]]:
        self.verifier_calls += 1
        nxt_id = int(opt["next_node"])
        nxt_roles = st.roles_placed | ({opt["role"]} if opt["role"] is not None else frozenset())
        nxt_st = CanvasState(
            node_id=nxt_id,
            roles_placed=nxt_roles,
            luminance_sum=st.luminance_sum + int(opt["dlum"]),
            balance_sum=st.balance_sum + int(opt["dbal"]),
            collisions=st.collisions + (1 if opt["collide"] else 0),
            steps=st.steps + 1,
        )
        if nxt_id == self.dead_sink_id or nxt_st.collisions > 0:
            return nxt_st, {
                "legal": True,
                "terminal": nxt_id == self.dead_sink_id,
                "solved": False,
                "dead_end": True,
                "irreversible": bool(opt["irreversible"]),
                "score": -90.0,
            }
        if nxt_id == self.terminal_id:
            lum_ok = self.lum_bounds[0] <= nxt_st.luminance_sum <= self.lum_bounds[1]
            bal_ok = self.bal_bounds[0] <= nxt_st.balance_sum <= self.bal_bounds[1]
            solved = (self.req_roles <= nxt_roles) and lum_ok and bal_ok and (nxt_st.collisions == 0)
            return nxt_st, {
                "legal": True,
                "terminal": True,
                "solved": solved,
                "dead_end": not solved,
                "irreversible": bool(opt["irreversible"]),
                "score": 100.0 if solved else -50.0,
            }
        # Local layer-act milestone check (no lookahead peek into unvisited trap nodes)
        milestone_ok = True
        if nxt_id in self.act_milestones:
            milestone_ok = self.act_milestones[nxt_id] <= nxt_roles
        dead_end = (not milestone_ok) or (nxt_st.steps > self.target_horizon + 4)
        prog, deficit = self.progress_metrics(nxt_st)
        role_gain = 3.5 if (opt["role"] is not None and opt["role"] in (self.req_roles - st.roles_placed)) else 0.0
        # Reward staying near expected per-step luminance (~50) and balance (~0)
        mean_l = float(nxt_st.luminance_sum) / float(max(1, nxt_st.steps))
        mean_b = float(nxt_st.balance_sum) / float(max(1, nxt_st.steps))
        aes_score = -0.04 * abs(mean_l - 50.0) - 0.05 * abs(mean_b)
        score = 2.0 * (1.0 - deficit) + role_gain + 0.5 * prog + aes_score - (1.0 if opt["irreversible"] else 0.0)
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

    def execute_ground_truth_action(self, st: CanvasState, opt_j: int) -> Tuple[CanvasState, Dict[str, Any]]:
        return self.step_and_verify(st, self.nodes[st.node_id]["options"][opt_j])

    def verify_final_route(
        self, route: Sequence[Tuple[int, int]], archived_witness: Optional[Sequence[Sequence[int]]] = None
    ) -> Dict[str, Any]:
        self.verifier_calls += 1
        st = self.initial_state()
        legal = True
        for nid, j in route:
            if nid != st.node_id or not (0 <= j < 3) or st.node_id in (self.terminal_id, self.dead_sink_id):
                legal = False
                break
            st, _ = self.execute_ground_truth_action(st, j)
        lum_ok = self.lum_bounds[0] <= st.luminance_sum <= self.lum_bounds[1]
        bal_ok = self.bal_bounds[0] <= st.balance_sum <= self.bal_bounds[1]
        roles_ok = self.req_roles <= st.roles_placed
        solved = legal and (st.node_id == self.terminal_id) and roles_ok and lum_ok and bal_ok and (st.collisions == 0)
        return {
            "legal_transitions": legal,
            "required_roles_met": roles_ok,
            "aesthetics_met": lum_ok and bal_ok and (st.collisions == 0),
            "all_verified": solved,
            "route_len": len(route),
        }


# =============================================================================
# Domain 3: Lean-Style Symbolic Tactic & Subgoal Proof Search
# =============================================================================

TACTIC_KINDS: Tuple[str, ...] = ("intro", "apply_lem", "rw_eq", "split_goal", "omega_close", "clear_bad")


@dataclass(frozen=True)
class LeanState:
    node_id: int
    open_goals: Tuple[int, ...]   # canonical tuple of remaining obligation IDs
    hyps: FrozenSet[int]          # proved local hypothesis IDs
    steps: int


def generate_lean_world(world_seed: int, target_horizon: int = 12) -> Dict[str, Any]:
    """Synthesize a verified Lean-style subgoal-discharge DAG with distractor generalizations."""
    rng = np.random.default_rng(world_seed + 83000)
    L = int(target_horizon)
    terminal_id = L
    dead_sink_id = L + 1
    next_id = L + 2

    trap_for_depth: Dict[int, int] = {}
    shared_traps: List[int] = []
    for d in range(L - 1):
        if float(rng.random()) < 0.45:
            if shared_traps and float(rng.random()) < 0.40:
                trap_for_depth[d] = shared_traps[-1]
            else:
                trap_for_depth[d] = next_id
                shared_traps.append(next_id)
                next_id += 1

    nodes: List[Dict[str, Any]] = [
        {
            "id": nid,
            "kind": (
                "terminal"
                if nid == terminal_id
                else ("dead_sink" if nid == dead_sink_id else ("spine" if nid < L else "distractor"))
            ),
            "options": [],
        }
        for nid in range(next_id)
    ]

    # Required lemmas / hypotheses gathered along proof spine
    n_req_hyps = 4 if L <= 12 else 5
    hyp_depths = sorted(set(int(round((k + 0.5) * L / n_req_hyps)) for k in range(n_req_hyps)))
    while len(hyp_depths) < n_req_hyps:
        c = int(rng.integers(0, L))
        if c not in hyp_depths:
            hyp_depths.append(c)
    hyp_depths.sort()
    depth_to_hyp = {d: k + 1 for k, d in enumerate(hyp_depths)}
    req_hyps = list(range(1, n_req_hyps + 1))

    # Each spine step discharges 1 obligation from initial goal list (L .. 1) -> ()
    witness_route: List[Tuple[int, int]] = []
    for d in range(L):
        true_j = int(rng.integers(0, 3))
        witness_route.append((d, true_j))
        clear_irrev = (d > 0) and (d % 6 == 0) and (d < L - 2)
        tactic = "omega_close" if d == L - 1 else ("intro" if d in depth_to_hyp else ("rw_eq" if d % 2 == 0 else "apply_lem"))
        true_opt = {
            "tactic": tactic,
            "add_hyp": depth_to_hyp.get(d, 0),
            "close_goal": True,
            "spawn_unprovable": False,
            "next_node": d + 1,
            "irreversible": clear_irrev,
        }

        alts = []
        if d in trap_for_depth:
            # Deceptive generalization tactic: closes current goal locally, collapses to unprovable inside trap
            alts.append(
                {
                    "tactic": "split_goal",
                    "add_hyp": 0,
                    "close_goal": True,
                    "spawn_unprovable": False,
                    "next_node": trap_for_depth[d],
                    "irreversible": float(rng.random()) < 0.20,
                }
            )
        else:
            alts.append(
                {
                    "tactic": "clear_bad",
                    "add_hyp": 0,
                    "close_goal": False,
                    "spawn_unprovable": True,
                    "next_node": d + 1,
                    "irreversible": False,
                }
            )
        alts.append(
            {
                "tactic": "rw_eq",
                "add_hyp": 0,
                "close_goal": False,
                "spawn_unprovable": True,
                "next_node": d + 1,
                "irreversible": False,
            }
        )

        opts: List[Dict[str, Any]] = [{}] * 3
        opts[true_j] = true_opt
        it = iter(alts)
        for j in range(3):
            if j != true_j:
                opts[j] = next(it)
        nodes[d]["options"] = opts

    for tid in shared_traps:
        nodes[tid]["options"] = [
            {
                "tactic": "clear_bad",
                "add_hyp": 0,
                "close_goal": False,
                "spawn_unprovable": True,
                "next_node": dead_sink_id,
                "irreversible": False,
            }
            for _ in range(3)
        ]

    reachable_hyps: Dict[int, FrozenSet[int]] = {terminal_id: frozenset(), dead_sink_id: frozenset()}
    for nid in list(reversed(shared_traps)) + list(range(L - 1, -1, -1)):
        acc = set()
        for opt in nodes[nid]["options"]:
            if opt["add_hyp"] > 0:
                acc.add(int(opt["add_hyp"]))
            acc |= reachable_hyps.get(opt["next_node"], frozenset())
        reachable_hyps[nid] = frozenset(acc)

    act_milestones = {
        m: sorted(h for hd, h in depth_to_hyp.items() if hd < m)
        for m in range(4, L, 4)
    }
    return {
        "world_id": f"lean_s{world_seed}_L{L}",
        "domain": "lean",
        "world_seed": int(world_seed),
        "target_horizon": L,
        "terminal_id": terminal_id,
        "dead_sink_id": dead_sink_id,
        "required_hyps": req_hyps,
        "act_milestones": {str(k): v for k, v in act_milestones.items()},
        "nodes": nodes,
        "reachable_hyps": {str(k): sorted(v) for k, v in reachable_hyps.items()},
        "_private_witness": {"route": [[int(a), int(b)] for a, b in witness_route]},
    }


def _encode_lean_opt(node_id: int, opt_j: int, opt: Dict[str, Any]) -> int:
    tac_id = TACTIC_KINDS.index(opt["tactic"])     # 3 bits
    add_h = int(opt["add_hyp"]) & 0xF              # 4 bits
    close_g = 1 if opt["close_goal"] else 0        # 1 bit
    spawn_u = 1 if opt["spawn_unprovable"] else 0  # 1 bit
    irrev = 1 if opt["irreversible"] else 0        # 1 bit
    nxt = int(opt["next_node"]) & 0x7F             # 7 bits
    tag = (node_id * 3 + opt_j) & 0x3FF            # 10 bits
    return ((((((tag << 7) | nxt) << 1 | irrev) << 1 | spawn_u) << 1 | close_g) << 4 | add_h) << 3 | tac_id


def _decode_lean_opt(node_id: int, opt_j: int, payload: int) -> Optional[Dict[str, Any]]:
    tac_id = payload & 0x7
    payload >>= 3
    add_h = payload & 0xF
    payload >>= 4
    close_g = bool(payload & 0x1)
    payload >>= 1
    spawn_u = bool(payload & 0x1)
    payload >>= 1
    irrev = bool(payload & 0x1)
    payload >>= 1
    nxt = payload & 0x7F
    tag = payload >> 7
    if tag != ((node_id * 3 + opt_j) & 0x3FF) or tac_id >= len(TACTIC_KINDS):
        return None
    return {
        "tactic": TACTIC_KINDS[tac_id],
        "add_hyp": int(add_h),
        "close_goal": close_g,
        "spawn_unprovable": spawn_u,
        "next_node": int(nxt),
        "irreversible": irrev,
    }


class LeanSearchEnv:
    """Verifier-backed Lean-style proof-state tactic search environment over external physical memory."""

    domain: str = "lean"

    def __init__(self, stripped_world: Dict[str, Any], mem: ExtendedPhysicalMemory, layout_seed: int):
        if "_private_witness" in stripped_world:
            raise ValueError("Witnesses MUST be stripped before instantiating LeanSearchEnv!")
        self.world = stripped_world
        self.world_id: str = stripped_world["world_id"]
        self.bucket: str = f"lean_{stripped_world['target_horizon']}"
        self.target_horizon: int = int(stripped_world["target_horizon"])
        self.terminal_id: int = int(stripped_world["terminal_id"])
        self.dead_sink_id: int = int(stripped_world["dead_sink_id"])
        self.req_hyps: FrozenSet[int] = frozenset(stripped_world["required_hyps"])
        self.nodes = stripped_world["nodes"]
        self.act_milestones = {int(k): frozenset(v) for k, v in stripped_world.get("act_milestones", {}).items()}
        self.reachable_hyps = {int(k): frozenset(v) for k, v in stripped_world["reachable_hyps"].items()}
        self.mem = mem.clone_shallow()
        self.verifier_calls: int = 0

        geom_mix = (0, 1, 2, 3)
        rng = np.random.default_rng(layout_seed + 29)
        used: set = set()
        self.hop_map: Dict[Tuple[int, int], HopSpec] = {}
        gc = 0
        for node in self.nodes:
            nid = int(node["id"])
            for j, opt in enumerate(node["options"]):
                hop = make_spatial_hop(rng, self.mem.width, self.mem.height, geom_mix[gc % len(geom_mix)], used)
                gc += 1
                self.mem.values[hop.target_record] = _encode_lean_opt(nid, j, opt)
                self.hop_map[(nid, j)] = hop

    def initial_state(self) -> LeanState:
        return LeanState(
            node_id=0,
            open_goals=tuple(range(1, self.target_horizon + 1)),
            hyps=frozenset(),
            steps=0,
        )

    def state_key(self, st: LeanState) -> Tuple[Any, ...]:
        return (st.node_id, st.open_goals, st.hyps)

    def option_queries(self, node_id: int) -> List[Tuple[int, HopSpec]]:
        return [(j, self.hop_map[(node_id, j)]) for j in range(3) if (node_id, j) in self.hop_map]

    def decode_record(self, node_id: int, opt_j: int, payload: int) -> Optional[Dict[str, Any]]:
        self.verifier_calls += 1
        return _decode_lean_opt(node_id, opt_j, payload)

    def progress_metrics(self, st: LeanState) -> Tuple[float, float]:
        closed_frac = 1.0 - float(len([g for g in st.open_goals if g > 0])) / float(max(1, self.target_horizon))
        hyp_frac = float(len(self.req_hyps & st.hyps)) / float(max(1, len(self.req_hyps)))
        has_bad = any(g < 0 for g in st.open_goals)
        prog = 0.60 * max(0.0, closed_frac) + 0.40 * hyp_frac
        deficit = 1.0 if has_bad else (1.0 - (0.55 * max(0.0, closed_frac) + 0.45 * hyp_frac))
        return float(np.clip(prog, 0.0, 1.0)), float(np.clip(deficit, 0.0, 1.0))

    def step_and_verify(self, st: LeanState, opt: Dict[str, Any]) -> Tuple[LeanState, Dict[str, Any]]:
        self.verifier_calls += 1
        nxt_id = int(opt["next_node"])
        goals = list(st.open_goals)
        if opt["close_goal"] and goals:
            goals.pop(0)
        if opt["spawn_unprovable"]:
            goals.append(-1)  # Unprovable goal obligation flagged by kernel!
        nxt_hyps = st.hyps | ({int(opt["add_hyp"])} if int(opt["add_hyp"]) > 0 else frozenset())
        nxt_st = LeanState(node_id=nxt_id, open_goals=tuple(goals), hyps=nxt_hyps, steps=st.steps + 1)

        if nxt_id == self.dead_sink_id or any(g < 0 for g in nxt_st.open_goals):
            return nxt_st, {
                "legal": True,
                "terminal": nxt_id == self.dead_sink_id,
                "solved": False,
                "dead_end": True,
                "irreversible": bool(opt["irreversible"]),
                "score": -90.0,
            }
        if nxt_id == self.terminal_id:
            solved = (len(nxt_st.open_goals) == 0) and (self.req_hyps <= nxt_hyps)
            return nxt_st, {
                "legal": True,
                "terminal": True,
                "solved": solved,
                "dead_end": not solved,
                "irreversible": bool(opt["irreversible"]),
                "score": 100.0 if solved else -50.0,
            }
        milestone_ok = True
        if nxt_id in self.act_milestones:
            milestone_ok = self.act_milestones[nxt_id] <= nxt_hyps
        dead_end = (not milestone_ok) or (nxt_st.steps > self.target_horizon + 4)
        prog, deficit = self.progress_metrics(nxt_st)
        hyp_gain = 3.5 if (int(opt["add_hyp"]) in (self.req_hyps - st.hyps)) else 0.0
        goal_gain = 2.0 if opt["close_goal"] else -1.5
        score = 2.0 * (1.0 - deficit) + hyp_gain + goal_gain + 0.5 * prog - (1.0 if opt["irreversible"] else 0.0)
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

    def execute_ground_truth_action(self, st: LeanState, opt_j: int) -> Tuple[LeanState, Dict[str, Any]]:
        return self.step_and_verify(st, self.nodes[st.node_id]["options"][opt_j])

    def verify_final_route(
        self, route: Sequence[Tuple[int, int]], archived_witness: Optional[Sequence[Sequence[int]]] = None
    ) -> Dict[str, Any]:
        self.verifier_calls += 1
        st = self.initial_state()
        legal = True
        for nid, j in route:
            if nid != st.node_id or not (0 <= j < 3) or st.node_id in (self.terminal_id, self.dead_sink_id):
                legal = False
                break
            st, _ = self.execute_ground_truth_action(st, j)
        solved = legal and (st.node_id == self.terminal_id) and (len(st.open_goals) == 0) and (self.req_hyps <= st.hyps)
        return {
            "legal_transitions": legal,
            "goals_discharged": len(st.open_goals) == 0,
            "required_hyps_met": self.req_hyps <= st.hyps,
            "all_verified": solved,
            "route_len": len(route),
        }
