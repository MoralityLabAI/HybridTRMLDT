"""Native Storyworld integration for MCP-07 (strictly separate from synthetic proxy).

Provides two distinct native evaluation tracks over MCP-05's `storyworld.hilbert.json`:
  1. `run_archived_witness_replay`: Replays the archived MCP-05 secret witnesses and
     adjacent-swap controls through bounded 32K multi-projection memory.
  2. `run_independent_route_discovery`: Strips all archived witnesses (`exact`,
     `balance.witnesses`, `balance.counterfactuals`) before execution. Controllers must
     retrieve scene-option transition records from bounded 32K physical memory and plan
     a valid secret-ending route using only retrieved records and native `transition`,
     `probability`, and `classify`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .curriculum import HopSpec, QueryRoot, coarse_scale
from .eval_harness import rollout_batch
from .memory import ALL_PROJECTIONS, SharedPhysicalMemory
from .models import AdaptiveGeometryController

DEFAULT_NATIVE_WORLD = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "mcp05"
    / "inky-hilbert-16k"
    / "storyworld.hilbert.json"
)


def strip_archived_witnesses(world_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of storyworld data with ALL archived witnesses removed."""
    cleaned = copy.deepcopy(world_data)
    cleaned.pop("exact", None)
    if "balance" in cleaned and isinstance(cleaned["balance"], dict):
        cleaned["balance"].pop("witnesses", None)
        cleaned["balance"].pop("counterfactuals", None)
    return cleaned


def _sign(v: int) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _make_mixed_geometry_hop(
    rng: np.random.Generator, w: int, h: int, geom_idx: int
) -> HopSpec:
    """Create one spatial retrieval hop cycling across all 4 structured geometries."""
    mode = geom_idx % 4
    if mode == 0:  # horizontal
        ax = int(rng.integers(0, w))
        ay = int(rng.integers(0, h))
        dx = int(rng.choice([-1, 1])) * int(rng.integers(16, 49))
        tx = int(np.clip(ax + dx, 0, w - 1))
        if (tx // 2) == (ax // 2):
            tx = (ax + 24) % w
        ty = ay
    elif mode == 1:  # vertical
        ax = int(rng.integers(0, w))
        ay = int(rng.integers(0, h))
        dy = int(rng.choice([-1, 1])) * int(rng.integers(16, 49))
        ty = int(np.clip(ay + dy, 0, h - 1))
        if ty == ay:
            ty = (ay + 24) % h
        tx = ax
    elif mode == 2:  # diagonal (Morton anti-diagonal Z-stroke)
        sz = int(rng.choice([16, 32]))
        half = sz // 2
        bx = int(rng.integers(0, w // sz)) * sz
        by = int(rng.integers(0, h // sz)) * sz
        ax = bx + half + int(rng.integers(half - 4, half))
        ay = by + int(rng.integers(half - 4, half))
        tx = bx + int(rng.integers(0, 4))
        ty = by + half + int(rng.integers(0, 4))
    else:  # isotropic (Hilbert 2D cluster)
        ax = int(rng.integers(12, w - 12))
        ay = int(rng.integers(12, h - 12))
        dx = int(rng.choice([-1, 1])) * int(rng.integers(3, 9))
        dy = int(rng.choice([-1, 1])) * int(rng.integers(3, 9))
        tx, ty = ax + dx, ay + dy

    return HopSpec(
        anchor=ay * w + ax,
        target_record=(ty * w + tx) // 2,
        hint_x=_sign(tx - ax),
        hint_y=_sign(ty - ay),
        scale_x=coarse_scale(tx - ax),
        scale_y=coarse_scale(ty - ay),
    )


def run_archived_witness_replay(
    world_path: Path,
    arm: str,
    model: Optional[AdaptiveGeometryController],
    seed: int = 107,
    budget: int = 16,
    corruption: float = 0.0,
    matched_rand_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """MCP-05 archived secret-ending witness replay bridge through 32K multi-projection memory."""
    from MCP.mcp05.hilbert_storyworld import START, classify, replay_path, transition

    raw_bytes = Path(world_path).read_bytes()
    world_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))

    mem = SharedPhysicalMemory(32768, 2800000 + seed)
    rng = np.random.default_rng(91000 + seed)
    endings_out: List[Dict[str, Any]] = []

    for k, (ending, entry) in enumerate(sorted(data["exact"]["secret_gates"].items())):
        route = entry["witness"]
        # 1. Verify archived witness & adjacent-swap control directly on native math
        state = START
        clues = frozenset()
        cur_i = 0
        for si, j in route:
            if si != cur_i:
                raise ValueError(f"Invalid archived witness scene transition {si} != {cur_i}")
            adv, state, clues = transition(data["scenes"][si]["options"][j], data["math"], state, clues)
            cur_i += adv
        if cur_i < len(data["scenes"]) or ending not in classify(data["math"], state, clues):
            raise ValueError(f"Archived witness failed native verification for {ending}")

        order_ctrl = entry.get("order_witness")
        swap_verified = False
        if order_ctrl:
            baseline, facts = replay_path(data["scenes"], data["math"], order_ctrl["witness"])
            swapped, same_facts = replay_path(
                data["scenes"], data["math"], order_ctrl["witness"], swap=order_ctrl["swap_decisions"]
            )
            if (
                facts != same_facts
                or ending not in classify(data["math"], baseline, facts)
                or ending in classify(data["math"], swapped, same_facts)
            ):
                raise ValueError(f"Archived swap control failed native verification for {ending}")
            swap_verified = True

        # 2. Memory-assisted witness replay (cycling all 4 structured geometries along the route)
        state = START
        clues = frozenset()
        cur_i = 0
        decision_receipts: List[Dict[str, Any]] = []
        used_recs = set()
        for step_idx, (si, j) in enumerate(route):
            hop = _make_mixed_geometry_hop(rng, mem.width, mem.height, step_idx + k)
            tries = 0
            while hop.target_record in used_recs and tries < 16:
                hop = _make_mixed_geometry_hop(rng, mem.width, mem.height, step_idx + k)
                tries += 1
            used_recs.add(hop.target_record)
            root = QueryRoot(family="native_witness", family_id=-1, hops=(hop,))
            old_val = int(mem.values[hop.target_record])
            mem.values[hop.target_record] = 3 * si + j
            try:
                recs, _ = rollout_batch(
                    mem,
                    [root],
                    arm=arm,
                    model=model,
                    budget=budget,
                    corruption=corruption,
                    matched_rand_profile=matched_rand_profile,
                    rng_seed=seed * 1000 + k * 100 + step_idx,
                )
            finally:
                mem.values[hop.target_record] = old_val

            rec = recs[0]
            decision_receipts.append(rec)
            if not rec["success"] or rec["value"] is None:
                break
            ri, rj = divmod(int(rec["value"]), 3)
            if ri != cur_i or (ri, rj) != (si, j):
                break
            adv, state, clues = transition(data["scenes"][ri]["options"][rj], data["math"], state, clues)
            cur_i += adv

        ok = (
            len(decision_receipts) == len(route)
            and all(r["success"] for r in decision_receipts)
            and cur_i >= len(data["scenes"])
            and ending in classify(data["math"], state, clues)
        )
        endings_out.append(
            {
                "ending": ending,
                "success": ok,
                "steps_completed": len(decision_receipts),
                "route_len": len(route),
                "mean_probes": float(np.mean([r["probes"] for r in decision_receipts])),
                "mean_charged_reads": float(np.mean([r["charged_reads"] for r in decision_receipts])),
                "mean_switches": float(np.mean([r["switches"] for r in decision_receipts])),
                "swap_control_verified": swap_verified,
            }
        )

    return {
        "track": "native_witness_replay",
        "world_sha256": world_sha256,
        "arm": arm,
        "seed": seed,
        "budget": budget,
        "corruption": corruption,
        "endings": endings_out,
        "all_passed": all(e["success"] for e in endings_out),
    }


def verify_discovered_route(
    scenes: Sequence[Dict[str, Any]],
    model_math: Dict[str, Any],
    route: Sequence[Tuple[int, int]],
    target_ending: str,
    archived_witness: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Independently verify all 5 Section-6 native Storyworld criteria on a discovered route."""
    from MCP.mcp05.hilbert_storyworld import (
        EPSILON,
        START,
        classify,
        counterfactual,
        pos_conditions,
        probability,
        replay_path,
        transition,
    )

    state = START
    clues: frozenset = frozenset()
    cur_i = 0
    legal_transitions = True
    for si, j in route:
        if si != cur_i or not (0 <= si < len(scenes)) or not (0 <= j < 3):
            legal_transitions = False
            break
        adv, state, clues = transition(scenes[si]["options"][j], model_math, state, clues)
        cur_i += adv
    if cur_i < len(scenes):
        legal_transitions = False

    rule = model_math["secrets"][target_ending]
    required_clues = set(rule["requires"])
    required_clues_met = required_clues <= set(clues)

    proj_report = {}
    projectors_met = True
    for proj, low, high in rule["projectors"]:
        val = float(probability(state, proj))
        ok = (low - EPSILON <= val <= high + EPSILON)
        proj_report[proj] = {"value": val, "interval": [low, high], "ok": ok}
        if not ok:
            projectors_met = False

    eligible_list = classify(model_math, state, clues) if legal_transitions else []
    secret_eligible = legal_transitions and (target_ending in eligible_list)

    cf_result = None
    cf_verified = False
    if secret_eligible:
        cf_result = counterfactual(scenes, model_math, list(route), target_ending)
        if cf_result is not None:
            base_st, base_cl = replay_path(scenes, model_math, list(route))
            swap_st, swap_cl = replay_path(
                scenes, model_math, list(route), swap=cf_result["swap_decisions"]
            )
            cf_verified = (
                base_cl == swap_cl
                and pos_conditions(rule, base_st, base_cl)
                and not pos_conditions(rule, swap_st, swap_cl)
            )

    differs_from_archived = None
    if archived_witness is not None:
        differs_from_archived = [list(x) for x in route] != [list(x) for x in archived_witness]

    return {
        "legal_transitions": legal_transitions,
        "required_clues_met": required_clues_met,
        "collected_clues": sorted(clues),
        "terminal_projectors_met": projectors_met,
        "projectors": proj_report,
        "secret_ending_eligible": secret_eligible,
        "eligible_endings": eligible_list,
        "noncommuting_counterfactual_verified": cf_verified,
        "counterfactual": cf_result,
        "all_five_verified": bool(
            legal_transitions
            and required_clues_met
            and projectors_met
            and secret_eligible
            and cf_verified
        ),
        "differs_from_archived_witness": differs_from_archived,
        "route_len": len(route),
        "route": [[int(a), int(b)] for a, b in route],
    }


def _projector_margin_score(state, rule: Dict[str, Any]) -> float:
    """Continuous heuristic score measuring how well `state` satisfies target projectors."""
    from MCP.mcp05.hilbert_storyworld import probability

    score = 0.0
    for proj, low, high in rule["projectors"]:
        p = float(probability(state, proj))
        mid = 0.5 * (low + high)
        half = max(1e-6, 0.5 * (high - low))
        if low <= p <= high:
            # Inside interval: reward being well inside the band
            score += 1.0 - 0.35 * abs(p - mid) / half
        else:
            dist = (low - p) if p < low else (p - high)
            score -= 2.5 * dist
    return score


def _build_storyworld_memory_layout(
    mem: SharedPhysicalMemory,
    scenes: Sequence[Dict[str, Any]],
    model_math: Dict[str, Any],
    seed: int,
) -> Tuple[Dict[Tuple[int, int], HopSpec], List[str], List[Optional[str]]]:
    """Encode all 90 native scene-option records into 32K physical memory with diverse geometries.

    Each scene option (si, j) is placed at a unique logical record in `mem`, with:
      - j=0: horizontal locality from scene anchor (favors row)
      - j=1: vertical locality from scene anchor (favors col)
      - j=2: alternating Morton diagonal / Hilbert isotropic locality
    Payload encodes (op_idx, clue_idx, advance) strictly without witness information.
    """
    op_names = sorted(model_math["operators"].keys())
    op_to_id = {name: i for i, name in enumerate(op_names)}
    clue_names: List[Optional[str]] = [None] + sorted(model_math["clues"].keys())
    clue_to_id = {name: i for i, name in enumerate(clue_names)}

    rng = np.random.default_rng(77000 + seed)
    used_recs = set()
    hop_map: Dict[Tuple[int, int], HopSpec] = {}

    for si, scene in enumerate(scenes):
        for j, opt in enumerate(scene["options"]):
            geom = j if j < 2 else (2 if (si % 2 == 0) else 3)
            hop = _make_mixed_geometry_hop(rng, mem.width, mem.height, geom)
            tries = 0
            while hop.target_record in used_recs and tries < 32:
                hop = _make_mixed_geometry_hop(rng, mem.width, mem.height, geom)
                tries += 1
            used_recs.add(hop.target_record)

            _, _, op_name, clue_name, advance = opt
            op_id = op_to_id[op_name]
            cl_id = clue_to_id[clue_name]
            # Packed payload: advance in {1,2}, cl_id in 0..6, op_id in 0..7
            payload = (op_id * 16 + cl_id) * 4 + int(advance)
            mem.values[hop.target_record] = payload
            hop_map[(si, j)] = hop

    return hop_map, op_names, clue_names


def _decode_option_payload(
    payload: int, op_names: Sequence[str], clue_names: Sequence[Optional[str]]
) -> Tuple[str, Optional[str], int]:
    advance = int(payload % 4)
    rem = int(payload // 4)
    cl_id = int(rem % 16)
    op_id = int(rem // 16)
    return op_names[op_id], clue_names[cl_id], advance


def discover_route_for_ending(
    world_without_witness: Dict[str, Any],
    target_ending: str,
    arm: str,
    model: Optional[AdaptiveGeometryController],
    seed: int = 107,
    retrieval_budget: int = 6,
    corruption: float = 0.0,
    matched_rand_profile: Optional[Dict[str, Any]] = None,
    blind_random: bool = False,
) -> Dict[str, Any]:
    """Discover a secret-ending route via bounded memory retrieval + native lookahead planning.

    Strictly receives `world_without_witness` (no `exact`, no archived witnesses).
    """
    from MCP.mcp05.hilbert_storyworld import START, apply, classify, counterfactual, transition

    if "exact" in world_without_witness:
        raise ValueError("Archived witnesses must be stripped before calling route discovery!")

    scenes = world_without_witness["scenes"]
    model_math = world_without_witness["math"]
    rule = model_math["secrets"][target_ending]
    req_clues = set(rule["requires"])
    n_scenes = len(scenes)

    mem = SharedPhysicalMemory(32768, 3900000 + seed)
    hop_map, op_names, clue_names = _build_storyworld_memory_layout(mem, scenes, model_math, seed)

    rng = np.random.default_rng(51000 + seed * 31 + (0 if target_ending.endswith("question") else 1))
    state = START
    clues: frozenset = frozenset()
    cur_i = 0
    route: List[Tuple[int, int]] = []

    total_queries = 0
    retrieved_queries = 0
    total_probes = 0
    total_charged_reads = 0
    total_switches = 0

    # Cache of scene options already retrieved during this planning episode (paid once!)
    retrieved_cache: Dict[Tuple[int, int], Tuple[str, Optional[str], int]] = {}

    def retrieve_scene_options(scene_idx: int) -> Dict[int, Tuple[str, Optional[str], int]]:
        nonlocal total_queries, retrieved_queries, total_probes, total_charged_reads, total_switches
        if scene_idx >= n_scenes or blind_random:
            return {}
        to_fetch = [j for j in range(3) if (scene_idx, j) not in retrieved_cache]
        if to_fetch:
            roots = [
                QueryRoot(family="native_plan", family_id=-1, hops=(hop_map[(scene_idx, j)],))
                for j in to_fetch
            ]
            recs, _ = rollout_batch(
                mem,
                roots,
                arm=arm,
                model=model,
                budget=retrieval_budget,
                corruption=corruption,
                matched_rand_profile=matched_rand_profile,
                rng_seed=seed * 10000 + scene_idx * 17,
            )
            for j, rec in zip(to_fetch, recs):
                total_queries += 1
                total_probes += int(rec["probes"])
                total_charged_reads += int(rec["charged_reads"])
                total_switches += int(rec["switches"])
                if rec["success"] and rec["value"] is not None:
                    retrieved_queries += 1
                    retrieved_cache[(scene_idx, j)] = _decode_option_payload(
                        int(rec["value"]), op_names, clue_names
                    )
        return {
            j: retrieved_cache[(scene_idx, j)]
            for j in range(3)
            if (scene_idx, j) in retrieved_cache
        }

    while cur_i < n_scenes:
        if blind_random:
            chosen_j = int(rng.integers(0, 3))
        else:
            opts_cur = retrieve_scene_options(cur_i)
            # Bounded 1-scene lookahead retrieval (at most 2 next scenes for non-commuting planning)
            opts_next1 = retrieve_scene_options(cur_i + 1) if cur_i + 1 < n_scenes else {}
            opts_next2 = retrieve_scene_options(cur_i + 2) if cur_i + 2 < n_scenes else {}

            if not opts_cur:
                chosen_j = int(rng.integers(0, 3))
            else:
                missing_clues = req_clues - set(clues)
                progress = float(cur_i) / float(max(1, n_scenes - 1))
                # Weight projector alignment more heavily in late acts; clue acquisition in early/mid acts
                proj_w = 0.6 + 2.6 * (progress ** 2)
                clue_w = 4.5

                best_j = -1
                best_val = -1e18
                for j in range(3):
                    if j not in opts_cur:
                        # Unretrieved option is unknown -> cannot evaluate reliably
                        val = -3.0 + float(rng.uniform(-0.05, 0.05))
                    else:
                        op_name, clue_name, adv = opts_cur[j]
                        st1 = apply(state, model_math["operators"][op_name])
                        cl1 = clues | ({clue_name} if clue_name else frozenset())
                        nxt_i = cur_i + adv

                        # Check immediate terminal eligibility if this step ends the story
                        if nxt_i >= n_scenes:
                            elig = target_ending in classify(model_math, st1, cl1)
                            # Bonus if route + (cur_i, j) also exhibits a noncommuting swap counterfactual
                            cf_bonus = 0.0
                            if elig:
                                trial_route = [*route, (cur_i, j)]
                                cf = counterfactual(scenes, model_math, trial_route, target_ending)
                                cf_bonus = 5.0 if cf is not None else 0.0
                            val = (
                                (20.0 if elig else 0.0)
                                + cf_bonus
                                + clue_w * len(req_clues & set(cl1))
                                + 3.0 * _projector_margin_score(st1, rule)
                            )
                        else:
                            # 1-step lookahead over retrieved next-scene options (if available)
                            nxt_opts = opts_next1 if adv == 1 else opts_next2
                            if nxt_opts:
                                best_la = -1e18
                                for _, (op2, cl2_name, adv2) in nxt_opts.items():
                                    st2 = apply(st1, model_math["operators"][op2])
                                    cl2 = cl1 | ({cl2_name} if cl2_name else frozenset())
                                    gain_clues = len(missing_clues & set(cl2))
                                    la_sc = (
                                        clue_w * gain_clues
                                        + proj_w * _projector_margin_score(st2, rule)
                                    )
                                    if cur_i + adv + adv2 >= n_scenes:
                                        if target_ending in classify(model_math, st2, cl2):
                                            la_sc += 18.0
                                    if la_sc > best_la:
                                        best_la = la_sc
                                val = best_la
                            else:
                                gain_clues = 1 if (clue_name in missing_clues) else 0
                                val = clue_w * gain_clues + proj_w * _projector_margin_score(st1, rule)

                            # Mild tie-break jitter to avoid degenerate loops & favor noncommuting diversity
                            val += float(rng.uniform(-0.02, 0.02))

                    if val > best_val:
                        best_val = val
                        best_j = j
                chosen_j = best_j

        # Execute the chosen option through native Storyworld transition
        route.append((cur_i, chosen_j))
        adv, state, clues = transition(scenes[cur_i]["options"][chosen_j], model_math, state, clues)
        cur_i += adv

    return {
        "target_ending": target_ending,
        "arm": "blind_random" if blind_random else arm,
        "seed": seed,
        "retrieval_budget": retrieval_budget,
        "corruption": corruption,
        "total_queries": total_queries,
        "retrieved_queries": retrieved_queries,
        "retrieval_rate": float(retrieved_queries) / float(max(1, total_queries)) if total_queries else 0.0,
        "total_probes": total_probes,
        "total_charged_reads": total_charged_reads,
        "total_switches": total_switches,
        "route": route,
    }


def run_independent_route_discovery(
    world_path: Path,
    arms_models: Dict[str, Optional[AdaptiveGeometryController]],
    seeds: Sequence[int] = (107, 109, 113),
    retrieval_budget: int = 6,
    corruptions: Sequence[float] = (0.0, 0.15),
    matched_rand_profiles: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run full independent witness-withheld Storyworld route discovery across arms & seeds."""
    raw_bytes = Path(world_path).read_bytes()
    world_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    full_data = json.loads(raw_bytes.decode("utf-8"))
    stripped_data = strip_archived_witnesses(full_data)
    assert "exact" not in stripped_data

    secret_endings = sorted(stripped_data["math"]["secrets"].keys())
    archived_witnesses = {
        e: full_data["exact"]["secret_gates"][e]["witness"] for e in secret_endings
    }

    episodes: List[Dict[str, Any]] = []
    # Include blind random baseline once per seed/ending/corruption
    for corruption in corruptions:
        for seed in seeds:
            for ending in secret_endings:
                res = discover_route_for_ending(
                    stripped_data,
                    ending,
                    arm="blind_random",
                    model=None,
                    seed=seed,
                    retrieval_budget=retrieval_budget,
                    corruption=corruption,
                    blind_random=True,
                )
                ver = verify_discovered_route(
                    full_data["scenes"],
                    full_data["math"],
                    res["route"],
                    ending,
                    archived_witness=archived_witnesses[ending],
                )
                episodes.append({**{k: v for k, v in res.items() if k != "route"}, "verification": ver})

        for arm, model_by_seed in arms_models.items():
            for seed in seeds:
                mdl = model_by_seed[seed] if isinstance(model_by_seed, dict) else model_by_seed
                prof = matched_rand_profiles.get(seed) if matched_rand_profiles else None
                for ending in secret_endings:
                    res = discover_route_for_ending(
                        stripped_data,
                        ending,
                        arm=arm,
                        model=mdl,
                        seed=seed,
                        retrieval_budget=retrieval_budget,
                        corruption=corruption,
                        matched_rand_profile=prof,
                        blind_random=False,
                    )
                    ver = verify_discovered_route(
                        full_data["scenes"],
                        full_data["math"],
                        res["route"],
                        ending,
                        archived_witness=archived_witnesses[ending],
                    )
                    episodes.append(
                        {**{k: v for k, v in res.items() if k != "route"}, "verification": ver}
                    )

    # Aggregate summary per (corruption, arm)
    by_arm: Dict[str, Any] = {}
    for corruption in corruptions:
        c_key = f"c={corruption:.2f}"
        by_arm[c_key] = {}
        arm_names = ["blind_random"] + list(arms_models.keys())
        for arm in arm_names:
            sub = [e for e in episodes if e["arm"] == arm and abs(e["corruption"] - corruption) < 1e-9]
            if not sub:
                continue
            by_arm[c_key][arm] = {
                "n_trials": len(sub),
                "legal_transitions_rate": float(np.mean([e["verification"]["legal_transitions"] for e in sub])),
                "required_clues_rate": float(np.mean([e["verification"]["required_clues_met"] for e in sub])),
                "projectors_rate": float(np.mean([e["verification"]["terminal_projectors_met"] for e in sub])),
                "secret_eligible_rate": float(np.mean([e["verification"]["secret_ending_eligible"] for e in sub])),
                "counterfactual_rate": float(
                    np.mean([e["verification"]["noncommuting_counterfactual_verified"] for e in sub])
                ),
                "all_five_verified_rate": float(np.mean([e["verification"]["all_five_verified"] for e in sub])),
                "novel_route_rate": float(
                    np.mean([bool(e["verification"]["differs_from_archived_witness"]) for e in sub])
                ),
                "mean_retrieval_rate": float(np.mean([e["retrieval_rate"] for e in sub])),
                "mean_charged_reads": float(np.mean([e["total_charged_reads"] for e in sub])),
                "mean_switches": float(np.mean([e["total_switches"] for e in sub])),
            }

    return {
        "track": "native_route_discovery",
        "witness_withheld_verified": True,
        "world_sha256": world_sha256,
        "retrieval_budget": retrieval_budget,
        "corruptions": list(corruptions),
        "seeds": list(seeds),
        "summary_by_arm": by_arm,
        "episodes": episodes,
    }
