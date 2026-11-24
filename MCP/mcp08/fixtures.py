"""MCP-08A deterministic CPU fixture generator for all 4 environments.

Strict security & scientific isolation:
- Every root returns `(public_root, private_witness)`.
- `public_root["federation_corpus"]` contains ONLY searchable documents (including deliberate
  contradictory/obsolete lore, renamed functions, conflicting commits, and distractors).
- `private_witness` contains ground-truth answers, mathematical witnesses, unseen test maps,
  and invariant oracles and is NEVER embedded in `public_root` or its corpus.
"""
from __future__ import annotations

import cmath
import hashlib
import json
import math
import random
from typing import Any


ENVIRONMENTS = ("flooded_city", "quantum_labyrinth", "arena_architect", "broken_archive")

NAMESPACE_OFFSETS = {
    "dev": 800000,
    "pilot": 805000,
    "eval": 810000,
}

ENV_OFFSETS = {
    "flooded_city": 0,
    "quantum_labyrinth": 100,
    "arena_architect": 200,
    "broken_archive": 300,
}


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Environment 1: The Flooded City (narrative memory under contradictory lore)
# ---------------------------------------------------------------------------

def _build_flooded_city(seed: int, root_id: str) -> tuple[dict, dict]:
    rng = random.Random(seed)
    districts = [
        "Sluicegate", "Bellspire", "Cistern Row", "Salt Market",
        "Lantern Dock", "Glass Arcade", "Tidewatch", "Old Foundry",
    ]
    factions = ["Hydrologists", "Bellkeepers", "Dredge Guild", "Tidewardens"]
    items = ["brass_key", "flood_ledger", "mercury_seal", "diving_bell_pass", "salt_cipher", "valve_crank"]
    npcs = ["Mira", "Tarek", "Olen", "Sera", "Voss", "Kestrel"]

    rng.shuffle(districts)
    rng.shuffle(items)
    rng.shuffle(npcs)

    tide_level = rng.randint(3, 6)  # meters above datum
    flooded_set = set(districts[:3])
    passable_set = set(districts[3:])

    ally_npc, rival_npc, archivist_npc = npcs[0], npcs[1], npcs[2]
    req_item = items[0]
    decoy_item = items[1]
    start_district = districts[3]
    mid_district = districts[4]
    goal_district = districts[5]
    submerged_district = districts[0]
    ally_faction = factions[seed % len(factions)]
    rival_faction = factions[(seed + 1) % len(factions)]

    # Build searchable federation corpus with canonical current records + stale/contradictory lore
    corpus = [
        {
            "doc_id": f"{root_id}_tide_old",
            "epoch": 12,
            "status": "obsolete",
            "title": "Harbor Almanac (Year 411 — Superseded)",
            "text": (
                f"Tide datum stands at 1m. {submerged_district} is dry and open to foot traffic. "
                f"{ally_npc} belongs to {rival_faction} and demands {decoy_item} for passage."
            ),
        },
        {
            "doc_id": f"{root_id}_tide_cur",
            "epoch": 19,
            "status": "canonical",
            "title": f"Tidewatch Gauge Dispatch (Epoch 19, Seed {seed})",
            "text": (
                f"Emergency surge: tide datum reached {tide_level}m. "
                f"Submerged (impassable without diving rig): {', '.join(sorted(flooded_set))}. "
                f"Passable high-causeway districts: {', '.join(sorted(passable_set))}."
            ),
        },
        {
            "doc_id": f"{root_id}_char_cur",
            "epoch": 19,
            "status": "canonical",
            "title": f"Faction Registry Update (Epoch 19)",
            "text": (
                f"Following the Sluice Schism, {ally_npc} defected to {ally_faction} and controls "
                f"the gate at {mid_district}, admitting bearers of `{req_item}`. "
                f"{rival_npc} of {rival_faction} holds a grudge against {ally_faction} and confiscates `{decoy_item}`."
            ),
        },
        {
            "doc_id": f"{root_id}_rumor_conflict",
            "epoch": 16,
            "status": "contradictory",
            "title": "Tavern Broadsheet (Unverified Rumor)",
            "text": (
                f"Dockside gossips claim {goal_district} sank beneath the waves and that `{decoy_item}` "
                f"opens the {mid_district} causeway. Harbor engineers dispute both claims."
            ),
        },
        {
            "doc_id": f"{root_id}_inventory_manifest",
            "epoch": 19,
            "status": "canonical",
            "title": "Courier Satchel & Vault Log (Epoch 19)",
            "text": (
                f"Courier departs {start_district} carrying `{req_item}` (qty=1) and 0 diving rigs. "
                f"Delivering `{req_item}` to {archivist_npc} at {goal_district} unlocks ending `archive_preserved`."
            ),
        },
    ]
    # Add realistic distractor lore records
    for j in range(6):
        corpus.append({
            "doc_id": f"{root_id}_distractor_{j}",
            "epoch": 10 + j,
            "status": "background",
            "title": f"Municipal Dredging Log #{seed}-{j}",
            "text": f"Routine silt clearance near {districts[(j + 2) % len(districts)]}; {rng.randint(12, 90)} barges logged.",
        })
    rng.shuffle(corpus)

    required_evidence = [f"{root_id}_tide_cur", f"{root_id}_char_cur", f"{root_id}_inventory_manifest"]
    witness_token = f"WITNESS_FC_{_sha([seed, 'flooded_city'])[:12]}"

    public_root = {
        "root_id": root_id,
        "env": "flooded_city",
        "seed": seed,
        "prompt": (
            f"Author a 3-scene branching narrative state plan starting in `{start_district}` that reaches "
            f"the valid terminal ending while respecting Epoch-19 flood conditions, faction allegiances, "
            f"and inventory conservation. Reject obsolete/contradictory lore."
        ),
        "initial_state": {
            "district": start_district,
            "inventory": [req_item],
            "tide_level_m": tide_level,
            "ending": None,
        },
        "federation_corpus": corpus,
    }
    private_witness = {
        "witness_id": witness_token,
        "root_id": root_id,
        "env": "flooded_city",
        "required_evidence_ids": required_evidence,
        "forbidden_stale_ids": [f"{root_id}_tide_old", f"{root_id}_rumor_conflict"],
        "flooded_districts": sorted(flooded_set),
        "passable_districts": sorted(passable_set),
        "start_district": start_district,
        "mid_district": mid_district,
        "goal_district": goal_district,
        "ally_npc": ally_npc,
        "ally_faction": ally_faction,
        "archivist_npc": archivist_npc,
        "required_item": req_item,
        "decoy_item": decoy_item,
        "valid_ending": "archive_preserved",
        "oracle_submission": {
            "cited_docs": required_evidence,
            "scenes": [
                {"id": "scene_0", "district": start_district, "npc": None, "inventory": [req_item], "action": "depart_causeway"},
                {"id": "scene_1", "district": mid_district, "npc": ally_npc, "faction": ally_faction, "inventory": [req_item], "action": f"present_{req_item}"},
                {"id": "scene_2", "district": goal_district, "npc": archivist_npc, "inventory": [], "action": f"deliver_{req_item}", "ending": "archive_preserved"},
            ],
            "ending": "archive_preserved",
        },
    }
    return public_root, private_witness


# ---------------------------------------------------------------------------
# Environment 2: Quantum Labyrinth (noncommuting 2-qubit Hilbert planning)
# ---------------------------------------------------------------------------

def _mat_vec(mat: list[list[complex]], vec: list[complex]) -> list[complex]:
    return [sum(mat[r][c] * vec[c] for c in range(len(vec))) for r in range(len(mat))]


def _hilbert_ops() -> dict[str, list[list[complex]]]:
    s2 = 1.0 / math.sqrt(2.0)
    # 4D (2-qubit) unitary operators that do NOT pairwise commute:
    # H0 = H \otimes I, CNOT, T1 = I \otimes diag(1, exp(i*pi/4)), SWAP, X1 = I \otimes X
    return {
        "H0": [
            [s2, 0, s2, 0],
            [0, s2, 0, s2],
            [s2, 0, -s2, 0],
            [0, s2, 0, -s2],
        ],
        "CNOT": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        "X0": [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        "X1": [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        "Z0": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ],
        "T1": [
            [1, 0, 0, 0],
            [0, cmath.exp(1j * math.pi / 4), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, cmath.exp(1j * math.pi / 4)],
        ],
        "SWAP": [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
    }


def apply_hilbert_sequence(ops_seq: list[str], init_state: list[complex] | None = None) -> list[complex]:
    ops = _hilbert_ops()
    psi = list(init_state) if init_state is not None else [1.0 + 0j, 0j, 0j, 0j]
    for name in ops_seq:
        if name not in ops:
            raise ValueError(f"Unknown operator: {name}")
        psi = _mat_vec(ops[name], psi)
    return psi


def state_probs(psi: list[complex]) -> list[float]:
    return [round(abs(a) ** 2, 6) for a in psi]


def _build_quantum_labyrinth(seed: int, root_id: str) -> tuple[dict, dict]:
    rng = random.Random(seed)
    # Construct guaranteed noncommuting target route of length 3 or 4 from |00>
    templates = [
        ["H0", "CNOT", "X1"],      # -> (|01> + |10>)/sqrt(2), probs [0, 0.5, 0.5, 0]
        ["X0", "H0", "CNOT"],      # -> (|00> - |11>)/sqrt(2), probs [0.5, 0, 0, 0.5]
        ["H0", "CNOT", "T1", "H0"],  # noncommuting interference
        ["X1", "H0", "CNOT", "SWAP"],
        ["H0", "SWAP", "CNOT"],    # -> (|00> + |11>)/sqrt(2) via SWAP vs CNOT order!
        ["X0", "SWAP", "H0", "CNOT"],
    ]
    chosen_seq = list(templates[seed % len(templates)])
    psi_target = apply_hilbert_sequence(chosen_seq)
    probs_target = state_probs(psi_target)

    # Verify noncommuting sensitivity (at least one adjacent swap changes probabilities or phase)
    swapped_diff = False
    for idx in range(len(chosen_seq) - 1):
        swapped = list(chosen_seq)
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        if state_probs(apply_hilbert_sequence(swapped)) != probs_target:
            swapped_diff = True
            break

    costs = {"H0": 2, "CNOT": 3, "X0": 1, "X1": 1, "Z0": 1, "T1": 2, "SWAP": 2}
    optimal_cost = sum(costs[op] for op in chosen_seq)
    resource_budget = optimal_cost + 2

    # Searchable federation corpus contains ONLY room/operator specifications — NEVER the solution route!
    corpus = []
    for op_name, op_cost in costs.items():
        corpus.append({
            "doc_id": f"{root_id}_op_{op_name}",
            "kind": "operator_card",
            "operator": op_name,
            "energy_cost": op_cost,
            "description": f"Chamber gate `{op_name}` (energy={op_cost}) acts on the 2-qubit state (|00>,|01>,|10>,|11>).",
        })
    corpus.append({
        "doc_id": f"{root_id}_terminal_spec",
        "kind": "ending_rule",
        "text": (
            f"Hidden ending `bell_lock_{seed}` unlocks iff starting from |00> the final measurement "
            f"probabilities equal {probs_target} (L1 tolerance 0.02) within energy budget <= {resource_budget} "
            f"and <= {len(chosen_seq) + 1} steps."
        ),
    })
    for j in range(5):
        corpus.append({
            "doc_id": f"{root_id}_decoy_note_{j}",
            "kind": "decoy_log",
            "text": f"Abandoned explorer note #{seed}-{j}: tried Z0 twice; state stayed at |00>.",
        })
    rng.shuffle(corpus)

    required_evidence = [f"{root_id}_terminal_spec"] + sorted({f"{root_id}_op_{op}" for op in chosen_seq})
    public_root = {
        "root_id": root_id,
        "env": "quantum_labyrinth",
        "seed": seed,
        "prompt": (
            f"Starting from |00> = [1,0,0,0], discover an ordered sequence of chamber operators within "
            f"energy budget <= {resource_budget} that achieves target probability vector {probs_target}. "
            f"Witness route is withheld."
        ),
        "initial_state_label": "|00>",
        "resource_budget": resource_budget,
        "max_steps": len(chosen_seq) + 1,
        "target_probabilities": probs_target,
        "federation_corpus": corpus,
    }
    private_witness = {
        "witness_id": f"WITNESS_QL_{_sha([seed, chosen_seq])[:12]}",
        "root_id": root_id,
        "env": "quantum_labyrinth",
        "required_evidence_ids": required_evidence,
        "oracle_sequence": chosen_seq,
        "optimal_cost": optimal_cost,
        "resource_budget": resource_budget,
        "max_steps": len(chosen_seq) + 1,
        "target_probabilities": probs_target,
        "noncommuting_verified": swapped_diff,
        "operator_costs": costs,
        "oracle_submission": {
            "cited_docs": required_evidence,
            "operators": chosen_seq,
            "ending": f"bell_lock_{seed}",
        },
    }
    return public_root, private_witness


# ---------------------------------------------------------------------------
# Environment 3: Arena Architect (executable JS combat & Canvas invariants)
# ---------------------------------------------------------------------------

def _build_arena_architect(seed: int, root_id: str) -> tuple[dict, dict]:
    rng = random.Random(seed)
    grid_w, grid_h = 10, 10
    damage = rng.randint(15, 30)
    enemy_hp = damage * rng.randint(2, 3)
    attack_range = rng.choice([1, 2])

    # Unseen maps held STRICTLY in private witness!
    unseen_maps = []
    for m_idx in range(2):
        wall_x = rng.randint(3, 6)
        walls = [[wall_x, y] for y in range(2, 7)]
        unseen_maps.append({
            "map_id": f"unseen_{seed}_{m_idx}",
            "width": grid_w,
            "height": grid_h,
            "walls": walls,
            "player_start": [1, 4],
            "enemy_start": [8, 4],
            "enemy_hp": enemy_hp,
            "damage": damage,
            "attack_range": attack_range,
        })

    buggy_source = f"""// Buggy Arena Engine (MCP-08 fixture {root_id})
function stepArena(state, action) {{
  // BUG 1: ignores wall collisions and arena bounds
  if (action === 'RIGHT') state.player.x += 1;
  if (action === 'LEFT') state.player.x -= 1;
  if (action === 'UP') state.player.y -= 1;
  if (action === 'DOWN') state.player.y += 1;
  // BUG 2: ATTACK does not check Manhattan distance <= attack_range or clamp enemy.hp >= 0
  if (action === 'ATTACK') {{
    state.enemy.hp -= {damage * 5};
  }}
  // BUG 3: Enemy AI never moves toward player around walls
  return state;
}}
function renderArena(ctx, state) {{
  // BUG 4: Never clears canvas or draws walls/player/enemy
}}
"""

    fixed_oracle_source = f"""function stepArena(state, action) {{
  const inBounds = (x, y) => x >= 0 && y >= 0 && x < state.width && y < state.height;
  const isWall = (x, y) => state.walls.some(w => w[0] === x && w[1] === y);
  const tryMove = (ent, dx, dy) => {{
    const nx = ent.x + dx, ny = ent.y + dy;
    if (inBounds(nx, ny) && !isWall(nx, ny)) {{ ent.x = nx; ent.y = ny; }}
  }};
  if (action === 'RIGHT') tryMove(state.player, 1, 0);
  else if (action === 'LEFT') tryMove(state.player, -1, 0);
  else if (action === 'UP') tryMove(state.player, 0, -1);
  else if (action === 'DOWN') tryMove(state.player, 0, 1);
  else if (action === 'ATTACK') {{
    const dist = Math.abs(state.player.x - state.enemy.x) + Math.abs(state.player.y - state.enemy.y);
    if (dist <= state.attackRange && state.enemy.hp > 0) {{
      state.enemy.hp = Math.max(0, state.enemy.hp - state.damage);
      state.hits = (state.hits || 0) + 1;
    }}
  }}
  // Enemy BFS one-step chase around walls while alive
  if (state.enemy.hp > 0) {{
    const q = [[state.enemy.x, state.enemy.y, null]];
    const seen = new Set([state.enemy.x + ',' + state.enemy.y]);
    let chosen = null;
    while (q.length) {{
      const [cx, cy, first] = q.shift();
      if (cx === state.player.x && cy === state.player.y) {{ chosen = first; break; }}
      for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {{
        const nx = cx + dx, ny = cy + dy;
        const key = nx + ',' + ny;
        if (!inBounds(nx, ny) || isWall(nx, ny) || seen.has(key)) continue;
        seen.add(key);
        q.push([nx, ny, first || [nx, ny]]);
      }}
    }}
    if (chosen && !(chosen[0] === state.player.x && chosen[1] === state.player.y)) {{
      state.enemy.x = chosen[0];
      state.enemy.y = chosen[1];
    }}
  }}
  state.status = state.enemy.hp <= 0 ? 'victory' : 'playing';
  return state;
}}
function renderArena(ctx, state) {{
  ctx.clearRect(0, 0, state.width * 32, state.height * 32);
  ctx.fillStyle = '#1b263b';
  ctx.fillRect(0, 0, state.width * 32, state.height * 32);
  ctx.fillStyle = '#778da9';
  for (const w of state.walls) ctx.fillRect(w[0] * 32, w[1] * 32, 32, 32);
  ctx.fillStyle = '#00f5d4';
  ctx.fillRect(state.player.x * 32, state.player.y * 32, 32, 32);
  ctx.fillStyle = '#f15bb5';
  ctx.fillRect(state.enemy.x * 32, state.enemy.y * 32, 32, 32);
}}
"""

    corpus = [
        {"doc_id": f"{root_id}_buggy_js", "kind": "source", "text": buggy_source},
        {
            "doc_id": f"{root_id}_spec",
            "kind": "contract",
            "text": (
                f"Requirements: (1) stepArena(state, action) must enforce bounds [0,width)x[0,height) and wall collisions; "
                f"(2) ATTACK deals state.damage ({damage}) only when Manhattan distance <= state.attackRange ({attack_range}), "
                f"clamping enemy.hp >= 0 and setting state.status='victory' when enemy.hp===0; "
                f"(3) enemy AI moves 1 orthogonal step per turn around walls toward player without overlapping player; "
                f"(4) renderArena(ctx, state) clears canvas and draws background, walls, player, and enemy with distinct fillStyle."
            ),
        },
        {
            "doc_id": f"{root_id}_stale_spec",
            "kind": "obsolete_contract",
            "text": "DEPRECATED v0.1 note: walls are decorative only and ATTACK deals 999 global damage.",
        },
    ]
    for j in range(4):
        corpus.append({"doc_id": f"{root_id}_changelog_{j}", "kind": "noise", "text": f"UI palette tweak #{seed}-{j}."})
    rng.shuffle(corpus)

    required_evidence = [f"{root_id}_buggy_js", f"{root_id}_spec"]
    public_root = {
        "root_id": root_id,
        "env": "arena_architect",
        "seed": seed,
        "prompt": "Repair stepArena(state, action) and renderArena(ctx, state) so they pass unseen map, adversarial input, AI pathfinding, and Canvas rendering checks.",
        "federation_corpus": corpus,
    }
    private_witness = {
        "witness_id": f"WITNESS_AA_{_sha([seed, unseen_maps])[:12]}",
        "root_id": root_id,
        "env": "arena_architect",
        "required_evidence_ids": required_evidence,
        "unseen_maps": unseen_maps,
        "oracle_submission": {
            "cited_docs": required_evidence,
            "code": fixed_oracle_source,
        },
    }
    return public_root, private_witness


# ---------------------------------------------------------------------------
# Environment 4: The Broken Archive (multi-hop federated repo research)
# ---------------------------------------------------------------------------

def _build_broken_archive(seed: int, root_id: str) -> tuple[dict, dict]:
    rng = random.Random(seed)
    old_fn = f"route_cells_v1_{seed % 7}"
    mid_fn = f"dispatch_projection_v2_{seed % 7}"
    new_fn = f"adaptive_page_switch_v3_{seed % 7}"
    config_key = f"SWITCH_PENALTY_BPS_{seed % 11}"
    stale_val = rng.randint(10, 45)
    canonical_val = rng.randint(55, 95)
    old_commit = _sha([seed, "old_commit"])[:10]
    revert_commit = _sha([seed, "revert_commit"])[:10]
    canonical_commit = _sha([seed, "canonical_commit"])[:10]

    corpus = [
        {
            "doc_id": f"{root_id}_repoA_readme_stale",
            "repo": "goblin-core",
            "commit": old_commit,
            "timestamp": "2026-08-01",
            "text": f"README (outdated): Call `{old_fn}()` with `{config_key} = {stale_val}`.",
        },
        {
            "doc_id": f"{root_id}_repoA_rename_log",
            "repo": "goblin-core",
            "commit": revert_commit,
            "timestamp": "2026-09-14",
            "text": f"Refactor commit {revert_commit}: renamed `{old_fn}` -> `{mid_fn}` and moved penalty config to `goblin-router/policy.toml`.",
        },
        {
            "doc_id": f"{root_id}_repoB_migration",
            "repo": "goblin-router",
            "commit": canonical_commit,
            "timestamp": "2026-10-28",
            "text": (
                f"Commit {canonical_commit} (HEAD): superseded `{mid_fn}` with `{new_fn}()`. "
                f"Set canonical `{config_key} = {canonical_val}` in `goblin-router/policy.toml`."
            ),
        },
        {
            "doc_id": f"{root_id}_repoC_pr_reverted",
            "repo": "goblin-experiments",
            "commit": _sha([seed, "abandoned"])[:10],
            "timestamp": "2026-10-10",
            "text": f"Draft PR (CLOSED/UNMERGED): proposed setting `{config_key} = {stale_val + 7}` inside `{mid_fn}`.",
        },
    ]
    for j in range(7):
        corpus.append({
            "doc_id": f"{root_id}_noise_commit_{j}",
            "repo": rng.choice(["goblin-core", "goblin-router", "goblin-ui"]),
            "commit": _sha([seed, "noise", j])[:10],
            "timestamp": f"2026-09-{10 + j:02d}",
            "text": f"Chores & formatting in module_{j}; touches unrelated metric `CACHE_TTL_{j} = {rng.randint(100, 999)}`.",
        })
    rng.shuffle(corpus)

    required_evidence = [f"{root_id}_repoA_rename_log", f"{root_id}_repoB_migration"]
    public_root = {
        "root_id": root_id,
        "env": "broken_archive",
        "seed": seed,
        "prompt": (
            f"Trace the current HEAD symbol that replaced legacy `{old_fn}()` across the federated repos, "
            f"return the canonical `{config_key}` value, the full dependency rename chain, and valid source doc IDs."
        ),
        "legacy_symbol": old_fn,
        "config_key": config_key,
        "federation_corpus": corpus,
    }
    private_witness = {
        "witness_id": f"WITNESS_BA_{_sha([seed, new_fn, canonical_val])[:12]}",
        "root_id": root_id,
        "env": "broken_archive",
        "required_evidence_ids": required_evidence,
        "stale_doc_ids": [f"{root_id}_repoA_readme_stale", f"{root_id}_repoC_pr_reverted"],
        "canonical_symbol": new_fn,
        "canonical_value": canonical_val,
        "canonical_commit": canonical_commit,
        "dependency_chain": [old_fn, mid_fn, new_fn],
        "oracle_submission": {
            "cited_docs": required_evidence,
            "canonical_symbol": new_fn,
            "canonical_value": canonical_val,
            "canonical_commit": canonical_commit,
            "dependency_chain": [old_fn, mid_fn, new_fn],
        },
    }
    return public_root, private_witness


BUILDERS = {
    "flooded_city": _build_flooded_city,
    "quantum_labyrinth": _build_quantum_labyrinth,
    "arena_architect": _build_arena_architect,
    "broken_archive": _build_broken_archive,
}


def generate_suite(split: str = "eval", roots_per_env: int = 8) -> tuple[list[dict], dict[str, dict]]:
    """Generate disjoint public roots (searchable federation) and private verifier witnesses."""
    if split not in NAMESPACE_OFFSETS:
        raise ValueError(f"Unknown split: {split}")
    base = NAMESPACE_OFFSETS[split]
    public_roots: list[dict] = []
    private_witnesses: dict[str, dict] = {}
    for env in ENVIRONMENTS:
        env_base = base + ENV_OFFSETS[env]
        builder = BUILDERS[env]
        for idx in range(roots_per_env):
            seed = env_base + idx
            root_id = f"mcp08_{split}_{env}_{idx:02d}"
            pub, wit = builder(seed, root_id)
            pub["split"] = split
            pub["index"] = idx
            pub["in_wide_oracle_subset"] = (split == "eval" and idx in (0, 4))
            public_roots.append(pub)
            private_witnesses[root_id] = wit
    return public_roots, private_witnesses


def assert_zero_witness_leakage(public_roots: list[dict], private_witnesses: dict[str, dict]) -> None:
    """Strictly verify no private witness ID, oracle sequence, or unseen test map appears in any searchable corpus."""
    for pub in public_roots:
        rid = pub["root_id"]
        wit = private_witnesses[rid]
        blob = json.dumps(pub["federation_corpus"], sort_keys=True)
        assert wit["witness_id"] not in blob, f"Witness ID leaked into corpus for {rid}"
        if pub["env"] == "quantum_labyrinth":
            seq_str = json.dumps(wit["oracle_sequence"])
            assert seq_str not in blob, f"Oracle sequence leaked into corpus for {rid}"
        elif pub["env"] == "arena_architect":
            for m in wit["unseen_maps"]:
                assert m["map_id"] not in blob, f"Unseen map {m['map_id']} leaked into corpus for {rid}"
        elif pub["env"] == "flooded_city":
            assert "oracle_submission" not in blob, f"Oracle submission leaked into {rid}"
        elif pub["env"] == "broken_archive":
            assert "WITNESS_BA_" not in blob, f"Broken archive witness leaked into {rid}"
