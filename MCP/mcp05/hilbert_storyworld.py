"""Exact authoring contracts and numerical replay for two-factor storyworlds."""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations, pairwise
from random import Random

from .storyworld_swmd import require

START = (1 + 0j, 0j, 0j, 0j)
BASIS = ("00", "01", "10", "11")
GATES = frozenset({"I", "CX01", "CX10"} | {f"{g}{q}" for g in "HXZS" for q in (0, 1)})
PROJECTORS = frozenset(BASIS) | {f"{axis}{q}{sign}" for axis in "XYZ" for q in (0, 1) for sign in "+-"}
EPSILON = 1e-9


def gate(state, name):
    """Qubit zero is the left, most-significant basis index."""
    if name == "I":
        return state
    result = list(state)
    if name in ("CX01", "CX10"):
        control, target = (2, 1) if name == "CX01" else (1, 2)
        for i, value in enumerate(state):
            result[i ^ target if i & control else i] = value
        return tuple(result)
    symbol, qubit = name
    bit = 2 >> int(qubit)
    for i in range(4):
        if i & bit:
            continue
        j = i | bit
        match symbol:
            case "H":
                result[i], result[j] = (state[i] + state[j]) / math.sqrt(2), (state[i] - state[j]) / math.sqrt(2)
            case "X":
                result[i], result[j] = state[j], state[i]
            case "Z":
                result[j] = -state[j]
            case "S":
                result[j] = 1j * state[j]
            case _:
                raise ValueError(f"unknown gate {name}")
    return tuple(result)


def apply(state, sequence):
    for primitive in sequence:
        state = gate(state, primitive)
    return state


def probability(state, projector):
    """Return the Born weight of a named computational or Pauli projector."""
    if projector in BASIS:
        return abs(state[BASIS.index(projector)]) ** 2
    require(projector in PROJECTORS, f"unknown projector {projector}")
    axis, qubit, sign = projector
    bit = 2 >> int(qubit)
    if axis == "Z":
        desired = bit if sign == "-" else 0
        return sum(abs(value) ** 2 for i, value in enumerate(state) if i & bit == desired)
    phase = -1j if axis == "Y" else 1
    if sign == "-":
        phase *= -1
    return sum(abs(state[i] + phase * state[i | bit]) ** 2 / 2 for i in range(4) if not i & bit)


def signature(state):
    pivot = next(value for value in state if abs(value) > EPSILON)
    phase = pivot.conjugate() / abs(pivot)
    return tuple((round((value * phase).real, 9), round((value * phase).imag, 9)) for value in state)


def order_changes_operator(first, second):
    """Compare both matrices modulo one global phase, independent of story prose."""
    columns = [tuple(complex(i == j) for i in range(4)) for j in range(4)]
    forward = tuple(value for state in columns for value in apply(state, (*first, *second)))
    reverse = tuple(value for state in columns for value in apply(state, (*second, *first)))
    return signature(forward) != signature(reverse)


def pos_conditions(rule, state, clues):
    return set(rule["requires"]) <= clues and all(
        low - EPSILON <= probability(state, projector) <= high + EPSILON
        for projector, low, high in rule["projectors"]
    )


def check_algebra(model, plan):
    require(set(model) == {"basis", "clues", "operators", "secrets"}, "model needs basis, clues, operators and secrets")
    require(set(model["basis"]) == set(BASIS) and len(set(model["basis"].values())) == 4,
            "name each computational basis state distinctly")
    for mapping in (model["basis"], model["clues"]):
        require(all(isinstance(v, str) and v.strip() for v in mapping.values()), "basis and clue descriptions need text")
    require(set(model["clues"]) == {f"c{i}" for i in range(6)}, "define clue IDs c0 through c5")
    operations = model["operators"]
    require(6 <= len(operations) <= 8, "define six to eight named operators")
    for name, sequence in operations.items():
        require(name.isidentifier() and 1 <= len(sequence) <= 4 and set(sequence) <= GATES,
                f"invalid gate sequence for {name}")
    require(len({tuple(sequence) for sequence in operations.values()}) == len(operations), "operator sequences repeat")
    order_pairs = [pair for pair in combinations(operations, 2)
                   if order_changes_operator(*(operations[name] for name in pair))]
    require(order_pairs, "operator catalog needs an order-sensitive pair")
    hidden = {row[0] for row in plan["endings"] if row[1]}
    require(set(model["secrets"]) == hidden, "provide a gate for both hidden endings")
    for secret, rule in model["secrets"].items():
        require(set(rule) == {"requires", "projectors"}, f"{secret} needs clues and projector intervals")
        clues = rule["requires"]
        tests = rule["projectors"]
        require(2 <= len(clues) <= 4 and len(set(clues)) == len(clues) and set(clues) <= model["clues"].keys(),
                f"{secret} needs two to four distinct defined clues")
        require(2 <= len(tests) <= 3 and len({row[0] for row in tests}) == len(tests),
                f"{secret} needs two to three different projector tests")
        for projector, lower, upper in tests:
            require(projector in PROJECTORS and type(lower) in (int, float) and type(upper) in (int, float)
                    and 0 <= lower <= upper <= 1 and upper - lower < 1, f"invalid interval for {projector}")
        require(any(row[0].startswith(("X", "Y")) for row in tests), f"{secret} needs a phase-sensitive projector")
    first, second = model["secrets"].values()
    require(first != second and set(first["requires"]) != set(second["requires"]), "secret clue routes must differ")
    return {"order_sensitive_operator_pairs": order_pairs}


def check_encounters(scenes, ids, model):
    require(isinstance(scenes, list) and [scene["id"] for scene in scenes] == list(ids), "encounter IDs or order differ")
    for scene in scenes:
        for key in ("title", "text"):
            require(isinstance(scene[key], str) and scene[key].strip(), f"missing {scene['id']} {key}")
        options = scene["options"]
        require(len(options) == 3 and len({option[0] for option in options}) == 3,
                f"{scene['id']} requires three distinct choices")
        for choice, reply, operator, clue, advance in options:
            require(isinstance(choice, str) and choice.strip() and isinstance(reply, str) and reply.strip()
                    and operator in model["operators"] and (clue is None or clue in model["clues"])
                    and type(advance) is int and advance in (1, 2), f"invalid choice in {scene['id']}")
        steps = {option[4] for option in options}
        wanted = {1, 2} if int(scene["id"][4:]) % 6 == 3 else {1}
        require(steps == wanted, f"{scene['id']} requires advances {sorted(wanted)}")
    return scenes


def check_scene_continuity(scenes, ids, model, earlier):
    check_encounters(scenes, ids, model)
    for field in ("title", "text"):
        texts = [scene[field].strip().casefold() for scene in (*earlier, *scenes)]
        require(len(texts) == len(set(texts)), f"repeated encounter {field}")


def transition(option, model, state, clues):
    _, _, operator, clue, advance = option
    state = apply(state, model["operators"][operator])
    require(abs(sum(abs(value) ** 2 for value in state) - 1) <= EPSILON, "unitary lost state norm")
    return advance, state, clues | {clue} if clue else clues


def classify(model, state, clues):
    return sorted(end for end, rule in model["secrets"].items() if pos_conditions(rule, state, clues))


def simulate(scenes, model, public, runs=5000, seed=523):
    rng = Random(seed)
    visits, endings, seen_secrets = Counter(), Counter(), Counter()
    witnesses, counterfactuals = {}, {}
    lengths = Counter()
    inspected = Counter()
    for _ in range(runs):
        i, state, clues, path = 0, START, frozenset(), []
        while i < len(scenes):
            scene = scenes[i]
            visits[scene["id"]] += 1
            selected = rng.randrange(3)
            path.append([i, selected])
            advance, state, clues = transition(scene["options"][selected], model, state, clues)
            i += advance
        eligible = classify(model, state, clues)
        lengths[len(path)] += 1
        seen_secrets.update(eligible)
        ending = rng.choice([*public, *eligible])
        endings[ending] += 1
        for name in eligible:
            witnesses.setdefault(name, path)
            if name not in counterfactuals and inspected[name] < 40:
                inspected[name] += 1
                changed = counterfactual(scenes, model, path, name)
                if changed:
                    counterfactuals[name] = {"witness": path, **changed}
    return {"runs": runs, "seed": seed, "encounter_visits": dict(visits), "endings": dict(endings),
            "secret_eligible": dict(seen_secrets), "witnesses": witnesses,
            "counterfactuals": counterfactuals, "route_lengths": dict(lengths)}


def exact_routes(scenes, model):
    """Merge paths reaching the same scene, ray and clue set, retaining the probability mass."""
    finish = len(scenes)
    layers = [{} for _ in range(finish + 1)]
    layers[0][signature(START), frozenset()] = (START, 1.0, [])
    for i, scene in enumerate(scenes):
        for (_, clues), (state, mass, witness) in layers[i].items():
            for j, option in enumerate(scene["options"]):
                advance, changed, facts = transition(option, model, state, clues)
                next_layer = layers[min(i + advance, finish)]
                key = signature(changed), facts
                initial, prior, path = next_layer.get(key, (changed, 0.0, [*witness, [i, j]]))
                next_layer[key] = (initial, prior + mass / len(scene["options"]), path)
    final = layers[-1]
    require(abs(sum(row[1] for row in final.values()) - 1) < EPSILON, "terminal probability mass differs from one")
    endings = {}
    for name, rule in model["secrets"].items():
        reachable = 0.0
        joint = Counter()
        witness = None
        order_witness = None
        for (_, clues), (state, mass, path) in final.items():
            if set(rule["requires"]) <= clues:
                probabilities = tuple(round(probability(state, row[0]), 9) for row in rule["projectors"])
                joint[probabilities] += mass
            if pos_conditions(rule, state, clues):
                reachable += mass
                witness = witness or path
                if order_witness is None:
                    trial = counterfactual(scenes, model, path, name)
                    if trial:
                        order_witness = {"witness": path, **trial}
        endings[name] = {"eligible_probability": reachable, "witness": witness, "order_witness": order_witness,
                         "clue_probability": sum(joint.values()),
                         "projector_distribution": [{"values": values, "probability": mass}
                                                    for values, mass in joint.most_common()]}
    return {"states_per_scene": [len(layer) for layer in layers], "secret_gates": endings}


def replay_path(scenes, model, path, swap=None):
    gates, clues = [], set()
    for i, j in path:
        _, _, operation, clue, _ = scenes[i]["options"][j]
        gates.append(model["operators"][operation])
        if clue:
            clues.add(clue)
    if swap:
        a, b = swap
        gates[a], gates[b] = gates[b], gates[a]
    state = START
    for sequence in gates:
        state = apply(state, sequence)
    return state, clues


def counterfactual(scenes, model, path, ending):
    """Swap neighboring operations, holding story facts and route fixed."""
    state, clues = replay_path(scenes, model, path)
    rule = model["secrets"][ending]
    require(pos_conditions(rule, state, clues), "witness is ineligible")
    for pair in pairwise(range(len(path))):
        changed, same_clues = replay_path(scenes, model, path, swap=pair)
        if not pos_conditions(rule, changed, same_clues):
            return {"swap_decisions": pair, "clues": sorted(clues),
                    "probabilities": {projector: [probability(state, projector), probability(changed, projector)]
                                      for projector, _, _ in rule["projectors"]}}
    return None
