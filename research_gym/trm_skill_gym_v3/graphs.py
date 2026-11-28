"""Extended graph strata for the retrieval and deduction skills (GYM-02 full campaign).

Every instance uses the GYM-01 tensor contract (``tasks.N`` slots) with a random
slot permutation.  Topology families:

* ``long_chain_<d>``       a single dependency chain of depth d plus a distractor chain
* ``branching``            branching DAG (fan-out 2-3 per level, depth 4-6)
* ``shared_subdeps``       diamonds: several parents share sub-dependencies
* ``distractors``          the query component plus 2-3 disjoint look-alike components
* ``misleading``           distractor nodes carry the degree statistics of relevant nodes
* ``ladder`` (withheld)    two interleaved chains with cross rungs, never trained or calibrated on

``hintfree`` variants of the original GYM-01 families neutralise the
label-correlated synthetic hint columns (retrieval col 1 lexical overlap,
deduction col 0 prior score).  Structural features computed from the public
graph (degrees, undominated flag) are kept: they are not label hints.

Graph size is bounded by the 16 candidate slots of the GYM-01 module contract;
larger graphs would need a new contract and are not attempted here.
"""
from __future__ import annotations

from typing import List

import numpy as np

from research_gym.trm_skill_gym import tasks as T

GRAPH_SKILLS = ("retrieval", "deduction")
TRAIN_TOPOLOGIES = ("long_chain_6", "long_chain_10", "branching", "shared_subdeps", "distractors", "misleading")
WITHHELD_TOPOLOGIES = ("ladder", "long_chain_14")
HINT_COLS = {"retrieval": (1,), "deduction": (0,)}


def _key(skill: str, family: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(f"v3/{skill}/{family}"))


def _edges(family: str, rng: np.random.Generator):
    """Return (n_nodes, edge list (child, parent), root/query node, in-query-component mask).

    Nodes are numbered so that parents have smaller indices than children (a DAG)."""
    edges: List[tuple] = []
    if family.startswith("long_chain_"):
        d = int(family.split("_")[-1])
        n = min(T.N, d + 1 + max(2, T.N - d - 1))
        for i in range(1, d + 1):
            edges.append((i, i - 1))
        for i in range(d + 2, n):  # distractor chain
            edges.append((i, i - 1))
        return n, edges, d
    if family == "branching":
        depth = int(rng.integers(4, 7))
        levels = [[0]]
        nxt = 1
        for _ in range(depth - 1):
            lvl = []
            for p in levels[-1]:
                for _ in range(int(rng.integers(1, 3))):
                    if nxt < T.N - 2:
                        lvl.append(nxt)
                        edges.append((nxt, p))
                        nxt += 1
            levels.append(lvl or [levels[-1][0]])
        # query sits above the deepest level node set: add a node depending on the leaves
        q = nxt
        for leaf in levels[-1][:3]:
            edges.append((q, leaf))
        n = q + 1
        for i in range(n, T.N):  # distractors
            edges.append((i, int(rng.integers(n, i))) if i > n else (i, i))
        edges = [(c, p) for c, p in edges if c != p]
        return T.N, edges, q
    if family == "shared_subdeps":
        base = int(rng.integers(3, 6))
        for i in range(1, base):
            edges.append((i, i - 1))
        mids = list(range(base, base + int(rng.integers(3, 5))))
        for m in mids:
            for p in rng.choice(base, size=min(base, 2), replace=False):
                edges.append((m, int(p)))
        q = mids[-1] + 1
        for m in mids:
            edges.append((q, m))
        n = q + 1
        for i in range(n + 1, T.N):
            edges.append((i, int(rng.integers(n, i))))
        return T.N, edges, q
    if family == "distractors":
        size = int(rng.integers(4, 6))
        comps = []
        start = 0
        while start + size <= T.N:
            comp = list(range(start, start + size))
            for i in range(1, size):
                edges.append((comp[i], comp[int(rng.integers(0, i))]))
            comps.append(comp)
            start += size
        q = comps[0][-1]
        return T.N, edges, q
    if family == "misleading":
        d = int(rng.integers(4, 7))
        for i in range(1, d + 1):
            edges.append((i, i - 1))
        q = d
        # distractor hub: many nodes depend on a hub node outside the query component
        hub = d + 1
        for i in range(d + 2, T.N):
            edges.append((i, hub))
            if rng.random() < 0.5 and i > d + 2:
                edges.append((i, i - 1))
        return T.N, edges, q
    if family == "ladder":
        rungs = int(rng.integers(5, 8))
        a = list(range(0, 2 * rungs, 2))
        b = list(range(1, 2 * rungs, 2))
        for i in range(1, rungs):
            edges.append((a[i], a[i - 1]))
            edges.append((b[i], b[i - 1]))
            if rng.random() < 0.6:
                edges.append((a[i], b[i - 1]))
        q = a[-1]
        n = 2 * rungs
        for i in range(n + 1, T.N):
            edges.append((i, i - 1))
        return T.N, edges, q
    raise KeyError(family)


def make_instance(skill: str, family: str, root: int) -> T.Instance:
    if skill not in GRAPH_SKILLS:
        raise ValueError("graph strata cover retrieval and deduction only")
    if family.endswith("_hintfree"):
        inst = T.make_instance(skill, family[: -len("_hintfree")], root)
        for c in HINT_COLS[skill]:
            inst.cand[:, c] = 0.5 * inst.mask
        return inst
    rng = np.random.default_rng([int(root), _key(skill, family)])
    n, edges, q = _edges(family, rng)
    perm = rng.permutation(T.N)
    rel = np.zeros((T.N, T.N), np.float32)
    for c, p in edges:
        rel[perm[c], perm[p]] = 1
    inst = T._empty()
    inst.mask[:] = 1
    inst.rel[:] = rel
    inst.glob[0] = 1.0
    if skill == "retrieval":
        qq = int(perm[q])
        reach = np.zeros(T.N, bool)
        stack = [qq]
        while stack:
            u = stack.pop()
            for v in np.nonzero(rel[u])[0]:
                if not reach[v]:
                    reach[v] = True
                    stack.append(int(v))
        inst.label[:] = reach
        inst.cand[qq, 0] = inst.aux[qq, 0] = 1
        inst.cand[:, 1] = 0.5
        inst.cand[:, 10] = rel.sum(1) / 4
        inst.cand[:, 11] = rel.sum(0) / 4
    else:
        # deduction: rel[i, j] = 1 means j dominates i; kernel by topological order of original numbering
        alive = np.zeros(T.N, bool)
        for orig in range(T.N):  # parents have smaller original indices, so this is a topological order
            slot = int(perm[orig])
            alive[slot] = not any(alive[j] for j in np.nonzero(rel[slot])[0])
        inst.label[:] = alive
        inst.cand[:, 0] = 0.5
        inst.cand[:, 1] = rel.sum(1) / 3
        inst.cand[:, 2] = rel.sum(0) / 3
        inst.cand[:, 3] = rel.sum(1) == 0
        inst.cand[:, 4:12] = rng.normal(0, 1, (T.N, 8))
    if skill == "retrieval":
        inst.cand[:, 2:10] = rng.normal(0, 1, (T.N, 8))  # uninformative type embedding, as in GYM-01
    return inst


def stack(skill: str, family: str, base: int, count: int):
    return T.stack([make_instance(skill, family, base + i) for i in range(count)])


def strata(skill: str, stage: str) -> List[str]:
    """Families per stage.  train/cal: dev families, their hint-free variants and the
    training topologies.  test adds the original withheld family and withheld topologies."""
    fams = list(T.FAMILIES[skill]["dev"])
    if skill in GRAPH_SKILLS:
        fams += [f + "_hintfree" for f in T.FAMILIES[skill]["dev"]] + list(TRAIN_TOPOLOGIES)
    if stage == "test":
        fams += list(T.FAMILIES[skill]["withheld"])
        if skill in GRAPH_SKILLS:
            fams += [f + "_hintfree" for f in T.FAMILIES[skill]["withheld"]] + list(WITHHELD_TOPOLOGIES)
    return fams


def instance(skill: str, family: str, root: int) -> T.Instance:
    if family in TRAIN_TOPOLOGIES or family in WITHHELD_TOPOLOGIES or family.endswith("_hintfree"):
        return make_instance(skill, family, root)
    return T.make_instance(skill, family, root)
