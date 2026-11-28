"""Shared, auditable controls for SKILL-02. No API or torch dependency."""
import hashlib
import json
import math


def paired_rank(rows, expected):
    """Rank only complete, identically stratified measurements; worst family first."""
    expected = set(expected)
    if set(rows) != expected or not expected:
        raise ValueError("incomplete or mismatched strata")
    vals = [rows[k]["exact"] for k in sorted(expected)]
    costs = [rows[k]["flops"] for k in sorted(expected)]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in vals):
        raise ValueError("invalid accuracy")
    if any(not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError("invalid work")
    return (min(vals), sum(vals) / len(vals), -sum(costs) / len(costs))


def strict_selected(text, valid, single=False):
    """Exactly one JSON object, no extra keys, booleans, duplicate slots or prose."""
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("duplicate JSON key")
            obj[k] = v
        return obj
    obj = json.loads(text, object_pairs_hook=unique)
    if not isinstance(obj, dict) or set(obj) != {"selected"}:
        raise ValueError("expected selected only")
    v = obj["selected"]
    if not isinstance(v, list) or any(type(i) is not int for i in v):
        raise ValueError("expected integer list")
    if len(set(v)) != len(v) or any(i not in valid for i in v) or (single and len(v) != 1):
        raise ValueError("invalid selection")
    return v


def graph_certificate(skill, parents, query=None, rounds=1):
    """Public graph only. +1 in, -1 out, 0 unknown. One synchronous round at a time.

    Retrieval uses directed reachability from query; unreachable nodes are certified
    only after the reached set stabilizes. Deduction assumes an acyclic graph.
    No labels, aux tensors, or verifier outcomes are accepted.
    Returns the certificate and a count of edge examinations (host work).
    """
    n = len(parents)
    if rounds < 0 or any(j < 0 or j >= n for ps in parents for j in ps):
        raise ValueError("bad graph/rounds")
    work = 0
    if skill == "retrieval":
        if query is None or not 0 <= query < n:
            raise ValueError("invalid query")
        reached = {query}
        stable = False
        for _ in range(rounds):
            nxt = reached | {j for i in reached for j in parents[i]}
            work += sum(len(parents[i]) for i in reached)
            if nxt == reached:
                stable = True
                break
            reached = nxt
        return [(-1 if i == query else 1 if i in reached else -1 if stable else 0)
                for i in range(n)], work
    if skill != "deduction":
        raise ValueError("unsupported graph skill")
    # Local kernel implications are sound on a DAG; unresolved cycles remain unknown.
    st = [0] * n
    for _ in range(rounds):
        nxt = st.copy()
        for i, ps in enumerate(parents):
            if st[i] != 0:
                continue
            work += len(ps)
            if any(st[j] == 1 for j in ps):
                nxt[i] = -1
            elif all(st[j] == -1 for j in ps):
                nxt[i] = 1
        if nxt == st:
            break
        st = nxt
    return st, work


def entropy(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def branches(p, likelihood):
    for outcome in (0, 1):
        weights = [a * (b if outcome else 1 - b) for a, b in zip(p, likelihood)]
        prob = sum(weights)
        if prob > 0:
            yield prob, [x / prob for x in weights]


def eig(p, likelihood):
    return entropy(p) - sum(prob * entropy(post) for prob, post in branches(p, likelihood))


def research_values(p, likelihoods, costs, horizon=2):
    """Expected entropy reduction minus accumulated costs, optional stopping.
    Binary outcomes conditionally independent given hypothesis. No action repeats."""
    if len(likelihoods) != len(costs) or horizon < 1:
        raise ValueError("bad research world")
    def qvals(belief, remaining, h):
        vals = {}
        for i in remaining:
            val = eig(belief, likelihoods[i]) - costs[i]
            if h > 1:
                rest = tuple(j for j in remaining if j != i)
                for prob, post in branches(belief, likelihoods[i]):
                    val += prob * max([0.0, *qvals(post, rest, h - 1).values()])
            vals[i] = val
        return vals
    return qvals(p, tuple(range(len(costs))), horizon)


def distill_rows(archive):
    """Keep all proposals; separate input/target and hold out whole skill families.
    Same skill/config/round can never cross the skill partition.
    """
    seen = set()
    for row in archive:
        key = (row["arm"], row["round"], row["proposal_index"])
        if key in seen:
            raise ValueError("duplicate proposal identity")
        seen.add(key)
        skill = row["skill"]
        prompt = {k: row.get(k) for k in ("skill", "family", "arm", "round", "incumbent",
                  "raw_config", "config", "incumbent_dev_exact", "edit_budget")}
        target = {k: row.get(k) for k in ("status", "decision", "rung0", "rung1",
                  "measured_dev_exact")}
        # No current-result diagnosis, retrospective rationale, or frozen outcome in prompt.
        yield {"id": "/".join(map(str, key)), "split": "test" if skill in
               {"research", "stopping"} else "train", "input": prompt, "target": target,
               "source_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}
