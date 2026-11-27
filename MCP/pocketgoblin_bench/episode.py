"""One PocketGoblin Bench episode: an MCP-02 world family, a compute budget and the trainer tools.

The model under test calls ``Episode.call(tool, arguments)`` (directly or through ``server.py``).
It sees the MCP catalog, train queries with their evidence, the mesh search space, and dev scores of
meshes it fitted. Test worlds are generated inside ``submit`` and never leave this module.

Budget unit: ``units`` approximate 25 ms of single-process CPU. ``fit`` charges training FLOPs (in
GFLOPs, counted by MCP-02's FlopCounterMode estimate) plus ``STEP_UNITS`` per optimizer step; ``evaluate``
charges inference GFLOPs plus ``EVAL_EPISODE_UNITS`` per dev episode. Charges depend only on the
candidate and the seeds, so a run is reproducible.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from MCP.mcp02.agent import make_control, run_episode
from MCP.mcp02.env import Env02
from MCP.mcp02.textutil import canonical_json
from MCP.mcp02.world import CONCEPTS, KIND_HOME, ROLE_WORDS, SCOPE, make_cases

from . import space

STEP_UNITS = 0.5
EVAL_EPISODE_UNITS = 2.0
BYTES_PER_TOKEN = 4.0
CONTROLS = ("lexical_bm25", "deterministic_rules", "sham_random", "oracle_upper_bound")
METRIC_KEYS = ("packet_recall", "accepted", "packet_tokens", "tool_calls")


class BenchError(Exception):
    """A tool call the episode refuses; ``code`` is stable for scripted clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate(episodes: list[dict]) -> dict:
    """Mean metrics over MCP-02 episode records. Recall counts only packets inside the handoff budget."""
    n = len(episodes)
    if n == 0:
        raise ValueError("no episodes")
    recall = [0.0 if e["handoff_budget_exceeded"] else float(e["grade"]["evidence_recall"]) for e in episodes]
    return {"episodes": n,
            "packet_recall": sum(recall) / n,
            "accepted": sum(1 for e in episodes if e["accepted"]) / n,
            "packet_tokens": sum(e["handoff_utf8_bytes"] for e in episodes) / n / BYTES_PER_TOKEN,
            "tool_calls": sum(e["tool_calls"] for e in episodes) / n,
            "router_gflops": sum(e["compute"]["flops"] for e in episodes) / n / 1e9}


def normalized(value: float, zero: float, one: float) -> float:
    """Position of ``value`` between the zero control and the oracle (unclipped)."""
    if one - zero <= 1e-12:
        return 1.0 if value >= one - 1e-12 else 0.0
    return (value - zero) / (one - zero)


class Episode:
    def __init__(self, config: dict, mcp_seed: int = 0, *, threads: int = 4):
        self.cfg = config
        self.mcp_seed = int(mcp_seed)
        self.threads = max(1, min(8, int(threads)))
        w, s = config["world"], config["seeds"]
        self.catalog_size = int(w["catalog_size"])
        self.surface = (w["arm"], w["layout"] if w["arm"].startswith("layered") else "")
        self.n = {k: int(v) for k, v in config["worlds"].items()}
        self.seed = {k: int(s[k]) + 1000 * self.mcp_seed for k in ("train", "dev", "test")}
        self.max_calls = int(config["limits"]["max_tool_calls"])
        self.handoff_budget = int(config["limits"]["handoff_budget_bytes"])
        self.prefilter_default = int(config["limits"]["prefilter_k"])
        self.total = float(config["budget"]["units"])
        self.max_fits = int(config["budget"]["max_fits"])
        self.max_evals = int(config["budget"]["max_evals"])
        self.spent = 0.0
        self.cpu_seconds = 0.0
        self.fits = self.evals = 0
        self.models: dict[str, dict] = {}
        self.dev_results: dict[str, dict] = {}
        self.submitted: dict | None = None
        self.log: list[dict] = []
        self._train_cases: list[dict] | None = None
        self._datasets: dict[int, dict] = {}
        self._flops: dict[str, float] = {}
        self._controls: dict[str, dict] = {}

    # -- public surface -------------------------------------------------------------------
    def public_task(self) -> dict:
        return {"service": "pocketgoblin-bench",
                "purpose": "train the best router mesh you can for one MCP within a compute budget",
                "mcp_seed": self.mcp_seed, "budget_units": self.total,
                "flow": ["describe_mcp", "sample_train_queries", "list_mesh_families", "estimate_cost",
                         "fit", "evaluate", "submit"]}

    def tools(self) -> list[dict]:
        def tool(name, desc, props=None, req=()):
            return {"name": name, "description": desc,
                    "inputSchema": {"type": "object", "properties": props or {}, "required": list(req),
                                    "additionalProperties": False}}
        cand = {"type": "object", "description": "Mesh candidate; see list_mesh_families. Defaults fill missing keys."}
        return [
            tool("describe_mcp", "Describe the MCP catalog family this episode trains a router for: servers by "
                 "role, record kinds, concepts, the tool surface the router routes over, split sizes, budget."),
            tool("sample_train_queries", "Return train queries with their gold evidence (the records a correct "
                 "answer must cite) and answer. Train split only.",
                 {"n": {"type": "integer", "minimum": 1, "maximum": 8},
                  "offset": {"type": "integer", "minimum": 0}}),
            tool("list_mesh_families", "List router families, their hyperparameter ranges and defaults, the "
                 "budget unit and how each tool is charged."),
            tool("estimate_cost", "Upper-bound the budget a fit of this candidate would charge. Free.",
                 {"candidate": cand}, ["candidate"]),
            tool("fit", "Train a mesh on the train worlds. Charges the budget; refused when the upper bound "
                 "exceeds what is left. Returns a candidate_id and training receipt.",
                 {"candidate": cand}, ["candidate"]),
            tool("evaluate", "Score a fitted mesh on the dev worlds against the BM25 control and the oracle. "
                 "Charges inference compute; repeat calls for one candidate_id are free.",
                 {"candidate_id": {"type": "string"}}, ["candidate_id"]),
            tool("submit", "Submit one fitted mesh. It is scored once on the hidden test worlds and the "
                 "episode ends.", {"candidate_id": {"type": "string"}}, ["candidate_id"]),
            tool("status", "Budget spent and remaining, fits and evaluations used, fitted candidate ids."),
        ]

    def call(self, name: str, arguments: dict | None = None) -> dict:
        arguments = arguments or {}
        entry = {"tool": name, "arguments": arguments, "t": time.time()}
        started = time.perf_counter()
        try:
            if self.submitted is not None and name != "status":
                raise BenchError("episode_over", "the episode ended at submit")
            handler = getattr(self, "tool_" + name, None)
            if handler is None:
                raise BenchError("unknown_tool", f"unknown tool {name}")
            result = handler(**arguments)
            entry["error"] = None
        except BenchError as error:
            result = {"error": error.message, "code": error.code}
            entry["error"] = error.code
        except ValueError as error:
            result = {"error": str(error), "code": "invalid_candidate"}
            entry["error"] = "invalid_candidate"
        entry["seconds"] = round(time.perf_counter() - started, 3)
        entry["spent_after"] = round(self.spent, 3)
        self.log.append(entry)
        return result

    # -- data -----------------------------------------------------------------------------
    @property
    def train_cases(self) -> list[dict]:
        if self._train_cases is None:
            self._train_cases = make_cases("train", self.seed["train"], self.n["train"], self.catalog_size)
        return self._train_cases

    def _dataset(self, worlds: int) -> dict:
        if worlds not in self._datasets:
            from MCP.mcp02.train import build_dataset
            arm, layout = self.surface
            self._datasets[worlds] = build_dataset(
                "train", worlds, self.seed["train"], self.catalog_size, [[arm, layout]],
                self.cfg["dataset"]["epsilons"], self.cfg["dataset"]["max_candidates"], self.max_calls)
        return self._datasets[worlds]

    def _cases(self, split: str) -> list[dict]:
        return make_cases(split, self.seed[split], self.n[split], self.catalog_size)

    # -- tools ----------------------------------------------------------------------------
    def tool_describe_mcp(self) -> dict:
        arm, layout = self.surface
        case = self.train_cases[0]
        env = Env02(case, arm, layout)
        kinds: dict[str, int] = {}
        for doc in case["docs"]:
            kinds[doc["kind"]] = kinds.get(doc["kind"], 0) + 1
        return {
            "catalog": {"documents_per_world": self.catalog_size, "servers": len(case["servers"]),
                        "server_roles": {role: ROLE_WORDS[role] for role in case["servers"]},
                        "record_kinds_in_train_world_0": kinds,
                        "home_scope_by_kind": {k: SCOPE[v] for k, v in KIND_HOME.items()},
                        "concepts": {k: list(v) for k, v in CONCEPTS.items()},
                        "world_features": ["ambiguous names", "replica and archive copies", "moved documents",
                                          "unit variants", "paginated collections", "mid-episode epoch change"],
                        "task_families": ["alias_lookup", "join_chain", "join_arith", "stale_moved",
                                          "epoch_roll", "conflict_authority", "missing", "paginated"]},
            "surface": {"arm": arm, "layout": layout, "tools": [{"name": t["name"], "description": t["description"]}
                                                                for t in env.tools()]},
            "splits": {"train_worlds": self.n["train"], "dev_worlds": self.n["dev"], "test_worlds": self.n["test"],
                       "note": "worlds differ per split in entity names, server names and request templates; "
                               "test worlds are never visible"},
            "task_example": self.train_cases[0]["task"]["request"],
            "scoring": {"primary": "packet_recall: evidence recall of the final packet, 0 when the packet "
                                   "exceeds the handoff budget",
                        "handoff_budget_bytes": self.handoff_budget, "max_tool_calls": self.max_calls,
                        "normalization": "(mesh - lexical_bm25) / (oracle - lexical_bm25)"},
            "budget": self._budget()}

    def tool_sample_train_queries(self, n: int = 4, offset: int = 0) -> dict:
        rows = []
        for index, case in enumerate(self.train_cases[:offset + n]):
            if index < offset:
                continue
            docs = {d["uri"]: d for d in case["docs"]}
            chains = [[docs[u] for u in chain] for chain in case["oracle"]["required_uris"]]
            rows.append({"index": index, "family": case["family"],
                         "request": case["task"]["request"], "answer": case["oracle"]["answer"],
                         "abstain": case["oracle"]["abstain"],
                         "evidence": [[{"uri": d["uri"], "server_role": d["role"], "kind": d["kind"],
                                        "name": d["name"], "region": d["region"], "scope": d["scope"],
                                        "status": d["status"], "fields": d["fields"],
                                        "links": d["links"]}
                                       for d in chain] for chain in chains]})
        return {"queries": rows, "offset": offset, "total_train_worlds": self.n["train"]}

    def tool_list_mesh_families(self) -> dict:
        return {"ranges": space.ranges(self.n["train"], self.max_calls),
                "families": {"dense": "dual encoder, no attention", "transformer": "stacked cross-encoder blocks",
                             "looped": "one shared block applied `passes` times",
                             "trm": "tiny recursive model: latent refinement over cycles",
                             "trm_ldt": "trm plus a learned exclusion head for the layered_ldt surface",
                             "hrm": "two-timescale recurrence"},
                "encoders": {"tokens": "hashed need and candidate tokens",
                             "match": "tokens plus an exact-match channel between need and candidate"},
                "candidate_example": space.default_candidate(self.n["train"], self.max_calls),
                "budget": self._budget(),
                "charges": {"fit": f"training GFLOPs + {STEP_UNITS} units per optimizer step",
                            "evaluate": f"inference GFLOPs + {EVAL_EPISODE_UNITS} units per dev episode",
                            "estimate_cost": "free", "submit": "free"}}

    def _per_candidate_gflops(self, router_cfg: dict) -> float:
        key = json.dumps(router_cfg, sort_keys=True)
        if key not in self._flops:
            import torch
            from MCP.mcp02.models import Router, flops_per_candidate
            torch.set_num_threads(self.threads)
            self._flops[key] = flops_per_candidate(Router(router_cfg))["flops"] / 1e9
        return self._flops[key]

    def _estimate(self, candidate: dict) -> dict:
        data = self._dataset(candidate["train"]["worlds"])
        t = candidate["train"]
        router_cfg = space.router_config(candidate)
        per = self._per_candidate_gflops(router_cfg)
        decide = sum(len(s["cands"]) for s in data["decide"])
        rows = sum(len(s["cands"]) for s in data["rows"]) if candidate["family"] == "trm_ldt" else 0
        steps = t["epochs"] * -(-len(data["decide"]) // t["batch_states"])
        gflops = 3 * per * (decide + rows) * t["epochs"]
        return {"upper_bound_units": round(gflops + STEP_UNITS * steps, 3), "gflops": round(gflops, 3),
                "steps": steps, "train_states": len(data["decide"]), "per_candidate_gflops": round(per, 6)}

    def _prepare(self, raw: Any) -> dict:
        return space.normalize(raw, self.n["train"], self.max_calls)

    def tool_estimate_cost(self, candidate: Any) -> dict:
        cand = self._prepare(candidate)
        return {"candidate": cand, **self._estimate(cand), "remaining_units": round(self.total - self.spent, 3)}

    def tool_fit(self, candidate: Any) -> dict:
        cand = self._prepare(candidate)
        if self.fits >= self.max_fits:
            raise BenchError("fit_limit", f"at most {self.max_fits} fits per episode")
        est = self._estimate(cand)
        remaining = self.total - self.spent
        if est["upper_bound_units"] > remaining + 1e-9:
            raise BenchError("budget_exceeded", f"fit needs up to {est['upper_bound_units']} units, "
                             f"{round(remaining, 3)} left; nothing was trained or charged")
        from MCP.mcp02.train import fit
        t = cand["train"]
        started = time.perf_counter()
        model, receipt = fit(space.router_config(cand), self._dataset(t["worlds"]), epochs=t["epochs"],
                             lr=t["lr"], batch_states=t["batch_states"], device="cpu", seed=t["seed"],
                             threads=self.threads, log=lambda *_: None)
        seconds = time.perf_counter() - started
        charge = receipt["train_flops_estimate"] / 1e9 + STEP_UNITS * receipt["steps"]
        self.spent += charge
        self.cpu_seconds += seconds
        self.fits += 1
        cid = f"c{self.fits}"
        self.models[cid] = {"candidate": cand, "model": model, "charge": charge, "steps": receipt["steps"],
                            "final_loss": receipt["loss_curve"][-1]["mean_loss"]}
        return {"candidate_id": cid, "candidate": cand, "charged_units": round(charge, 3),
                "steps": receipt["steps"], "final_train_loss": round(receipt["loss_curve"][-1]["mean_loss"], 5),
                "parameters": receipt["parameters"]["total"], "train_seconds": round(seconds, 2),
                "budget": self._budget()}

    def _controls_for(self, split: str, filters: dict | None = None) -> dict:
        filters = filters or {"prefilter_k": self.prefilter_default, "max_tool_calls": self.max_calls}
        key = f"{split}/{filters['prefilter_k']}/{filters['max_tool_calls']}"
        if key not in self._controls:
            cases = self._cases(split)
            self._controls[key] = {name: aggregate(self._run(make_control(name), cases, filters)) for name in CONTROLS}
        return self._controls[key]

    def _run(self, policy: Any, cases: list[dict], filters: dict) -> list[dict]:
        arm, layout = self.surface
        return [run_episode(case, arm=arm, layout=layout, policy=policy, max_tool_calls=filters["max_tool_calls"],
                            prefilter_k=filters["prefilter_k"], handoff_budget_bytes=self.handoff_budget)
                for case in cases]

    def _score(self, entry: dict, split: str) -> dict:
        from MCP.mcp02.models import RouterPolicy
        cand = entry["candidate"]
        controls = self._controls_for(split)
        policy = RouterPolicy("pgb", entry["model"], device="cpu")
        import torch
        torch.set_num_threads(self.threads)
        episodes = self._run(policy, self._cases(split), cand["filters"])
        metrics = aggregate(episodes)
        zero, one = controls["lexical_bm25"], controls["oracle_upper_bound"]
        norm = normalized(metrics["packet_recall"], zero["packet_recall"], one["packet_recall"])
        return {"metrics": metrics, "controls": controls,
                "normalized_packet_recall": round(norm, 4), "score": round(min(1.0, max(0.0, norm)), 4),
                "normalized_accepted": round(normalized(metrics["accepted"], zero["accepted"], one["accepted"]), 4),
                "episodes": episodes}

    def _entry(self, candidate_id: Any) -> dict:
        entry = self.models.get(candidate_id)
        if entry is None:
            raise BenchError("unknown_candidate", f"unknown candidate_id {candidate_id}; fitted: {sorted(self.models)}")
        return entry

    def _eval_bound(self, entry: dict) -> float:
        """Upper bound on one dev evaluation: every decision scores a full prefilter slate."""
        from MCP.mcp02.models import RouterPolicy  # noqa: F401  (torch import is required for scoring)
        f = entry["candidate"]["filters"]
        per = self._per_candidate_gflops(space.router_config(entry["candidate"]))
        return self.n["dev"] * (EVAL_EPISODE_UNITS + 1.5 * per * f["prefilter_k"] * f["max_tool_calls"])

    def tool_evaluate(self, candidate_id: str) -> dict:
        entry = self._entry(candidate_id)
        if candidate_id in self.dev_results:
            return dict(self.dev_results[candidate_id], cached=True)
        if self.evals >= self.max_evals:
            raise BenchError("eval_limit", f"at most {self.max_evals} dev evaluations per episode")
        bound = self._eval_bound(entry)
        if bound > self.total - self.spent + 1e-9:
            raise BenchError("budget_exceeded", f"dev evaluation needs up to {round(bound, 3)} units, "
                             f"{round(self.total - self.spent, 3)} left; nothing was run or charged")
        started = time.perf_counter()
        scored = self._score(entry, "dev")
        episodes = scored.pop("episodes")
        charge = sum(e["compute"]["flops"] for e in episodes) / 1e9 + EVAL_EPISODE_UNITS * len(episodes)
        self.spent += charge
        self.cpu_seconds += time.perf_counter() - started
        self.evals += 1
        result = {"candidate_id": candidate_id, "split": "dev", "charged_units": round(charge, 3), **scored,
                  "budget": self._budget(), "cached": False}
        self.dev_results[candidate_id] = result
        return result

    def tool_submit(self, candidate_id: str) -> dict:
        entry = self._entry(candidate_id)
        scored = self._score(entry, "test")
        scored.pop("episodes")
        self.submitted = {"candidate_id": candidate_id, "candidate": entry["candidate"], **scored,
                          "budget": self._budget()}
        return {"submitted": candidate_id, "split": "test", **{k: v for k, v in self.submitted.items()
                                                               if k not in ("candidate_id", "candidate")}}

    def tool_status(self) -> dict:
        return {"budget": self._budget(), "fitted": {cid: {"family": e["candidate"]["family"],
                                                           "dev_score": (self.dev_results.get(cid) or {}).get("score")}
                                                      for cid, e in self.models.items()},
                "submitted": self.submitted is not None}

    def _budget(self) -> dict:
        return {"unit": "units", "total": self.total, "spent": round(self.spent, 3),
                "remaining": round(self.total - self.spent, 3), "fits_used": self.fits, "fits_max": self.max_fits,
                "evals_used": self.evals, "evals_max": self.max_evals}

    # -- receipt --------------------------------------------------------------------------
    def receipt(self) -> dict:
        """Episode outcome. A run that never submitted scores zero."""
        sub = self.submitted
        return {"benchmark": "pocketgoblin-bench", "config": self.cfg["name"], "mcp_seed": self.mcp_seed,
                "surface": {"arm": self.surface[0], "layout": self.surface[1]},
                "status": "submitted" if sub else "no_submission",
                "score": sub["score"] if sub else 0.0,
                "normalized_packet_recall": sub["normalized_packet_recall"] if sub else None,
                "test_metrics": sub["metrics"] if sub else None,
                "controls_test": sub["controls"] if sub else None,
                "candidate": sub["candidate"] if sub else None,
                "budget": self._budget(), "cpu_seconds": round(self.cpu_seconds, 2),
                "dev": {cid: {"score": r["score"], "packet_recall": r["metrics"]["packet_recall"]}
                        for cid, r in self.dev_results.items()},
                "tool_calls": [{k: e[k] for k in ("tool", "error", "spent_after")} | {"arguments_sha": _sha(e["arguments"])}
                               for e in self.log]}


def _sha(value: Any) -> str:
    import hashlib
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:12]
