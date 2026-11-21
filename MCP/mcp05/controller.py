"""Small CPU SetTRM policy. Checkpoints contain tensor weights and public metadata."""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn

from MCP.mcp02.agent import ITEM_TYPES
from MCP.mcp02.features import OFFSET, STRUCT_DIM, VOCAB_BUCKETS, encode_pair, encode_text, utility_needs
from MCP.mcp02.models import Block, Router
from .context import candidate_features

ACTIONS = ("expand", "read_symbol", "reuse_cached", "inspect_edit", "run_verifier", "submit_to_solver")
WIDTH = 32
VERSION = 1


class SetTRM(nn.Module):
    """Set attention over the slate and two recurrent states, with shared transitions."""

    def __init__(self, mode: str, depth: int):
        super().__init__()
        if mode not in {"shared", "dual", "recurrent", "shuffled"} or not 1 <= depth <= 8:
            raise ValueError("invalid controller configuration")
        self.mode, self.depth = mode, depth
        self.tokens = nn.Embedding(OFFSET + VOCAB_BUCKETS, WIDTH, padding_idx=0)
        self.action = nn.Embedding(len(ACTIONS), WIDTH)
        self.numeric = nn.Linear(10, WIDTH)
        self.refine = Block(WIDTH, 4, ffn_mult=2)
        self.update = nn.GRUCell(WIDTH, WIDTH)
        self.utility = nn.Linear(WIDTH, 2 if mode == "dual" else 1)
        self._embeddings = lru_cache(maxsize=2048)(self.embed)

    def embed(self, text: str) -> torch.Tensor:
        ids = encode_text(text, 48)
        return self.tokens(torch.tensor(ids or [0])).mean(dim=0)

    def cache_size(self) -> int:
        return self._embeddings.cache_info().currsize

    def step(self, observation: dict, latent=None):
        context = observation["context"]
        slate = observation["slate"]
        embed = self.embed if self.training else self._embeddings
        task = embed(context["task"] + " " + context.get("subgoal", "")
                     + " " + context.get("verifier_status", ""))
        embedded, features, types = [], [], []
        for item in slate:
            text, numeric = candidate_features(context["task"], item)
            embedded.append(embed(text))
            features.append(numeric + [
                float(item.get("seen", False)),
                min(context["calls_left"] / 24, 1),
                min(context["tokens_left"] / 2048, 1),
                float(context["dirty"]),
                float(context["failed"]),
                float(context["expanded"]),
            ])
            types.append(ACTIONS.index(item["action"]))
        kinds = torch.tensor(types)
        x = torch.stack(embedded) + self.action(kinds) + self.numeric(torch.tensor(features)) + task
        nodes = torch.cat((task[None], x), dim=0)[None]
        mask = torch.ones((1, nodes.shape[1]), dtype=torch.bool)
        z = torch.zeros_like(task) if latent is None else latent
        for _ in range(self.depth):
            conditioned = nodes + z[None, None, :]
            refined = self.refine(conditioned, mask)[0]
            z = self.update(refined[0], z)
        scores = self.utility(refined[1:]).squeeze(-1)
        if self.mode == "dual":
            scores = scores.gather(1, (kinds == ACTIONS.index("read_symbol")).long()[:, None])[:, 0]
        return scores, z


class RU02(nn.Module):
    """The original MCP-02 looped encoder and RU02 heads on MCP-05 action slates."""

    def __init__(self, mode: str, depth: int):
        super().__init__()
        self.mode = mode
        self.router = Router({"arch": "looped", "d": 64, "heads": 4, "passes": depth, "encoder": "match",
                              **({"single_utility": True} if mode == "shared"
                                 else {"utility_heads": "expand_read"})})
        self._pairs = lru_cache(maxsize=2048)(encode_pair)

    def cache_size(self) -> int:
        return self._pairs.cache_info().currsize

    def step(self, observation: dict, _latent=None):
        context = observation["context"]
        needs = utility_needs({"remaining_calls": context["calls_left"],
                               "needs": [context["task"], context["subgoal"], context["verifier_status"]]})
        need = " ; ".join(needs)
        ids, segments, structs = [], [], []
        for item in observation["slate"]:
            text, numeric = candidate_features(context["task"], item)
            state = ("seen" if item.get("seen") else "unseen", "dirty" if context["dirty"] else "clean",
                     "failed-verifier" if context["failed"] else "no-verifier-failure",
                     f"tokens-{context['tokens_left'] // 256}")
            pair = self._pairs(need, " ".join((item["action"], *state, text)), "match")
            ids.append(pair[0])
            segments.append(pair[1])
            struct = [0.] * STRUCT_DIM
            kind = "doc" if item["action"] in {"read_symbol", "reuse_cached"} else "branch"
            struct[ITEM_TYPES.index(kind)] = 1.
            struct[len(ITEM_TYPES):] = numeric[:STRUCT_DIM - len(ITEM_TYPES)]
            structs.append(struct)
        scores, _ = self.router(torch.tensor(ids), torch.tensor(segments), torch.tensor(structs))
        return scores, scores.new_zeros(0)


def observation(working, *, task: str, subgoal: str, calls_left: int, tokens_left: int,
                dirty: bool, failed: bool, expanded: bool, verifier_status: str = "",
                limit: int = 12) -> dict:
    # The same slate builder runs in collection and in GDK inference.
    query = working.query(task + " " + subgoal, limit=limit)
    read_set = set(working.reads)
    items = [{"action": "read_symbol", **row, "seen": row["uri"] in read_set}
             for row in query["candidates"] if row.get("uri") and row["uri"] not in read_set
             and working.env.symbols[row["uri"]].kind != "module"]
    if not items:
        items = [{"action": "read_symbol", **row, "seen": False}
                 for row in query["candidates"] if row.get("uri") and row["uri"] not in read_set]
    if not expanded:
        items.append({"action": "expand", "node_id": "d:", "uri": "d:"})
    for body in list(working.bodies.values())[-1:] if failed else []:
        items.append({"action": "reuse_cached", "seen": True, **{
            key: body[key] for key in ("uri", "path", "handle")}})
    if dirty:
        items.extend({"action": action} for action in ("inspect_edit", "run_verifier"))
    items.append({"action": "submit_to_solver"})
    return {"context": {"task": task, "subgoal": subgoal, "calls_left": calls_left,
                        "tokens_left": tokens_left, "dirty": dirty, "failed": failed,
                        "expanded": expanded, "verifier_status": verifier_status},
            "revision": query["revision"], "slate": items}


class Controller:
    def __init__(self, checkpoint, depth: int | None, reset: bool, threads: int):
        torch.set_num_threads(threads)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["version"] != VERSION:
            raise ValueError("unsupported MCP-05 controller checkpoint")
        model = RU02 if data.get("architecture") == "ru02" else SetTRM
        self.model = model(data["mode"], depth or data["depth"])
        self.model.load_state_dict(data["state"])
        self.model.eval()
        self.reset = reset or data["mode"] in {"shared", "dual"}
        self.latent = None

    @torch.inference_mode()
    def propose(self, obs: dict) -> dict:
        scores, latent = self.model.step(obs, self.latent)
        if not self.reset:
            self.latent = latent
        return dict(obs["slate"][int(scores.argmax())], utility=float(scores.max()),
                    candidates=len(scores), latent_bytes=latent.numel() * latent.element_size(),
                    weights_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters()),
                    cached_embeddings=self.model.cache_size())
