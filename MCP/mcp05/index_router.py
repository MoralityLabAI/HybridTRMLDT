"""Fit and evaluate a small recursive set router on disjoint source paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from contextlib import closing
from pathlib import Path, PurePosixPath

import torch

from .controller import VERSION, Controller
from .federation import Federation, readonly, terms
from .train import RAM_LIMIT, check_ram, fit


def observation(query: str, candidates: list[dict]) -> dict:
    return {"context": {"task": query, "calls_left": 8, "tokens_left": 1024,
                        "dirty": False, "failed": False, "expanded": False},
            "slate": [dict(candidate, action="read_symbol") for candidate in candidates]}


class IndexRouter:
    def __init__(self, checkpoint, threads=2):
        self.controller = Controller(checkpoint, None, False, threads)

    @torch.inference_mode()
    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        scores, self.controller.latent = self.controller.model.step(
            observation(query, candidates), self.controller.latent,
        )
        ordering = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), index))
        return [candidates[index] for index in ordering]


def source_rows(federation: Federation, seed: int):
    chooser = random.Random(seed)
    for name in sorted(federation.repos):
        with closing(readonly(federation.root / f"{name}.sqlite")) as shard:
            paths = [row[0] for row in shard.execute("SELECT path FROM files ORDER BY path")]
        chooser.shuffle(paths)
        seen_names = set()
        picked = 0
        for source_path in paths:
            path = PurePosixPath(source_path)
            words = terms(path.stem)
            if len(words) < 2 or path.stem.casefold() in seen_names:
                continue
            seen_names.add(path.stem.casefold())
            queries = ("find " + " ".join(words), "find " + path.parent.name + " " + " ".join(words))
            steps = []
            for query in queries:
                candidates, _ = federation.shortlist(query, terms(query))
                if len(candidates) < 2 or name not in {row["id"] for row in candidates}:
                    break
                steps.append({"observation": observation(query, candidates),
                              "utility": [float(row["id"] == name) for row in candidates]})
            if len(steps) == len(queries):
                yield picked < 18, {"target": name, "source_path": source_path, "steps": steps}
                picked += 1
            if picked == 24:
                break
        if picked != 24:
            raise ValueError(f"insufficient eligible routing examples in {name}")


def grade(episodes, router=None, *, reset=False) -> dict:
    top1 = top2 = queries = 0
    elapsed = 0.
    for episode in episodes:
        if router:
            router.controller.latent = None
        for step in episode["steps"]:
            if router and reset:
                router.controller.latent = None
            obs = step["observation"]
            slate = obs["slate"]
            if router:
                start = time.monotonic()
                slate = router.rank(obs["context"]["task"], slate)
                elapsed += time.monotonic() - start
            top1 += slate[0]["id"] == episode["target"]
            top2 += episode["target"] in {row["id"] for row in slate[:2]}
            queries += 1
    return {"queries": queries, "index_recall_at_1": top1 / queries,
            "index_recall_at_2": top2 / queries, "rank_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=523)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    federation = Federation(args.indexes)
    train, test = [], []
    for training, episode in source_rows(federation, args.seed):
        (train if training else test).append(episode)
    model, losses = fit(train, "recurrent", depth=3, epochs=4, seed=args.seed)
    checkpoint = args.out / "router.pt"
    torch.save({"version": VERSION, "mode": "recurrent", "depth": 3,
                "state": model.state_dict()}, checkpoint)
    router = IndexRouter(checkpoint)
    receipt = {"index_revision": federation.revision, "seed": args.seed, "cpu_threads": 2,
               "train_paths": len(train), "held_out_paths": len(test),
               "lexical": grade(test), "recurrent": grade(test, router),
               "reset": grade(test, router, reset=True),
               "first_loss": losses[0], "last_loss": losses[-1],
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "peak_rss_bytes": check_ram(), "ram_cap_bytes": RAM_LIMIT}
    (args.out / "training.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
