"""Collect verified replay episodes and fit matched CPU controller ablations."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .context import WorkingSet
from .controller import VERSION, RU02, SetTRM, observation
from .gym import clone, dump, source_root, verifier_manifest
from .service import RepairWorkspace

RAM_LIMIT = 3 * 1024**3
MODES = ("shared", "dual", "recurrent", "shuffled")


def peak_rss() -> int:
    if os.name != "nt":
        import resource
        import sys
        scale = 1 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale

    import ctypes
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (field, ctypes.c_size_t) for field in ("peak", "working", "paged_peak", "paged",
                                                "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    measure = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    measure.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
    measure.restype = wintypes.BOOL
    if not measure(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak


def check_ram(limit: int = RAM_LIMIT) -> int:
    peak = peak_rss()
    if peak > limit:
        raise MemoryError(f"trainer peak process RSS {peak} exceeds limit {limit}")
    return peak


def utilities(obs, gold: set[str], pending: set[str], phase: str) -> list[float]:
    targets = []
    has_gold = any(item.get("uri") in pending for item in obs["slate"])
    for item in obs["slate"]:
        action = item["action"]
        score = -0.4
        if action == "read_symbol":
            # Marginal credit for unseen edit evidence. Test bodies carry a lower auxiliary value.
            score = 1.0 if item["uri"] in pending else 0.08 if item["path"].startswith("tests/") else 0.
        elif action == "expand":
            score = 0.38 if pending and not has_gold else 0.12
        elif action == "run_verifier":
            score = 1.0 if phase == "edited" else -0.2
        elif action == "inspect_edit":
            score = 0.48 if phase == "failed" else 0.06
        elif action == "reuse_cached":
            score = 0.78 if phase == "failed" and item["uri"] in gold else -0.12
        elif action == "submit_to_solver":
            score = 1.0 if phase in {"failed", "verified"} or not pending else 0.16
        targets.append(score)
    return targets


def collect(stage: Path, case: dict, work: Path) -> dict:
    root = work / case["instance_id"]
    clone(source_root(stage, case), root)
    working = WorkingSet(root)
    workspace = RepairWorkspace(working, verifier_manifest())
    gold = set(case["_receipt"]["gold_symbols"])
    original = {path: (root / path).read_bytes() for path in
                dict.fromkeys(uri.split("::")[0] for uri in sorted(gold))}
    task = case["problem_statement"]
    steps = []
    pending = set(gold)
    phase = "retrieve"
    expanded = False
    verification = workspace.verify()
    if verification["exit_code"] != 1:
        raise ValueError("fixture did not fail its native regression")
    recent = verification["output"].splitlines()[-3:]

    def record(action: str, detail=None):
        slate = observation(working, task=task, subgoal="Repair the reported regressions", calls_left=24-len(steps),
                            tokens_left=max(0, 1024 - sum(len(x["source"]) // 3 for x in working.bodies.values())),
                            dirty=phase in ("edited", "failed", "verified"), failed=phase == "failed",
                            expanded=expanded, verifier_status="\n".join(recent))
        steps.append({"observation": slate, "utility": utilities(slate, gold, pending, phase),
                      "action": action, "result": detail})

    record("expand")
    working.expand("d:")
    expanded = True
    for uri in sorted(gold):
        record("read_symbol", uri)
        working.read(uri=uri)
        pending.discard(uri)

    # Apply a deliberate non-fix first, followed by the gold repair after observing failure.
    path = next(iter(original))
    src = original[path].decode("utf-8")
    eol = "\r\n" if "\r\n" in src else "\n"
    (root / path).write_bytes((src + eol + "# plausible attempted correction" + eol).encode())
    phase = "edited"
    record("run_verifier")
    failed = workspace.verify()
    if failed["passed"] or failed["exit_code"] != 1:
        raise ValueError("failed patch rollout did not leave a native regression")
    phase, recent = "failed", failed["output"].splitlines()[-6:]
    record("inspect_edit", {"failed": failed["exit_code"], "patch_sha256": failed["patch_sha256"]})
    (root / path).write_bytes(original[path])
    import subprocess
    applied = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=root,
                             input=case["patch"].encode(), capture_output=True)
    if applied.returncode:
        raise ValueError(applied.stderr.decode(errors="replace"))
    phase = "edited"
    record("run_verifier")
    success = workspace.verify()
    if not success["passed"]:
        raise ValueError("gold repair failed native verification")
    phase, recent = "verified", []
    record("submit_to_solver", {"verified": True, "patch_sha256": success["patch_sha256"]})
    return {"case": case["instance_id"], "steps": steps}


def fit(episodes: list[dict], mode: str, *, depth: int, epochs: int, seed: int,
        mms_options=None, mms_receipt=None, ram_limit_bytes=RAM_LIMIT):
    torch.manual_seed(seed)
    model_class = RU02 if mode in {"shared", "dual"} else SetTRM
    model = model_class(mode, depth)
    memory = None
    if mms_options and mms_options.get('mib',0):
        from .mms import EpisodeMemory
        if mms_options['mib']*2**20 >= ram_limit_bytes:
            raise ValueError('MMS bank must leave headroom inside process RAM limit')
        memory = EpisodeMemory(model,episodes,depth,mms_options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    chronology = random.Random(seed)
    shuffle = random.Random(seed + 1)
    order = list(range(len(episodes)))
    losses = []
    for _ in range(epochs):
        chronology.shuffle(order)
        for index in order:
            optimizer.zero_grad()
            context = memory.capture(index,len(losses)) if memory else nullcontext()
            with context:
                latent = None
                loss = []
                for step in episodes[index]["steps"]:
                    target = step["utility"].copy()
                    if mode == "shuffled":
                        shuffle.shuffle(target)
                    scores, next_latent = model.step(step["observation"], latent)
                    latent = next_latent if mode in {"recurrent", "shuffled"} else None
                    loss.append(F.smooth_l1_loss(scores, torch.tensor(target)))
                objective = torch.stack(loss).mean()
                objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(objective.detach()))
            if memory is not None:
                check_ram(ram_limit_bytes)
        check_ram(ram_limit_bytes)
    if memory is not None and mms_receipt is not None:
        mms_receipt.update(memory.memory.receipt())
    return model, losses


def grade(model: SetTRM | RU02, episodes: list[dict], *, reset=False) -> dict:
    model.eval()
    regret, total, hits = 0., 0, 0
    with torch.inference_mode():
        for episode in episodes:
            latent = None
            for step in episode["steps"]:
                scores, next_latent = model.step(step["observation"], latent)
                latent = next_latent if model.mode in {"recurrent", "shuffled"} and not reset else None
                target = step["utility"]
                best = max(target)
                selected = target[int(scores.argmax())]
                hits += selected == best
                regret += best - selected
                total += 1
    return {"top_utility_hits": hits, "slates": total, "mean_regret": regret / max(total, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--depth", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    stage = json.loads((args.stage / "sealed.json").read_text(encoding="utf-8"))
    work = args.out / "rollouts"
    work.mkdir()
    train = [collect(args.stage, case, work) for case in stage["train"]]
    test = [collect(args.stage, case, work) for case in stage["test"]]
    replay = args.out / "training-replay.json"
    dump(replay, train)
    replay_bytes = replay.read_bytes()
    receipt = {"seed": args.seed, "threads": torch.get_num_threads(), "depth": args.depth,
               "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(), "replay_bytes": len(replay_bytes),
               "train_roots": len(train), "holdout_roots": len(test),
               "models": {}}
    for mode in MODES:
        model, losses = fit(train, mode, depth=args.depth, epochs=args.epochs, seed=args.seed)
        architecture = "ru02" if isinstance(model, RU02) else "settrm"
        torch.save({"version": VERSION, "mode": mode, "depth": args.depth, "state": model.state_dict(),
                    "architecture": architecture},
                   args.out / f"{mode}.pt")
        metrics = grade(model, test)
        receipt["models"][mode] = {"architecture": architecture,
                                   "fit_updates": len(losses), "final_loss": losses[-1], **metrics}
        if mode == "recurrent":
            receipt["models"]["reset"] = grade(model, test, reset=True)
    receipt.update(peak_rss_bytes=check_ram(), ram_limit_bytes=RAM_LIMIT, seconds=time.monotonic()-started)
    dump(args.out / "training.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
