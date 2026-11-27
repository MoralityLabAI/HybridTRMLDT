# Verbatim copy of LoopedTransformers experiments/molt_goblin_v1/pilot.py at c42044c5ea0927382dd910a173ef9142a8d49d02 (body below this header is unchanged).
"""MoLT-inspired strict-prefix recurrence pilot, independent of upstream kernels."""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn


class PrefixBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, query, memory, offset: int = 0):
        # A fixed zero-valued null entry prevents all-masked softmax rows.
        b, t, d = query.shape
        m = memory.shape[1]
        q = self.q(query)
        k = self.k(memory)
        v = self.v(memory)
        null = q.new_zeros((b, 1, d))
        k = torch.cat((null, k), dim=1)
        v = torch.cat((null, v), dim=1)
        i = torch.arange(offset, offset + t, device=query.device)[:, None]
        j = torch.arange(m, device=query.device)[None, :]
        allowed = torch.cat((torch.ones((t, 1), dtype=torch.bool, device=query.device), j < i), dim=1)
        scores = (q @ k.transpose(-1, -2)) * d ** -0.5
        a = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        z = self.norm(query + self.o(a @ v))
        return z + self.ff(z)


class PrefixReasoner(nn.Module):
    def __init__(self, width: int = 32, vocab: int = 7):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.block = PrefixBlock(width)
        self.head = nn.Linear(width, vocab)

    def states(self, x, loops: int, arm: str):
        if loops < 1 or arm not in {"prefix_kv", "residual", "one_pass"}:
            raise ValueError("invalid loops or arm")
        base = self.embedding(x)
        h = base
        for _ in range(1 if arm == "one_pass" else loops):
            h = self.block(h if arm == "residual" else base, h)
        return h

    def sequential(self, x):
        base = self.embedding(x)
        h = base[:, :0, :]
        for i in range(x.shape[1]):
            nxt = self.block(base[:, i:i + 1, :], h, offset=i)
            h = torch.cat((h, nxt), dim=1)
        return h

    def forward(self, x, loops: int, arm: str):
        return self.head(self.states(x, loops, arm))


def batch(seed: int, size: int, length: int, device: str):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, 7, (size, length), generator=g).to(device)
    return x, x.cumsum(dim=1).remainder(7)


@torch.no_grad()
def evaluate(model, arm: str, loops: int, length: int, device: str, seed: int):
    model.eval()
    x, y = batch(seed, 64, length, device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    z = model(x, loops, arm)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    correct = z.argmax(-1).eq(y)
    result = {"length": length, "loops": loops, "token_accuracy": correct.float().mean().item(),
              "sequence_accuracy": correct.all(1).float().mean().item(), "elapsed_s": elapsed}
    if arm == "prefix_kv":
        h = model.states(x, length, arm)
        fixed = model.sequential(x)
        result["stabilization_max_abs"] = (h - fixed).abs().max().item()
    return result


def train_arm(arm: str, seed: int, steps: int, loops: int, device: str):
    torch.manual_seed(seed)
    model = PrefixReasoner().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        model.train()
        x, y = batch(seed * 100000 + step, 32, 8, device)
        optim.zero_grad(set_to_none=True)
        z = model(x, loops, arm)
        loss = nn.functional.cross_entropy(z.flatten(0, 1), y.flatten())
        loss.backward()
        optim.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    return {"arm": arm, "seed": seed, "params": sum(p.numel() for p in model.parameters()),
            "train_tokens": steps * 32 * 8, "train_block_visits": steps * (1 if arm == "one_pass" else loops),
            "train_elapsed_s": elapsed, "peak_cuda_allocated_bytes": peak,
            "evaluations": [evaluate(model, arm, r, n, device, seed + 2000000000 + n)
                            for n in (8, 16) for r in (1, loops, n)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not (1 <= args.steps <= 2000 and 2 <= args.loops <= 8 and 1 <= args.threads <= 8
            and 0 <= args.seed <= 10000):
        parser.error("outside pilot bounds")
    torch.set_num_threads(args.threads)
    arms = [train_arm(a, args.seed, args.steps, args.loops, args.device)
            for a in ("prefix_kv", "residual", "one_pass")]
    print(json.dumps({"status": "development_only", "torch": torch.__version__, "device": args.device,
                      "threads": args.threads, "results": arms}, indent=2))


if __name__ == "__main__":
    main()
