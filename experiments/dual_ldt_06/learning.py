"""Fixed-recipe, capacity-matched shared-GRU trajectory value students."""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter, process_time
import platform

import numpy as np
import torch
from torch import nn

from experiments.dual_ldt_06.environment import (
    PERSONAS, PROGRAMS, DENOMINATOR, canonical, world, world_ids, evaluate, input_features,
)

SEEDS = (17, 23, 41)
STEPS = 800
SCALE = 315.0


class ValueStudent(nn.Module):
    """Same parameter count for one/four latent iterations. New weights, CPU."""
    def __init__(self, passes: int):
        super().__init__()
        self.passes = passes
        self.encoder = nn.Linear(77, 32)
        self.recurrent = nn.GRUCell(32, 32)
        self.readout = nn.Linear(32, 1)

    def forward(self, x, passes=None):
        z = torch.tanh(self.encoder(x))
        h = torch.zeros_like(z)
        for _ in range(self.passes if passes is None else passes):
            h = self.recurrent(z, h)
        return self.readout(h).squeeze(-1)


def data(split: str):
    x, y, groups, cases, labels = [], [], [], [], []
    for wid in world_ids(split):
        w = world(wid)
        evaluated = [evaluate(w, p) for p in PROGRAMS]
        for persona, theta in PERSONAS.items():
            for i, result in enumerate(evaluated):
                if result.feasible:
                    value = result.numerator(theta) / DENOMINATOR
                    x.append(input_features(w, theta, PROGRAMS[i]))
                    y.append(value / SCALE)
                    groups.append((wid, persona, i))
                    labels.append(result.numerator(theta))
        cases.append(w.record())
    X = np.array(x, dtype=np.float32)
    Y = np.array(y, dtype=np.float32)
    return X, Y, {"split": split, "world_ids": list(world_ids(split)),
                   "rows": len(y), "world_manifest_sha256": sha256(canonical(cases)).hexdigest(),
                   "labels_sha256": sha256(canonical([groups, labels])).hexdigest()}


def serial_weights(model):
    return {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()}


def load(path: Path):
    record = json.loads(path.read_text())
    m = ValueStudent(record["passes"])
    m.load_state_dict({name: torch.tensor(v, dtype=torch.float32)
                       for name, v in record["state_dict"].items()})
    m.eval()
    return m, record


def train_all(out: Path):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    X, Y, manifest = data("train")
    xt, yt = torch.from_numpy(X), torch.from_numpy(Y)
    files, records = {}, []
    for seed in SEEDS:
        for passes in (1, 4):
            torch.manual_seed(seed)
            model = ValueStudent(passes)
            generator = torch.Generator().manual_seed(seed + 1000)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            losses = []
            start, cpu = perf_counter(), process_time()
            for step in range(STEPS):
                idx = torch.randint(len(xt), (512,), generator=generator)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xt[idx])
                loss = ((pred - yt[idx]) ** 2).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("training loss nonfinite")
                loss.backward()
                optimizer.step()
                if step in (0, 99, 399, 799):
                    losses.append({"step": step+1, "mse_scaled": float(loss.detach())})
            model.eval()
            wall_s, cpu_s = perf_counter()-start, process_time()-cpu
            payload = {"seed": seed, "passes": passes, "steps": STEPS,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "learning_rate": 0.003, "training_manifest": manifest,
                       "recorded_losses": losses, "train_seconds": wall_s, "cpu_seconds": cpu_s,
                       "state_dict": serial_weights(model)}
            raw = canonical(payload)
            name = f"k{passes}_seed{seed}.json"
            (out / name).write_bytes(raw)
            files[name] = sha256(raw).hexdigest()
            records.append({k: v for k, v in payload.items() if k != "state_dict"})
            print("TRAIN", name, payload["parameters"], round(wall_s, 3), losses[-1], flush=True)
    freeze = {"status": "all_model_weights_frozen_before_report",
              "torch": torch.__version__, "python": platform.python_version(),
              "threads": 2, "seed_list": list(SEEDS), "files": files,
              "training_manifest": manifest, "models": records,
              "total_seconds_including_dataset": perf_counter()-t0}
    (out / "FREEZE.json").write_bytes(canonical(freeze))
    return freeze
