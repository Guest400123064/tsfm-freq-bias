import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bias.data import make_phase_map
from bias.model import SimTFM

BANDS = (0.5, 2.5, 4.5, 6.5)
LEVELS = (0, 1, 2, 3)
BETA = 0.1
CTX = 512
PATCH = 16
HIDDEN = 64
STEPS = 32_768
N_TRAIN = 4096
N_PROBE = 512
BATCH = 128
LR = 0.01


def evaluate(model, probe):
    with torch.no_grad():
        pred, target = model.predict_next(probe)
    variance = ((target - target.mean()) ** 2).mean()
    mse = F.mse_loss(pred, target)
    return {
        "mse": float(mse),
        "var_explained": float(1 - mse / variance),
    }


def train_arm(level, seed, train_x, probe, device, steps):
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=2,
        num_attn_heads=4,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    sampler = torch.Generator().manual_seed(seed)
    train_x = train_x.to(device)
    probe = probe.to(device)

    checkpoints = set(range(128, steps + 1, 128))
    records = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        idx = torch.randint(len(train_x), (BATCH,), generator=sampler).to(device)
        x = train_x[idx]
        loss = F.mse_loss(model(x)[0][:, :-1], model.pat(x)[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step in checkpoints:
            model.eval()
            records.append(
                {
                    "step": step,
                    "loss": float(loss),
                    **evaluate(model, probe),
                }
            )
            model.train()
            if step % 1024 == 0:
                print(
                    f"m={level} step={step} loss={float(loss):.6f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return {"level": level, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "this experiment requires CUDA"
    device = torch.device("cuda")

    arms = []
    for level in LEVELS:
        train_x = make_phase_map(
            BANDS, [level] * len(BANDS), BETA, N_TRAIN, CTX, PATCH, 1000 + args.seed
        )
        probe = make_phase_map(
            BANDS, [level] * len(BANDS), BETA, N_PROBE, CTX, PATCH, 2000 + args.seed
        )
        arms.append(train_arm(level, args.seed, train_x, probe, device, args.steps))

    config = {
        "bands": list(BANDS),
        "beta": BETA,
        "ctx": CTX,
        "patch": PATCH,
        "hidden": HIDDEN,
        "steps": args.steps,
        "n_train": N_TRAIN,
        "batch": BATCH,
        "lr": LR,
        "seed": args.seed,
    }
    args.output.write_text(
        json.dumps({"config": config, "arms": arms}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
