import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bias.data import make_pm_profile
from bias.model import SimTFM
from bias.probes import band_coefficients

N_BANDS = 32
LEVELS = 8
CTX = 512
PATCH = 64
HIDDEN = 128
STEPS = 32_768
N_TRAIN = 4096
BATCH = 256
LR = 0.04


def draw_task(seed):
    """Draw a stratified frequency grid and an independent random power pairing."""
    gen = torch.Generator().manual_seed(seed)
    jitter = (torch.rand(N_BANDS, generator=gen) - 0.5) * 0.5
    bands = torch.arange(N_BANDS) + 0.5 + jitter
    levels = torch.arange(LEVELS).repeat_interleave(N_BANDS // LEVELS)
    levels = levels[torch.randperm(N_BANDS, generator=gen)]
    weights = 2.0**levels
    powers = 0.5 * weights / weights.sum()
    return bands, levels, powers


def evaluate(model, x, bands):
    with torch.no_grad():
        pred, target = model.predict_next(x)
    variance = ((target - target.mean()) ** 2).mean()
    z_true = band_coefficients(target, bands, PATCH)
    z_pred = band_coefficients(pred, bands, PATCH)
    return {
        "var_explained": float(1 - F.mse_loss(pred, target) / variance),
        "r": (z_pred.abs() / z_true.abs()).mean(dim=0).tolist(),
        "dk": ((z_pred - z_true).abs() / z_true.abs()).mean(dim=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "this experiment requires CUDA"
    device = torch.device("cuda")

    bands, levels, powers = draw_task(args.seed)
    print(
        f"seed={args.seed} bands=" + " ".join(f"{band:.2f}" for band in bands.tolist()),
        flush=True,
    )
    freqs = bands * CTX / PATCH
    train_x = make_pm_profile(
        freqs, powers, [0.0] * N_BANDS, N_TRAIN, CTX, PATCH, 1000 + args.seed
    )[0].to(device)
    probe = make_pm_profile(
        freqs, powers, [0.0] * N_BANDS, 512, CTX, PATCH, 2000 + args.seed
    )[0].to(device)

    torch.manual_seed(args.seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=2,
        num_attn_heads=4,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    sampler = torch.Generator().manual_seed(args.seed)
    checkpoints = set(range(128, STEPS + 1, 128))
    records = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
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
                    "seed": args.seed,
                    "step": step,
                    "loss": float(loss),
                    "frequencies": bands.tolist(),
                    "levels": levels.tolist(),
                    "powers": powers.tolist(),
                    **evaluate(model, probe, bands),
                }
            )
            model.train()
            if step % 1024 == 0:
                print(
                    f"seed={args.seed} step={step} loss={float(loss):.6f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    config = {
        "n_bands": N_BANDS,
        "ctx": CTX,
        "patch": PATCH,
        "hidden": HIDDEN,
        "steps": STEPS,
        "n_train": N_TRAIN,
        "batch": BATCH,
        "lr": LR,
        "seeds": [args.seed],
        "output": str(args.output),
    }
    args.output.write_text(
        json.dumps({"config": config, "records": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
