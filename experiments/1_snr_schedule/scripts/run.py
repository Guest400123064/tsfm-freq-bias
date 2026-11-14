import argparse
import json
import math
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
SIGMA_RELS = (0.0, 0.25, 0.5, 1.0)
WEAKEST_SNR = 1.0


def draw_bands(seed):
    gen = torch.Generator().manual_seed(seed)
    jitter = (torch.rand(N_BANDS, generator=gen) - 0.5) * 0.5
    return torch.arange(N_BANDS) + 0.5 + jitter


def make_powers(levels):
    weights = 2.0 ** torch.as_tensor(levels)
    return 0.5 * weights / weights.sum()


def make_corpus(bands, powers, seed):
    freqs = bands * CTX / PATCH
    train_x = make_pm_profile(
        freqs, powers, [0.0] * N_BANDS, N_TRAIN, CTX, PATCH, 1000 + seed
    )[0]
    probe = make_pm_profile(
        freqs, powers, [0.0] * N_BANDS, 512, CTX, PATCH, 2000 + seed
    )[0]
    return train_x, probe


def calibrated_sigma(powers):
    weak_power = float(torch.as_tensor(powers).min())
    return math.sqrt(PATCH * weak_power / (2 * WEAKEST_SNR))


def band_errors(pred, target, bands):
    z_true = band_coefficients(target, bands, PATCH)
    z_pred = band_coefficients(pred, bands, PATCH)
    return {
        "r": (z_pred.abs() / z_true.abs()).mean(dim=0).tolist(),
        "dk": ((z_pred - z_true).abs() / z_true.abs()).mean(dim=0).tolist(),
    }


def evaluate(model, clean_probe, noisy_probe, bands):
    target = model.pat(clean_probe)[:, -1]
    with torch.no_grad():
        clean_pred = model(clean_probe[:, :CTX])[0][:, -1]
        noisy_pred = model(noisy_probe[:, :CTX])[0][:, -1]
    variance = ((target - target.mean()) ** 2).mean()
    clean_errors = band_errors(clean_pred, target, bands)
    clean_errors["var_explained"] = float(1 - F.mse_loss(clean_pred, target) / variance)
    return {f"clean_{key}": value for key, value in clean_errors.items()} | {
        f"noisy_{key}": value
        for key, value in band_errors(noisy_pred, target, bands).items()
    }


def train_arm(arm, bands, powers, sigma, seed, train_x, probe, device):
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
    noise_gen = torch.Generator(device=device).manual_seed(5000 + seed)
    train_x = train_x.to(device)
    probe = probe.to(device)
    noisy_probe = probe + sigma * torch.randn(
        probe.shape, generator=noise_gen, device=device
    )

    checkpoints = set(range(128, STEPS + 1, 128))
    records = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = torch.randint(len(train_x), (BATCH,), generator=sampler).to(device)
        x = train_x[idx]
        if sigma:
            x = x + sigma * torch.randn(x.shape, device=device)
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
                    **evaluate(model, probe, noisy_probe, bands),
                }
            )
            model.train()
            if step % 1024 == 0:
                print(
                    f"{arm} step={step} loss={float(loss):.6f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return {
        "arm": arm,
        "sigma": sigma,
        "frequencies": bands.tolist(),
        "powers": powers.tolist(),
        "records": records,
    }


def main():
    global STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("snr", "assignment"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    STEPS = args.steps
    assert torch.cuda.is_available(), "this experiment requires CUDA"
    device = torch.device("cuda")
    bands = draw_bands(args.seed)
    arms = []

    if args.suite == "snr":
        powers = make_powers(torch.zeros(N_BANDS))
        train_x, probe = make_corpus(bands, powers, args.seed)
        for sigma_rel in SIGMA_RELS:
            sigma = sigma_rel * math.sqrt(0.5)
            arms.append(
                train_arm(
                    f"snr_{sigma_rel:g}",
                    bands,
                    powers,
                    sigma,
                    args.seed,
                    train_x,
                    probe,
                    device,
                )
            )
    else:
        sorted_levels = torch.arange(LEVELS).repeat_interleave(N_BANDS // LEVELS)
        assignments = {
            "red": sorted_levels.flip(0),
            "anti_red": sorted_levels,
        }
        for name, levels in assignments.items():
            powers = make_powers(levels)
            train_x, probe = make_corpus(bands, powers, args.seed)
            arms.append(
                train_arm(
                    name,
                    bands,
                    powers,
                    calibrated_sigma(powers),
                    args.seed,
                    train_x,
                    probe,
                    device,
                )
            )

    config = {
        "suite": args.suite,
        "seed": args.seed,
        "n_bands": N_BANDS,
        "ctx": CTX,
        "patch": PATCH,
        "hidden": HIDDEN,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
    }
    args.output.write_text(
        json.dumps({"config": config, "arms": arms}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
