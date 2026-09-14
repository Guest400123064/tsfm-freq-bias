r"""`0_init` smoke: train the two-tone mixture, read the §4 metrics back.

Run from the repo root: ``.venv/bin/python experiments/0_init/scripts/smoke.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from fbias.cli import train as train_cli
from fbias.data import b_of, make_mixture
from fbias.model import SimTFM
from fbias.probes import band_r2, fit_oracle, phase_error, retention

CTX, PATCH = 512, 32
FREQS = [2.0, 128.0]  # b = 0.125 and b = 8
N_PER = 256
SEED = 0
PROBE_POS = -1
RUNS = Path(__file__).resolve().parents[1] / "runs"


def metrics(pred, true, b, k):
    r = retention(pred, true, b, k)
    pe = phase_error(pred, true, b, k)
    pe = (pe + math.pi) % (2 * math.pi) - math.pi  # wrap to (-pi, pi]
    return (
        r.mean().item(),
        r.std().item(),
        pe.abs().mean().item(),
        band_r2(pred, true, b, k),
    )


def main():
    args = argparse.Namespace(
        ctx=CTX,
        patch=PATCH,
        hidden=32,
        layers=2,
        heads=4,
        steps=2000,
        batch=64,
        lr=1e-2,
        seed=SEED,
        n_per=N_PER,
        freqs=FREQS,
        out=str(RUNS / "smoke.pt"),
        probe_pos=PROBE_POS,
        device="cpu",
    )
    t0 = time.time()
    train_cli.main(args)
    wall = time.time() - t0

    sidecar = json.loads((RUNS / "smoke.json").read_text())
    model = SimTFM(**sidecar["config"])
    model.load_state_dict(torch.load(RUNS / "smoke.pt", map_location="cpu"))
    model.eval()

    train_x = make_mixture(FREQS, N_PER, CTX, PATCH, SEED)
    probe_x = make_mixture(FREQS, N_PER, CTX, PATCH, SEED + 1)
    with torch.no_grad():
        pred, _, z = model(probe_x)
        patches = model.pat(probe_x)
        _, _, z_train = model(train_x)
        patches_train = model.pat(train_x)
        f_hat = pred[:, :-1][:, PROBE_POS]
        true = patches[:, 1:][:, PROBE_POS]

    rows = []
    for f in FREQS:
        b = b_of(f, PATCH, CTX)
        oracle = fit_oracle(z_train, patches_train, z, patches, b, PATCH)
        rows.append(
            {
                "f": f,
                "b": b,
                "dphi": (2 * math.pi * b % (2 * math.pi)) / math.pi,
                "model": metrics(f_hat, true, b, PATCH),
                "oracle": metrics(oracle["pred"][:, PROBE_POS], true, b, PATCH),
                "oracle_pooled_r2": oracle["r2"],
            }
        )

    hist = sidecar["loss_history"]
    print("\n== 0_init smoke ==")
    print(
        f"steps {args.steps} batch {args.batch} lr {args.lr} | "
        f"loss {hist[0][1]:.4f} @{hist[0][0]} -> {hist[-1][1]:.6f} @{hist[-1][0]}"
    )
    print("loss " + " ".join(f"{loss:.4f}" for _, loss in hist))
    print(f"wall {wall:.1f}s, probe position {PROBE_POS} of {CTX // PATCH}\n")

    print(f"{'f':>6} {'b':>7} {'dphi/pi':>8} {'r_model':>18} {'r2_model':>9}", end="")
    print(
        f" {'|phase err|':>11} {'r_oracle':>18} {'r2_oracle':>10} {'r2_orc_pool':>11}"
    )
    print("-" * 105)
    for row in rows:
        r, r_std, pe, r2 = row["model"]
        or_ret, or_std, _, or_r2 = row["oracle"]
        print(
            f"{row['f']:>6.1f} {row['b']:>7.3f} {row['dphi']:>8.3f} "
            f"{r:>11.4f}±{r_std:<6.4f} {r2:>9.4f} {pe:>11.4f} "
            f"{or_ret:>11.4f}±{or_std:<6.4f} {or_r2:>10.4f} "
            f"{row['oracle_pooled_r2']:>11.4f}"
        )


if __name__ == "__main__":
    main()
