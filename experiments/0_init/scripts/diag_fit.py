r"""Can ``SimTFM`` fit the broad marginal at all, or was it just under-trained?

Three training marginals, one seed each, evaluated at 2000/5000/10000/20000
steps. Run from the repo root:
``.venv/bin/python experiments/0_init/scripts/diag_fit.py``.

The gate is variance explained on a held-out corpus of the same marginal;
retention is only read once that is well above zero.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, make_broad, make_mixture
from fbias.model import SimTFM
from fbias.probes import band_r2, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 20000, 64, 1e-2
N_PER = 256
SEED = 0
PROBE_FREQS = [2.0, 128.0]
PROBE_SEED = 123
BANDS = [b_of(f, PATCH, CTX) for f in PROBE_FREQS]
CHECKPOINTS = (2000, 5000, 10000, 20000)
RUNS = Path(__file__).resolve().parents[1] / "runs"

# Each entry draws 2 * N_PER windows of one marginal; the first half trains and
# the second half is held out. Splitting one draw keeps a fixed-frequency arm's
# 32 tones identical across the split while the windows (hence phases) are new.
ARMS = {
    "N   mixture[2,128]": lambda: make_mixture(
        PROBE_FREQS, 2 * N_PER, CTX, PATCH, SEED, normalize=True
    ),
    "BR  broad32 redraw": lambda: make_broad(
        32, 2 * N_PER, CTX, PATCH, SEED, resample_freqs=True
    ),
    "BF  broad32 fixed": lambda: make_broad(
        32, 2 * N_PER, CTX, PATCH, SEED, resample_freqs=False
    ),
}


@torch.no_grad()
def variance_explained(model, held):
    r"""``1 - mse / var(target)`` for the shifted next-patch loss on ``held``."""
    pred, _, _ = model(held)
    target = model.pat(held)[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    return 1 - mse / var, mse, var


@torch.no_grad()
def probe_metrics(model, probe):
    r"""Retention at ``b = 0.125, 8`` at the last trained position."""
    model.eval()
    f_hat, true = model(probe)[0][:, -2], model.pat(probe)[:, -1]
    m = {}
    for b in BANDS:
        m[f"r@{b}"] = float(retention(f_hat, true, b, PATCH).mean())
        m[f"r2@{b}"] = band_r2(f_hat, true, b, PATCH)
    m["r8_over_r1"] = m[f"r@{BANDS[1]}"] / m[f"r@{BANDS[0]}"]
    return m


def train(arm, probe):
    torch.manual_seed(SEED)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data = ARMS[arm]()
    train_data, held = data[:N_PER], data[N_PER:]
    gen = torch.Generator().manual_seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    rows = []
    for step in range(1, STEPS + 1):
        x = train_data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in CHECKPOINTS:
            ve, mse, var = variance_explained(model, held)
            row = {
                "step": step,
                "train_loss": float(loss),
                "held_mse": mse,
                "held_var": var,
                "var_explained": ve,
            }
            row.update(probe_metrics(model, probe))
            model.train()
            rows.append(row)
            print(
                f"  {arm} {step:>6}  loss {row['train_loss']:.4f}  "
                f"var_expl {ve:>+7.3f}  r@0.125 {row['r@0.125']:.4f}  "
                f"r@8 {row['r@8.0']:.4f}  ratio {row['r8_over_r1']:.3f}",
                flush=True,
            )
    return rows


def main():
    t0 = time.time()
    print("== diag_fit: is the broad marginal fittable at this scale? ==")
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )
    print(
        f"probe make_mixture({PROBE_FREQS}, normalize=True), seed {PROBE_SEED}, "
        f"position pred[:, -2] vs pat(x)[:, -1]"
    )

    probe = make_mixture(PROBE_FREQS, N_PER, CTX, PATCH, PROBE_SEED, normalize=True)
    arms = {name: train(name, probe) for name in ARMS}

    head = (
        f"{'arm':<20} {'step':>6} {'train loss':>10} {'var expl':>9} "
        f"{'r@0.125':>8} {'r@8.0':>8} {'r2@0.125':>9} {'r2@8.0':>8} {'r8/r1':>7}"
    )
    print("\n" + head)
    print("-" * len(head))
    for name, rows in arms.items():
        for r in rows:
            print(
                f"{name:<20} {r['step']:>6} {r['train_loss']:>10.4f} "
                f"{r['var_explained']:>+9.3f} {r['r@0.125']:>8.4f} {r['r@8.0']:>8.4f} "
                f"{r['r2@0.125']:>9.4f} {r['r2@8.0']:>8.4f} {r['r8_over_r1']:>7.3f}"
            )

    record = {
        "config": {
            "ctx": CTX,
            "patch": PATCH,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": HEADS,
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "n_per": N_PER,
            "seed": SEED,
            "checkpoints": list(CHECKPOINTS),
            "device": "cpu",
        },
        "probe": {
            "freqs": PROBE_FREQS,
            "b": [round(b, 6) for b in BANDS],
            "seed": PROBE_SEED,
            "normalize": True,
            "position": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "arms": arms,
    }
    out = RUNS / "diag_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
