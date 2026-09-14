r"""Does the training marginal alone change probe retention on the high band?

One fixed probe, a ladder of training corpora, three seeds each. Run from the
repo root: ``.venv/bin/python experiments/0_init/scripts/diag_marginal.py``.
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
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
PROBE_FREQS = [2.0, 128.0]  # b = 0.125 and 8.0, dphi = pi/4 and 0
PROBE_SEED = 123
BANDS = [b_of(f, PATCH, CTX) for f in PROBE_FREQS]
RUNS = Path(__file__).resolve().parents[1] / "runs"

D1_FREQS = [16.0 * i for i in range(1, 9)]  # b = 1..8, the D1 corpus

ARMS = {
    "A  mixture[2,128]": lambda seed: make_mixture(
        PROBE_FREQS, N_PER, CTX, PATCH, seed
    ),
    "B  broad32": lambda seed: make_broad(32, N_PER, CTX, PATCH, seed),
    "C  mixture[16..128]": lambda seed: make_mixture(D1_FREQS, N_PER, CTX, PATCH, seed),
}
ARM_D = (
    "D  broad32 high",
    lambda seed: make_broad(32, N_PER, CTX, PATCH, seed, 8.0, 16.0),
)


def train(corpus, seed):
    r"""The smoke training loop: SGD on the shifted MSE, same seed plumbing."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data = corpus(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss = torch.tensor(0.0)
    for _ in range(STEPS):
        x = data[torch.randint(data.shape[0], (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, float(loss)


@torch.no_grad()
def probe_metrics(model, probe):
    r"""Metrics at the last trained position: ``pred[:, -2]`` vs the last patch."""
    model.eval()
    f_hat, true = model(probe)[0][:, -2], model.pat(probe)[:, -1]
    m = {"mse": float(F.mse_loss(f_hat, true))}
    for b in BANDS:
        r = retention(f_hat, true, b, PATCH)
        m[f"r@{b}"] = float(r.mean())
        m[f"r_std@{b}"] = float(r.std())
        m[f"r2@{b}"] = band_r2(f_hat, true, b, PATCH)
    m["r8_over_r1"] = m[f"r@{BANDS[1]}"] / m[f"r@{BANDS[0]}"]
    return m


def summarize(runs):
    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    out = {}
    for key in keys:
        vals = torch.tensor([r[key] for r in runs], dtype=torch.float64)
        out[key] = (float(vals.mean()), float(vals.std()))
    return out


def run_arm(name, corpus):
    runs = []
    for seed in SEEDS:
        t0 = time.time()
        model, loss = train(corpus, seed)
        m = probe_metrics(
            model, make_mixture(PROBE_FREQS, N_PER, CTX, PATCH, PROBE_SEED)
        )
        m["seed"], m["final_loss"], m["wall"] = seed, loss, time.time() - t0
        runs.append(m)
        print(f"  {name} seed {seed}  loss {loss:.6f}  {m['wall']:.1f}s", flush=True)
    return runs


def control():
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SimTFM(
            context_size=CTX,
            patch_size=PATCH,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            num_attn_heads=HEADS,
        )
        m = probe_metrics(
            model, make_mixture(PROBE_FREQS, N_PER, CTX, PATCH, PROBE_SEED)
        )
        m["seed"] = seed
        runs.append(m)
    return runs


def print_tables(arms, ctrl, names):
    head = (
        f"{'arm':<20} {'seed':>4} {'r@0.125':>8} {'r@8.0':>8} {'r8/r1':>7} "
        f"{'r2@0.125':>9} {'r2@8.0':>8} {'probe mse':>10} {'final loss':>11}"
    )
    print("\nper-seed\n" + head)
    print("-" * len(head))
    for name in names:
        for r in arms[name]:
            loss = f"{r['final_loss']:.6f}" if "final_loss" in r else "-"
            print(
                f"{name:<20} {r['seed']:>4} {r['r@0.125']:>8.4f} {r['r@8.0']:>8.4f} "
                f"{r['r8_over_r1']:>7.3f} {r['r2@0.125']:>9.4f} {r['r2@8.0']:>8.4f} "
                f"{r['mse']:>10.4f} {loss:>11}"
            )
    for r in ctrl:
        print(
            f"{'control untrained':<20} {r['seed']:>4} {r['r@0.125']:>8.4f} "
            f"{r['r@8.0']:>8.4f} {r['r8_over_r1']:>7.3f} {r['r2@0.125']:>9.4f} "
            f"{r['r2@8.0']:>8.4f} {r['mse']:>10.4f} {'-':>11}"
        )

    head = (
        f"{'arm':<20} {'r@0.125':>16} {'r@8.0':>16} {'r2@0.125':>10} "
        f"{'r2@8.0':>9} {'probe mse':>11} {'final loss':>12}"
    )
    print("\nmean +- std over 3 seeds\n" + head)
    print("-" * len(head))
    for name in names + ["control untrained"]:
        s = summarize(arms[name]) if name in arms else summarize(ctrl)
        loss = (
            f"{s['final_loss'][0]:.6f}±{s['final_loss'][1]:.5f}"
            if "final_loss" in s
            else "-"
        )
        print(
            f"{name:<20} {s['r@0.125'][0]:>8.4f}±{s['r@0.125'][1]:<6.4f} "
            f"{s['r@8.0'][0]:>8.4f}±{s['r@8.0'][1]:<6.4f} "
            f"{s['r2@0.125'][0]:>10.4f} {s['r2@8.0'][0]:>9.4f} "
            f"{s['mse'][0]:>11.4f} {loss:>12}"
        )


def main():
    t0 = time.time()
    probe_b = [round(b, 6) for b in BANDS]
    print("== diag_marginal: probe fixed, training marginal varies ==")
    print(f"probe make_mixture({PROBE_FREQS}), seed {PROBE_SEED}, b = {probe_b}")
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )

    arms = {name: run_arm(name, corpus) for name, corpus in ARMS.items()}

    s_a, s_b = summarize(arms["A  mixture[2,128]"]), summarize(arms["B  broad32"])
    drop = s_a["r@8.0"][0] - s_b["r@8.0"][0]
    spread = s_a["r@8.0"][1] + s_b["r@8.0"][1]
    d_damps = drop > max(spread, 0.05)
    print(
        f"\nB damping check at b=8: A {s_a['r@8.0'][0]:.4f} -> B {s_b['r@8.0'][0]:.4f} "
        f"(drop {drop:.4f}, seed spread {spread:.4f}) -> "
        f"{'RUN arm D' if d_damps else 'no arm D'}"
    )
    names = list(ARMS)
    if d_damps:
        arms[ARM_D[0]] = run_arm(*ARM_D)
        names.append(ARM_D[0])

    ctrl = control()
    print_tables(arms, ctrl, names)

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
            "seeds": list(SEEDS),
            "device": "cpu",
        },
        "probe": {
            "freqs": PROBE_FREQS,
            "b": probe_b,
            "seed": PROBE_SEED,
            "position": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "arms": {name: {"per_seed": arms[name]} for name in names},
        "control_untrained": {"per_seed": ctrl},
    }
    out = RUNS / "diag_marginal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
