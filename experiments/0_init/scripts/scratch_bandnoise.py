r"""Is band-selective training noise a *prior*, or just correct Bayes shrinkage?

The clean probe is identical in every arm and every seed: two noiseless tones
``f = 2`` (``b = 0.125``) and ``f = 128`` (``b = 8``). Only the training
corpus's noise changes, at constant total noise power. A retention deficit on
the clean probe therefore cannot be a response to noise *in the probe*: it is
carry-over from the training marginal. The injected noise/signal power in each
probe bin is printed too, since it bounds how much damping the marginal could
justify in the first place.

Run from the repo root:
``.venv/bin/python experiments/0_init/scripts/scratch_bandnoise.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, band_noise, make_mixture
from fbias.model import SimTFM
from fbias.probes import band_r2, fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
PROBE_FREQS = [2.0, 128.0]  # b = 0.125 and 8.0, dphi = pi/4 and 0
PROBE_SEED = 123
BANDS = [b_of(f, PATCH, CTX) for f in PROBE_FREQS]
NOISE_REL = 0.7  # noise RMS as a fraction of the signal RMS
NOISE_SEED = 1000  # one white draw per seed, shared by the three noise arms
LO_BAND, HI_BAND = (1.0, 32.0), (96.0, 160.0)
ARMS = {
    "clean": None,
    "noise_hi": HI_BAND,
    "noise_lo": LO_BAND,
    "noise_all": (0.0, CTX / 2),
}
RUNS = Path(__file__).resolve().parents[1] / "runs"


def corpus(seed, band):
    r"""The two tones plus ``band``-limited noise; returns both parts."""
    signal = make_mixture(PROBE_FREQS, 2 * N_PER, CTX, PATCH, seed, normalize=True)
    if band is None:
        return signal, torch.zeros_like(signal)
    rms = NOISE_REL * float(signal.square().mean().sqrt())
    noise = band_noise(signal, band[0], band[1], rms, CTX, NOISE_SEED + seed)
    return signal + noise, noise


@torch.no_grad()
def injection(band, seed):
    r"""Noise/signal power in the probes' own patch bins, from the corpus alone.

    How strong the injection is in the band the probe reads: at a ratio of
    ``v`` a single-patch estimator is optimally damped to ``1 / (1 + v)``,
    so a large ``v`` is what would license real shrinkage in that band.
    """
    signal, noise = corpus(seed, band)
    p, q = signal.unfold(1, PATCH, PATCH), noise.unfold(1, PATCH, PATCH)
    return {
        f"noise_bin_ratio@{b}": float(
            matched_amp(q, b, PATCH).abs().square().mean()
            / matched_amp(p, b, PATCH).abs().square().mean()
        )
        for b in BANDS
    }


def train(seed, band):
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data, noise = corpus(seed, band)
    train_data, held = data[:N_PER], data[N_PER:]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss = torch.tensor(0.0)
    for _ in range(STEPS):
        x = train_data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, held, noise[N_PER:], float(loss)


@torch.no_grad()
def evaluate(model, held, held_noise, probe, train_loss):
    r"""Fit on a held corpus of the training marginal, then probe clean."""
    model.eval()
    pred, _, _ = model(held)
    target = model.pat(held)[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    noise_mse = float(model.pat(held_noise)[:, 1:].square().mean())
    row = {
        "train_loss": train_loss,
        "held_mse": mse,
        "held_var": var,
        "var_explained": 1 - mse / var,
        # best any function of the context can do: the target's noise part
        "noise_floor": 1 - noise_mse / var,
    }
    p_hat, _, z = model(probe)
    true = model.pat(probe)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        row[f"r2@{b}"] = band_r2(f_hat, true[:, -1], b, PATCH)
        row[f"r_oracle@{b}"] = fit_oracle(z, true, z, true, b, PATCH)["retention"]
    row["r8_over_r1"] = row[f"r@{BANDS[1]}"] / row[f"r@{BANDS[0]}"]
    return row


def summarize(runs):
    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    out = {}
    for key in keys:
        vals = torch.tensor([r[key] for r in runs], dtype=torch.float64)
        out[key] = (float(vals.mean()), float(vals.std()))
    return out


def print_tables(arms):
    head = (
        f"{'arm':<10} {'seed':>4} {'var expl':>9} {'floor':>7} {'r@0.125':>8} "
        f"{'r@8.0':>8} {'r8/r1':>7} {'r2@0.125':>9} {'r2@8.0':>8} {'r_or@8':>7}"
    )
    print("\nper-seed (probe is clean in every row)\n" + head)
    print("-" * len(head))
    for name, rows in arms.items():
        for r in rows:
            print(
                f"{name:<10} {r['seed']:>4} {r['var_explained']:>+9.3f} "
                f"{r['noise_floor']:>+7.3f} {r['r@0.125']:>8.4f} {r['r@8.0']:>8.4f} "
                f"{r['r8_over_r1']:>7.3f} {r['r2@0.125']:>9.4f} {r['r2@8.0']:>8.4f} "
                f"{r['r_oracle@8.0']:>7.3f}"
            )

    head = (
        f"{'arm':<10} {'var expl':>16} {'r@0.125':>17} {'r@8.0':>17} "
        f"{'r8/r1':>15} {'r2@8.0':>9}"
    )
    print("\nmean +- std over 3 seeds\n" + head)
    print("-" * len(head))
    for name, rows in arms.items():
        s = summarize(rows)
        print(
            f"{name:<10} {s['var_explained'][0]:>+8.3f}±{s['var_explained'][1]:<6.3f} "
            f"{s['r@0.125'][0]:>8.4f}±{s['r@0.125'][1]:<7.4f} "
            f"{s['r@8.0'][0]:>8.4f}±{s['r@8.0'][1]:<7.4f} "
            f"{s['r8_over_r1'][0]:>8.3f}±{s['r8_over_r1'][1]:<6.3f} "
            f"{s['r2@8.0'][0]:>9.4f}"
        )
    return {name: summarize(rows) for name, rows in arms.items()}


def contrasts(s):
    print("\nnoisy band vs clean control on the clean probe")
    for b, arm in ((BANDS[1], "noise_hi"), (BANDS[0], "noise_lo")):
        key = f"r@{b}"
        change = s[arm][key][0] - s["clean"][key][0]
        spread = s["clean"][key][1] + s[arm][key][1]
        verdict = "beyond" if abs(change) > spread else "inside"
        print(
            f"  {key:<8} clean {s['clean'][key][0]:.4f} -> {arm} {s[arm][key][0]:.4f}  "
            f"change {change:+.4f}  seed spread {spread:.4f}  {verdict} spread"
        )
    for key in ("r@0.125", "r@8.0", "r8_over_r1", "var_explained"):
        vals = "  ".join(f"{a} {s[a][key][0]:+.4f}±{s[a][key][1]:.4f}" for a in ARMS)
        print(f"  {key:<12} {vals}")


def main():
    t0 = time.time()
    print("== scratch_bandnoise: band-selective training noise, clean probe ==")
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )
    probe = make_mixture(PROBE_FREQS, N_PER, CTX, PATCH, PROBE_SEED, normalize=True)
    signal_rms = float(probe.square().mean().sqrt())
    print(
        f"probe make_mixture({PROBE_FREQS}, normalize=True) seed {PROBE_SEED}, "
        f"b = {BANDS}, position pred[:, -2] vs pat(x)[:, -1]"
    )
    print(
        f"training signal = same two tones (RMS {signal_rms:.4f}) + noise RMS "
        f"{NOISE_REL} * signal RMS = {NOISE_REL * signal_rms:.4f}; "
        f"lo band {LO_BAND} hi band {HI_BAND} (cycles/window)"
    )

    print("\ninjection strength, noise/signal power in the probes' own patch bins")
    injected = {}
    for name, band in ARMS.items():
        per_seed = [injection(band, seed) for seed in SEEDS]
        injected[name] = {
            k: sum(r[k] for r in per_seed) / len(per_seed) for k in per_seed[0]
        }
        cells = "  ".join(f"{k} {v:.4f}" for k, v in injected[name].items())
        print(f"  {name:<10} {cells}")

    arms = {}
    for name, band in ARMS.items():
        rows = []
        for seed in SEEDS:
            t = time.time()
            model, held, held_noise, loss = train(seed, band)
            row = evaluate(model, held, held_noise, probe, loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            rows.append(row)
            print(
                f"  {name:<10} seed {seed}  loss {loss:.4f}  "
                f"var expl {row['var_explained']:+.3f}  r@0.125 {row['r@0.125']:.4f}  "
                f"r@8 {row['r@8.0']:.4f}  [{row['wall']:.0f}s]",
                flush=True,
            )
        arms[name] = rows

    s = print_tables(arms)
    contrasts(s)

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
            "noise_rel_rms": NOISE_REL,
            "noise_seed": NOISE_SEED,
            "signal_rms": signal_rms,
            "noise_rms": NOISE_REL * signal_rms,
            "bands_cycles_per_window": {
                "lo": LO_BAND,
                "hi": HI_BAND,
                "all": [0.0, CTX / 2],
            },
            "device": "cpu",
        },
        "probe": {
            "freqs": PROBE_FREQS,
            "b": BANDS,
            "seed": PROBE_SEED,
            "normalize": True,
            "position": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "injection": injected,
        "arms": {
            name: {"per_seed": rows, "mean_std": s[name]} for name, rows in arms.items()
        },
    }
    out = RUNS / "scratch_bandnoise.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
