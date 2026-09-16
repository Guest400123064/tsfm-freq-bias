r"""Is the clean-probe deficit a gradient in theta, or a cliff at theta = 0?

PLAN.md §4's second definition of theta: the fraction of a band's power that is
coherent and persistent. Band power is held fixed and the high band's share is
moved between a coherent tone at ``f = 128`` (fraction theta) and incoherent
energy confined to ``[96, 160]`` (fraction ``1 - theta``). P0's additive noise
on top of a *persisting* tone could not work: i.i.d. noise averages down over
~16 context patches, so ``r = 1`` was the correct answer there and the null
discriminated nothing. At ``theta = 0`` the band has no persistent component.

Arms: theta 1.0 (the clean baseline), 0.5, 0.2, 0.0, and ``off_support``, where
the second tone sits at ``f = 64`` instead -- same tone count, same amplitudes,
same RMS, and nothing at all inside ``[96, 160]``. ``theta = 1.0`` with the band
power fixed *is* the old clean arm distributionally: ``make_coherent`` gives
every tone mean power ``0.5 / n_tones = 0.25``, the same as
``make_mixture(..., normalize=True)``, so both are RMS ``sqrt(0.5)`` and the
probe is RMS-matched to the training signal in every arm (contract 9). The RMS
table prints this rather than assuming it.

The clean probe is identical in every arm and seed: two noiseless tones
``f = 2`` (``b = 0.125``) and ``f = 128`` (``b = 8``), seed 123.

Run from the repo root:
``.venv/bin/python experiments/0_init/scripts/scratch_coherent.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, make_coherent, make_mixture
from fbias.model import SimTFM
from fbias.probes import band_r2, fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
PROBE_FREQS = [2.0, 128.0]  # b = 0.125 and 8.0
PROBE_SEED = 123
HI_BAND = (96.0, 160.0)
BANDS = [b_of(f, PATCH, CTX) for f in PROBE_FREQS]
BASELINE = "theta_1.0"
ARMS = {
    "theta_1.0": ([2.0, 128.0], 1.0),
    "theta_0.5": ([2.0, 128.0], 0.5),
    "theta_0.2": ([2.0, 128.0], 0.2),
    "theta_0.0": ([2.0, 128.0], 0.0),
    "off_support": ([2.0, 64.0], 1.0),
}
RUNS = Path(__file__).resolve().parents[1] / "runs"


def corpus(freqs, theta, seed):
    r"""The arm's training marginal, plus the part of it that is unpredictable."""
    return make_coherent(freqs, HI_BAND, theta, 2 * N_PER, CTX, PATCH, seed)


def mean_or_none(vals):
    r"""Mean over the seeds that have a value; ``None`` if none do."""
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


@torch.no_grad()
def bin_ratio(freqs, theta, seed):
    r"""Coherent/incoherent power in the probe's own patch bins, from the corpus.

    A persistent tone puts all its power in one bin while the incoherent energy
    spreads over the band, so this is much larger than ``theta / (1 - theta)``.
    The model sees ~16 context patches, which multiplies it again.
    """
    x, noise = corpus(freqs, theta, seed)
    tone, q = (x - noise).unfold(1, PATCH, PATCH), noise.unfold(1, PATCH, PATCH)
    out = {}
    for b in BANDS:
        coh = float(matched_amp(tone, b, PATCH).abs().square().mean())
        inc = float(matched_amp(q, b, PATCH).abs().square().mean())
        out[f"bin_ratio@{b}"] = coh / inc if inc > 1e-12 else None
    return out


@torch.no_grad()
def rms_table():
    r"""Corpus RMS per arm, against the probe the model is evaluated on."""
    out = {}
    for name, (freqs, theta) in ARMS.items():
        vals = [float(corpus(freqs, theta, s)[0].square().mean().sqrt()) for s in SEEDS]
        out[name] = (sum(vals) / len(vals), max(vals) - min(vals))
    return out


def train(seed, freqs, theta):
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data, noise = corpus(freqs, theta, seed)
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
    r"""Fit gate on the training marginal, then the identical clean probe."""
    model.eval()
    pred, _, z_held = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    floor = float(model.pat(held_noise)[:, 1:].square().mean())
    row = {
        "train_loss": train_loss,
        "held_var": var,
        "var_explained": 1 - mse / var,
        # what no function of the context could explain: the target's own noise
        "unpredictable_floor": 1 - floor / var,
    }
    p_hat, _, z = model(probe)
    true = model.pat(probe)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        row[f"r2@{b}"] = band_r2(f_hat, true[:, -1], b, PATCH)
        # out-of-sample ridge: held corpus -> probe, so not the P0 degenerate one
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
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
        f"{'arm':<12} {'seed':>4} {'var expl':>9} {'floor':>7} {'r@0.125':>8} "
        f"{'r@8.0':>8} {'r8/r1':>7} {'r2@0.125':>9} {'r2@8.0':>8} {'r_or@8':>7}"
    )
    print("\nper-seed (probe is clean in every row)\n" + head)
    print("-" * len(head))
    for name, rows in arms.items():
        for r in rows:
            print(
                f"{name:<12} {r['seed']:>4} {r['var_explained']:>+9.3f} "
                f"{r['unpredictable_floor']:>+7.3f} {r['r@0.125']:>8.4f} "
                f"{r['r@8.0']:>8.4f} {r['r8_over_r1']:>7.3f} "
                f"{r['r2@0.125']:>9.4f} {r['r2@8.0']:>8.4f} "
                f"{r['r_oracle@8.0']:>7.3f}"
            )

    head = (
        f"{'arm':<12} {'var expl':>16} {'r@0.125':>17} {'r@8.0':>17} "
        f"{'r8/r1':>15} {'r2@8.0':>9}"
    )
    print("\nmean +- std over 3 seeds\n" + head)
    print("-" * len(head))
    for name, rows in arms.items():
        s = summarize(rows)
        print(
            f"{name:<12} {s['var_explained'][0]:>+8.3f}±{s['var_explained'][1]:<6.3f} "
            f"{s['r@0.125'][0]:>8.4f}±{s['r@0.125'][1]:<7.4f} "
            f"{s['r@8.0'][0]:>8.4f}±{s['r@8.0'][1]:<7.4f} "
            f"{s['r8_over_r1'][0]:>8.3f}±{s['r8_over_r1'][1]:<6.3f} "
            f"{s['r2@8.0'][0]:>9.4f}"
        )
    return {name: summarize(rows) for name, rows in arms.items()}


def contrasts(s):
    print(f"\nhigh band against the {BASELINE} baseline (clean probe)")
    for key in (f"r@{BANDS[1]}", f"r2@{BANDS[1]}", "r8_over_r1"):
        base, base_sd = s[BASELINE][key]
        print(f"  {key:<12} {BASELINE} {base:+.4f}  seed spread {base_sd:.4f}")
        for arm in ARMS:
            if arm == BASELINE:
                continue
            v, sd = s[arm][key]
            spread = base_sd + sd
            verdict = "beyond" if abs(v - base) > spread else "inside"
            print(
                f"    {arm:<12} {v:+.4f}±{sd:.4f}  change {v - base:+.4f}  "
                f"{verdict} spread ({spread:.4f})"
            )
    print("\nlow band (internal control; the baseline overshoot is quotable)")
    for arm in ARMS:
        v, sd = s[arm][f"r@{BANDS[0]}"]
        print(f"  {arm:<12} {v:+.4f}±{sd:.4f}")


def main():
    t0 = time.time()
    print("== scratch_coherent: theta = coherent fraction of the high band ==")
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )
    probe = make_mixture(PROBE_FREQS, N_PER, CTX, PATCH, PROBE_SEED, normalize=True)
    probe_rms = float(probe.square().mean().sqrt())
    print(
        f"probe make_mixture({PROBE_FREQS}, normalize=True) seed {PROBE_SEED}, "
        f"b = {BANDS}, RMS {probe_rms:.4f}, pred[:, -2] vs pat(x)[:, -1]"
    )
    print(
        f"high band {HI_BAND} cycles/window; arms = (freqs, theta): "
        + ", ".join(f"{n}={v}" for n, v in ARMS.items())
    )

    print("\nRMS of each arm's corpus (contract 9) and the probe")
    rms = rms_table()
    for name, (mean, spread) in rms.items():
        print(
            f"  {name:<12} {mean:.4f}  seed spread {spread:.2e}  "
            f"{mean / probe_rms - 1:+.2%} of probe"
        )

    print("\ncoherent/incoherent power in the probe's patch bins (nominal theta)")
    ratios = {}
    for name, (freqs, theta) in ARMS.items():
        per_seed = [bin_ratio(freqs, theta, seed) for seed in SEEDS]
        ratios[name] = {k: mean_or_none([r[k] for r in per_seed]) for k in per_seed[0]}
        cells = "  ".join(
            f"{k} {'n/a' if v is None else f'{v:.3f}'}" for k, v in ratios[name].items()
        )
        print(f"  {name:<12} theta {theta:<4} {cells}")

    arms = {}
    for name, (freqs, theta) in ARMS.items():
        rows = []
        for seed in SEEDS:
            t = time.time()
            model, held, held_noise, loss = train(seed, freqs, theta)
            row = evaluate(model, held, held_noise, probe, loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            rows.append(row)
            print(
                f"  {name:<12} seed {seed}  loss {loss:.4f}  "
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
            "hi_band": list(HI_BAND),
            "arms": {n: {"freqs": f, "theta": th} for n, (f, th) in ARMS.items()},
            "device": "cpu",
        },
        "probe": {
            "freqs": PROBE_FREQS,
            "b": BANDS,
            "seed": PROBE_SEED,
            "normalize": True,
            "rms": probe_rms,
            "position": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "corpus_rms": {n: {"mean": m, "spread": sp} for n, (m, sp) in rms.items()},
        "bin_ratio": ratios,
        "arms": {
            name: {"per_seed": rows, "mean_std": s[name]} for name, rows in arms.items()
        },
    }
    out = RUNS / "scratch_coherent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
