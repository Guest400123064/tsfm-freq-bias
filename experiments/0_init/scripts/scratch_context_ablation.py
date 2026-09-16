r"""P1b Step 1: does the theta=0.5 model use its context at all?

P1 found the trained model's clean-probe retention at ``b = 8.0`` sitting at
single-patch Wiener shrinkage (0.82 against 0.81) while its 16-patch context
would allow ~0.94 (PLAN.md §5, P1b). This is the eval-only ablation: re-evaluate
the P1 config with the context left-truncated to its most recent
``N in {1, 2, 4, 8, 16}`` patches, feeding the last ``(N + 1) * k`` timesteps so
the target patch is still last and the readout is still ``pred[:, -2]`` against
``pat(x)[:, -1]``. Left-truncation drops whole patches from the front, so every
relative position is preserved and RoPE transfers; ``N = 16`` is the trained
length exactly, so that row reproduces P1.

Two readings, on different corpora, because neither alone discriminates:

- **Training marginal** (the theta=0.5 corpus, held out): next-patch MSE. The
  direct test of "does the model use history". The incoherent band's in-bin
  coherent/incoherent ratio is ~4 there, so ``N`` patches buy a ~20% MSE
  reduction for an estimator that averages coherently across them -- the oracle
  row measures how much, and is exactly 0 at every ``N`` on a clean corpus.
- **Clean probe** (the P0/P1 ``make_mixture([2.0, 128.0])`` probe): retention
  and band r2 at ``b = 8.0``. One patch already determines the phase of a clean
  persistent tone, so ``N`` is irrelevant *by construction* here -- the oracle is
  0 for every ``N`` -- and this reading alone would be a false negative.

``scratch_coherent.py`` does not save state_dicts, so the two arms are retrained
here with its exact ``train()`` and the weights written to ``runs/``.

Run from the repo root:
``.venv/bin/python experiments/0_init/scripts/scratch_context_ablation.py``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, make_coherent, make_mixture
from fbias.model import SimTFM
from fbias.probes import band_r2, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
NS = (1, 2, 4, 8, 16)
PROBE_FREQS = [2.0, 128.0]  # b = 0.125 and 8.0
PROBE_SEED = 123
HI_BAND = (96.0, 160.0)
BANDS = [b_of(f, PATCH, CTX) for f in PROBE_FREQS]
ARMS = {"theta_0.5": 0.5, "theta_1.0": 1.0}  # P1 arm in question, clean control
MARGINAL = "theta_0.5"  # the shared held-out corpus both arms are scored on
RUNS = Path(__file__).resolve().parents[1] / "runs"


def corpus(theta, seed):
    r"""The arm's training marginal, ``(2 * N_PER, CTX + PATCH)``."""
    return make_coherent(PROBE_FREQS, HI_BAND, theta, 2 * N_PER, CTX, PATCH, seed)


def truncate(x, n):
    r"""The last ``(n + 1) * PATCH`` timesteps: ``n`` context patches + target."""
    return x[:, -(n + 1) * PATCH :]


def train(seed, theta):
    r"""P1's training loop, verbatim, so the weights match ``scratch_coherent``."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data, _ = corpus(theta, seed)
    train_data = data[:N_PER]
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
    return model, float(loss)


@torch.no_grad()
def next_patch_mse(model, x, n):
    r"""MSE of the one-patch forecast, made from the last ``n`` context patches."""
    w = truncate(x, n)
    pred, _, _ = model(w)
    return float(F.mse_loss(pred[:, -2], model.pat(w)[:, -1]))


@torch.no_grad()
def probe_row(model, probe, n):
    r"""Clean-probe retention and band r2 at every probe frequency."""
    w = truncate(probe, n)
    pred = model(w)[0][:, -2]
    true = model.pat(w)[:, -1]
    row = {}
    for b in BANDS:
        row[f"r@{b}"] = float(retention(pred, true, b, PATCH).mean())
        row[f"r2@{b}"] = band_r2(pred, true, b, PATCH)
    return row


@torch.no_grad()
def fit_gate(model, held):
    r"""Variance explained on the full held window, all positions (PLAN.md §3.8)."""
    pred, _, _ = model(held)
    target = model.pat(held)[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    return {"held_mse": mse, "var_explained": 1 - mse / var}


def _basis(t, bands):
    r"""``(len(t), 2 * len(bands))``: cosine and sine at each band."""
    cols = []
    for b in bands:
        a = 2 * math.pi * b * t / PATCH
        cols += [torch.cos(a), torch.sin(a)]
    return torch.stack(cols, -1)


@torch.no_grad()
def oracle_mse(x, n, bands=None):
    r"""Best linear estimator of the target patch from the ``n`` context patches.

    Least squares of the known tone set on the context patches only -- the
    target patch is never in the fit -- so this is the ceiling for coherently
    averaging ``n`` patches, and it is 0 at every ``n`` on a clean corpus.
    ``bands`` restricts the fit to a subset, which shows how much of the
    available gain each tone carries (the un-fitted band is predicted as 0, so
    compare variants by their change in ``n``, not by their level).
    """
    ctx = x[:, -(n + 1) * PATCH : -PATCH].double()
    t = torch.arange(ctx.shape[1], dtype=torch.float64)
    coef = torch.linalg.lstsq(_basis(t, bands or BANDS), ctx.mT).solution
    tt = torch.arange(ctx.shape[1], ctx.shape[1] + PATCH, dtype=torch.float64)
    pred = (_basis(tt, bands or BANDS) @ coef).mT
    return float(F.mse_loss(pred, x[:, -PATCH:].double()))


@torch.no_grad()
def persistence_mse(x):
    r"""Error of copying the last context patch: no history, no N-dependence."""
    p = x.unfold(1, PATCH, PATCH)
    return float(F.mse_loss(p[:, -2], p[:, -1]))


@torch.no_grad()
def band_err_row(model, x, n):
    r"""In-band error contribution ``0.5 * |dZ|^2`` of a forecast from ``n`` patches.

    Splits the next-patch error between the two probe bands, so a fall in total
    MSE can be attributed. Approximate at ``b = 0.125``: the matched filter
    there is phase-dependent (see ``matched_amp``).
    """
    w = truncate(x, n)
    pred = model(w)[0][:, -2]
    true = model.pat(w)[:, -1]
    return {
        f"errb@{b}": 0.5
        * float(
            (matched_amp(pred, b, PATCH) - matched_amp(true, b, PATCH))
            .abs()
            .square()
            .mean()
        )
        for b in BANDS
    }


@torch.no_grad()
def bin_gamma(theta, seed):
    r"""In-bin coherent/incoherent power ratio of a corpus (P1's ``bin_ratio``).

    Per patch, in the probe's own bins. A persisting tone puts all its power in
    one bin while the incoherent energy spreads over the band, so this is much
    larger than ``theta / (1 - theta)``. The N-patch coherent average shrinks by
    ``gamma * N / (1 + gamma * N)``.
    """
    x, incoherent = make_coherent(
        PROBE_FREQS, HI_BAND, theta, 2 * N_PER, CTX, PATCH, seed
    )
    tone = (x - incoherent).unfold(1, PATCH, PATCH)
    noise = incoherent.unfold(1, PATCH, PATCH)
    return {
        b: float(
            matched_amp(tone, b, PATCH).abs().square().mean()
            / matched_amp(noise, b, PATCH).abs().square().mean()
        )
        for b in BANDS
    }


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def table(title, keys, sums):
    r"""One table per field: rows are the config's seeds, columns are ``N``."""
    for f, label in keys.items():
        print(f"\n{title}: {label}")
        head = f"{'arm':<10} {'seed':>4} " + " ".join(
            f"{'N=' + str(n):>12}" for n in NS
        )
        print(head + "\n" + "-" * len(head))
        for name, rows in sums.items():
            for r in rows:
                print(
                    f"{name:<10} {r['seed']:>4} "
                    + " ".join(f"{r[f'{f}@{n}']:>12.5f}" for n in NS)
                )


def main():
    t0 = time.time()
    print(
        "== scratch_context_ablation: P1b Step 1, context length on P1 checkpoints =="
    )
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )
    RUNS.mkdir(parents=True, exist_ok=True)
    probe = make_mixture(PROBE_FREQS, N_PER, CTX, PATCH, PROBE_SEED, normalize=True)
    print(
        f"clean probe make_mixture({PROBE_FREQS}, normalize=True) seed {PROBE_SEED}, "
        f"b = {BANDS}; arms {ARMS}; N = {list(NS)} patches"
    )
    print(
        f"truncation = last (N + 1) * {PATCH} timesteps; readout pred[:, -2] vs "
        f"pat(x)[:, -1]; N = {NS[-1]} is the trained length"
    )
    held = {th: {s: corpus(th, s)[0][N_PER:] for s in SEEDS} for th in ARMS.values()}

    arms, gates, checkpoints = {}, {}, {}
    for name, theta in ARMS.items():
        rows, gates[name] = [], []
        for seed in SEEDS:
            t = time.time()
            model, loss = train(seed, theta)
            path = RUNS / f"ctx_ablation_{name}_seed{seed}.pt"
            torch.save(model.state_dict(), path)
            checkpoints[f"{name}_seed{seed}"] = path.name
            model.eval()
            own, shared = held[theta][seed], held[0.5][seed]
            row = {"seed": seed, "train_loss": loss} | fit_gate(model, own)
            gates[name].append({"seed": seed} | fit_gate(model, own))
            for n in NS:
                row[f"mse@0.5@{n}"] = next_patch_mse(model, shared, n)
                row[f"mse@own@{n}"] = next_patch_mse(model, own, n)
                row |= {f"{k}@{n}": v for k, v in probe_row(model, probe, n).items()}
                for key, v in band_err_row(model, shared, n).items():
                    row[f"{key}@{n}"] = v
            row["wall"] = time.time() - t
            rows.append(row)
            print(
                f"  {name:<10} seed {seed}  loss {loss:.4f}  "
                f"var expl {row['var_explained']:+.3f}  "
                f"mse@0.5 N=1 {row['mse@0.5@1']:.5f} -> N=16 {row['mse@0.5@16']:.5f}  "
                f"r@8 N=1 {row[f'r@{BANDS[1]}@1']:.4f} -> N=16 "
                f"{row[f'r@{BANDS[1]}@16']:.4f}  [{row['wall']:.0f}s]",
                flush=True,
            )
        arms[name] = rows

    keys = {
        "mse@0.5": "next-patch MSE on the shared theta=0.5 marginal",
        "mse@own": "next-patch MSE on each arm's own marginal",
        f"errb@{BANDS[1]}": (
            f"in-band error contribution 0.5*|dZ|^2 at b={BANDS[1]}, shared marginal"
        ),
        f"errb@{BANDS[0]}": (
            f"in-band error contribution 0.5*|dZ|^2 at b={BANDS[0]}, shared marginal"
        ),
        f"r@{BANDS[1]}": f"retention at b={BANDS[1]} on the clean probe",
        f"r@{BANDS[0]}": f"retention at b={BANDS[0]} on the clean probe",
        f"r2@{BANDS[1]}": f"band r2 at b={BANDS[1]} on the clean probe",
    }
    table("per-seed", keys, arms)

    s = {
        name: {
            f"{f}@{n}": mean_std([r[f"{f}@{n}"] for r in rows])
            for f in keys
            for n in NS
        }
        for name, rows in arms.items()
    }
    for f, label in keys.items():
        print(f"\nmean±std over 3 seeds: {label}")
        for name in ARMS:
            print(
                f"  {name:<10} "
                + " ".join(
                    f"{s[name][f'{f}@{n}'][0]:.5f}±{s[name][f'{f}@{n}'][1]:.5f}"
                    for n in NS
                )
            )

    print("\nfit gate (contract 8), variance explained on each arm's own marginal")
    for name, rows in gates.items():
        m, sd = mean_std([r["var_explained"] for r in rows])
        print(f"  {name:<10} {m:+.4f}±{sd:.4f}")

    ref = {
        "marginal_oracle": {
            n: mean_std([oracle_mse(held[0.5][sd], n) for sd in SEEDS]) for n in NS
        },
        "own_oracle_theta_1.0": {
            n: mean_std([oracle_mse(held[1.0][sd], n) for sd in SEEDS]) for n in NS
        },
        "probe_oracle": {
            n: mean_std([oracle_mse(probe, n) for _ in SEEDS]) for n in NS
        },
        "marginal_persistence": mean_std(
            [persistence_mse(held[0.5][sd]) for sd in SEEDS]
        ),
    }
    var = [
        float(c.unfold(1, PATCH, PATCH)[:, 1:].var(unbiased=False))
        for c in held[0.5].values()
    ]
    print("\nreference: best linear estimator from the N context patches (oracle)")
    print(f"  held target var (theta=0.5 marginal)     {mean_std(var)[0]:.5f}")
    for label, key in (
        ("theta=0.5 marginal", "marginal_oracle"),
        ("clean theta=1.0     ", "own_oracle_theta_1.0"),
        ("clean probe         ", "probe_oracle"),
    ):
        print(f"  {label}  " + " ".join(f"{ref[key][n][0]:>9.5f}" for n in NS))
    p = ref["marginal_persistence"]
    print(f"  persistence on the marginal (N-independent) {p[0]:.5f}±{p[1]:.5f}")
    print(
        "  oracle with the fit restricted to one tone (levels differ by the un-fitted"
        "\n  tone's power 0.25, so the change in N is what compares)"
    )
    for i, (label, subset) in enumerate(
        ((f"b={BANDS[0]} only", slice(0, 1)), (f"b={BANDS[1]} only", slice(1, 2)))
    ):
        ref[f"marginal_oracle_{'lo' if i == 0 else 'hi'}"] = {
            n: mean_std([oracle_mse(held[0.5][sd], n, BANDS[subset]) for sd in SEEDS])
            for n in NS
        }
        row = ref[f"marginal_oracle_{'lo' if i == 0 else 'hi'}"]
        print(f"    {label:<10} " + " ".join(f"{row[n][0]:>9.5f}" for n in NS))

    gamma = {b: mean_std([bin_gamma(0.5, s)[b] for s in SEEDS]) for b in BANDS}
    ref["gamma_theta_0.5"] = {str(b): gamma[b] for b in BANDS}
    ref["wiener_probe_r8"] = {
        str(n): gamma[BANDS[1]][0] * n / (1 + gamma[BANDS[1]][0] * n) for n in NS
    }
    print(
        "\nreference: the high band's in-bin coherent/incoherent ratio gamma in the"
        "\n  theta=0.5 marginal, and the shrinkage gamma*N/(1 + gamma*N) that an"
        "\n  estimator calibrated to *N patches of that marginal* would apply"
    )
    for b in BANDS:
        print(f"  gamma at b={b:<6} {gamma[b][0]:.3f}±{gamma[b][1]:.3f}")
        print(
            "    gamma*N/(1 + gamma*N)  "
            + " ".join(f"{gamma[b][0] * n / (1 + gamma[b][0] * n):>9.4f}" for n in NS)
        )
    print(
        "  On the clean probe the tone is recoverable from one patch, so an estimator\n"
        "  that adapts to the evidence it has would read r = 1 at every N; one that\n"
        "  calibrates to the marginal's N-patch SNR would rise along the row above."
    )

    print("\nN = 1 -> N = 16 (mean +- std over 3 seeds)")
    for name in ARMS:
        m1, m16 = s[name]["mse@0.5@1"], s[name]["mse@0.5@16"]
        r1, r16 = s[name][f"r@{BANDS[1]}@1"], s[name][f"r@{BANDS[1]}@16"]
        o1, o16 = ref["marginal_oracle"][1], ref["marginal_oracle"][16]
        print(
            f"  {name:<10} mse@0.5 {m1[0]:.5f}±{m1[1]:.5f} -> "
            f"{m16[0]:.5f}±{m16[1]:.5f} ({(m16[0] - m1[0]) / m1[0]:+.2%})   "
            f"r@8 {r1[0]:.4f}±{r1[1]:.4f} -> "
            f"{r16[0]:.4f}±{r16[1]:.4f} ({r16[0] - r1[0]:+.4f})"
        )
        print(
            f"  {'':<10} oracle mse {o1[0]:.5f} -> {o16[0]:.5f} "
            f"({(o16[0] - o1[0]) / o1[0]:+.2%})"
        )
    for name, rows in arms.items():
        for r in rows:
            print(
                f"  {name:<10} seed {r['seed']}  mse@0.5 {r['mse@0.5@1']:.5f} -> "
                f"{r['mse@0.5@16']:.5f}   r@8 {r[f'r@{BANDS[1]}@1']:.4f} -> "
                f"{r[f'r@{BANDS[1]}@16']:.4f}"
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
            "seeds": list(SEEDS),
            "device": "cpu",
            "truncation": "last (n + 1) * patch timesteps, left-truncate only",
        },
        "arms": {n: {"theta": th} for n, th in ARMS.items()},
        "n_patches": list(NS),
        "marginal": {
            "arm": MARGINAL,
            "freqs": PROBE_FREQS,
            "theta": 0.5,
            "hi_band": list(HI_BAND),
            "held_target_var": mean_std(var)[0],
        },
        "probe": {
            "freqs": PROBE_FREQS,
            "b": BANDS,
            "seed": PROBE_SEED,
            "normalize": True,
            "position": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "checkpoints": checkpoints,
        "per_seed": {
            name: [{k: v for k, v in r.items()} for r in rows]
            for name, rows in arms.items()
        },
        "mean_std": s,
        "fit_gate": gates,
        "reference": {
            k: {str(n): v for n, v in d.items()} if isinstance(d, dict) else d
            for k, d in ref.items()
        },
        "wall": time.time() - t0,
    }
    out = RUNS / "scratch_context_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
