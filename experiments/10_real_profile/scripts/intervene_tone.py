r"""T5 step 4: adding pure tones -- is it the band's share, or the band?

The practical question is what to mix into a real corpus. The natural answer,
and the one this run tests, is **pure tones**: windows that are nothing but a
single clean tone at a chosen band. They are not mimic windows.

That choice has a price and a payoff, and both are the point.

  Price: a pure-tone window has no energy at the other fifteen bands, so mixing
  them in dilutes every other band's marginal. There is no arm in which some band
  is untouched, and step 3's identification problem would come back -- except for
  what follows.

  Payoff: one tone window in ten, each a single unit-variance tone, puts the same
  ~0.10 of corpus power at whichever band it occupies, and leaves the other
  twelve (b16 is Nyquist) at exactly 0.9x the real corpus identically in every
  arm. The arms differ in exactly one thing: which frequency carries that fixed
  dose. The dose is gentle on purpose -- the corpus's total matched-band energy
  moves only from 0.686 to 0.718 -- but it takes the corpus's marginal from a
  537x range to 5.9x, which is the flattening the intervention is for.

Arms:

  real      the real training half alone (the anchor);
  tone_b6   real windows + pure tones at band 6 (2.67 h);
  tone_b10  the same at band 10 (1.60 h);
  tone_b14  the same at band 14 (1.14 h).

Each arm's corpus marginal is ~0.101 at its own tone band -- equal across the
three arms to within 2.5 % *in absolute power* and to within 2.5 % in **share**
(0.0809 / 0.0791 / 0.0789) -- and exactly 0.9x the real corpus at the other
twelve bands, identically in all three arms. So the three tone bands end the
intervention with the same share even though their real shares differed by 3.8x,
and the question is whether their retention ends up equal too. Equal share, three
frequencies, twelve identical control bands: the project's central question in
its cleanest available form, *is it the share or the band?*

The tones are pure (beta = 0), so the added band is also made fully predictable.
A positive result is therefore not attributable to power alone, and a
matched-beta control arm is the follow-up -- it only needs running if this one
moves.

Probes, all on held-out windows:

  real   the real held-out half (the primary reading);
  mimic  a fixed synthetic corpus drawn from the real profile, never trained on
         -- does any gain transfer to a synthetic window whose band content is
         the real corpus's?
  own    a fresh draw from the arm's own profile -- was the added tone learned
         at all?

Pre-registered, before the run:

  1. the manipulation landed: outside the tone bands the three arms' corpus
     marginals agree at exactly 0.9x the real corpus, and the dose at the three
     tone bands agrees to within a few percent in both absolute power and share;
  2. internal control: at the twelve non-tone bands the three tone arms agree
     with *each other* within three seed-paired sd, on every probe. ``real`` is
     not a control there -- it is not halved outside the tone bands -- so it is
     the anchor only;
  3. **the diagonal**: in the 3 x 3 table of res/floor at (arm, tone band), each
     column's minimum is the arm that carries the dose at that band, on the real
     probe;
  4. **equal dose, equal effect**: the three diagonal values agree with each
     other within seed noise, i.e. the improvement is predicted by the corpus
     power, which is equal, and not by the frequency. A systematic gradient
     along the diagonal is the signature of a residual frequency effect beyond
     share, and is reported as such.

Run from the repo root:
``.venv/bin/python experiments/10_real_profile/scripts/intervene_tone.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from fbias.data import make_pm_profile  # noqa: E402
from fbias.model import SimTFM  # noqa: E402
from fbias.probes import band_component  # noqa: E402

BASE_PATH = Path(__file__).resolve().parent / "baseline.py"
RUNS = ROOT / "experiments/10_real_profile/runs"

K, CTX = 32, 512
BANDS = tuple(float(b) for b in range(1, 17))
WELL = BANDS[:-1]  # b16 is Nyquist: its matched filter is its own mirror
FREQS = torch.tensor([b * CTX / K for b in BANDS])

W = 0.1  # tone windows per real training window
TONE_E = (
    1.0  # a tone window's power: one unit-variance tone, the scale of a real window
)
TONE_BANDS = (6.0, 10.0, 14.0)
SEEDS = (0, 1, 2, 3)
SYNTH_SEED = 4242
PROBE_N, MIMIC_SEED, OWN_SEED = 512, 4222, 5222
ORACLE_FIT = 512
CROSS = 2.0
MARK_STEPS = (100, 200, 500, 1000, 2000, 5000, 10000)
TOL_SD = 3.0


def load_baseline():
    spec = importlib.util.spec_from_file_location("bl", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bl"] = mod
    spec.loader.exec_module(mod)
    return mod


def tone_arm(band):
    return f"tone_b{int(band)}"


def tone_profile(band, dose):
    r"""One tone: all ``dose`` of the window's power at ``band``, nothing elsewhere."""
    prof = np.zeros(len(BANDS), dtype=np.float64)
    prof[int(band) - 1] = dose
    return prof


def synth_corpus(profile, betas, n, seed=SYNTH_SEED):
    r"""``(n, n_patch, k)`` phase-diffusing windows with per-band power ``profile``."""
    x, _ = make_pm_profile(FREQS, profile, betas, n, CTX, K, seed)
    return x.reshape(n, -1, K).numpy()


@torch.no_grad()
def band_res(model, x):
    r"""Per-band residual ``mean|Z_pred - Z_true|^2 / 2`` on a probe, keyed by band."""
    model.eval()
    pred, _, _ = model(x)
    true = model.pat(x)[:, -1]
    f_hat = pred[:, -2]
    out = {}
    for b in BANDS:
        d = band_component(f_hat, b, K) - band_component(true, b, K)
        out[b] = float((d**2).mean() / 2)
    model.train()
    return out


@torch.no_grad()
def probe_mse(model, x):
    r"""Whole-window next-patch MSE: was the arm's own corpus learned at all?"""
    model.eval()
    pred, _, _ = model(x)
    m = float(F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:]))
    model.train()
    return m


def probe_floor(bl, x):
    r"""A synthetic probe's own per-band floor, from its own ridge and its own power."""
    n = len(x)
    bound = bl.band_bound(x, BANDS, np.arange(n // 2), np.arange(n // 2, n))
    power = bl.band_power(x, BANDS, K)
    return bound, {b: float((1 - bound[str(b)]) * p) for b, p in zip(BANDS, power)}


def train(bl, data, tr, ev, seed, args, probes, own_x):
    r"""The baseline's loop, plus a ``res/floor`` reading on every probe."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=K,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_attn_heads=bl.HEADS,
    )
    patches, naive, ok = data[0], data[1], data[2]
    x_tr = torch.from_numpy(patches[tr].reshape(len(tr), -1).astype(np.float32))
    held = torch.from_numpy(patches[ev].reshape(len(ev), -1).astype(np.float32))
    fit_x = torch.from_numpy(
        patches[tr[:ORACLE_FIT]]
        .reshape(min(ORACLE_FIT, len(tr)), -1)
        .astype(np.float32)
    )
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=bl.LR)
    marks = []
    for step in range(1, args.steps + 1):
        x = x_tr[torch.randint(len(x_tr), (bl.BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        due = (step <= args.ckpt_fine_until and step % args.ckpt_fine == 0) or (
            step % args.ckpt == 0
        )
        if due:
            row = bl.evaluate(model, held, naive[ev], ok[ev], fit_x, BANDS)
            for name, x_probe, floor in probes:
                res = band_res(model, x_probe)
                for b in BANDS:
                    row[f"{name}@{b}"] = float(res[b] / floor[b])
            row["own_mse"] = probe_mse(model, own_x)
            row["step"] = step
            marks.append(row)
            print(
                f"      step {step:>6}  mse {row['mse']:.4f}  mase "
                f"{row['mase']:.4f}  var expl {row['var_explained']:+.4f}",
                flush=True,
            )
    return marks


def cross_step(marks, key, b, level=CROSS):
    for m in marks:
        if m[f"{key}@{b}"] <= level:
            return int(m["step"])
    return None


def paired(out, a, b, key, band):
    r"""Mean and sd of ``a - b`` over seeds: one init, one batch schedule."""
    d = [
        x[-1][f"{key}@{band}"] - y[-1][f"{key}@{band}"]
        for x, y in zip(out["arms"][a]["per_seed"], out["arms"][b]["per_seed"])
    ]
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return float(np.mean(d)), sd


def last(out, a, key, b):
    return out["arms"][a]["marks"][-1][f"{key}@{b}"]


def table(head, rows):
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def print_setup(out):
    arms = list(out["arms"])
    ref = out["arms"]["real"]["corpus_power"]
    print("\n== setup ==")
    print(
        f"  1 061 real training windows + {int(W * 1061)} pure-tone windows in every "
        f"arm but `real`; tone bands {[int(b) for b in TONE_BANDS]}; "
        f"seeds {out['config']['seeds']}; dose {out['dose']:.4f} per tone window"
    )
    print("  corpus marginal relative to the real corpus (* marks a tone band)")
    table(
        [
            f"  {'b':>4}",
            f"{'period_h':>8}",
            f"{'real power':>11}",
            *[f"{a:>10}" for a in arms],
        ],
        [
            [
                f"  {int(b):>4}" + ("*" if b in TONE_BANDS else " "),
                f"{K / b * 0.5:>8.2f}",
                f"{ref[j]:>11.5f}",
                *[f"{out['arms'][a]['corpus_power'][j] / ref[j]:>10.3f}" for a in arms],
            ]
            for j, b in enumerate(BANDS)
        ],
    )
    print(
        "  probes: real = the real held-out half; mimic = fixed synthetic real profile;"
    )
    print(
        "          own = a fresh draw from each arm's own profile (was the tone learnt)"
    )


def print_dose(out):
    arms = list(out["arms"])
    doses = [out["arms"][tone_arm(b)]["corpus_power"][int(b) - 1] for b in TONE_BANDS]
    print("\n== pre-registered checks ==")
    print(
        "  1. equal dose at the tone bands: "
        + ", ".join(f"b{int(b)} {d:.4f}" for b, d in zip(TONE_BANDS, doses))
        + f"   spread {100 * (max(doses) / min(doses) - 1):.2f}%"
    )
    tones = [a for a in arms if a != "real"]
    print("     outside the tone bands the tone arms halve every band identically:")
    worst = 0.0
    for j, b in enumerate(WELL):
        if b in TONE_BANDS:
            continue
        vals = [out["arms"][a]["corpus_power"][j] for a in tones]
        worst = max(worst, (max(vals) - min(vals)) / max(vals))
    outside = [
        out["arms"][tones[0]]["corpus_power"][j]
        for j, b in enumerate(WELL)
        if b not in TONE_BANDS
    ]
    real_out = [
        out["arms"]["real"]["corpus_power"][j]
        for j, b in enumerate(WELL)
        if b not in TONE_BANDS
    ]
    ratio = float(np.mean(np.array(outside) / np.array(real_out)))
    print(
        f"     worst spread among tone arms {100 * worst:.3f}%; they sit at "
        f"{ratio:.3f}x `real` there (real is not halved, so it is the anchor only)"
    )


def print_matrix(out, key, label):
    arms = list(out["arms"])
    print(f"\n== {label}: res/floor at the tone bands, bands x arms ==")
    table(
        [f"  {'b':>4}", *[f"{a:>10}" for a in arms], f"{'best':>10}"],
        [
            [
                f"  {int(b):>4}*",
                *[
                    f"{last(out, a, key, b):>10.3f}"
                    + ("*" if a == tone_arm(b) else " ")
                    for a in arms
                ],
                f"{min(arms, key=lambda a: last(out, a, key, b)):>10}",
            ]
            for b in TONE_BANDS
        ],
    )


def print_schedule(out, key, label):
    arms = list(out["arms"])
    marks = {a: {m["step"]: m for m in out["arms"][a]["marks"]} for a in arms}
    steps = [s for s in MARK_STEPS if s in marks[arms[0]]]
    print(f"\n== {label}: res/floor by step at the tone bands ==")
    for b in TONE_BANDS:
        print(f"\n  band {int(b)} (period {K / b * 0.5:.2f} h)")
        table(
            [f"    {'step':>6}", *[f"{a:>11}" for a in arms]],
            [
                [f"    {s:>6}", *[f"{marks[a][s][f'{key}@{b}']:>11.2f}" for a in arms]]
                for s in steps
            ],
        )


def print_diagonal(out):
    print("\n  3. the diagonal, per seed-paired contrast (negative = tone arm better)")
    for key, label in (("real", "real probe"), ("mimic", "mimic probe")):
        print(f"     {label}")
        for b in TONE_BANDS:
            parts = []
            for other in TONE_BANDS:
                if other == b:
                    continue
                m, sd = paired(out, tone_arm(b), tone_arm(other), key, b)
                parts.append(f"vs tone_b{int(other)} {m:+.3f}±{sd:.3f}")
            print(f"       b{int(b):>2}: " + "   ".join(parts))
        diag = [last(out, tone_arm(b), key, b) for b in TONE_BANDS]
        print(
            "       diagonal values: "
            + ", ".join(f"b{int(b)} {v:.3f}" for b, v in zip(TONE_BANDS, diag))
            + f"   spread {max(diag) - min(diag):.3f}"
        )


def print_control(out):
    r"""The twelve non-tone bands, tone arm against tone arm.

    ``real`` is *not* a control for these arms: it is not halved outside the tone
    bands, so a difference there is expected and is the anchor, not a null.
    """
    arms = [a for a in out["arms"] if a != "real"]
    unmanip = [b for b in WELL if b not in TONE_BANDS]
    print("\n  2. internal control: the twelve non-tone bands, tone arm vs tone arm")
    print("     (`real` is not halved there, so it is the anchor and not a null)")
    for key in ("real", "mimic"):
        worst, at = 0.0, None
        for b in unmanip:
            for i, a in enumerate(arms):
                for other in arms[i + 1 :]:
                    m, sd = paired(out, a, other, key, b)
                    z = abs(m) / max(sd, 1e-9)
                    if z > worst:
                        worst, at = z, (a, other, b, m, sd)
        print(
            f"     {key:<6} worst |delta| = {worst:.2f} sd, {at[0]} vs {at[1]} at "
            f"b{int(at[2])} (delta {at[3]:+.3f}, sd {at[4]:.3f}; "
            f"tolerance {TOL_SD:g} sd)"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--ckpt", type=int, default=500)
    ap.add_argument("--ckpt-fine", type=int, default=100)
    ap.add_argument("--ckpt-fine-until", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    bl = load_baseline()
    prof = bl.load_profile()
    pred = prof.load_pred()
    _, raw = pred.load_monash(pred.DATASET)
    series = [v for _, _, v in raw]
    patches, naive, ok, tr, ev = bl.build(K, CTX, series, "random")
    power = bl.band_power(patches, BANDS, K)
    bound = bl.band_bound(patches, BANDS, tr, ev)
    floor = {b: float((1 - bound[str(b)]) * p) for b, p in zip(BANDS, power)}
    betas = torch.tensor([math.sqrt(-math.log(bound[str(b)])) for b in BANDS])
    zeros = torch.zeros(len(BANDS))

    dose = TONE_E
    mimic = synth_corpus(power, betas, PROBE_N, seed=MIMIC_SEED)
    _, mimic_floor = probe_floor(bl, mimic)
    real_held = torch.from_numpy(patches[ev].reshape(len(ev), -1).astype(np.float32))
    mimic_x = torch.from_numpy(mimic.reshape(PROBE_N, -1).astype(np.float32))

    profiles = {"real": None}
    for b in TONE_BANDS:
        profiles[tone_arm(b)] = tone_profile(b, dose)
    if args.arms:
        profiles = {a: v for a, v in profiles.items() if a in args.arms}
    m = int(W * len(tr))

    print(
        f"corpus: {len(patches)} real windows ({len(tr)} train / {len(ev)} eval), "
        f"+{m} pure-tone windows ({W:g} per real) in every arm but `real`"
    )
    print(
        f"  tone windows carry power {dose:.4f} each, so a tone band's corpus power is"
        f" ~{dose * W:.4f}"
    )
    print(
        "  real marginal power P(b), bands 1..16: "
        + " ".join(f"{p:.2e}" for p in power)
    )

    out = {"config": vars(args), "dose": dose, "arms": {}}
    out["config"]["tone_bands"] = list(TONE_BANDS)
    for name, prof in profiles.items():
        t0 = time.time()
        base = power if prof is None else prof
        own = synth_corpus(
            base, betas if prof is None else zeros, PROBE_N, seed=OWN_SEED
        )
        own_x = torch.from_numpy(own.reshape(PROBE_N, -1).astype(np.float32))
        if prof is None:
            mix, tr_mix = patches, tr
        else:
            synth = synth_corpus(prof, zeros, m)
            mix = np.concatenate([patches, synth])
            tr_mix = np.concatenate([tr, np.arange(len(patches), len(patches) + m)])
        probes = [("real", real_held, floor), ("mimic", mimic_x, mimic_floor)]
        corpus_power = bl.band_power(mix[tr_mix], BANDS, K)
        print(f"\n== arm {name} ==", flush=True)
        per_seed = [
            train(bl, (mix, naive, ok), tr_mix, ev, s, args, probes, own_x)
            for s in args.seeds
        ]
        out["arms"][name] = {
            "corpus_power": [float(v) for v in corpus_power],
            "corpus_share": [float(v / corpus_power.sum()) for v in corpus_power],
            "marks": bl.mean_marks(per_seed),
            "per_seed": per_seed,
            "wall": time.time() - t0,
        }

    out["probe"] = {
        "power": [float(v) for v in power],
        "bound": bound,
        "floor": floor,
        "mimic_floor": mimic_floor,
    }
    print_setup(out)
    print_dose(out)
    for key, label in (("real", "real probe"), ("mimic", "mimic probe")):
        print_matrix(out, key, label)
    print(
        "\n== own probe: whole-window MSE on a fresh draw from the arm's own corpus =="
    )
    print("  (a pure tone is fully predictable: this should fall towards 0 if learned)")
    for a in out["arms"]:
        print(
            f"  {a:<9} "
            + " ".join(
                f"{int(m['step'])}:{m['own_mse']:.4f}"
                for m in out["arms"][a]["marks"]
                if m["step"] in MARK_STEPS
            )
        )
    print_schedule(out, "real", "real probe")
    print_diagonal(out)
    print_control(out)
    print("\n  crossing step of res/floor <= 2.0 on the real probe")
    arms = list(out["arms"])
    table(
        [f"    {'b':>4}", *[f"{a:>10}" for a in arms]],
        [
            [
                f"    {int(b):>4}" + ("*" if b in TONE_BANDS else " "),
                *[
                    (
                        f"{cross_step(out['arms'][a]['marks'], 'real', b):>10}"
                        if cross_step(out["arms"][a]["marks"], "real", b)
                        else f"{'-':>10}"
                    )
                    for a in arms
                ],
            ]
            for b in WELL
        ],
    )

    RUNS.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = RUNS / f"intervene_tone{tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
