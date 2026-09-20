r"""T5 step 3: does raising a band's marginal power share speed up its learning?

The baseline closed the headroom question *for its own corpus*: at 10 000 steps
the model already sits at or below the per-band linear rule at 15 of 16 bands,
so there is nothing left to win there. What is left is the **schedule** -- the
share-to-learning-speed law the synthetic programme measured (P8/P12/P13/P15),
which says a band's loss share, not its frequency, sets how fast its retention
climbs. This run asks whether that law survives on a real corpus under a real
probe.

Design. The real windows are never touched. To each arm's training half we add
the same number of synthetic windows -- half as many as the real training
windows, in every arm -- built by ``make_pm_profile`` so that band ``b`` carries
a chosen mean power and the fraction of it any function of the context can
recover is ``exp(-beta_b**2)``, matched band by band to the corpus's own measured
``P(b)``. Power and predictability are separate knobs in that generator, so the
added windows move power and nothing else:

  real          the real training half alone (the baseline anchor);
  dilute        every synthetic window given the real profile exactly, so the
                mixture's marginal *is* the real marginal and only dilution and
                the change of source move;
  bal_mid       fraction 0.4 of each synthetic window's matched-band budget
                spread equally over the bands the real corpus leaves quiet;
  bal_hard      all of it, the flattest marginal pure addition can build.

Each synthetic window is given the real corpus's total matched-band power, so an
arm can never win by carrying more energy than the real windows -- only by
putting it in different bands. ``dilute`` is the control: it doubles the corpus
exactly as ``bal_*`` do, with the marginal held fixed, so ``bal_* - dilute`` is
the marginal's effect and not "more data"'s.

Reading. ``floor_b = (1 - P(b)) * P_b`` is what an optimal linear predictor from
the band's own history leaves on the **real held-out windows**; it is a property
of the probe, so it is the same number in every arm. ``res_b / floor_b`` starts
above 1 at the weak bands and falls towards 1 as the model catches up with that
rule, so its *decline rate* is the learning speed. Pre-registered, before the
run:

  1. the manipulation landed -- each ``bal_*`` corpus marginal is flatter than
     ``dilute``'s, and its quiet bands carry far more power;
  2. dilution is not a free lunch -- ``dilute`` is no better than ``real`` at any
     band (doubling the corpus with windows of a different source leaves the real
     windows at half weight, so this is one-sided and not an equality);
  3. at the bands a ``bal_*`` arm lifts, ``res/floor`` declines faster than in
     ``dilute``, which holds the corpus size, the real-window count and the
     synthetic windows' total energy fixed -- the marginal's effect alone. The
     ordering of decline rates should follow the ordering of the corpus share,
     and whether a ``bal_*`` arm also beats the undiluted ``real`` is reported
     band by band rather than predicted, since that contrast mixes the marginal's
     gain with dilution's cost;
  4. at 10 000 steps the arms agree -- power writes a schedule, not a ceiling.

A band that fails (3) falsifies the transfer; a corpus where (2) fails is not a
controlled comparison and is reported as such.

Run from the repo root:
``.venv/bin/python experiments/10_real_profile/scripts/intervene.py``.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from fbias.data import make_pm_profile  # noqa: E402

BASE_PATH = Path(__file__).resolve().parent / "baseline.py"
RUNS = ROOT / "experiments/10_real_profile/runs"

K, CTX = 32, 512
BANDS = tuple(float(b) for b in range(1, 17))
FREQS = torch.tensor([b * CTX / K for b in BANDS])
W = 0.5  # synthetic windows per real training window, every arm
LAMBDA = {"bal_mid": 0.4, "bal_hard": 1.0}  # flat-profile fraction, see below
SYNTH_SEED = 4242  # the same phases in every arm: only the profile differs
CROSS = 2.0  # res/floor level whose crossing step is the speed reading
MARK_STEPS = (100, 200, 500, 1000, 2000, 5000, 10000)
WELL = BANDS[:-1]  # b16 is Nyquist: its matched filter is its own mirror


def load_baseline():
    spec = importlib.util.spec_from_file_location("bl", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bl"] = mod
    spec.loader.exec_module(mod)
    return mod


def synth_profile(p_real, lam):
    r"""The synthetic marginal: the real profile blended towards a flat one.

    Every synthetic window is given the real corpus's **total** matched-band
    power, so only the distribution over bands moves and an arm never wins by
    simply carrying more energy than the real windows do. Fraction ``lam`` of
    that budget is spread equally over the bands the real corpus leaves below
    the equal-split level ``sum(p_real) / 16`` -- bands 3..16 here -- and the
    rest keeps the real profile. ``lam = 0`` is the dilution control: the
    synthetic marginal *is* the real one, so the mixture's marginal is
    unchanged and only the source of the windows differs.
    """
    s = p_real.sum()
    quiet = p_real < s / len(p_real)
    quiet[-1] = False  # b16 is Nyquist: its matched filter reads twice its power
    flat = np.where(quiet, s / quiet.sum(), 0.0)
    return (1 - lam) * p_real + lam * flat


def synth_corpus(profile, betas, n, seed=SYNTH_SEED):
    r"""``(n, n_patch, k)`` phase-diffusing windows with per-band power ``profile``."""
    x, _ = make_pm_profile(FREQS, profile, betas, n, CTX, K, seed)
    return x.reshape(n, -1, K).numpy()


def check_predictability(bl, profile, betas, n=1024):
    r"""Measured per-band ``P(b)`` on a synthetic corpus, by the baseline's own ridge.

    The claim that the added windows move power and nothing else rests on
    ``exp(-beta_b**2)`` actually coming back out of the estimator the real
    corpus is measured with, so it is checked here rather than assumed.
    """
    x = synth_corpus(profile, betas, n, seed=SYNTH_SEED + 1)
    return bl.band_bound(x, BANDS, np.arange(n // 2), np.arange(n // 2, n))


def add_ratios(marks, floor):
    for m in marks:
        for b, f in zip(BANDS, floor):
            m[f"ratio@{b}"] = float(m[f"res@{b}"] / f)
    return marks


def cross_step(marks, b, level=CROSS):
    r"""First checkpoint where ``res/floor`` at ``b`` is at or below ``level``."""
    for m in marks:
        if m[f"ratio@{b}"] <= level:
            return int(m["step"])
    return None


def band_speed(rec, b):
    r"""Decline rate of ``res/floor``: log drop from the first mark to ``CROSS``.

    Read between the first checkpoint and the crossing, because ``res/floor``
    is not separable in step and a fitted slope over a curve that is still on
    the floor would be describing the floor. Larger is faster.
    """
    first = rec["marks"][0]
    t = cross_step(rec["marks"], b)
    if t is None or first[f"ratio@{b}"] <= CROSS:
        return None
    return float(
        math.log(first[f"ratio@{b}"] / CROSS) / max(t - first["step"], 1) * 1000
    )


def table(head, rows):
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def print_manipulation(out):
    print("\n== manipulation check: corpus marginal, per band ==")
    arms = list(out["arms"])
    ref = out["arms"].get("real", {"corpus_power": out["probe"]["power"]})
    ref = ref["corpus_power"]
    print("  power in normalized-window units; other columns are relative to real")
    table(
        [
            f"  {'b':>4}",
            f"{'period_h':>8}",
            f"{'real power':>11}",
            *[f"{a:>10}" for a in arms],
        ],
        [
            [
                f"  {int(b):>4}",
                f"{K / b * 0.5:>8.2f}",
                f"{ref[j]:>11.5f}",
                *[f"{out['arms'][a]['corpus_power'][j] / ref[j]:>10.3f}" for a in arms],
            ]
            for j, b in enumerate(BANDS)
        ],
    )
    print("  corpus marginal range (the arm's own band 1 over its weakest band 2..15):")
    for a in arms:
        cp = out["arms"][a]["corpus_power"]
        print(f"    {a:<9} {cp[0] / min(cp[1:15]):>8.1f}x")
    print("  corpus share per band (b1..b4, then b15):")
    for a in arms:
        print(
            f"    {a:<9} "
            + " ".join(f"{s:.4f}" for s in out["arms"][a]["corpus_share"][:4])
            + " ... "
            + f"{out['arms'][a]['corpus_share'][14]:.4f}"
        )


def print_schedule(out):
    arms = list(out["arms"])
    marks = {a: {m["step"]: m for m in out["arms"][a]["marks"]} for a in arms}
    steps = [s for s in MARK_STEPS if s in marks[arms[0]]]
    print("\n== res/floor by step, the weak bands ==")
    for b in (6.0, 8.0, 12.0, 15.0):
        print(f"\n  band {int(b)} (period {K / b * 0.5:.1f} h)")
        table(
            [f"    {'step':>6}", *[f"{a:>11}" for a in arms]],
            [
                [
                    f"    {s:>6}",
                    *[f"{marks[a][s][f'ratio@{b}']:>11.2f}" for a in arms],
                ]
                for s in steps
            ],
        )


def print_verdict(out):
    arms = list(out["arms"])
    last_step = out["arms"][arms[0]]["marks"][-1]["step"]
    print("\n== pre-registered checks (b16 excluded throughout: Nyquist) ==")
    last_step = out["arms"][arms[0]]["marks"][-1]["step"]
    last = {a: out["arms"][a]["marks"][-1] for a in arms}
    paired = "dilute" in arms and "real" in arms
    if paired:
        print(f"  2. dilute vs real: res/floor at {last_step} steps, per band")

        def gap(b):
            return last["dilute"][f"ratio@{b}"] - last["real"][f"ratio@{b}"]

        table(
            [
                f"  {'b':>4}",
                *[f"{a:>12}" for a in arms],
                f"{'dil-real':>12}",
            ],
            [
                [
                    f"  {int(b):>4}",
                    *[f"{last[a][f'ratio@{b}']:>12.3f}" for a in arms],
                    f"{gap(b):>+12.3f}",
                ]
                for b in WELL
            ],
        )
    else:
        print(f"  2. res/floor at {last_step} steps, per band ({', '.join(arms)})")
        table(
            [f"  {'b':>4}", *[f"{a:>12}" for a in arms]],
            [
                [
                    f"  {int(b):>4}",
                    *[f"{last[a][f'ratio@{b}']:>12.3f}" for a in arms],
                ]
                for b in WELL
            ],
        )
    print("\n  3. crossing step of res/floor <= 2.0 (blank = never crossed)")
    table(
        [f"  {'b':>4}", *[f"{a:>10}" for a in arms]],
        [
            [
                f"  {int(b):>4}",
                *[
                    (
                        f"{cross_step(out['arms'][a]['marks'], b):>10}"
                        if cross_step(out["arms"][a]["marks"], b)
                        else f"{'-':>10}"
                    )
                    for a in arms
                ],
            ]
            for b in WELL
        ],
    )
    print("\n  decline rate of res/floor, per 1000 steps (larger is faster)")
    table(
        [f"  {'b':>4}", *[f"{a:>10}" for a in arms]],
        [
            [
                f"  {int(b):>4}",
                *[
                    (
                        f"{band_speed(out['arms'][a], b):>10.2f}"
                        if band_speed(out["arms"][a], b)
                        else f"{'-':>10}"
                    )
                    for a in arms
                ],
            ]
            for b in WELL
        ],
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
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
    floor = np.array([(1 - bound[str(b)]) * p for b, p in zip(BANDS, power)])
    betas = torch.tensor([math.sqrt(-math.log(bound[str(b)])) for b in BANDS])

    shapes = {"real": None, "dilute": synth_profile(power, 0.0)}
    for name, lam in LAMBDA.items():
        shapes[name] = synth_profile(power, lam)
    if args.arms:
        shapes = {a: v for a, v in shapes.items() if a in args.arms}
    m = len(tr)

    cal = check_predictability(bl, shapes["dilute"], betas)
    print("\n== predictability calibration: beta_b = sqrt(-ln P(b)) ==")
    table(
        [f"  {'b':>4}", f"{'target P':>9}", f"{'measured':>9}", f"{'diff':>8}"],
        [
            [
                f"  {int(b):>4}",
                f"{bound[str(b)]:>9.4f}",
                f"{cal[str(b)]:>9.4f}",
                f"{cal[str(b)] - bound[str(b)]:>+8.4f}",
            ]
            for b in WELL
        ],
    )

    print(
        f"corpus: {len(patches)} real windows ({len(tr)} train / {len(ev)} eval), "
        f"+{m} synthetic ({W:g} per real window) in every arm but `real`",
        flush=True,
    )
    print(
        "  real marginal power P(b), bands 1..16: "
        + " ".join(f"{p:.2e}" for p in power)
    )
    for name, shape in shapes.items():
        if shape is not None:
            print(
                f"  {name:<9} synthetic marginal (total {shape.sum():.4f}): "
                + " ".join(f"{s:.4f}" for s in shape)
            )

    out = {"config": vars(args), "calibration": cal, "arms": {}}
    for name, shape in shapes.items():
        t0 = time.time()
        if shape is None:
            mix, tr_mix = patches, tr
        else:
            synth = synth_corpus(shape, betas, m)
            mix = np.concatenate([patches, synth])
            tr_mix = np.concatenate([tr, np.arange(len(patches), len(patches) + m)])
        corpus_power = bl.band_power(mix[tr_mix], BANDS, K)
        print(f"\n== arm {name} ==", flush=True)
        per_seed = [bl.train((mix, naive, ok), tr_mix, ev, s, args) for s in args.seeds]
        marks = add_ratios(bl.mean_marks(per_seed), floor)
        out["arms"][name] = {
            "corpus_power": [float(v) for v in corpus_power],
            "corpus_share": [float(v / corpus_power.sum()) for v in corpus_power],
            "marks": marks,
            "wall": time.time() - t0,
        }

    out["probe"] = {
        "power": [float(v) for v in power],
        "bound": bound,
        "floor": [float(v) for v in floor],
    }
    print_manipulation(out)
    print_schedule(out)
    print_verdict(out)

    RUNS.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = RUNS / f"intervene{tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
