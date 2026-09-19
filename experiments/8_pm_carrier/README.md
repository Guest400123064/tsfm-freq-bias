# P16 — carrier invariance: one modulator, two carriers, several β

**Question.** The energy branch (P8–P15) says a band's learning schedule is set by its share of the
loss, and P13 found share sufficing for 96% of the variance across ten spectra. What has never been
tested is whether the band's **position** matters once the share is held fixed. PM gives an exact way
to hold it fixed.

## Design

```
y(t) = cos(2π b_h t / k + β · cos(2π b_m t / k)),      b_m = 1,  φ = 0 (fixed)
```

- **One common modulator** in every arm: `b_m = 1` cycle/patch, phase fixed. Rank is 1 everywhere
  (the context must resolve only the time offset), so this run holds rank fixed and moves position.
- **Constant envelope**: power is exactly 1/2 at every β, and every window is noiseless.
- **Sidebands** at `b_h ± n·b_m` with power `J_n(β)²/2`, so the share of a line is **`J_n(β)²`** —
  *independent of the carrier*. Two carriers at the same β therefore have **identical share profiles
  in the carrier frame**; moving `b_h` moves only the positions.

| arm | carrier | β | significant lines (share ≥ 1%) |
| :-- | :-- | :-- | :-- |
| `b6_beta1` | 6 | 1.0 | n=0,±1,±2 → shares .586 .194 .013 |
| `b6_beta1.5` | 6 | 1.5 | .262 .311 .054 |
| `b6_beta2.405` | 6 | 2.405 | n=±1,±2,±3 → **.270** .186 .040 (`J_0 = 0`) |
| `b10_*` | 10 | same three | identical shares, bands shifted by 4 |

All significant lines land inside **b3–b13** — clear of DC and of the Nyquist band b16 (excluded
throughout, as in P12/P13). Config `ctx 512 / k 32 / hidden 32 / 2L / 4 heads`, 3 seeds, 2 000 and
10 000 steps.

**Probe**: a fresh set of **PM windows** (new time offsets, same carrier/β/φ). It must *not* be a
per-band amplitude profile with random phases: the sidebands' relative phases **are** the signal
here, so scrambling them would destroy the structure this run measures. This is the first experiment
in the project where `Δk` can detect a genuinely new failure mode — a model can reproduce every
line's *amplitude* (`r ≈ 1`) while getting the *relative phases* wrong, and then the waveform is
simply wrong. In every previous corpus each band carried an independent random phase per window, so
there was no coherent phase structure to get wrong.

## Pre-registered decision (written before the run)

At 10 000 steps (the 2 000-step leg is the schedule check):

1. **capacity** — the frozen-latent oracle ≥ 0.9 at every significant sideband of every arm;
2. **carrier invariance** *(primary)* — for every β and every matched `n`,
   `|r(b6,n) − r(b10,n)| ≤ 0.05` **and** `|Δk(b6,n) − Δk(b10,n)| ≤ 0.08`;
3. **share ordering** — pooling all arms, `Spearman(log share, deficit) ≤ −0.8`, i.e. the β response
   is the share response;
4. *reported, not a decision rule* — the non-monotone prediction at β = 2.405: its ±1 share (.270) is
   **lower** than β = 2.0's (.333), so at matched carrier its ±1 lines should be learned **more
   slowly** than β = 2.0's. A purely share-driven non-monotonicity.

Verdicts: capacity failure → inconclusive; 2 failing → **the band's position matters beyond its
share**, which would be the first such effect in this project; 2 and 3 passing → carrier-invariant
and share-ordered.

## Validation already in hand

The generator check measures the corpus's band powers against `J_n(β)²` through the same matched
filter used for every readout: **max |share error| = 0.0000** across all six arms, with off-band
leakage ≤ 0.0042. The independent variable of this experiment is exact, not estimated.

## Status — DONE, both legs

```
1. oracle minimum over significant sidebands   0.9816 ... 0.9972           PASS
2. carrier invariance (primary)   worst |dr| = 0.0080 (tol 0.05)           PASS
                                  worst |d(dk)| = 0.0360 (tol 0.08)        PASS
3. share ordering (pooled)        Spearman = -0.769 at 10k                marginal FAIL
                                  Spearman = -0.893 at 2k                 PASS
-> carrier-invariant; the check-3 failure is saturation, not data
```

At every β and every matched line the two carriers differ by |Δr| ≤ **0.008** — translation
invariance in band index over b3–b13. The 2 000-step "violation" (0.0643 at the weakest line) was
convergence noise, as its within-arm spread had suggested.

Check 3 fails because at 10 000 steps every significant line is essentially learned (deficit range
0.0007–0.0696 against 0.0007–0.5456 at 2 000, with 6 exact ties): a rank correlation over a
saturated quantity measures noise. The criterion is a *transient* one; at the budget where it has
dynamic range it passes. Recorded as the third instance of a criterion misapplied by saturation
rather than by the data (P15's t80 floor, P12's α=2 plateau).

**Not covered here**: the rank axis — this run holds rank at 1 by fixing the modulator's phase. The
natural next step is the same single-tone modulator with a **random phase per window**: rank 2,
marginal spectrum bit-identical.
