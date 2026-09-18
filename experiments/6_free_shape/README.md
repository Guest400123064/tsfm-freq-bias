# P13 — free spectral shapes: is loss-share a *sufficient* predictor of retention?

**Question.** P12's α ladder gave one curve `r = G(share)` across five power-law spectra, but a
power-law family has a built-in limitation: **share is perfectly collinear with band identity**
(b1 is always the loudest, b16 always the quietest), so "share" and "frequency" can only be pulled
apart by cross-arm comparisons at matched share — and those left a residual in the deepest octave.
This experiment breaks the collinearity directly, and adds the mirrored (violet) case.

## Design

Sixteen carriers `b = 1..16` filling the `k = 32` patch, fully coherent, no noise, total power 0.5
per window, homogeneous corpora (one profile per arm), each probed clean with **its own** profile.
`hidden 64` (P12 showed 16 bands need it), 3 seeds, both 2 000 and 10 000 steps.

| profile | what it is | band carrying the largest share |
| :-- | :-- | :-- |
| `v1`, `v2` | power laws α = −1, −2 — **violet**, high bands loud | b16 (excluded), b16 |
| `r1..r5` | random log-normal: `log P_b = 1.2 z_b`, `z ~ N(0,1)` | b11, b16, b11, b1, b5 |
| `bump1`, `bump2` | a random centre and width, everything else floored | b16, b7 |
| `flat` | the α = 0 anchor | — |

**Every profile is floored at a 1% share.** The floor is set by the readout, not by taste: P7
measured an absolute emitted-amplitude floor of ≈0.015 at a band with no content, so a band whose
true amplitude is `sqrt(2·0.5·share)` is floor-dominated below share ≈ 0.25%. The floor keeps every
included band four times above that. It also truncates the violet power laws at their bottom end.

The profiles are chosen so that **each band takes a turn carrying the largest share** — the loudest
band lands on b1, b5, b7, b11, b16 across the set — which is what a power-law family can never do.

## Pre-registered decision (written before the run)

1. **Capacity** — the frozen-latent oracle reads ≥ 0.9 at every band of every arm.
2. **Share is sufficient** — the isotonic (least-squares monotone) fit of `r` on `log share`, pooled
   over all arms and bands, has **R² ≥ 0.9**.
3. **Frequency adds nothing** — in equal-count share bins, the correlation between `r` and band
   index is **|corr| ≤ 0.5** in every bin with ≥ 8 points, and the within-bin spread of `r` is
   ≤ 0.15 (the same tolerance P12's collapse used).

Verdicts: 1 fails → inconclusive/capacity; 2 fails → share is not sufficient; 2 holds but 3 fails →
share plus a residual frequency dependence; both hold → **share is the sufficient statistic**.

b16 (Nyquist) is excluded from every statistic, as in P12.

## Status

**DONE — both budgets. Verdict at 10 000 steps: `share is the sufficient statistic`.**

```
1. oracle minimum (>= 0.9)                     0.9282 ... 0.9874    PASS
2. share-only monotone fit R^2 (>= 0.9)        0.9558              PASS
3. worst |corr(r, band)| in a bin (<= 0.5)     0.482               PASS
   worst within-bin r spread (<= 0.15)         0.088               PASS

post-hoc residual:  residual sd 0.0122,  corr(residual, band) = -0.070
                    (2k: 0.0205 and +0.084 -- the sign flips, i.e. noise)
```

One monotone curve in loss-share predicts retention across **150 points from ten structurally
different spectra** — violet power laws, random log-normals, random bumps, flat — in which the
loudest band lands on b1, b5, b7, b11 and b16. Within-arm band trends are unstable across budgets
(r3 +0.59 → −0.05), so the residual reads as transient rather than structural. At 2 000 steps the
verdict was `capacity` (two oracles at 0.882 / 0.891).
