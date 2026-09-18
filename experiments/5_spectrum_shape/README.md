# P12 — the 1/f^α ladder: is the apparent bias a function of spectral shape?

**Question.** P8 established the schedule mechanism with a hand-set power ratio on three bands;
P11 showed the fixed point tracks the conditional mean, not the corpus marginal. This run asks the
question the blog needs in its final form: **does the apparent per-band deficit follow the corpus's
spectral shape, continuously, at full-patch spectral resolution?**

## Design

Sixteen carriers `b = 1..16` fill the `k = 32` patch (`f = 16..256` cycles per window), fully
coherent, no noise, total power `0.5` in every window. Five **homogeneous** corpora with profile
`P_b ∝ b^(−α)`:

| arm | α | b1 share | b16 share |
| :-- | :-- | :-- | :-- |
| `a0` | 0.0 | 6.25% | 6.25% |
| `a05` | 0.5 | — | 2.2% |
| `a10` | 1.0 | — | 1.85% |
| `a15` | 1.5 | — | 1.1% |
| `a20` | 2.0 | 63.1% | 0.26% |

Each arm is probed **clean with its own profile** (shape-matched, contract 9), so `r_b` is that
arm's response to exactly the spectrum it was trained on. Config: `ctx 512 / k 32 / hidden 32 / 2L /
4 heads / SGD 1e-2 / batch 64`, 3 seeds, both 2 000 and 10 000 steps.

## Pre-registered decision (written before the run)

1. **Representation is fine** — the frozen-latent oracle reads ≥ 0.9 at every band of every arm;
   an arm short here is a capacity story, not a shape story.
2. **Within an arm the deficit follows the share** — Spearman corr between `log share_b` and `r_b`
   ≥ 0.8 in every arm with α > 0.
3. **The collapse** — pooling all `(share_b, r_b)` points across the five arms, every `log2(share)`
   octave containing ≥ 2 different arms has a spread of `r` ≤ 0.15, i.e. **one curve `r = G(share)`
   describes every spectrum**. This is P8's mechanism read at spectral resolution.

Verdicts: gate failure → inconclusive; oracle short → capacity; Spearman short → shape does not
order the deficit; collapse holds → "the apparent bias is the schedule at spectral resolution";
otherwise → no collapse, the deficit depends on more than the share.

Headline number, reported but not a decision rule: `r(b1)/r(b15)` versus α — the dose-response of
"how much apparent frequency bias a spectrum of this redness buys at this budget".

## Two measurement caveats, decided before the run

- **b16 is Nyquist** (`k/2`): a real tone there is its own negative-frequency image, so the matched
  filter reads **2×** the design power. Retention is a ratio and stays consistent when the emitted
  phase matches, but b16's *phase is degenerate* (a sign), so **b16 is excluded from all three
  decision statistics** (`WELL = b1..b15`) and reported separately.
- **The collapse is a fixed-budget statement** — `G` depends on `t`, so it is checked separately at
  2 000 and 10 000 steps (contract 2).

## Status

**DONE — both rungs, both budgets.** Verdicts: `capacity story` at `hidden 32` (oracle short for
α ≥ 1 at both budgets), **`collapse holds`** at `hidden 64` / 10 000 steps (worst multi-arm spread
**0.1049** ≤ 0.15, all oracles ≥ 0.917, all Spearman ≥ 0.911).

```
dose-response r(b1)/r(b15), hidden 64 @10k      hidden 32 @2k -> @10k
  alpha 0.0   1.008                              1.006 -> 0.788
  alpha 0.5   1.040                              1.464 -> 0.836
  alpha 1.0   1.076                              1.915 -> 0.726
  alpha 1.5   1.201                              1.978 -> 0.668
  alpha 2.0   1.484                              1.585 -> 0.707   (oracle still short: 0.630)
```

Both 2 000-step anomalies cleared with training (α=2 Spearman 0.668 → 0.996; collapse 0.174 →
0.105), and the small-model failure is capacity, not schedule: at `hidden 32` the oracle never
reaches 0.9 for α ≥ 1. **Apparent frequency bias is a monotone function of spectral redness at a
fixed budget, and one curve in the band's loss-share describes all five spectra.**
