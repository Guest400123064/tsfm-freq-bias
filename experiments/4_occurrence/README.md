# P11 — marginal-matched homogeneous vs heterogeneous allocation

**Question.** P8 showed the deficit is a *schedule* ordered by loss-share, never a frequency
property — but P8's corpus was homogeneous (every window the same allocation), which is exactly
what a real signal generating process fixes for you and what data mixing cannot change. So: with
the corpus **marginal held fixed**, does changing only the *distribution of allocation across
windows* change the bias? This is the experiment that turns P8's mechanism into data-mixing
guidance: is the target "flatten the marginal" or "give the target pattern enough loss-share"?

**Why the design looks this way.** With per-window total power fixed at 0.5, a band's corpus
marginal is algebraically identical to its occurrence-weighted share — so the two accounts cannot
be separated by moving one band's occurrence. The separation is whether the model **conditions on
the context-visible allocation**:

- **Conditional account** (ours): converged model copies any allocation that occurred in training,
  marginal irrelevant except through the schedule (each pattern's loss-share).
- **Marginal-prior account**: the model's per-band response scales with the corpus marginal
  regardless of context.

## Design

Three carriers `b = 2, 4, 8`, fully coherent, no noise, total power 0.5 per window. All
discrimination arms share the corpus marginal **m = (0.24, 0.24, 0.02)** exactly (verified in the
generator check, `target_marginal` column):

| arm | windows | probe |
| :-- | :-- | :-- |
| `x` | every window `(0.24, 0.24, 0.02)` — b8 moderate | same |
| `y4` | 4% × `(0, 0, 0.5)`, 96% × `(0.25, 0.25, 0)` | the loud window |
| `y6` | 6% × `(1/12, 1/12, 1/3)`, 94% × weak | the loud window |
| `y8` | 8% × `(0.125, 0.125, 0.25)`, 92% × weak | the loud window |
| `c` | every window `(0.245, 0.245, 0.01)` — P8's `x25_w2` corpus | same |
| `a` | every window flat `(1/6, 1/6, 1/6)` | same |
| `b` | uniform orbit of `(0.40, 0.07, 0.03)` — marginal exactly flat | one orbit pattern |

The ladder's key property: `q · R = m8 = 0.02` in every Y arm, so the occurrence-weighted
**schedule share** of b8 is 4% in `x` and **all** Y arms alike — schedule is matched by
construction, and only the fixed-point prediction varies. Marginal-prior account predicts
`r(b8 on loud probe) = m8 / R` = **0.04 / 0.06 / 0.08** at *every* budget; the conditional account
predicts `r → 1` at convergence for all three.

**Controls.** `a` (flat homogeneous, no bias expected) and `b` (flat heterogeneous — does
heterogeneity itself cost anything?); `c` anchors to P8 (`x25_w2` corpus; probe `(0.245, 0.245,
0.01)` reads like P8's ρ=25 arm).

## Pre-registered decision (written before the run)

At 10 000 steps, with `r_q` the retention of b8 on Yq's loud probe:

1. any arm failing the fit gate (`var_explained ≤ 0.5` on the pattern-conditional held set) →
   **inconclusive**;
2. `min(r_q) ≥ 0.9` → **conditional account**: marginal-matched heterogeneity is free of bias;
   the operative variable is occurrence × in-pattern power (loss-share); data mixing should target
   that, not the marginal;
3. `r_q ≈ m8/R` (within a factor of 2) → **a marginal prior exists**; flattening the marginal is
   necessary;
4. otherwise → intermediate; report the dose curve.

Contract 2 binds: the 2 000-step grid checks the schedule match (`x` and the Y arms should read
alike there); the fixed point is read at 10 000 steps. No knob is tuned after seeing results.

## Status

**2 000-step grid (7 arms × 3 seeds): DONE** — the schedule check passes exactly as
designed (x and all y arms read r@8 ≈ 0.83; q·R = m₈ holds, so the ladder isolates the
fixed point), controls clean (flat arms ≈ 0.995), and the marginal account's fixed-point
signature (~0.04–0.08 on the loud probes) is absent at every point of the ladder — the y
arms emit ~0.83 of a band whose marginal share is 2%. Reporting artifact noted: y4's
pure-b8 probe makes its b2/b4 retention cells a division-by-zero display artifact; the
decision uses r@8 only.

**10 000-step leg (x, y4, y6, y8 × 3 seeds): DONE — conditional account wins.** Every Y
arm converges to 0.97–0.98 on a probe band whose corpus marginal is 2% — 12–24× the
marginal-prior prediction (0.04/0.06/0.08), excluded at every ladder point. The dose curve
is flat (schedule matched by q·R = m₈, so all Y arms move together). Data-mixing rule:
target the loss-share of the patterns you want reproduced; flattening the marginal is
neither necessary nor the mechanism.

**Scale-up path (discussed, not yet designed):** the generator is N-band generic; the 16-band
version (`b = 1..16` filling the k=32 patch) needs new arm allocations (e.g. red ∝ 1/b² vs orbit
shifts in log-b) and attention to per-tone shares (flat arm → 1/32 share each), fit gate, and
budget.

## Sibling run: `scripts/occurrence16.py` (P11-16)

The same X/Y discrimination at **16 bands** (`b = 1..16`, filling the k=32 patch), marginal fixed
to the α=1 profile (`RED16`), target band b16 whose corpus share is **1.85%** — inside the
slow-schedule regime P8 characterised (share 2% → r ≈ 0.51 at 2k, 0.96 at 10k).

| arm | windows | probe |
| :-- | :-- | :-- |
| `x16` | every window `RED16` (b16 at 1.85% throughout) | `RED16` |
| `y16` | 10% of windows boost b16 to `R = 0.0926` (18.5% of the window), others ×0.830; 90% drop b16 to zero, others ×1.019 | the loud-window profile |

Marginals match exactly by construction (`0.9c + 0.1d = 1` for `b < 16`, `0.1R = RED16[15]`), and
b16's occurrence-weighted schedule share is 1.85% in **both** arms. Predictions at 10 000 steps:
conditional account `r(b16) → ~1`; marginal-prior account `r(b16) = m16/R = 0.10`.

**Nyquist caveat (decided before the run).** b16 = k/2 is its own negative-frequency image, so the
matched filter reads 2× the design power on the *power* columns (retention, a ratio, stays
consistent). The generator check therefore excludes b16 from the marginal-deviation statistic and
reports its ratio separately.

Status: both legs **DONE**, and with the target band moved to a well-conditioned band (b15) at the
S3 rung (`--target 15 --hidden 64`) the discrimination is clean:

```
              r@15            dk@15          oracle     2k -> 10k
  x15        0.9144±0.0173   0.2965±0.0134   0.9722     0.715 -> 0.914
  y15        0.9160±0.0146   0.1270±0.0071   0.9958     0.667 -> 0.916
  -> conditional account generalises  (marginal-account prediction 0.100)
```

The two arms converge to the same number to three digits from opposite occurrence structure, and
the concentrated corpus locks phase better (`dk` 0.127 vs 0.297). The original Nyquist-targeted
run's levels are unusable (x16 2.03 → 1.57, y16 1.01 → 1.09) though its direction agreed.
