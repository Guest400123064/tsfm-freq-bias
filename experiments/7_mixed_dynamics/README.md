# P14 — learning dynamics on one mixed corpus: is *speed* set by the band's power?

**Question.** P8, P12 and P13 all read retention at fixed budgets. The schedule claim is a
statement about *dynamics*, so this run makes it dynamic: **one corpus, mixed spectral shapes,
sixteen bands, a retention curve per band sampled every 1 000 steps.** Two predictions:

1. **Ordering** — the time a band needs to reach a retention threshold is ordered by that band's
   power share in the corpus (higher power → earlier).
2. **Time collapse** — rescaling each band's curve by its share, i.e. `r_b` as a function of
   `u = step × share_b`, puts every band on **one** curve. That is the schedule law in its strongest
   form: learning time inversely proportional to share, with a single universal curve.

## Corpus

Closer to a TSFM setting than any previous run: **one corpus, mixed shapes**. Each window draws its
own spectrum, `log P_b = log w_b + σ z_b` with `w_b ∝ b^(−α)`, so every window has a different shape
while the corpus marginal stays red. Per-window total power 0.5; every band's phase drawn
independently per window.

- **Why not average several fixed profiles**: the marginal of a convex mixture is flatter than its
  parts — the ten P13 profiles average to only a **4.4×** share spread, which weakens the ordering
  test. Jittering around a red mean keeps the marginal red (measured spread **~15×** over b1..b15).
- **Why α = 1 and σ = 0.8**: with α = 2 the weakest included band sits near the **≈0.015 absolute
  emission floor** measured in P7, which inflates its early retention and would *invert* the
  ordering. α = 1 puts the weakest band at share ≈ 2 %, four times above the floor-dominated regime.
- **Phases are randomised per window per band**, as in every run here. A phase fixed across windows
  would let the model emit a standing tone **without reading the context**, turning "copy what the
  context shows" into a degenerate task; per-window randomisation makes the context the only source
  of phase information.
- **Probe**: clean, carrying the corpus's **mean** spectrum, so the probe's per-band power is
  proportional to the band's marginal share (the same shape-matched convention as P7/P8/P12/P13).

Config: `ctx 512 / k 32 / hidden 64 / 2L / 4 heads / SGD 1e-2 / batch 64`, 3 seeds, 20 000 steps,
checkpoint every 1 000.

## Pre-registered decision (written before the run)

1. **Capacity** — fit gate passed and the frozen-latent oracle ≥ 0.9 at the final checkpoint. (The
   oracle is *expected to start low* — the representation itself must learn the weak bands; that
   rise is the dynamic form of P12's capacity finding and is reported as a curve, not a gate.)
2. **Ordering** — `Spearman(log share_b, t80_b) ≥ 0.8`, where `t80_b` is the first checkpoint at
   which band `b` reaches `r = 0.8`, censored at 40 000 if never reached.
3. **Time collapse** — with `u = step × share_b`, all points with `r < 0.9` fall on one curve:
   within-bin spread of `r` ≤ 0.15. The restriction to the rising regime is pre-registered because a
   band that has not converged by 20 000 steps leaves the common asymptote, and the collapse would
   then fail for a reason that is not the schedule.

Verdicts: gate/oracle failure → inconclusive; 2 failing → speed is not ordered by power; 2 holding
but 3 failing → power sets the order but not one time constant; both holding → **learning speed is
set by power share through one universal curve**.

b16 (Nyquist) is excluded from every statistic, as in P12/P13.

## Status

Running: 3 seeds × 20 000 steps, checkpoint every 1 000.
