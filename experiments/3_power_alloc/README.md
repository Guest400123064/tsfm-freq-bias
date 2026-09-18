# P8 — band power allocation, on its own

**Question.** Does a band's **share of the spectrum's energy** change how much of it the model
retains, when nothing else varies? This is Fredformer's stated mechanism (PLAN.md §2.4), and it is the
one step of the programme the user fixed before this run: *control total power, move only the
allocation.*

**Why it is not a replication of their Case 1.** Their Case 1 is a three-component synthetic series
forecast with PatchTST for 50 epochs. PLAN.md §2.4 records what the paper does **not** state: the
component frequencies, the sampling rate, the amplitudes, whether total power was held fixed, the
model config, and the split. Appendix A documents only Case 2 — which is not synthetic at all. So there
is nothing to match. This run tests the **claim** (`−|Δ_k| ∝ P(ã_k)`) in a setup specified completely.

## Design

| | |
| :-- | :-- |
| Bands | three carriers `b = 2, 4, 8` (`f = 32, 64, 128` at `ctx 512 / k 32`) |
| Corpus | all three coherent, **no noise, no phase walk**; total power `0.5` in every arm |
| Consequence | every band is perfectly predictable, so the Bayes optimum is `r* = 1` at every band and **any shortfall is a failure to fit, not a correct shrink** |
| Axis | one band carries `P_w = 0.5 / (2ρ + 1)`, the other two carry `ρ·P_w` each; `ρ ∈ {5, 25, 100, 1000}`, plus an equal-power `ρ = 1` reference |
| Rotation | at each `ρ` the three arms are **permutations** of one another — same power multiset, different assignment to `b = 2, 4, 8` |
| Probe | clean and shape-matched to its own arm, per-band power ratio `1.0000` |
| Model | `ctx 512 / k 32 / hidden 32 / 2L / 4 heads / SGD 1e-2 / batch 64`, 3 seeds |
| Reported | `r` (ours, primary) and `Δ_k` (Fredformer's), both on the clean probe; the oracle on frozen latents; the marginal for the fit gate |

Integer carriers make the matched filters exact (`Δφ ≡ 0`) and orthogonal over a patch, so a loud band
cannot leak into a weak one. That is measured in the generator check, not assumed.

## Pre-registered decision (written before the run)

Let `r` be a band's clean-probe retention.

- **Power effect** — at fixed `ρ`, `gap(ρ) = mean r at the two loud bands − r at the weak band`,
  paired within an arm. Pooled over the three rotations.
- **Frequency effect** — each band's `r` pooled over the whole `ρ` family. Inside a family every band
  spends one arm weak and two loud, so the power effect **cancels exactly** in these means and any
  spread left is frequency.

Verdicts, in order:

1. any arm failing the fit gate (`var_explained ≤ 0.5`) → **inconclusive**;
2. `max_ρ gap(ρ) ≤ 0.05` → **no power effect: the energy account is not supported**;
3. a gap that does not follow the assigned power (band spread pooled over `ρ` > 0.05) →
   **frequency-locked**;
4. otherwise a gap > 0.05 that does follow the assignment → **power-locked**.

**Contract 2 is the binding constraint.** P7 read this same corpus (its `x25_b0`) at `r@8 = 0.51` after
2 000 steps and `0.958` after 10 000 — so a 2 000-step grid will read "power-locked" for reasons that
may be entirely convergence. Any deficit must therefore also be shown to shrink with steps, and the
oracle must be consulted before the deficit is attributed to a prior. **No knob is tuned after seeing
results.**

## Prediction, ours

`r ≈ 1` and `Δ_k ≈ 0` at every band, flat in `ρ`; the oracle at 1.000 in every cell. A deficit that
tracks whichever band is weak, **survives convergence**, and is **present in the oracle** would be the
energy prior Fredformer claims — and the first representation-side power effect observed in this
project.

## Reproduction anchor

The generator seeds are P7's, so arm **`x25_w2` is P7's `x25_b0` corpus and probe**. Seed 0 must read
`r@8 = 0.5091` exactly (verified in the smoke run before the grid was launched).

## Status

**2 000-step grid (13 arms × 3 seeds): DONE** — power-locked deficit (gap up to 0.72 at ρ = 100),
frequency spread 0.0012 (nil), oracle ≈ 1.000 in every cell (head-side), `Δ` at weak bands
0.52–1.12 bracketing Fredformer's 0.95. Pre-registered verdict: "power-locked" — with contract 2
noted as binding, since this is the transient reading. Full detail in `experiments/README.md`.

**10 000-step leg (ρ ≥ 25, 3 rotations × 3 seeds): DONE** — three regimes, one mechanism:
ρ=25 closes (r_weak 0.96, gap inside tolerance); ρ=100 half-learned (0.54, τ̂ ≈ 20–50k steps);
ρ=1000 below the SGD noise floor (r fell 0.53 → 0.30, wrong-phase emission signature). The oracle
reads ≈ 1.000 in **every** cell at 10k too, and the frequency spread stays ≤ 0.007 at both budgets.
Conclusion: power writes the learning **schedule**, never the **fixed point**; the deficit is
power-locked, head-side, and frequency-independent across four orders of magnitude of disparity.
Full detail in `experiments/README.md`. Open follow-up: per-ρ transients to fit τ(ρ) and locate the
noise-floor boundary.
