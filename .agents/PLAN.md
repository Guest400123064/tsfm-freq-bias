# Plan — ICLR 2027 blog post

## Objective

Write a short, evidence-first blog that disambiguates "frequency bias" in time-series
forecasting. The post is not a new architecture paper and does not claim to solve the
problem. It should leave the reader with a cleaner vocabulary, controlled evidence
about which variable actually moves the error profile, and an explicit list of open
questions.

Target: 4–7 pages, readable in about 15 minutes. Core narrative and limitations should
survive even if the appendix and interactive figures are removed.

## Current claim

"Frequency bias" is used for at least three different objects:

1. **signal frequency** — where spectral energy sits;
2. **effective power / SNR** — how much coherent signal a band carries relative to
   what the model must predict;
3. **map complexity** — how hard the history-to-future transformation is, independent
   of where its carrier sits.

The current controlled evidence says:

> A large part of what looks like a fixed preference for low frequencies is a
> data-driven learning schedule. Power allocation and SNR move the time needed to
> reach a given error, not the identity of the band that is intrinsically preferred.
> Map complexity is a separate axis and must not be conflated with either.

Do not claim that architecture never matters. Do not claim an asymptotic result unless
the learning curves have plateaued.

## Experiment status

| group | question | status | headline evidence |
| --- | --- | --- | --- |
| `0_power_schedule` | does assigned power set the learning schedule? | complete | final `Δk` Spearman `−0.992` with log power, `−0.090` with frequency; `t ∝ P^-1.083`, `R²=0.999` |
| `1_snr_schedule` | does white noise slow learning, and is high frequency only a proxy for low SNR? | complete | `t(Δk≤0.5)` rises 723 → 3,119 steps as noise/signal RMS rises 0→1; red and anti-red level curves almost coincide |
| `2_map_complexity` | does nonlinear phase-map order alone change the schedule? | complete | nonlinear maps slower than pure rotation; final variance monotone in `m` (0.9997 / 0.9994 / 0.9989 / 0.9964); `m=0` vs `m=3` separated at `t(NMSE≤0.02)`, `m=1`/`m=2` close |

### Group 0 details

- 32 dense tones, fixed total power, no noise.
- Eight power levels, four bands each, random frequency-power pairing across three
  runs.
- Per-band coefficients recovered by joint cos/sin inversion.
- Per-run scaling exponents: `1.09`, `1.10`, `1.06`.
- Main limitation: fastest level reaches the first 128-step checkpoint, so the
  exponent should be refined with earlier probes before camera-ready.

### Group 1 details

- Equal-power SNR dose: clean-probe attenuation and time-to-threshold both worsen with
  noise.
- Red vs anti-red assignment swaps which carrier frequencies are loud while holding the
  power multiset and noise fixed. Matched levels differ by at most `0.017` in final
  `Δk`; frequency residual correlates `0.036` with error.
- Missing control: a scale-matched noiseless arm is still needed to separate noise from
  the larger input RMS at high noise.
- Clean-probe attenuation is a learned prior or finite-budget effect, not an
  information-theoretic floor: a clean input determines the tone.

### Group 2 details

- Three fixed carriers `{2.5, 4.5, 6.5}`, `k=16`, `ctx=512`, `H=64`, two layers.
  The earlier set included `b=0.5`; there `ω ≈ 0.196` is comparable to the modulation
  peak `β√(2m) ≤ 0.245`, so some harmonic-phase draws lock the phase (`ω+g ≤ 0`) and
  turn that carrier into a constant. At `b ≥ 2.5`, `min(ω+g) ≥ 0.78`; `make_phase_map`
  asserts this.
- Generalized circle map per timestep with `m` harmonics and fixed phase-modulation RMS
  `β=0.1`.
- One model per `m`; all carriers in a model share the same `m`. Within a seed the four
  models share carriers, initial phases, weights, and batch order.
- The experiment tests map order at fixed power and scalar phase-state dimension. It
  does not make a cross-carrier invariance claim.
- Circle maps do not obey the external-PM Bessel decomposition. Measure realized
  bandwidth and entropy rather than assuming sideband fractions.
- The treatment is the latent map `F_m: θ_t→θ_{t+1}`, whose Fourier series has exactly
  `m` harmonics. The observable map `G_m(θ)=cos(F_m(θ))` is generally dense, and the
  full history-to-next map `H_m` also contains phase inference. We therefore claim
  control of latent map order, not ambient `R^L` Fourier rank; circle maps are chosen
  for power and state-dimension control.
- Result (safe carriers): nonlinear maps are slower than the pure rotation, and final
  variance is monotone in `m`: `0.9997 > 0.9994 > 0.9989 > 0.9964` for `m=0..3`. At
  `t(NMSE≤0.02)`, `m=0` takes 384–512 steps and `m=3` 768; `m=1` and `m=2` are close
  (512–640). Seed variance is small. All arms still improve at 32k.
- The earlier `b=0.5` run produced a spurious non-monotone ordering because phase
  locking turned one carrier into a constant. Do not report those numbers.

## Immediate work

1. For group 2, report the monotone final-variance ordering and the `m=0` vs `m=3`
   separation; do not overclaim an ordering between `m=1` and `m=2`.
2. Build the group-2 blog figure around the learning curves and the `m=0` vs `m=3`
   separation. Note that the earlier non-monotone result was a locking artifact.
3. Add the scale-matched noiseless control for SNR before making a strong causal noise
   claim.
4. Re-run group 0 with early checkpoints for an unbiased scaling exponent.
5. Convert the real-data power spectrum plus noise floor into an illustrative step
   estimate. Keep it explicitly an extrapolation from synthetic controls.
6. Rewrite `README.md` from the current narrative; it is still an older draft.
7. Move final numbers and figures into the ICLR post under `docs/_posts/`.

## Scope and limitations

- Evidence comes from a small input-space `SimTFM`, not a production TSFM. The blog can
  motivate a re-evaluation of TSFM evaluation; it cannot claim that every pretrained
  checkpoint behaves this way without a checkpoint study.
- Group 1's clean attenuation is at a fixed 32k-step budget; training loss was still
  falling.
- Group 2 fixes carriers; carrier-frequency claims belong to groups 0 and 1.
- Raw experiment JSON is reproducible but gitignored. Final figures and summaries are
  committed.
- `experiments/_archive/` is historical material, not a maintained reproduction target.

## Source map

- `docs/README.md`: current positioning, related work, and narrative.
- `experiments/0_power_schedule/`: power schedule results.
- `experiments/1_snr_schedule/`: noise and red/anti-red controls.
- `experiments/2_map_complexity/`: map-order experiment and pilot.
- `docs/_posts/`: ICLR post draft.
- `experiments/_archive/README.md`: exploratory run log.
