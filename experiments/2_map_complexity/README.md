# Map complexity and learning schedules

## TLDR

Adding nonlinearity to the one-step phase map slows learning relative to a pure
rotation, and the slowdown is largest at the highest order once the carrier is in the
no-lock regime. With power, carriers, state dimension, and initialization fixed,
`m=0` reaches the highest final variance and `m=3` the lowest; `m=1` and `m=2` are
close.

## What it does

Each carrier is a constant-envelope phase oscillator sampled at every time step:

\[
x_i(t)=A_i\cos\theta_i(t),
\qquad
\theta_i(t+1)=\theta_i(t)+\omega_i+
\sum_{k=1}^{m}\gamma_k\sin(k\theta_i(t)+\phi_k).
\]

- `ω_i` sets the base carrier frequency.
- `m` is the discrete complexity level: 0 is a pure rotation; 1, 2, and 3 add
  nonlinear phase harmonics.
- Phase advances are defined per timestep, independent of patch boundaries.
- Harmonic amplitudes are normalized as `γ_k = β√(2/m)`, fixing RMS phase modulation
  at `β=0.1` across levels.
- Each window is RMS-normalized to total power 0.5. A circle map's invariant phase
  density need not be uniform, so constant instantaneous amplitude alone does not fix
  finite-window mean power.
- Phase recurrence is evaluated in float64; float32 accumulation creates spurious
  sidebands over long records.

Fixed carriers are `b = {2.5, 4.5, 6.5}` cycles/patch. They are shared by every seed
and every `m`. An earlier set included `b=0.5`, where `ω = 2πb/k ≈ 0.196` is comparable
to the maximum modulation `β√(2m) ≤ 0.245`; some harmonic-phase draws then satisfy
`ω + g(θ) ≤ 0`, the phase locks, and that carrier becomes a constant. At `b ≥ 2.5`,
`min(ω + g) ≥ 0.78`, so no carrier can lock; the realized rotation number is within 1%
of nominal. `make_phase_map` asserts this condition.

- `k=16`, `ctx=512`, `H=64`, two layers, four heads.
- 32,768 steps, batch 128, checkpoints every 128 steps.
- Each model contains all three carriers at one `m`; models are not mixtures of map
  orders. A mixed model would make full-waveform error impossible to attribute.
- Within a seed, `m=0,1,2,3` share carriers, initial phases, map phases, initial
  weights, and batch order. The three seeds vary phases, initialization, and batch
  order.
- Primary readout: full next-patch waveform MSE and variance explained. Phase
  modulation moves energy into sidebands, so a carrier-bin coefficient is not a
  sufficient readout.

## What is controlled, and what is not

| quantity | controlled? | mechanism |
| --- | --- | --- |
| total power | yes | constant envelope plus per-window RMS normalization |
| intrinsic state dimension | yes | one scalar phase per carrier; `m` adds harmonics, not state variables |
| carrier set | yes | three fixed carriers across all seeds and levels |
| map nonlinear order | manipulated | `m=0..3` |
| observed spectral bandwidth | **not fixed** | phase modulation creates sidebands; measured and reported |
| waveform complexity | changes with `m` | this is the treatment |

The experiment supports a bounded statement:

> Nonlinear phase maps slow learning relative to a pure rotation, and the slowdown is
> largest at the highest order in the no-lock regime. Adjacent orders `m=1` and `m=2`
> are not resolved.

It does **not** support a cross-carrier invariance claim. A phase-modulated signal is
not a single carrier line: most power may sit near a carrier, but sidebands change the
effective input spectrum. Claims about carrier frequency itself are handled by the
power and SNR experiments.

A standard external PM signal would admit a Bessel decomposition and carrier-independent
sideband fractions. The autonomous generalized circle map used here does not: its
nonlinear term feeds back into the phase, so there is no exact `J_n(β)` decomposition.
The pilot measures the realized spectrum instead.

### Which complexity is controlled

The intervention is on the **latent phase-transition map**:

\[
F_m:\theta_t\mapsto\theta_{t+1}
\]

Its Fourier series in `θ` has exactly `m` nonlinear harmonics. This is the formal
complexity axis.

It is not yet the function the model sees. There are three nested objects:

1. `F_m(θ)`: the latent map; exactly `m` harmonics.
2. `G_m(θ)=cos(F_m(θ))`: the next observable sample as a function of current phase.
   This is generally **not** `m`-sparse. For `m=1`, Jacobi–Anger already gives
   \(e^{i\beta\sin\theta}=\sum_n J_n(\beta)e^{in\theta}\), so the observed function
   has infinitely many coupled sidebands.
3. `H_m: (x_{t-L+1},...,x_t) \mapsto x_{t+1}`: the full history-to-next map that the
   model learns. It composes phase inference, `F_m`, and the observation function; it
   has no canonical `m`-term Fourier decomposition in the ambient `R^L` coordinates.

So circle maps are chosen for **control**, not because `m` is the Fourier rank of the
ambient target function: constant envelope fixes power, one scalar phase fixes the
intrinsic state dimension, and `m` changes the latent one-step rule. The induced
observable complexity is measured, not assumed. A linear mixture of pure tones would
instead add independent oscillators, changing state dimension and spectral support
along with the target.

`G_m` can be used as a diagnostic link between `F_m` and the observed next sample: its
Fourier/Bessel components show how a sparse latent map becomes a spectrally dense
observable function. It is not the full target map, because the model does not observe
`θ` directly.

## Data pilot

![data pilot](figs/data_pilot.png)

*Pilot at `b=1` so the carrier sits on an FFT bin; the experiment uses
`b={2.5,4.5,6.5}`. (a) Waveforms for `m=0..3`, vertically offset. (b) Phase-map
derivative over one phase period. (c) Spectra.*

| `m` | min derivative | phase RMS | spectral entropy | 95% bandwidth | RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.707 |
| 1 | 0.859 | 0.100 | 0.033 | 1.735 | 0.707 |
| 2 | 0.741 | 0.100 | 0.028 | 0.150 | 0.707 |
| 3 | 0.603 | 0.100 | 0.146 | 0.133 | 0.707 |

- All maps remain orientation-preserving (`min derivative > 0`). This checks
  invertibility, not phase advance; the generator separately asserts `min(ω+g)>0` for
  the actual carriers.
- Total RMS is exactly fixed at `sqrt(0.5)`.
- Neither 95% bandwidth nor spectral entropy increases monotonically with `m`. The
  learning result is therefore about the latent map, not simply the observed sideband
  width.

## Results

![map complexity](figs/map.png)

*Three independent harmonic-phase realizations. Thin curves are individual seeds and
thick curves are means. (a) Held-out normalized MSE. (b) Steps to normalized-MSE
thresholds; marks are seeds, dashes are means. (c) Final error.*

| `m` | variance at 4096 | final variance | final range | `t(NMSE≤0.02)`, seeds |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.9983 | 0.9997 | 0.9997–0.9998 | 384, 512, 384 |
| 1 | 0.9964 | 0.9994 | 0.9994–0.9995 | 640, 640, 512 |
| 2 | 0.9958 | 0.9989 | 0.9989–0.9990 | 640, 640, 640 |
| 3 | 0.9926 | 0.9964 | 0.9960–0.9968 | 768, 768, 768 |

- `m=0` is fastest and reaches the highest final variance; `m=3` is slowest and lowest.
  Final variance is monotone in `m`: `0.9997 > 0.9994 > 0.9989 > 0.9964`.
- Every nonlinear map is slower than the pure rotation at 4096 steps: `0.9964–0.9926`
  versus `0.9983`.
- The separation is modest at loose thresholds: at `NMSE≤0.05` every arm reaches by 256
  steps. At `NMSE≤0.02`, `m=0` takes 384–512 steps, `m=3` takes 768, while `m=1` and
  `m=2` are close (512–640). Do not overclaim an ordering between adjacent `m`.
- Seed variance is small; the three harmonic-phase realizations agree on the ordering.
  The earlier `b=0.5` run was different because phase locking turned one carrier into a
  constant; that run is not evidence about `m`.
- All arms improve through 32k steps; there is no evidence for a converged floor.

## Blog visualisation

The pilot is a good candidate for an interactive HTML figure: a slider over `m` can
show the same carrier becoming less tone-like while power stays fixed. A second view
can show the carrier-free phase deviation, which is more legible than the raw waveform.

## How to run

```bash
make data                # regenerate pilot JSON and figure
make smoke               # 256-step pipeline check
make -j3                 # full three-seed m sweep
make clean
```
