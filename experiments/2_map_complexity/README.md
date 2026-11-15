# Map complexity and learning schedules

## TLDR

This experiment isolates next-step phase-map complexity from carrier frequency. In one
seed, four models share the same four carriers, initial phases, weights, and batch
order; only the harmonic order `m` changes. The data pilot and training smoke pass.
Full training has not yet completed.

## What it does

Each carrier is a constant-envelope phase oscillator sampled at every time step:

\[
x_i(t)=A_i\cos\theta_i(t),
\qquad
\theta_i(t+1)=\theta_i(t)+\omega_i+
\sum_{k=1}^{m}\gamma_k\sin(k\theta_i(t)+\phi_k).
\]

- `ω_i` sets the carrier frequency.
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

Fixed carriers are `b = {0.5, 2.5, 4.5, 6.5}` cycles/patch. They are shared by every
seed and every `m`.

- `k=16`, `ctx=512`, `H=64`, two layers, four heads.
- 32,768 steps, batch 128, checkpoints every 128 steps.
- Each model contains all four carriers at one `m`; models are not mixtures of map
  orders. A mixed model would make full-waveform error impossible to attribute.
- Within a seed, `m=0,1,2,3` share carriers, initial phases, initial weights, and
  batch order. The three seeds vary phases, initialization, and batch order.
- Primary readout: full next-patch waveform MSE and variance explained. Phase
  modulation moves energy into sidebands, so a carrier-bin coefficient is not a
  sufficient readout.

## What is controlled, and what is not

| quantity | controlled? | mechanism |
| --- | --- | --- |
| total power | yes | constant envelope plus per-window RMS normalization |
| intrinsic state dimension | yes | one scalar phase per carrier; `m` adds harmonics, not state variables |
| carrier set | yes | four fixed carriers across all seeds and levels |
| map nonlinear order | manipulated | `m=0..3` |
| observed spectral bandwidth | **not fixed** | phase modulation creates sidebands; measured and reported |
| waveform complexity | changes with `m` | this is the treatment |

The experiment therefore supports:

> With power, carriers, and phase-state dimension fixed, increasing the nonlinear order
> of the one-step phase map changes the learning schedule.

It does **not** support a cross-carrier invariance claim. A phase-modulated signal is
not a single carrier line: most power may sit near a carrier, but sidebands change the
effective input spectrum. Claims about carrier frequency itself are handled by the
power and SNR experiments.

A standard external PM signal would admit a Bessel decomposition and carrier-independent
sideband fractions. The autonomous generalized circle map used here does not: its
nonlinear term feeds back into the phase, so there is no exact `J_n(β)` decomposition.
The pilot measures the realized spectrum instead.

## Data pilot

![data pilot](figs/data_pilot.png)

*One carrier at `b=1` with identical initial phase. (a) Waveforms for `m=0..3`,
vertically offset. (b) Phase-map derivative over one phase period. (c) Spectra.*

| `m` | min derivative | phase RMS | spectral entropy | 95% bandwidth | RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.707 |
| 1 | 0.859 | 0.100 | 0.057 | 2.000 | 0.707 |
| 2 | 0.729 | 0.100 | 0.165 | 2.000 | 0.707 |
| 3 | 0.655 | 0.100 | 0.154 | 2.000 | 0.707 |

- All maps remain orientation-preserving (`min derivative > 0`).
- Total RMS is exactly fixed at `sqrt(0.5)`.
- The 95% bandwidth diagnostic saturates at about 2 cycles/patch for every nonlinear
  level, so bandwidth does not increase monotonically with `m`. Spectral entropy also
  is not monotone. The pilot does not support a simple “more sidebands” explanation.
- Power outside the original carrier bin is effectively 100% for every `m>0`; this is
  expected for phase modulation and is not a generator failure.

## Smoke test

`make smoke` trains all four `m` values for 256 steps on one seed. It checks data
generation, training, checkpointing, and evaluation only; its error levels are not a
result.

## Results

Full 32k-step training pending.

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
