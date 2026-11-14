# Map complexity and learning schedules

## TLDR

Design scaffold for the third experiment. It isolates the complexity of the next-step
phase map from carrier frequency: higher Fourier-harmonic order `m` should require more
steps even when carrier set, total power, and modulation strength are fixed. No code or
run exists yet.

## What it does

Each carrier is a constant-envelope phase oscillator sampled at every time step:

\[
x_i(t)=A_i\cos\theta_i(t),
\qquad
\theta_i(t+1)=\theta_i(t)+\omega_i+
\sum_{k=1}^{m}\gamma_k\sin(k\theta_i(t)+\phi_k).
\]

- `ω_i` sets the mean carrier frequency.
- `m` is the discrete complexity level: 0 is a pure rotation; 1, 2, and 3 add
  nonlinear phase harmonics.
- Phase advances are defined per timestep, independent of patch boundaries.
- The amplitude is constant, so total power does not change with `m`.
- Harmonic amplitudes are normalized as `γ_k = β√(2/m)` so total modulation RMS is
  fixed at `β` across levels. Candidate values are `β=0.4` and
  `m ∈ {0,1,2,3}`; these are pending a data-only stability check.

Proposed model and sampling:

- 8 carriers uniformly covering `(0,8)` cycles/patch.
- `k=16`, `ctx=512`, `H=64`, two layers, four heads.
- 32,768 steps, batch 128, checkpoints every 128 steps.
- Each `m` appears on two carriers per run. Pairings rotate across three runs so that
  no carrier is permanently tied to one complexity level.

The primary readout is full next-patch waveform error and variance explained. A
per-band carrier coefficient is not appropriate: phase modulation moves energy into
sidebands, so the carrier bin no longer summarises the signal.

## Energy spreading

Pure phase modulation preserves total power but changes the spectrum. Before training,
a data-only diagnostic will report for every `m`:

- 95% occupied bandwidth;
- spectral entropy;
- power outside the original carrier bin.

If bandwidth increases strongly and monotonically with `m`, the result will be reported
as “nonlinear phase-map complexity and its induced bandwidth”, not as a pure
complexity effect. A bandwidth-matched variant will be added only if the pilot shows
that the confound dominates.

## Blog visualisation

This section is a good candidate for an interactive HTML figure: a slider over `m`
showing the same carrier becoming progressively less tone-like as harmonics are added.
The asset will live under `docs/assets/html/2027-04-30-disambiguating-freq-bias-in-tsfm/`
if the experiment survives.

## Results

Pending. Implementation waits for design review.

## How to run

Not runnable yet. The Makefile, signal generator, diagnostics, and training script will
be added after the map family and `β` are confirmed.
