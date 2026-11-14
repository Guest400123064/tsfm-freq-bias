# SNR and learning schedules

## TLDR

This experiment tests whether full-spectrum observation noise increases the number of
steps needed to learn deterministic tone components, and whether carrier frequency
matters once power and SNR are held fixed. No run has been started.

## What it does

Each run uses the same dense 32-band model as the power experiment: `k=64`,
`ctx=512`, `H=128`, two layers, four heads. Frequencies are drawn with the same
conditioned jittered grid. Training adds independent per-sample Gaussian noise to both
context and target. Noise is never synthesized from a window's own FFT coefficients.

`snr` suite: all tones have equal power. Sweep noise amplitude
`sigma_rel ∈ {0, 0.25, 0.5, 1.0}`, where `sigma_rel` is noise RMS divided by signal
RMS. Carrier frequency and signal power are constant, so every band has the same
nominal SNR.

`assignment` suite: hold the same power multiset and noise variance, but mirror the
power-frequency assignment.

- `red`: high power at low frequency, low power at high frequency.
- `anti_red`: the reverse.

Noise is calibrated so the weakest level has in-band SNR 1 in both assignments. Any
difference between matched levels is therefore attributable to carrier placement, not
power or SNR.

For both suites, evaluate two probes:

- clean context -> clean target: measures learned deterministic signal coefficients;
- noisy context -> clean target: measures denoising and finite-context estimation.

Metrics are per-band `r` and `Δk` from joint cos/sin coefficient recovery, plus
variance explained against the clean target.

Planned controls, not yet implemented:

- a scale-matched noiseless arm at each noise dose;
- an analytic linear-Gaussian reference for the clean-probe attenuation floor.

## Results

Pending.

## How to run

```bash
make -j3 snr             # three runs, all noise doses
make -j3 assignment      # three runs, red vs anti-red
make                     # both suites
make clean
```

Each suite writes per-run JSON under `outputs/` and a merged file for plotting.
