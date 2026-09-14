# Run log

Append-only record of what was run, what came out, and what it means.
Design contract: [`PLAN.md`](../PLAN.md). Per-experiment intent: `experiments/<id>/README.md`.

---

## 2026-09-13 — `0_init`: harness, smoke, first reproduction

Intent: [`0_init/README.md`](0_init/README.md).

### Built

| File | Role |
| :-- | :-- |
| `src/fbias/data.py` | `make_mixture` (sum of tones, per-window amp/phase, `amp_jitter`/`phase_jitter`), `b_of`, and the `PLAN.md` §3.4 `Δb ≥ 1` guard |
| `src/fbias/probes.py` | `matched_amp`, `retention`, `phase_error`, `band_component`/`band_r2`, `fit_oracle` (ridge on frozen `z`), `ring_dim12`, `plane_overlap` |
| `src/fbias/cli/train.py` | train CLI: shifted MSE on `ctx + k` windows, SGD, checkpoint + sidecar JSON |
| `experiments/0_init/scripts/smoke.py` | driver |
| `tests/test_probes.py` | closed-form matched filter, retention, `Δb` guard |

Also fixed a pre-existing `NameError` in `src/fbias/cli/__init__.py`: the module-level `registry`
annotation referenced `_SubParsersAction`, which is only imported under `TYPE_CHECKING`, so
*every* CLI invocation raised before dispatch. One `from __future__ import annotations` fixes it.

### Smoke — PASS

`ctx 512 / k 32 / hidden 32 / 2L / 4 heads / SGD 1e-2 / batch 64 / 2000 steps`, two tones
`b = 0.125, 8` (`f = 2, 128`), deterministic. Wall **47.3 s** on CPU, loss `1.3006 → 0.003558`.
Checkpoint + sidecar in `0_init/runs/`.

```
     f       b  dphi/pi        r_model   r2_model  |phase err|      r_oracle  r2_oracle
   2.0   0.125    0.250      1.0193±0.0574   0.9987      0.0137     1.0003     1.0000
 128.0   8.000    0.000      1.0026±0.0305   0.9988      0.0129     1.0001     1.0000
```

### Reproduction check — FAILED, and the specification was wrong

The README's recorded check (`r_{b=8} < r_{b=0.125}`) did not hold: 1.0026 vs 1.0193, a 1.7%
difference well inside the ±0.03–0.06 window-to-window spread. Neither band shrank, and the
oracle had no gap to recover (1.0000 vs the model's 0.9988).

This is **my spec bug, not a harness failure or evidence against C1**. The corpus was specified
as deterministic, and `PLAN.md` **C4** — which I put there from lab It.29-30 — says shrinkage
vanishes at convergence on deterministic corpora. The spec asked a deterministic on-support
mixture to demonstrate damping, which C4 forbids. The two sections of the plan contradicted
each other.

The metric is not the problem. An untrained model at the same position reads retention
0.269 / 0.129 and r² +0.034 / −0.123, so `r_f` discriminates.

### Diagnostic D1 — no damping on a deterministic on-support corpus

8 tones `b = 1..8`, 2000 steps, same config, loss `0.01637`:

```
   b        r       r2          (all on-support)
 1.000   0.9870   +0.9943
 2.000   0.9851   +0.9942
 3.000   0.9840   +0.9958
 4.000   0.9924   +0.9953
 5.000   0.9858   +0.9948
 6.000   0.9780   +0.9950
 7.000   0.9923   +0.9958
 8.000   0.9883   +0.9955
```

Confirms the C4 reading: flat across `b`, ~0.98–0.99, no band structure. Consistent with C4.

### Diagnostic D2 — damping on an off-support frequency

Train on 8 tones `b = 1..8`; probe on 8 tones `b = 0.5, 2..8` (same tone count, same
amplitudes, `b=1` swapped for the off-support `b=0.5`), loss `0.01637`:

```
     f       b  support        r        r2
     8   0.500      OFF   0.6593   -0.7780
    32   2.000       on   1.0702   +0.8323
    48   3.000       on   1.0210   +0.8587
    64   4.000       on   1.0510   +0.9096
    80   5.000       on   1.0017   +0.9808
    96   6.000       on   1.0173   +0.9363
   112   7.000       on   1.0132   +0.9292
   128   8.000       on   1.0176   +0.9374
```

The off-support band is damped to **0.66×** with band r² **−0.78** (worse than predicting the
mean), while every on-support band sits at ~1.0 with r² 0.83–0.98. **The harness detects
frequency-selective damping.** This is a miniature C5.

**Two confounds to note, neither resolved here:**

1. **Amplitude/scale mismatch.** A first version of D2 probed 3 tones against an 8-tone training
   corpus and read retention ~1.7 across the *on-support* bands. `make_mixture` gives every tone
   amplitude 1, so total RMS scales as `sqrt(n_tones)` (√(8/3) ≈ 1.63, matching the 1.7 observed).
   The probe corpus must match the training corpus's tone count and amplitude distribution. This
   is a concrete instance of the `PLAN.md` §3.3 marginal-control requirement, and the design must
   state it explicitly.
2. **`b = 0.5` is simultaneously off-support *and* the `Δφ = π` case.** Lab It.29-30 identified
   `Δφ = π` as the transient laggard, so this single run cannot separate coverage from the `Δφ`
   ordering. Separating them is `3_deterministic_dphi`'s job (vary `Δφ` at fixed coverage), and
   `6_coverage`'s job (vary coverage at fixed `Δφ`).

### Verdict — corrected after reading It.27 / It.29 directly

**The attribution above is superseded.** It.27's damping was measured on `monash-direct-h32.pt` and
`geom-rich-input.pt`, **neither of which was trained on the 2-tone mixture it was probed on**;
`geom-rich-input` is deterministic within a window and still erased the band at r² = −0.46. It.29's
null came from a model trained and probed on the *same* pure-tone corpus (0.988–0.998).

So the discriminator is the **training marginal**, not determinism. `0_init`'s corpus is
deterministic, structurally simple, and *is* the probe — it sits in It.29's regime, which is why it
showed no damping. The spec bug is that it combined It.27's probe signal with It.29's training
condition; that combination appears nowhere in the lab.

### Diagnostic D3 — the marginal ladder (`diag_marginal.py`)

Probe identical in every arm (`make_mixture([2.0, 128.0])`, seed 123, `b = 0.125, 8.0`); only the
training marginal varies. 3 seeds each, 2000 steps, `ctx 512 / k 32 / hidden 32`.

```
arm                   r@0.125            r@8.0        r2@8.0   probe mse  final loss
A  mixture[2,128]   1.0258±0.0102    1.0009±0.0018    +0.9982    0.0038    0.004181
B  broad32          0.0339±0.0158    0.0407±0.0167    −0.0230    1.0468    0.499700
C  mixture[16..128] 0.1878±0.0414    2.1750±0.0859    −0.5047    2.0658    0.016213
D  broad32 high     0.0502±0.0173    0.1523±0.0559    −0.0056    1.0570    0.496606
   untrained ctrl   0.2489±0.0650    0.1835±0.0553    −0.1142    1.4081          —
```

**Arms B and D never fit.** B's final loss 0.4997 against a target variance of 0.5021 is **+0.5%**
variance explained — the model outputs the window mean. Its `r ≈ 0.04` at both bands means "no
signal anywhere", not "high band erased". First appearance of contract §3.8.

Arm C's `r@8 = 2.175` is the amplitude-scale artifact: its 8-tone corpus has RMS 2.0 against the
probe's 1.0, and a perfectly-tracking model reads ≈2.0 (its `band r2 = −0.50` confirms the reading
is not meaningful retention). Second appearance of the trap now recorded as §3.9.

### Diagnostic D4 — is the broad arm merely under-trained? (`diag_fit.py`)

One seed per arm, 20k steps with checkpoints, probe RMS-matched to every training corpus:

```
arm                        step   var expl (held)   r@0.125   r@8.0   r2@0.125   r2@8.0   r8/r1
N   mixture[2,128]         2000        +0.993        1.0268   0.9987    0.9967    0.9984   0.973
N                          20000       +0.999        1.0019   1.0000    0.9999    0.9999   0.998
BR  broad32, freqs redrawn 2000        −0.011        0.0164   0.0266    0.0044    0.0209   1.621
BR                         20000       −0.005        0.0213   0.0441   −0.0109    0.0614   2.073
BF  broad32, freqs fixed   2000        +0.178        0.5005   0.3084   −0.5096    0.2087   0.616
BF                         10000       +0.529        0.9385   0.5692   −1.0703    0.2826   0.606
BF                         20000       +0.691        1.1378   0.8022   −1.1103    0.4125   0.705
```

1. **BR is not under-trained; it is unlearnable at this scale.** Variance explained runs −0.011 →
   −0.005 across 2000→20000 with held-out MSE ≈ held-out variance throughout: it converged to the
   mean predictor, the optimum for an unpredictable target. The D3-B reading is confirmed worthless.
2. **BF fits** (+0.691 at 20k, still rising: 0.178 → 0.388 → 0.529 → 0.691) and shows
   `r8/r1 = 0.705`, ~40× the seed spread. **But `band_r2` does not support a clean high-band-damping
   story**: `r@0.125 = 1.14` with `r2@0.125 = −1.11` means the low band is *over*-predicted and
   phase-wrong, so the ratio is as much low-band overshoot as high-band suppression. BF had also
   not converged.
3. The BR/BF gap localises the D3-B failure: it was **per-window frequency identification**
   (32 unknown frequencies redrawn every window), not marginal breadth. Contract §3.10.

### Open

- `1_pred_side` still has no corpus it can run on. BF is the closest candidate, but it needs
  training to convergence and the low-band overshoot explained before `r8/r1` carries the meaning
  the lab contrast implies.
- The lab's rich corpus draws frequencies from **12 named clusters** — a low-dimensional family —
  which is plausibly why the lab's model fit its marginal while a uniform redraw could not.
  Reproducing that structure is the untried option (§3.10).
- `PLAN.md` §4's transient metric (`r_f(t)`) is still not implemented beyond four discrete
  checkpoints.
- The `b = 0.5` / `Δφ = π` confound from D2 remains unseparated.
