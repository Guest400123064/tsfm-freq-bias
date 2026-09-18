# `1_realdata` — a corpus's per-band conditional predictability `P(b)`

**T1** of `PLAN.md` §6.2. Design contract: [`../../PLAN.md`](../../PLAN.md) §1.2, §3, §6.
Run log: [`../README.md`](../README.md). The bulk of this file is T1 and its follow-up
T1b: pure data analysis, whose product is the *explaining variable* of the whole claim.
**T2 — the fourth section — does train models** (6 runs on this corpus), and its
result contradicts T1b's prediction: the per-band retention is strongly sloped, not flat.
**T2b — the last section — is T2's shape-hypothesis control** (15 runs: the same corpus
spectrally flattened, the same probe), and it shows that slope was the corpus's spectral
*shape*, not its per-band predictability.

## Intent

`PLAN.md` §6.1 predicts that a next-patch model's per-band amplitude retention `r_b`
equals `sqrt(P(b))`, where `P(b)` is the fraction of the next patch's band-`b` content
that the observed context determines. T1 exists to measure `P(b)` **without training
anything**, so that T2 has a prediction to test.

Concretely, for the chosen corpus:

1. survey four Monash candidates and pick one, with the resolution stated;
2. state the window-corpus construction explicitly (`PLAN.md` contracts A3, A9);
3. estimate `P(b)` on a 32-band grid — own-band and full-context, on the real corpus,
   on a phase-randomized surrogate of it, and on the clock-folded residual;
4. report the power spectrum in the same band units, so energy and predictability can
   be compared (the Fredformer question, `PLAN.md` §2);
5. run the four morphology diagnostics of `PLAN.md` §6.3.

## Config

| | |
| :-- | :-- |
| corpus | `australian_electricity_demand_dataset` (Monash, already in the local HF cache) |
| resolution | **30 min** per sample. `b = k / period_samples`, `k = 32` |
| grid | 32 bands, `b = 0.5 … 16.0` cycles/patch, Nyquist `b = 16` |
| window | `ctx + k = 544` samples = 16 context patches + 1 target patch |
| period ↔ band | `T(h) = 16 / b`, so `b = 1` is 16 h, `b = 16` is 1 h |
| estimator | complex ridge, train/eval split by window, `λ` chosen inside train |
| cost | CPU only, **~60 s** total for the three T1 scripts, **+74 s** for the `k` sweep below |
| `k` sweep | `scripts/aperture.py` re-runs the estimator at `k ∈ {32, 64, 128}`; see the follow-up section |
| **T2** | `scripts/t2.py`: **6 trainings** (2 arms × 3 seeds, 20 k steps, 2 torch threads each), **1 640 s** wall for the whole grid in parallel, plus **13.5 s** for `--relation`; see the T2 section at the end |
| **T2b** | `scripts/t2b.py`: **15 trainings** (3 corpus variants × 2 apertures × 3 seeds, 20 k steps), **4 249 s** wall for the whole grid in three waves of 6/6/3, plus **80 s** for `--report` (the filter and `P(b)` diagnostics, no training); see the T2b section at the very end |

## Step 0 — survey

`.venv/bin/python experiments/1_realdata/scripts/survey.py` → `runs/survey.json`
(6.1 s). Every candidate is complete (no NaNs) and needs no download.

| dataset | series | length med (max) | interval | `b` of 24 h | windows @544 | scale ratio | start field |
| :-- | --: | --: | :-- | --: | --: | --: | :-- |
| `solar_10_minutes_dataset` | 137 | 52 560 | 10 min | 0.222 | 13 152 | 10.7 | ok (all 137 share `2006-01-01 00-00-01`) |
| **`australian_electricity_demand`** | **5** | **230 736** (232 272) | **30 min** | **0.667** | **2 122** | **9.24** | **metadata: `start` holds `NSW`/`VIC`/…** |
| `wind_4_seconds_dataset` | 1 | 7 397 147 | 4 s | 0.0015 | 13 597 | 1.0 | ok (`2019-08-01 00-00-03`) |
| `oikolab_weather_dataset` | 8 | 100 057 | 1 h | 1.333 | 1 464 | 5 841 | ok (all share `2010-01-01 00:00:00`) |

NaN fraction is exactly **0.0** in all four, and no series in any of them carries
missing values.

Two survey findings worth keeping:

- **The loader's `start` field is not always a start time.** In
  `australian_electricity_demand` the data line is
  `T1:NSW:2002-01-01 00-00-00:5714.0,…`, so `load_monash` returns `"NSW"` as the start
  (it is the series' state). All five real starts are `2002-01-01`, aligned, but the
  records end at different dates (three lengths: 230 736 / 230 784 / 232 272). The
  loader is used unmodified and the timestamp is recovered from the raw line.
- `solar_10_minutes` gives all 137 series the *same* start, and wind appends a 3 s
  offset (`00-00-03`) to a nominal midnight — both consistent with an hourly-to-second
  conversion upstream. They are internally consistent (starts land exactly on the
  sampling grid), so they are trustworthy; they are just not midnight.

### Why this corpus

- **The clock is in band.** `b = k / period_samples`, so at `k = 32` the grid reaches
  periods from 64 samples (`b = 0.5`) down to 2 samples (`b = 16`). The 24 h clock sits
  at `b = 32/48 = 0.667` — inside the grid's lowest band — and the 7 d clock at
  `b = 32/336 = 0.095`. `solar_10_minutes` puts its diurnal at `b = 0.222` (below the
  grid: only its harmonics are visible), `oikolab` puts it at `b = 1.333` (good, but see
  the next point), and `wind_4_seconds` puts it at `b = 0.0015` with **99.65 % of all its
  power below `b = 0.125`** — a wind window is one band wide, so there is no per-band
  contrast to measure at all.
- **Two incommensurate clocks (24 h and 7 d)** make the deterministic part *structured*
  rather than a single line: in the whole-record periodogram of series T1, 30 % of the
  power sits at `b < 0.125` and 50 % in `b ∈ [0.5, 1)`, with further coherent harmonics
  of the diurnal all the way to Nyquist.
- **A clean contrast with weather/measurement-driven content**: the mid bands carry
  temperature- and calendar-driven load structure, and the high bands carry the fast
  small-amplitude component, whose power is **460×** below the lowest band's
  (`3.24e-1` at `b = 0.5` vs `7.0e-4` at `b = 16`).
- **Statistics**: 2 122 non-overlapping `ctx + k` windows (5 series × 424–426). Thin
  but workable, and the five states are *different* series rather than one weather
  realisation seen five times.
- **The diagnostics are well posed.** `solar_10_minutes` was the closest rival on
  window count (13 152) and on weather-driven high-frequency content, but **50 % of its
  samples are exactly 0** (night). A band-limited "carrier" is then zero half the time,
  which makes Step 3's envelope and phase diagnostics degenerate and turns the high
  bands into a day/night mask. `oikolab` was rejected because its 8 series are 8
  climate variables **at one site**, so its 1 464 windows are far fewer than 1 464
  independent samples, and its per-series scale ratio is 5 841 (pressure in Pa vs
  humidity in 0–1).

## Step 1 — corpus construction

Every choice below is a contract-relevant decision, not an implementation detail.

| decision | choice | why |
| :-- | :-- | :-- |
| pooling vs one long series | **pool windows from all 5 series**, windowed independently | 5 series × 424 windows is 5× the data of any one state, and each series is long enough (13 years) that the marginal is not dominated by any single regime. Windows from the same state are serially correlated, so the *effective* sample size is nearer 424 than 2 122 — stated, not corrected (T2 will face the same corpus). |
| scale (contract **A9**) | **per-series centre and unit-variance over the whole series** | per-series std spans 147 → 1 362 MW (ratio 9.24); pooling raw would make the corpus's power spectrum the spectrum of NSW. Normalising on the *whole* series is not RevIN: the target patch's own scale is not removed, so no target information leaks into the context. |
| start times | **not aligned**; each series is windowed on its own index | pooling makes every window an i.i.d. draw from the marginal, and alignment would only matter for a multivariate corpus. |
| window stride | **544 samples (non-overlapping)** for 2 122 windows | overlapping windows would place the same samples in train and eval, and the train/eval split has to be a genuine held-out set. |
| NaN policy | **none needed** — NaN fraction is 0.0 everywhere | for a corpus with gaps, the policy would be to drop any window containing one, and to say so. |
| patch demeaning | **subtract each patch's own mean before the matched filter** | without it the patch's local level leaks into every non-integer band (measured: ×2.10 band power at `b = 0.5`, ×1.31 at `b = 15.5`). It leaves integer bands exactly unchanged, so it cannot alter the patch's DFT bins. |
| band grid | **32 bands, Δb = 0.5** | the brief's fine grid; `2b` is an integer at every centre so the matched filter is exact (`probes.py`). See the resolution trap below for what it costs. |
| de-clocking | subtract the per-series mean profile folded over **336 samples (7 d)** | one fold removes the 24 h profile, the 7 d profile and their interaction. It explains 61.9 % of the variance (`PLAN.md` §6.2's "after phase-folding out clock-locked components"). |
| split | random half of the **windows** (seed 0), `λ` selected on an internal half of train | non-overlapping windows make a random split leak-free. Train `P` is reported beside every eval `P` as an over-fitting check. |

## Step 2 — `P(b)`, and the two traps that set the design

`.venv/bin/python experiments/1_realdata/scripts/predictability.py` → `runs/predictability.json`
(42–47 s). Per band: band-limit each patch to `b`, matched-filter it to one complex
coefficient per (window, patch), ridge-regress the **target patch's** coefficient on the
context patches' coefficients — `own` = band `b` only, `full` = all 32 bands,
`last` = the newest patch only (the patch-local reference) — then
`P = 1 - residual_var / var(target band)`.

### Trap 1 — a 32-sample patch cannot isolate a band

The matched filter of a `k = 32` patch has a mainlobe `±0.5` cycles/patch, so a
rectangular-window patch coefficient at band `b` contains everything the aperture's
sidelobes pick up. The `in-band` column below is
`power(series band-limited to |b'-b| < 0.5) / power(raw patch coefficient)`: the fraction
of a band's measured power that is genuinely that band. On white noise (where nothing
is concentrated anywhere) the same statistic averages **0.774** — so the aperture's own
spread accounts for a lot of leakage and only the *shortfall* is a corpus effect.
Measured: `0.96` at `b = 1`, then `0.34 / 0.15 / 0.10` at `b = 2 / 4 / 8`. **The corpus's
clock-locked profiles carry 62 % of the variance and their sidelobes are comparable to
the high bands' own content**, so the high-band numbers below are dominated by leakage
into the bin, not by the band's own predictability. They are reported because they are
what a 32-sample-aperture model actually sees, and flagged because they are not a
per-band property of the signal.

### Trap 2 — pre-filtering the leakage away is degenerate

The obvious fix — band-limit the series to `|b' - b| < 0.25` first, then take patch
coefficients — **does not work, and fails silently**. A 544-sample window band-limited to
0.5 cycles/patch contains only `0.5 × 17 ≈ 8.5` complex degrees of freedom, while the
regression asks 16 complex context coefficients to predict the 17th. The coefficients
are linearly dependent by construction, so the ridge fits the train half exactly and the
held-out half too: **white noise reads `P_own = 0.998`** (printed as the oversampling
control). Two consequences that shape the design:

- the *only* way to make finer bands leakage-free is to pre-filter them, and that is
  degenerate; so the `Δb = 0.5` grid is reported together with its `in-band` column
  instead of being cleaned up, and no band is claimed to be a per-band measurement when
  `in-band` is small;
- the white-noise control above (raw patches, no pre-filter, `P_own mean = -0.002`,
  range `[-0.006, +0.000]`) is what shows the *delivered* estimator is not degenerate.

### `P(b)` — table A: the real corpus

`power` = mean `|Z|²/2` over every window and patch, in units of the corpus's variance.

| b | T (h) | power | frac | `P_last` | `P_own` | `P_full` | in-band |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 0.5 | 32.0 | 3.24e-1 | 0.304 | 0.265 | **0.894** | 0.937 | 0.894 |
| 1.0 | 16.0 | 4.32e-1 | 0.404 | 0.291 | **0.917** | 0.939 | 0.962 |
| 1.5 | 10.7 | 1.39e-1 | 0.130 | 0.276 | 0.882 | 0.923 | 0.845 |
| 2.0 | 8.0 | 6.13e-2 | 0.057 | 0.069 | 0.826 | 0.900 | 0.343 |
| 2.5 | 6.4 | 3.82e-2 | 0.036 | 0.433 | 0.885 | 0.926 | 0.325 |
| 3.0 | 5.3 | 1.42e-2 | 0.013 | 0.312 | 0.722 | 0.837 | 0.759 |
| 3.5 | 4.6 | 1.19e-2 | 0.011 | 0.205 | 0.824 | 0.883 | 0.348 |
| 4.0 | 4.0 | 8.68e-3 | 0.008 | 0.079 | 0.749 | 0.881 | 0.146 |
| 4.5 | 3.6 | 6.98e-3 | 0.007 | 0.210 | 0.830 | 0.900 | 0.221 |
| 5.0 | 3.2 | 4.49e-3 | 0.004 | 0.175 | 0.759 | 0.862 | 0.297 |
| 5.5 | 2.9 | 3.09e-3 | 0.003 | 0.077 | 0.790 | 0.859 | 0.145 |
| 6.0 | 2.7 | 2.93e-3 | 0.003 | 0.054 | 0.722 | 0.863 | 0.154 |
| 6.5 | 2.5 | 2.22e-3 | 0.002 | 0.095 | 0.791 | 0.860 | 0.192 |
| 7.0 | 2.3 | 2.17e-3 | 0.002 | 0.194 | 0.770 | 0.856 | 0.182 |
| 7.5 | 2.1 | 1.57e-3 | 0.002 | 0.097 | 0.784 | 0.852 | 0.130 |
| 8.0 | 2.0 | 1.48e-3 | 0.001 | 0.097 | 0.699 | 0.831 | 0.103 |
| 8.5 | 1.9 | 1.24e-3 | 0.001 | 0.091 | 0.776 | 0.845 | 0.119 |
| 9.0 | 1.8 | 1.36e-3 | 0.001 | 0.194 | 0.756 | 0.845 | 0.139 |
| 9.5 | 1.7 | 1.09e-3 | 0.001 | 0.132 | 0.800 | 0.861 | 0.124 |
| 10.0 | 1.6 | 1.04e-3 | 0.001 | 0.143 | 0.712 | 0.845 | 0.064 |
| 10.5 | 1.5 | 9.05e-4 | 0.001 | 0.121 | 0.803 | 0.854 | 0.060 |
| 11.0 | 1.5 | 9.69e-4 | 0.001 | 0.179 | 0.757 | 0.858 | 0.080 |
| 11.5 | 1.4 | 8.23e-4 | 0.001 | 0.128 | 0.803 | 0.854 | 0.083 |
| 12.0 | 1.3 | 8.57e-4 | 0.001 | 0.166 | 0.715 | 0.840 | 0.047 |
| 12.5 | 1.3 | 8.27e-4 | 0.001 | 0.148 | 0.793 | 0.844 | 0.067 |
| 13.0 | 1.2 | 8.09e-4 | 0.001 | 0.200 | 0.757 | 0.832 | 0.086 |
| 13.5 | 1.2 | 7.37e-4 | 0.001 | 0.118 | 0.767 | 0.841 | 0.075 |
| 14.0 | 1.1 | 7.65e-4 | 0.001 | 0.117 | 0.679 | 0.820 | 0.075 |
| 14.5 | 1.1 | 7.20e-4 | 0.001 | 0.123 | 0.791 | 0.849 | 0.106 |
| 15.0 | 1.1 | 8.09e-4 | 0.001 | 0.199 | 0.762 | 0.857 | 0.106 |
| 15.5 | 1.0 | 6.71e-4 | 0.001 | 0.114 | 0.800 | 0.848 | 0.094 |
| 16.0 | 1.0 | 7.00e-4 | 0.001 | 0.092 | 0.718 | 0.845 | 0.096 |

Train in-sample means are `0.799` (own) / `0.903` (full) against eval `0.782` / `0.864`,
so the ridge is not memorising the train half.

### Table B — the de-clocked residual (clock profiles removed)

| b | T (h) | power | frac | `P_last` | `P_own` | `P_full` |
| --: | --: | --: | --: | --: | --: | --: |
| 0.5 | 32.0 | 5.09e-2 | 0.262 | 0.230 | **0.604** | 0.691 |
| 1.0 | 16.0 | 6.79e-2 | 0.350 | 0.216 | 0.668 | 0.696 |
| 1.5 | 10.7 | 2.86e-2 | 0.147 | 0.205 | 0.661 | 0.714 |
| 2.0 | 8.0 | 1.41e-2 | 0.072 | 0.040 | 0.615 | 0.747 |
| 3.0 | 5.3 | 5.68e-3 | 0.029 | 0.285 | 0.642 | 0.703 |
| 4.0 | 4.0 | 3.01e-3 | 0.016 | 0.049 | 0.577 | 0.709 |
| 5.0 | 3.2 | 2.02e-3 | 0.010 | 0.229 | 0.675 | 0.739 |
| 6.0 | 2.7 | 1.09e-3 | 0.006 | 0.014 | 0.535 | 0.682 |
| 7.0 | 2.3 | 7.47e-4 | 0.004 | 0.196 | 0.619 | 0.669 |
| 8.0 | 2.0 | 4.69e-4 | 0.002 | 0.035 | 0.445 | 0.582 |
| 10.0 | 1.6 | 3.26e-4 | 0.002 | 0.093 | 0.469 | 0.579 |
| 12.0 | 1.3 | 2.51e-4 | 0.001 | 0.112 | 0.450 | 0.577 |
| 14.0 | 1.1 | 2.19e-4 | 0.001 | 0.107 | 0.368 | 0.526 |
| 16.0 | 1.0 | 2.09e-4 | 0.001 | 0.117 | 0.441 | 0.572 |

Full 32-row tables for A, B and C are in `runs/predictability.json`.

### Table C — phase-randomized surrogates

`S` = surrogate of the raw corpus, `S'` = surrogate of the de-clocked residual.
`P_real − P_surrogate` is the part not implied by the linear spectrum.

| b | T (h) | `P_own` | `S_own` | `P−S` | `D_own` | `S'_own` | `D−S'` |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 0.5 | 32.0 | 0.894 | 0.872 | +0.022 | 0.604 | 0.571 | +0.032 |
| 1.0 | 16.0 | 0.917 | 0.902 | +0.015 | 0.668 | 0.653 | +0.015 |
| 2.0 | 8.0 | 0.826 | 0.874 | −0.048 | 0.615 | 0.645 | −0.030 |
| 4.0 | 4.0 | 0.749 | 0.847 | −0.098 | 0.577 | 0.558 | +0.019 |
| 8.0 | 2.0 | 0.699 | 0.854 | −0.155 | 0.445 | 0.494 | −0.049 |
| 12.0 | 1.3 | 0.715 | 0.856 | −0.141 | 0.450 | 0.455 | −0.005 |
| 16.0 | 1.0 | 0.718 | 0.837 | −0.119 | 0.441 | 0.462 | −0.021 |

`P_own − S_own` is **≈ +0.02 at the two lowest bands and negative from `b = 1.5` upward**
(mean −0.082 above `b = 1.5`, worst −0.166 at `b = 14`). The real corpus is therefore
*less* predictable at high bands than a stationary process with the same power spectrum:
its high-band content carries structure the spectrum does not encode (weather, load
anomalies, half-hourly heterogeneity). The de-clocked residual shows the same sign but
much smaller (−0.023 on average, and +0.03/+0.02 at `b = 0.5/1.0`): the clock-locked part
is entirely spectrum-implied, as it must be.

## Step 3 — the four morphology diagnostics

`.venv/bin/python experiments/1_realdata/scripts/morphology.py` → `runs/morphology.json`
(6.5 s). Bands: `clock` `b ∈ [0.542, 0.792]` (20–30 h, contains the diurnal at `2/3`),
`mid` `b ≈ 3`, `high` `b ≈ 8`. The surrogate column is the null for each statistic.

**1. Narrowband envelope CV** (`std/mean` of the band envelope; oscillator/PM ≈ 0,
stationary Rayleigh ≈ 0.523, but the band width sets the null):

| band | real | surrogate |
| :-- | --: | --: |
| clock | 0.307 [0.271, 0.440] | 0.279 [0.180, 0.383] |
| mid | 0.557 [0.512, 0.628] | 0.518 [0.495, 0.531] |
| high | 0.531 [0.382, 0.575] | 0.526 [0.340, 0.541] |

The clock band's envelope is **not constant** — it fluctuates as much as the surrogate's
(0.307 vs 0.279; both sit below the 0.523 Rayleigh reference only because the band is
dominated by the coherent diurnal line, whose own envelope is constant). The corpus's
periodic component is amplitude-modulated day to day, so it is **not PM-like**.

**2. Phase diffusion vs phase slips** (*the decisive one*). `D(τ)` = mean squared
unwrapped phase change over τ samples; `gamma` = log-log slope on lags 256–4096
(1 = diffusing, 0 = saturated); `slip/env` = mean envelope at the largest 1 % of
per-sample phase jumps, divided by the mean envelope.

| band | | gamma | phase sd (rad) | slip/env |
| :-- | :-- | --: | --: | --: |
| clock | real | **0.17** | **0.27** | 0.606 |
| clock | surrogate | 0.44 | 0.35 | 0.522 |
| mid | real | 1.13 | 89.6 | 0.122 |
| mid | surrogate | 1.19 | 69.6 | 0.121 |
| high | real | 1.01 | 14.5 | 0.142 |
| high | surrogate | 1.06 | 9.26 | 0.135 |

`D(τ)` for the clock band, lags `[1,4,16,64,256,1024,4096]`:
real `[0.00, 0.00, 0.00, 0.03, 0.05, 0.05, 0.13]`,
surrogate `[0.00, 0.00, 0.00, 0.03, 0.08, 0.07, 0.23]`.

The clock component's **phase does not diffuse**: over 13 years its unwrapped phase
spreads by only 0.27 rad (≈15°) and `D(τ)` saturates (gamma 0.17) instead of growing
linearly. At the two upper bands there is no carrier to lock to — the phase wanders by
90 rad and 15 rad respectively, and there the large jumps *do* sit on envelope fades
(`slip/env = 0.12–0.14`, the stationary-Gaussian signature), exactly as the surrogate
does. So the corpus is **clock-locked in the clock band and stationary-like elsewhere**,
which is precisely the `PLAN.md` §6.3 expectation and the opposite of an autonomous
oscillator.

**3. Linewidth vs record length** (second moment of the mean periodogram within
`±0.05` cycles/patch of the diurnal carrier, in `b` units):

| record N | real | surrogate | real × N | surrogate × N |
| --: | --: | --: | --: | --: |
| 14 421 | 8.14e-6 | 9.67e-6 | 0.117 | 0.139 |
| 28 842 | 1.19e-5 | 1.04e-5 | 0.344 | 0.300 |
| 57 684 | 1.39e-5 | 1.41e-5 | 0.800 | 0.811 |
| 115 368 | 1.40e-5 | 1.53e-5 | 1.618 | 1.767 |
| 230 736 | 1.49e-5 | 1.73e-5 | 3.439 | 3.981 |

A phase-locked line would keep narrowing as `1/L` (its second moment falling as `1/L²`).
Instead the width **converges** to a fixed value — a second moment of 1.5e-5, i.e. a
width of 3.9e-3 cycles/patch and a coherence time of ~260 patches (≈5 days) — and the
surrogate, which has no clock at all, behaves identically. So this diagnostic **does not
discriminate on this corpus**: the line's finite width is already set by slow
amplitude/phase modulation present in the power spectrum itself, not by the clock.
Reported as a degenerate step rather than dressed up.

**4. Is the periodic component phase locked?** Variance explained by folding:

| fold | real | surrogate |
| :-- | --: | --: |
| 24 h | 0.4692 [0.353, 0.646] | 0.4692 [0.353, 0.646] |
| 7 d | **0.6188** [0.486, 0.743] | 0.5270 [0.460, 0.693] |

The 24 h fold's `r²` is **exactly invariant under phase randomization** (it depends only
on the magnitudes of the 24 h harmonics), so the two columns *must* agree and that row
carries no information by itself. The 7 d fold does discriminate: the real corpus
explains **+0.092** more than its surrogate, i.e. it has a 7-day phase-locked component
that the power spectrum alone does not imply. Both folds together remove 61.9 % of the
variance, and the residual's power share per band (printed by the script, 32 rows in the
JSON) is ≈0.19 at `b = 0.5` and falls to ≈1e-5 at `b = 16`.

## Answer: does predictability vary with band?

**Yes, but weakly, and not monotonically in any clean sense.**

- `P_own` on the raw corpus runs **0.92 at `b = 1` → 0.68–0.83 for `b ≥ 3`**, with a
  wobble (local dips at `b = 3`, 6, 8, 14; the strongest single band is `b = 1.0`, the
  16 h band). `P_full` is uniformly higher, 0.82–0.94, i.e. cross-band information adds
  **0.05–0.15** at every band.
- On the de-clocked residual the same shape is cleaner and lower: **0.60 at `b = 0.5` →
  ~0.40–0.48 at `b ≥ 8`**, again with a wobble, and again `full > own` by ~0.06–0.13.
- Both curves are **dominated by a floor, not by structure**: at a 32-sample patch the
  corpus's clock lines and its steep red continuum leak into every bin, so `P(b)` is
  high *everywhere*. The in-band column says how much of each number is the band's own
  content, and it is below 0.5 for every band from `b = 2` up except `b = 3`.
- The **single-patch** reference is the informative contrast: `P_last` is 0.27 at low
  bands and 0.05–0.31 above them, versus `P_own` 0.68–0.92. Almost all of the
  predictability is context, not the last patch — the opposite of the synthetic result
  P1b found for the learned prior.

So the honest headline for T1 is: **this corpus's per-band conditional predictability is
high and roughly flat, with a mild decline toward the Nyquist band**, and the flatness is
partly an aperture artifact that is quantified rather than removed. The energy and
predictability curves are *not* the same object — power falls 460× from `b = 0.5` to
`b = 16` while `P_own` falls only ~1.25× — which is the contrast `PLAN.md` §2 wants from
the Fredformer question.

## Follow-up: is that flatness an aperture artifact? (the `k` sweep)

Added 2026-09-16. `scripts/aperture.py` → `runs/aperture.json` (73.5 s). No model is
trained; T1's estimator is imported and re-run unchanged, varying only `k`. This is the
"**open before T2**" question of `PLAN.md` §6.2: *is the near-flatness of `P(b)` a property
of the corpus, or an artifact of the 32-sample aperture?*

### What a `k` sweep can and cannot hold fixed

Three facts decide the design, and two of them contradict the natural reading of the
question.

1. **In `b` units the aperture is nearly the same object at every `k`.** The matched
   filter's relative response is `|W(Δb)| = |sin(πΔb)| / (k|sin(πΔb/k)|)`, whose nulls are
   at the *same* place (every integer `Δb`) for *every* `k`, with a mainlobe of `±1`
   cycle/patch. So the estimator reads a function of the **pair `(k, b)`**, and the band
   grid only chooses which pairs are looked at. Holding `b` fixed while growing `k` does
   **not** bring a better aperture to bear on a band; it slides the corpus's physical
   spectrum along the `b` axis — a relabel, exact at the nulls and near-exact between them,
   where the only difference is the far sidelobes (relative gain at `Δb = 20.5` is 0.034 at
   `k = 32` and 0.016 at `k = 128`), i.e. the longer patch has the *lower* far sidelobes.
2. **A physical band is `b = k / (2T)`.** Holding the band (period `T`) is what makes the
   relative passband `1/b` narrow, and that needs `b ∝ k`. Both readings are run, because
   only the second is an aperture test: `grid = "t1"` (T1's `b` values) and
   `grid = "phys"` (`b` scaled by `k/32`, T1's physical periods).
3. **The forecast horizon is `k` samples and cannot be held fixed.** Predicting the next
   patch means predicting the next 16 h at `k = 32` and the next 64 h at `k = 128`; the
   target patch is that long too. A patch-based construction cannot separate aperture from
   horizon, so no arm here is a pure aperture manipulation.

| arm | grid | ctx / k | P | window | `b` | physical periods | windows (train/eval) | split rms |
| :-- | :-- | --: | --: | --: | --: | :-- | --: | --: |
| **k32/t1** (T1) | t1 | 512 / 32 | 16 | 544 | 0.5–16 | 32 h – 1 h | 2 122 (1 061/1 061) | 0.014 |
| k64/t1 | t1 | 1024 / 64 | 16 | 1088 | 0.5–16 | 64 h – 2 h | 1 061 (530/531) | 0.022 |
| k128/t1 | t1 | 1024 / 128 | 8 | 1152 | 0.5–16 | 128 h – 4 h | 1 001 (500/501) | 0.021 |
| k64/phys | phys | 1024 / 64 | 16 | 1088 | 1–32 | 32 h – 1 h | 1 061 (530/531) | 0.021 |
| k128/phys | phys | 1024 / 128 | 8 | 1152 | 2–64 | 32 h – 1 h | 1 001 (500/501) | 0.032 |

`t1` and `phys` coincide at `k = 32`, so that arm is run once. The two grids are not
different measurements: they sample the same function of `(k, b)`. "split rms" is the
RMS band-to-band difference between two random train/eval splits — the thin-arm guard:
the two larger apertures are 1.5× and 2.3× noisier than T1, and still well inside the
effects below. Both larger-`k` arms also **fail T1's own pass criterion 3** (≥ 1 000
windows *per split*: they have 530 and 500).

### 1. `in-band` vs `b` — it improves in the high bands, and the low bands are a trap

Same `b`, different `k` (the brief's literal arms — but each column is a *different*
physical band):

| `b` | period at k32 / k64 / k128 | in-band k32 | k64 | k128 |
| --: | :-- | --: | --: | --: |
| 4 | 4 h / 8 h / 16 h | 0.146 | 0.651 | 0.065 |
| 8 | 2 h / 4 h / 8 h | 0.103 | 0.359 | 0.869 |
| 16 | 1 h / 2 h / 4 h | 0.096 | 0.271 | 0.641 |

At a **matched physical period** (the aperture test):

| period | in-band k32 | k64 | k128 |
| :-- | --: | --: | --: |
| 8 h | 0.343 | 0.651 | 0.869 |
| 4 h | 0.146 | 0.359 | 0.641 |
| 2 h | 0.103 | 0.271 | 0.546 |
| 1 h | 0.096 | 0.253 | 0.522 |
| 32 h | 0.894 | 0.886 | 0.205 |
| 16 h | 0.962 | 0.095 | 0.065 |

**At `T ≤ 8 h` the answer is yes, unambiguously**, and it is stronger than the numbers
look: the `±0.5`-`b` window is a fixed *relative* width `1/(2b)`, so at 2 h it is 6.25%
wide at `k = 32` and 1.6% wide at `k = 128`. A four-fold *narrower* window contains
**5.3× more** of the measured power (0.103 → 0.546). That is the aperture concentrating
the band, not the window getting lucky.

**The low bands invert the ranking, and the metric is at fault.** A `±0.5`-`b` window is
not a fixed physical filter, so at 32 h it encloses the 7 d line (`Δb = 0.41` at `k = 32`,
1.6 at `k = 128`) and the 24 h line (`Δb = 0.17` / 0.67). The `k = 32` reading
(0.894/0.962) counts those clock lines as "in-band"; the `k = 128` one (0.205/0.065) counts
them as leakage — even though the aperture's *gain on the same line* has fallen from
**0.75 to 0.19**. So `in-band` is quotable within one aperture (as T1 used it) but it is
**not a cross-`k` purity measure**, and at the low bands it ranks the better aperture worse.

One corollary the k128 arm makes visible: at `k = 128` the in-band column reads a **line
comb, not a continuum**. The bands that are exact diurnal harmonics (8, 4, 2.67, 2, 1.6,
1.33, 1.14, 1 h → `b = 8, 16, 24, 32, 40, 48, 56, 64`) read in-band 0.29–0.87, while every
band between them reads 0.05–0.07 — consistent with T1's survey finding coherent harmonics
of the diurnal "all the way to Nyquist".

### 2. Dynamic range of `P_own` — the curve does *not* gain structure

Over the arm's own grid (`max−min` / `max/min`), and over T1's 32 physical bands:

| arm | `P_own` range | max/min | matched range | matched max/min |
| :-- | --: | --: | --: | --: |
| k32/t1 | 0.239 | 1.352 | 0.239 | 1.352 |
| k64/phys | 0.286 | 1.453 | 0.286 | 1.453 |
| k128/phys | 0.479 | 2.296 | 0.479 | 2.296 |
| k64/t1 | 0.419 | 1.823 | — | — |
| k128/t1 | 0.609 | 3.430 | — | — |

The extra spread at `k = 128` is **entirely** at the two bands that same arm shows to be
leakage-dominated (32 h: 0.493, in-band 0.205; 16 h: 0.369, in-band 0.065). The clean
comparison is the six physical bands that the `k = 128` aperture actually isolates
(diurnal harmonics 8, 4, 2.67, 2, 1.14, 1 h), where the mean in-band rises **0.153 →
0.352 → 0.613** across the three apertures:

| period | k32 | k64 | k128 |
| :-- | --: | --: | --: |
| 8 h | 0.826 | 0.823 | 0.792 |
| 4 h | 0.749 | 0.706 | 0.713 |
| 2.67 h | 0.722 | 0.684 | 0.702 |
| 2 h | 0.699 | 0.679 | 0.700 |
| 1.14 h | 0.679 | 0.656 | 0.757 |
| 1 h | 0.718 | 0.663 | 0.848 |
| **range** | **0.148** | **0.167** | **0.148** |
| **max/min** | **1.218** | **1.255** | **1.211** |

A 4.0× change in how much of the measured band power is genuinely in-band moves `P_own` by
≤ 0.05 at five of the six bands and leaves the spread at **1.21–1.26×**, i.e. T1's flatness
exactly. This is the answer to the headline question. (`P_full` is flat the same way,
1.08–1.22× at these bands; `P_last` is 0.09 / 0.00 / 0.19 on average at `k = 32 / 64 / 128`,
so the context is still doing the work.)

One column is *not* comparable across arms: the `power` column. A longer patch's demean
removes more low-frequency variance, and the matched filter's variance on a red spectrum
scales with `k`, so the same physical band reads a different absolute power in each arm
(over T1's 32 physical bands, `max/min` is 643 / 1 759 / 1 713 at `k = 32 / 64 / 128`). The
energy/predictability contrast T1 quotes is a within-arm statement and stays valid there.

### 3. `P_own − S_own` — the sign survives, the magnitude does not

| arm | mean over 32 matched bands | over the 6 well-isolated bands | at `T = 16 h / 32 h` |
| :-- | --: | --: | --: |
| k32 | −0.082 | −0.121 | −0.016 / −0.022 |
| k64 | −0.123 | −0.116 | −0.010 / −0.002 |
| k128 | −0.035 | **+0.009** | **−0.410 / −0.258** |

"Real is less predictable than its own linear spectrum implies" is **net-negative at every
`k`** (T1's −0.082 is reproduced exactly at `k = 32`), and at `k = 32`/`k = 64` it is
6–8× the split noise. At `k = 128` it is not resolvable on the well-isolated bands
(mean +0.009, mixed signs, split noise 0.032) and what deficit remains is concentrated at
the two bands where that arm's in-band column collapses. So the *sign* is a corpus
property; the *size* is entangled with leakage in both directions and should not be quoted
to two decimals.

Secondary observation, reported because it cuts the other way: the **de-clocked residual**
*does* gain slope with `k` over the same six bands — `D_own` spans 0.368–0.615 (1.67×) at
`k = 32`, 0.206–0.712 at `k = 64`, 0.289–0.826 (2.86×) at `k = 128`, monotonically
decreasing in `T` at the largest aperture. Part of Table B's flatness was aperture mixing.
That is the residual, not the marginal a model trains on; the raw `P_own` — the curve T2
compares against — is the aperture-insensitive one.

### 4. Null controls at every `k`

| arm | white-noise `P_own` | pre-filtered (0.5-wide) control | in-band real / white | windows per split |
| :-- | --: | --: | :-- | --: |
| k32 | −0.002 [−0.006, +0.000] | 0.998 | 0.233 / 0.774 | 1 061 |
| k64 | −0.005 [−0.018, −0.001] | 0.998 | 0.431 / 0.770 | 530 |
| k128 | −0.005 [−0.021, +0.000] | 0.995 | 0.454 / 0.770 | 500 |

The delivered estimator stays clean at every aperture, so nothing above is degeneracy. The
low-DOF trap is **arm-independent**: it is triggered by pre-filtering to 0.5-wide bands
before taking patch coefficients, whatever `k` is, because a window then holds only
`P + 1` complex degrees of freedom. Every arm here is on raw patches, so none of them falls
into it. The white-noise in-band fraction is 0.770–0.775 at all three apertures — the
aperture's own spread, unchanged by `k`, which is the measurement confirming fact 1 above.

### Answer

**No — the near-flatness of `P(b)` is a property of the corpus, not an artifact of the
32-sample aperture.** Where the aperture is demonstrably better (a 4× narrower relative
window, in-band 0.15 → 0.61 on the same six physical bands), `P_own` moves by ≤ 0.05 and
its spread stays 1.21–1.26× — T1's own arm reads 1.22× on those six bands and 1.30× over
all 29 bands at `T ≤ 8 h`. Where spread *does* grow (`k = 128`, up to 2.30×) it sits
exactly at the two bands that arm is shown to be leakage-dominated: leakage manufactures
apparent structure rather than hiding it. The aperture inflates the **level** of `P`
(T1's point stands: above `b = 2` most of the measured power is not in the band, at every
`k` here) but it does not create the flatness.

**Recommended aperture for T2.** The binding invariant is that **`P(b)` and the model's
`r_b` go through the same aperture** — the same leakage biases both — not the value of `k`.

- **`k = 32` (T1's arm) is safe.** `P(b)` is aperture-insensitive, so the T2 comparison is
  valid as designed; keep the in-band column beside every `r_b` and say that the high-band
  readouts are 5–35% in-band for `P` *and* for `r`.
- **`k = 64` is the better readout** if the model's patch size is free: on the same physical
  bands the high-band in-band share rises 1.9–2.6× (0.10 → 0.27 at 2 h, 0.15 → 0.36 at 4 h,
  0.34 → 0.65 at 8 h) with `P(b)` unchanged, so the `r_b` vs `sqrt(P(b))` comparison is
  better conditioned. Cost: half the windows (530 per split, below T1's criterion), 1.5×
  the split noise, and a 32 h horizon instead of 16 h.
- **Do not use `k = 128`**: 500 windows per split, 8 context patches, 2.3× the split noise,
  no information gain, and its apparent per-band structure is leakage.
- **Do not re-read the in-band column across apertures** (see item 1). It is a valid
  within-aperture leakage report and an invalid cross-aperture purity metric.

## What T2/T3 would need here

**T2 was then run — see the T2 section at the bottom of this file.** This section is kept
as the design note it was; two of its premises did not survive: the fit gate turned out to
be easy (the model reaches +0.92 variance explained against a mean-predictor baseline of
+0.15), and the `in-band` caveat below turned out to be the binding one.

- **Window count**: 2 122 non-overlapping (1 061 train / 1 061 eval). A stride of 272
  doubles it to ~4 244 (and a time-blocked split is then required to keep train and eval
  sample-disjoint); stride 32 gives ~36 000 but the windows are then ~17× redundant.
- **Fit gate (contract A8)**: the corpus is 62 % clock-locked, so a next-patch model has
  a large, learnable deterministic share — a mean predictor is *not* the optimum here,
  which is exactly the failure mode that made the `broad32` synthetic arm unquotable.
  A per-band gate is feasible: `P_full` is 0.83–0.94 at every band, so no band is
  unpredictable to the point of being unmeasurable.
- **T3 (causal)**: per-patch phase randomization preserves band power exactly, and the
  de-clocked residual already provides the "clock-free" arm. Both are cheap here.
- **Caveat for T2's readout**: the same leakage that inflates `P(b)` also inflates a
  matched-filter `r_b` measured on the model's 32-sample output, since both go through
  the same aperture. Decide *before* the run whether `r_b` is to be read through the
  patch aperture (aperture-consistent with `P(b)`) or through a longer one.
  **Resolved by the `k` sweep above**: the aperture must match the model's patch, the flat
  `P(b)` is not an aperture artifact, and a longer readout aperture is not worth its cost
  (it moves the horizon, halves the window count and does not change `P`).

## Pass criteria

Stated so the result can fail; outcome recorded.

| # | criterion | outcome |
| --: | :-- | :-- |
| 1 | the estimator reads `P ≈ 0` on a white-noise corpus at every band | **pass** — `P_own` mean −0.002, range `[−0.006, +0.000]` |
| 2 | a finer-than-resolvable grid must not be reported | **pass, as a guard** — the pre-filtered 0.5-wide variant reads 0.998 on white noise and is therefore excluded |
| 3 | ≥ 1 000 non-overlapping windows per split | **pass** — 1 061 |
| 4 | the clock-locked component is identifiable and removable | **pass** — 7 d fold `r² = 0.619`; de-clocking lowers `P_own` by 0.08–0.42 (median 0.25) |
| 5 | every reported band is mostly its own content (`in-band > 0.5`) | **fail** — holds only for `b ≤ 1.5` and `b = 3`; elsewhere 0.05–0.35. Tables A and B are therefore reported with the leakage column rather than as clean per-band measurements. |

Follow-up (`k` sweep), stated before the run:

| # | criterion | outcome |
| --: | :-- | :-- |
| 6 | every aperture keeps a clean white-noise null | **pass** — `P_own` mean −0.002 / −0.005 / −0.005 at `k = 32 / 64 / 128` |
| 7 | an arm too thin to estimate the spectrum is flagged, not smoothed over | **pass, with a failure recorded** — split-seed rms 0.014 / 0.021 / 0.032; the `k = 64` and `k = 128` arms fail criterion 3 (530 and 500 windows per split) and are reported with that caveat |
| 8 | bigger `k` is not reported as improving `P(b)` unless it does | **pass** — it does not: over the six physical bands the `k = 128` aperture isolates, `P_own` spans 1.22 / 1.26 / 1.21× at `k = 32 / 64 / 128` |

## Scripts, reproduce, wall time

| script | role | wall |
| :-- | :-- | --: |
| `scripts/survey.py` → `runs/survey.json` | Step 0 survey of the four candidates | 6.1 s |
| `scripts/predictability.py` → `runs/predictability.json` | Step 2: `P(b)`, spectra, surrogates, controls | 42–47 s |
| `scripts/morphology.py` → `runs/morphology.json` | Step 3: the four diagnostics | 6.5 s |
| `scripts/aperture.py` → `runs/aperture.json` | the `k` sweep (5 arms, no model) | 73.5 s |
| `scripts/t2.py` → `runs/t2_<arm>_s<seed>.json` | **T2**: one arm/seed (corpus + 20 k steps) | 1 640 s |
| `scripts/t2.py --summarize` → `runs/t2_summary.json` | T2's tables, pooled over seeds | <1 s |
| `scripts/t2.py --relation` → `runs/t2_relation.json` | T1's ridge read as a retention (no model) | 13.5 s |
| `scripts/t2b.py --report` → `runs/t2b/flatten.json` | **T2b**: the flattening filter, corpus flatness and `P(b)` per variant | 80 s |
| `scripts/t2b.py --variant <A,B,C> --arm <k32,k64> --seed <0,1,2>` → `runs/t2b/<v>_<arm>_s<seed>.json` | **T2b**: one variant/arm/seed (corpus + 20 k steps) | 860–1 690 s |
| `scripts/t2b.py --summarize` → `runs/t2b/summary.json` | T2b's tables, the decision, and arm A's check against T2 | <1 s |

```
.venv/bin/python experiments/1_realdata/scripts/survey.py
.venv/bin/python experiments/1_realdata/scripts/predictability.py
.venv/bin/python experiments/1_realdata/scripts/morphology.py
.venv/bin/python experiments/1_realdata/scripts/aperture.py
OMP_NUM_THREADS=2 .venv/bin/python experiments/1_realdata/scripts/t2.py --arm k32 --seed 0
.venv/bin/python experiments/1_realdata/scripts/t2.py --summarize
.venv/bin/python experiments/1_realdata/scripts/t2.py --relation
.venv/bin/python experiments/1_realdata/scripts/t2b.py --report
# T2b, one process per (variant, arm, seed): 6 in parallel (= 1 695 s), then 6
# (= 1 683 s), then the 3 dose runs (= 871 s), 4 249 s of wall for all 15.
OMP_NUM_THREADS=2 .venv/bin/python experiments/1_realdata/scripts/t2b.py --variant A --arm k32 --seed 0
.venv/bin/python experiments/1_realdata/scripts/t2b.py --summarize
```

`predictability.py` takes `--k`, `--ctx` (default `16 * k`) and `--grid {t1,phys}`, and
writes `runs/predictability_k<k>_<grid>.json` for any arm other than T1's; its defaults are
T1's and reproduce `runs/predictability.json` bit-for-bit. `aperture.py` imports that
estimator rather than re-implementing it.

≈60 s for T1, +74 s for the sweep, CPU only, no downloads (`load_monash` reads the local HF
cache; the `Salesforce/GiftEvalPretrain` dataset is not touched). `src/fbias/realdata.py`
is a copy of the lab loader with `ruff format` applied and no logic changes.

## Where this contradicts or refines the brief

1. **"Aim for 16–32 bands" is not freely achievable at `k = 32`.** 32 bands at `Δb = 0.5`
   can be computed from the raw patches, but they are neither independent (a 32-sample
   patch has 17 independent complex coefficients) nor leakage-free, and the one
   clean-looking fix makes the estimator degenerate (trap 2). Only the 16 integer bands
   (`Δb = 1`, the patch's own DFT bins) can be called per-band measurements in any clean
   sense; the delivered 32-band table must be read with its `in-band` column.
2. **`P(b)` here is an `R²`, not a retention.** The brief defines
   `P = 1 - residual_var / var(target)`. On the synthetic PM corpus of `PLAN.md` §4 the
   same estimator gives exactly `r*²` (`z_p = A e^{iφ_p}` is AR(1) with coefficient
   `r*`), so the claim's per-band retention should read `sqrt(P(b))`, not `P(b)`. Any
   direct T2 comparison must square before comparing.
3. **`PLAN.md` §6.2 says `P(b)` is measured "after phase-folding out clock-locked
   components"; the brief's Step 2 asks for `P(b)` on the corpus as it stands.** Both are
   reported (tables A and B) because they answer different questions — B is the cleaner
   per-band measurement of the stochastic part, A is what a model trained on this corpus
   would actually see.
4. **Diagnostic 3 is degenerate on this corpus** and is reported as such: the surviving
   linewidth converges at the same value for the real corpus and for its surrogate.

From the `k` sweep (the section above):

5. **"Vary only `k`" is not achievable in a patch-based construction.** The aperture *is*
   the forecast horizon: predicting the next patch is predicting the next 16 h at
   `k = 32` and the next 64 h at `k = 128`, and the target patch is that long. No arm here
   is a pure aperture manipulation, so the sweep is reported as two readings (same `b`;
   same physical period) and neither one isolates the aperture alone.
6. **"`P = ctx/k` held at 16 keeps the band grid in `b` comparable" understates the
   problem.** Nothing makes the grid comparable except `b` itself, and in `b` units the
   matched filter is the *same object* at every `k` (nulls at integer `Δb`, mainlobe
   `±1`). Growing `k` at fixed `b` therefore slides the physical spectrum along the `b`
   axis — a relabel — and only `b ∝ k` (the `phys` grid) holds the band. What `P = 16`
   does buy is a constant number of context patches for `k = 32/64`.
7. **The `in-band` column, designated the key column, is not comparable across `k`.** It
   uses a `±0.5`-`b` window, i.e. a *relative* width `1/(2b)`, so a longer patch at a
   matched physical band narrows it; at the low bands that flips the ranking (at 32 h the
   `k = 128` aperture's gain on the 7 d line is 0.19 against `k = 32`'s 0.76, yet in-band
   reads 0.205 against 0.894). It remains a valid within-aperture leakage report.
8. **The sweep contradicts the worry that motivated it, in the useful direction.** The
   flatness is *not* aperture-limited, so T2's `r_b` vs `sqrt(P(b))` comparison is fine as
   designed at T1's `k = 32`; the one real cost of the 32-sample aperture is that both
   numbers are leakage-dominated, not that either is biased flat.

---

# T2 — a model trained on this corpus: is its per-band attenuation flat?

Added 2026-09-16. `scripts/t2.py` → `runs/t2_<arm>_s<seed>.json` (6 trainings, 3 seeds ×
2 arms), pooled into `runs/t2_summary.json`, plus `runs/t2_relation.json` (model-free).
**This is the section that trains models** — everything above it is data analysis.

T1/T1b established the explaining variable: on this corpus `P(b)` is nearly flat across a
32× range of `b` (32 h → 1 h) while band power falls 463×. T2's prediction was therefore
sharp: *a model trained on this corpus should show nearly flat per-band attenuation*, not
the "high frequency is worse" pattern the field reports.

## Answer, up front

**The clean-probe curve is not flat. It falls monotonically, by 7.9× (k32) / 5.8× (k64)
from `b = 1` to `b = 12`, while `P_own` over the same bands spans 1.31× / 1.32×.** The
spread across bands is **18× (k32) and 33× (k64) the pooled per-band seed sd**, so it is
not seed noise; the monotone shape is already in place at step 1 000 and the per-band
values move by at most 0.13 to step 20 000 (worst: `b = 2`, 0.501 → 0.636 at `k32`), so it
is not under-training either.

So T1b's prediction is **falsified on this corpus and this model**, in the direction of the
field's expectation: the model attenuates the high bands far more than the corpus's
per-band predictability justifies. In distribution the same model matches T1's ridge band
by band to within ~0.1 (below), so this is not a "the model did not learn the corpus"
result — it is a statement about the **gain applied to clean, genuine content at each
band** (with the caveat that the in-distribution comparison is itself leakage-carried above
`b = 2`).

Two caveats that a reader must carry, both quantified below and neither of which removes
the effect:

- **The probe is spectrally far out of distribution.** Contract A9 matches the probe's
  *total* RMS to the corpus, and on this corpus that leaves the probe's per-band power at
  0.126 in *every* band — **0.29×** the corpus's lowest band and **180×** its highest
  (power table below). A9 controls the total, not the spectral shape; on a red corpus the
  two are not the same, and the clean probe is therefore testing the model's response to a
  *flat-spectrum* input, which it never saw.
- **The high bands' `in-band` is small** (0.10 at 2 h, 0.05 at 1.33 h at `k = 32`), so the
  corpus's own high-band "content" is 90%+ leakage and `P(b)`'s flatness there is
  leakage-carried. At the `k = 64` arm's `b = 4` (8 h, `in-band` **0.65**) the deficit is
  still there — retention 0.52 against `sqrt(P) = 0.91` — so leakage alone does not
  explain the slope.

## The two arms, and the horizons

The aperture **is** the horizon (T1b), so these are two different forecasting problems and
are reported separately, never merged.

| arm | ctx / k | P | window | patches | horizon | `b` grid | windows (train/eval) |
| :-- | --: | --: | --: | --: | :-- | :-- | --: |
| `k32` | 512 / 32 | 16 | 544 | 16 + 1 | **next 16 h** | 0.5–16 (T1's) | **2 122** (1 061/1 061) |
| `k64` | 1024 / 64 | 16 | 1088 | 16 + 1 | **next 32 h** | 0.5–16 (T1's) | **1 061** (530/531) |

Band↔period: `T(h) = 16/b` at `k = 32` and `32/b` at `k = 64`, so a fixed `b` is a
different physical band in the two arms — T1b's relabel, reported as such.

## Corpus construction — T1's, asserted rather than re-implemented

Same 5 series (`australian_electricity_demand_dataset`, 30 min), same per-series centre and
unit variance over the whole series (contract 9), same non-overlapping stride (so the split
stays leak-free), same window split (`predictability.SPLIT_SEED`, half train / half eval),
same **per-patch demeaning before the matched filter**. `t2.py` imports T1's estimator and
**asserts** that its own training windows reproduce T1's coefficients band by band
(`check_aperture`, `atol = 1e-9`), so the demeaning convention — the binding invariant, not
the value of `k` — cannot drift. The exported `P_own`, `P_full`, `P_last` and `in_band`
columns reproduce `runs/aperture.json`'s `k32/t1` and `k64/t1` arms exactly.

The model sees the window **raw** (only the per-series normalisation of T1's corpus), so
nothing about the target patch's own scale is removed — no RevIN (contract A14). Readouts
are taken through T1's matched filter with T1's per-patch demeaning on **both** numerator
and denominator.

## The clean probe — contracts A4 and A9

`b ∈ {1, 2, 4, 6, 8, 10, 12, 16}` with `f = 16 b` cycles per `ctx`, the brief's suggested
set: all integers, so `2b` is an integer and the matched filter is exact, and the minimum
gap is **exactly 1 cycle/patch** (contract A4, no slack). Physical periods are 16 h…1 h at
`k32` and 32 h…2 h at `k64`. 256 windows, one fixed seed (123), identical in every arm and
every model seed.

**RMS matching (contract A9).** The probe is scaled to the corpus's window RMS:

| arm | corpus window RMS | probe RMS | ratio | probe band power |
| :-- | --: | --: | --: | --: |
| `k32` | 0.99995 | 0.99995 | 1.0000 | 0.1257 in every band (`b = 16`: 0.2405) |
| `k64` | 0.98448 | 0.98448 | 1.0000 | 0.1212 in every band |

Two facts about that probe that the reader needs:

- **`b = 16` at `k = 32` is the sampling Nyquist** (`f = 256` cycles per 512 samples), where
  the matched filter's positive and negative frequency images coincide and a pure tone of
  amplitude `A` reads `|Z| = 2A`. Ratios stay valid — both sides go through the same filter
  — but the probe's own bin power reads 0.2405 against 0.1257 everywhere else, so the
  `b = 16` row is reported separately wherever it could matter. At `k = 64`, `b = 16` is at
  0.25 cycles/sample and no such degeneracy applies.
- **The corpus's per-band power is much lower than the probe's at every band above the
  first**, and much of the apparent slope sits in that gap (see "the one real confound").

## Fit gate and step count (contracts A2, A8)

Stated before the run: an arm is quotable only if its across-seed mean held-out variance
explained beats **both** interpretable baselines — persistence (copy the last context
patch) and the mean predictor — by at least 0.02 absolute *and* 3× the across-seed sd.
Both arms pass by a mile.

| arm | seed 0 | seed 1 | seed 2 | mean ± sd | persistence | mean predictor | verdict |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| `k32` | +0.9145 | +0.9155 | +0.9183 | **+0.9161 ± 0.0016** | −0.8995 | +0.1526 | **PASS** (margin +0.7635) |
| `k64` | +0.8753 | +0.8790 | +0.8770 | **+0.8771 ± 0.0015** | −1.0028 | +0.1316 | **PASS** (margin +0.7455) |

(variance explained at the trained position; all-position values are +0.9124 / +0.8672.)

**The persistence baseline is negative, and that is a property of this corpus, not a bug.**
The context patch is 16 h (k32) / 32 h (k64) before the target, which is neither a multiple
of 24 h nor of 7 d, so "copy the last patch" is close to anti-correlated with the target:
MSE = 1.9× the target variance. The mean predictor is the meaningful weak baseline
(+0.15 / +0.13) and is itself far from zero because 62% of the corpus's variance is
clock-locked (T1).

**Per-band variance explained** (the bands `r_b` is read from) is positive at every band in
both arms — 0.95/0.90/0.80/0.74/0.68/0.69/0.68/0.62 for `k32` and
0.91/0.90/0.87/0.57/0.63/0.53/0.52/0.34 for `k64` — so no band is unmeasurable and the
per-band readout is not noise.

**20 000 steps, and why.** 2 000 is too few for real data (contract A2; P3's warning). The
transient is logged at every checkpoint:

| step | `k32` ve_last | `k32` mean probe `r` | `k64` ve_last | `k64` mean probe `r` |
| --: | --: | --: | --: | --: |
| 1 000 | +0.7900 | 0.405 | +0.6971 | 0.302 |
| 2 000 | +0.8371 | 0.411 | +0.7489 | 0.304 |
| 5 000 | +0.8783 | 0.407 | +0.8109 | 0.330 |
| 10 000 | +0.9003 | 0.400 | +0.8493 | 0.353 |
| 15 000 | +0.9123 | 0.393 | +0.8671 | 0.370 |
| 20 000 | +0.9161 | 0.389 | +0.8771 | 0.375 |

`k32` has saturated (+0.0018 over the last 5 000 steps, inside its own seed sd of 0.0016);
`k64` is still rising at +0.006/5 000. **The shape the experiment is about does not depend
on this**: the per-band probe `r_b` at `k32` is already 1.006/0.501/0.339/0.244/0.258/0.229/
0.266/0.420 at step 1 000 and ends at 1.382/0.636/0.358/0.223/0.185/0.183/0.175/0.319 —
the same monotone decline, present from the first checkpoint. Wall was **1 640 s per run**
(6 concurrent, 2 threads each, CPU only).

## Table A — power, predictability, retention (3 seeds, mean ± sd)

`power` = mean `|Z|²/2` of the demeaned corpus patches (T1's units); `in-band` = T1's
leakage column; `r_probe` = clean probe; `r_corpus` = held-out corpus windows; `or_*` = the
contract-5 oracle.

### `k32` — horizon 16 h, 2 122 windows

| b | T(h) | power | `P_own` | `P_full` | `sqrt(P)` | in-band | `r_probe` | `r_corpus` | `or_probe` | `or_corpus` |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 16.0 | 4.32e-1 | 0.917 | 0.939 | 0.958 | 0.962 | **1.3816 ± 0.1398** | 1.0906 ± 0.0100 | 2.790 | 1.134 |
| 2 | 8.0 | 6.13e-2 | 0.826 | 0.900 | 0.909 | 0.343 | **0.6364 ± 0.0362** | 1.0243 ± 0.0117 | 1.175 | 1.018 |
| 4 | 4.0 | 8.68e-3 | 0.749 | 0.881 | 0.866 | 0.146 | **0.3585 ± 0.0454** | 1.1361 ± 0.0055 | 0.585 | 1.116 |
| 6 | 2.67 | 2.93e-3 | 0.722 | 0.863 | 0.849 | 0.154 | **0.2233 ± 0.0059** | 1.1405 ± 0.0337 | 0.420 | 1.111 |
| 8 | 2.0 | 1.48e-3 | 0.699 | 0.831 | 0.836 | 0.103 | **0.1851 ± 0.0342** | 1.2934 ± 0.0227 | 0.219 | 1.163 |
| 10 | 1.6 | 1.04e-3 | 0.712 | 0.845 | 0.844 | 0.064 | **0.1833 ± 0.0039** | 1.2394 ± 0.0223 | 0.151 | 1.091 |
| 12 | 1.33 | 8.57e-4 | 0.715 | 0.840 | 0.846 | 0.047 | **0.1750 ± 0.0210** | 1.2277 ± 0.0174 | 0.147 | 1.086 |
| 16 | 1.0 | 7.00e-4 | 0.718 | 0.845 | 0.847 | 0.096 | **0.3194 ± 0.1018** | 3.2666 ± 0.4161 | 0.317 | 2.133 |

`P_last` (the newest context patch only) is 0.29 / 0.07 / 0.08 / 0.05 / 0.10 / 0.14 / 0.17 /
0.09 — almost all of the predictability is context, not the last patch, exactly as T1 found.

### `k64` — horizon 32 h, 1 061 windows

| b | T(h) | power | `P_own` | `P_full` | `sqrt(P)` | in-band | `r_probe` | `r_corpus` | `or_probe` | `or_corpus` |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 32.0 | 3.51e-1 | 0.850 | 0.889 | 0.922 | 0.886 | **1.0154 ± 0.0484** | 1.0196 ± 0.0202 | 1.605 | 1.007 |
| 2 | 16.0 | 1.14e-1 | 0.849 | 0.904 | 0.921 | 0.095 | **0.6154 ± 0.0394** | 1.1735 ± 0.0260 | 0.766 | 1.222 |
| 4 | 8.0 | 3.02e-2 | 0.823 | 0.880 | 0.907 | **0.651** | **0.5198 ± 0.0130** | 1.1566 ± 0.0412 | 0.773 | 1.116 |
| 6 | 5.33 | 4.10e-3 | 0.645 | 0.783 | 0.803 | 0.183 | **0.2405 ± 0.0031** | 1.1361 ± 0.0342 | 0.384 | 1.153 |
| 8 | 4.0 | 3.11e-3 | 0.706 | 0.842 | 0.840 | 0.359 | **0.2214 ± 0.0141** | 1.0939 ± 0.0097 | 0.440 | 1.071 |
| 10 | 3.2 | 1.25e-3 | 0.698 | 0.760 | 0.835 | 0.087 | **0.1764 ± 0.0289** | 1.0664 ± 0.0654 | 0.211 | 0.992 |
| 12 | 2.67 | 1.11e-3 | 0.684 | 0.816 | 0.827 | 0.379 | **0.1756 ± 0.0035** | 1.8374 ± 0.0981 | 0.288 | 1.607 |
| 16 | 2.0 | 5.06e-4 | 0.679 | 0.769 | 0.824 | 0.271 | **0.1567 ± 0.0138** | 1.6009 ± 0.1204 | 0.160 | 1.151 |

## 1. Is the clean-probe curve flat? **No.**

| | `k32` | `k64` |
| :-- | --: | --: |
| `spread_b` (max−min over the 8 bands, seed mean) | **1.2066** | **0.8587** |
| pooled per-band seed sd | 0.0661 | 0.0258 |
| **`spread_b` / seed sd** | **18.3** | **33.3** |
| `corr(log2 b, r_probe)` | **−0.87** | **−0.96** |
| max/min over the bands | 7.9 (7.9 excl. `b=16`) | 6.5 (5.8 excl. `b=16`) |
| mean `r_probe` | 0.4328 | 0.3902 |

Against T1's calibration on the same corpus, `P_own` over these 8 bands spans only
**1.31× / 1.32×**. A flat curve is excluded at 18–33× the seed noise, and the trend is
monotone in `log b` apart from a single upturn at the Nyquist band of `k32` (`b = 16`:
0.319 against 0.175 at `b = 12` — the one band whose target `|Z|` is not constant, §2).

Two sub-results worth separating:

- **The lowest band is over-emitted, not under-emitted.** `r_probe = 1.38` at `b = 1` in the
  `k32` arm and 1.02 in `k64` — gain *inflation*, P0's artifact, now at the band carrying
  40% of the corpus power. If the low band is excluded the remaining decline is still
  3.6× (`k32`: 0.636 → 0.175) and 3.9× (`k64`: 0.615 → 0.157) against pooled seed sds of
  0.047 / 0.021 over the remaining seven bands. **The slope is not a low-band artifact.**
- **The deficit is largest exactly where the corpus is quietest.** At `b = 8…12` `r_probe`
  is 0.175–0.185 (`k32`) and 0.157–0.221 (`k64`), against `sqrt(P) = 0.83–0.85` — a factor
  of **4.5–5.3** — and 4.5× below what T1's own ridge achieves in distribution on the same
  bands (0.80–0.85, `--relation`).

## 2. Which relation holds between `P` and `r`? — `r ≈ sqrt(P)`, measured model-free

`--relation` re-runs T1's estimator and reads **its own** amplitude retention on the same
held-out half (`r_ms` = pooled RMS ratio `sqrt(Σ|Z_pred|²/Σ|Z_ev|²)`, `r_own` = mean of the
per-window ratios), so the relation is measured without training anything:

| | `k32` | `k64` |
| :-- | --: | --: |
| `r_ms / sqrt(P_own)`, all 32 bands | **mean 0.998**, range 0.971–1.027 | **mean 0.994**, range 0.938–1.105 |
| `r_ms / sqrt(P_full)`, all 32 bands | range 0.954–1.038 | range 0.968–1.072 |
| `r_own / sqrt(P_own)` (mean-of-ratios) | mean **1.362** | mean **1.263** |
| `r_own / P_own` | mean 1.55 | mean 1.50 |

**So the relation is `r ≈ sqrt(P)`, and the R²-vs-amplitude-ratio worry T1 flagged is real
but small and convention-dependent:** the *pooled* amplitude retention equals `sqrt(P)` to
±5% band by band at both apertures, while the *mean-of-ratios* convention reads a
systematic **+26…36%** above it (small `|Z_target|` windows inflate the ratio; the corpus
has heavy-tailed per-window band coefficients, `r_own` reaches 2.73 at `k32`'s `b = 16`).

For the **probe**, the two conventions coincide: every probe tone sits at an integer `b`
with a fixed amplitude, so `|Z_target, b|` is the *same in every window* — verified:
exactly 0.5014 at `k32` and 0.5000 at `k64` in every window and every band, **except
`b = 16` at `k32`**, where the Nyquist degeneracy makes it vary (mean 0.626, sd 0.299,
range 0.005–1.003). So for `b ≤ 12` there is no small-denominator tail and no bias, and
`r_probe` is directly comparable to `sqrt(P)` — where it is 4–5× below it at the top
bands. The `k32` `b = 16` row alone carries a mean-of-ratios bias and is read separately.

## 3. Power vs predictability vs retention — T1's headline, model-side

Over the 8 probe bands (`b = 1 → 16`):

| arm | band power | `P_own` | `sqrt(P_own)` | clean-probe `r_b` |
| :-- | --: | --: | --: | --: |
| `k32` | **617×** (4.32e-1 → 7.00e-4) | 1.31× (0.917 → 0.699) | 1.14× | **7.9×** (1.382 → 0.175) |
| `k64` | **694×** (3.51e-1 → 5.06e-4) | 1.32× (0.849 → 0.645) | 1.15× | **5.8×** (1.015 → 0.176) |

T1's contrast was energy 463× against predictability 1.25×. The model-side counterpart is
this table's last column: the model's per-band retention varies **6–8×**, i.e. it is not
the flat curve the predictability curve predicts. It is also not the power curve — power
falls 617–694×, which no retention curve bounded near 1 can reproduce. What the retention
curve tracks is closer to the corpus's *band amplitude* than to either: `r_probe` ratios
against `b = 1` are 1.00/0.46/0.26/0.16/0.13/0.13/0.13 at `k32` while `sqrt(power)` ratios
are 1.00/0.38/0.14/0.08/0.06/0.05/0.04 — correlated but not equal, and flattened at the
top (the model floors at ~0.13 rather than continuing down).

## 4. The `in-band` column, and the one real confound

`in-band` (T1's leakage fraction, within-aperture only) sits beside `r_b` in both tables.
At `k = 32` it is 0.96 at `b = 1`, 0.34 at `b = 2`, and 0.05–0.15 for `b ≥ 4`: the
corpus's "high bands" are 85–95% leakage from the clock lines and the red continuum. The
same leakage runs through `P(b)` and through the model's `r_b`, which is why the two stay
comparable — but it means:

- **`r_corpus` is not evidence about the high bands.** At `b ≥ 4` both `Z_pred` and
  `Z_target` are carried by *the same leaked low-frequency content*, which the model
  reproduces well, so the ratio reads ≈1 (1.02–1.29 at `b = 1…12`, and 3.27 at the
  Nyquist band). That ≈1 is a leakage cancellation, not retention of the band's own
  content. The informative in-distribution bands are `b ≤ 2`.
- **The model is still *at* T1's ridge in distribution**, band by band: `r_corpus` 1.09 /
  1.02 / 1.14 / 1.14 / 1.29 / 1.24 / 1.23 / 3.27 against the ridge's own `r_own` 1.12 /
  1.02 / 1.16 / 1.08 / 1.16 / 1.17 / 1.13 / 2.73 (`--relation`). Nothing in the model's
  in-distribution per-band behaviour is below what the best linear estimator of this
  corpus achieves.
- **The `k = 64` arm gives the one clean high-band datapoint**: `b = 4` there is 8 h with
  `in-band` **0.651** — the corpus really does carry that band — and the clean probe still
  reads 0.52 against `sqrt(P) = 0.91`.

**The confound that remains, stated plainly.** Contract A9 matches the probe's total RMS to
the corpus and nothing more. On this corpus that leaves the probe with ~0.126 power in
*every* band: 0.29× the corpus's own lowest band and **180×** the corpus's own highest band.
A probe that is flat-spectrum against a corpus that is 617× red is out of distribution in
spectral *shape*, so a frequency-dependent response to it is not by itself the band-wise
prior the synthetic experiments measured — in P2b/P3 the training corpus had equal power in
every band, so the probe was spectrally in distribution and only the phase structure
changed. T2 as designed therefore cannot separate "a band-wise prior calibrated to
predictability" from "a learned response to spectral shape / power". **T3 (per-patch phase
randomization, which preserves band power exactly and moves only predictability) is the
experiment that separates them**, and this result is the argument for running it.

## 5. The oracle (contract A5)

A ridge readout of the same frozen latents, fitted on the corpus's held-out windows and
read at the probe (`or_probe`) and at the corpus (`or_corpus`), is in both tables. It
bounds what a *linear* readout of this representation could do.

- On the corpus it reads 1.01–1.22, and 2.13 at `k32`'s `b = 16` — i.e. it saturates at
  the same ≈1 the head reads, with the same interpretation.
- On the probe it **reproduces the head's decline** — at `k32`
  2.79/1.17/0.58/0.42/0.22/0.15/0.15/0.32 against the head's
  1.38/0.64/0.36/0.22/0.19/0.18/0.18/0.32, and at `k64`
  1.61/0.77/0.77/0.38/0.44/0.21/0.29/0.16 against 1.02/0.62/0.52/0.24/0.22/0.18/0.18/0.16.
  Both are far below the `sqrt(P) ≈ 0.80–0.96` the marginal implies, and neither shows a
  flat curve. So the head's shortfall is **not** a head-only optimisation gap of P3's kind:
  a closed-form linear readout of the same frozen latents is at least as damped at the top
  of the range (`k32` `b = 10, 12`: 0.151/0.147 against the head's 0.183/0.175).
- Caveat, contract 5's own: the oracle is itself fitted to minimise corpus MSE, and the
  corpus is red, so its high-band inability is *also* a consequence of the marginal's
  power spectrum. The oracle rules out "the head alone is at fault"; it does not rule out
  "the marginal's spectral shape, rather than its predictability, set the gain".

## Escalation: **not needed**

The brief's escalation triggers if an arm does not clearly beat both baselines. Both arms
beat both baselines by **+0.76 / +0.75** against a required margin of 0.02, with seed sds of
0.0016, and per-band variance explained positive at every band. One larger config
(`hidden 64 / 4 layers`, same `ctx`/`k`, same steps) would have cost ≈2.4× per step
(51 ms against 22 ms, measured before the run) — ≈17 min per run, 6 runs — and would have
answered a question the gate does not raise, so it was not run. Recorded here so the
decision is visible: **the escalation was available and skipped on the gate's own terms,
not because of the result.**

## Pass criteria (stated before the run)

| # | criterion | outcome |
| --: | :-- | :-- |
| 9 | both arms clear contract A8's gate against persistence and the mean predictor | **pass** — +0.9161/+0.8771 against −0.8995/−1.0028 and +0.1526/+0.1316 |
| 10 | the training corpus is T1's, aperture and demeaning included | **pass** — `check_aperture` asserts band-by-band equality at `atol 1e-9`; `P` and `in-band` reproduce `aperture.json` |
| 11 | the probe satisfies contract A4 (`Δb ≥ 1`) and A9 (RMS match) | **pass** — minimum gap exactly 1; probe/corpus RMS ratio 1.0000 both arms |
| 12 | 3 seeds per arm, spread reported beside every reading | **pass** — 6 runs; `spread_b`/seed-sd 18.3 and 33.3 |
| 13 | the converged state is reported and the transient logged (A2) | **pass** — 6 checkpoints per run; the probe curve's shape is fixed by step 1 000 |
| 14 | nothing is tuned to make the curve flat or sloped | **pass** — one config, chosen before the run; the result is *sloped*, and reported as such |
| 15 | the clean-probe curve is flat | **FAIL** — 7.9× / 5.8× decline, 18× / 33× the seed sd |

## Scripts, reproduce, wall time

| script | role | wall |
| :-- | :-- | --: |
| `scripts/t2.py --arm k32 --seed 0` → `runs/t2_k32_s0.json` | one arm/seed (data + 20 k steps) | **1 640 s** |
| `scripts/t2.py --summarize` → `runs/t2_summary.json` | the tables above, pooled | <1 s |
| `scripts/t2.py --relation` → `runs/t2_relation.json` | T1's ridge as a retention, model-free | **13.5 s** |

```
OMP_NUM_THREADS=2 .venv/bin/python experiments/1_realdata/scripts/t2.py --arm k32 --seed 0
OMP_NUM_THREADS=2 .venv/bin/python experiments/1_realdata/scripts/t2.py --arm k64 --seed 0
# ... one process per (arm, seed); 6 in parallel = 1 640 s wall for the whole grid
.venv/bin/python experiments/1_realdata/scripts/t2.py --summarize
.venv/bin/python experiments/1_realdata/scripts/t2.py --relation
```

CPU only, no downloads, `Salesforce/GiftEvalPretrain` untouched. Batch wall **1 640 s** for
the 6 trainings (each process 2 torch threads; 6 concurrent on 12 cores). T1's estimator
inside each run costs 39–47 s and is re-verified against `aperture.json` on every run.

## Where this contradicts or refines the brief

9. **The prediction in the brief is contradicted.** "A model trained on this corpus should
   show nearly flat per-band attenuation" is false here: the clean-probe curve declines 7.9×
   (`k32`) / 5.8× (`k64`) with `spread_b` at 18–33× the seed sd, established by step 1 000.
   Reported as the more interesting outcome the brief asks for, not smoothed.
10. **Contract A9 is not sufficient on a red corpus.** Matching the probe's *total* RMS
    leaves it 180× above the corpus's own high-band power. The synthetic experiments never
    hit this because their corpora were spectrally flat by construction. The clean-probe
    readout on a red corpus therefore tests the response to spectral *shape* as well as a
    band-wise prior, and this experiment cannot separate them.
11. **The `in-band` column is what keeps the comparison honest — and it also caps it.** At
    `k = 32`, `b ≥ 4` has `in-band ≤ 0.15`, so both `P(b)` and `r_b` there are
    leakage-carried; `r_corpus ≈ 1` at those bands is a leakage cancellation and must not
    be read as retention. The one well-isolated high band the design offers is `k64`'s
    `b = 4` (8 h, `in-band` 0.65), where the deficit is 0.52 against 0.91.
12. **`P` and `r` are related by `r ≈ sqrt(P)`, measured rather than assumed.** T1's ridge
    read as a retention gives `r_ms/sqrt(P) = 0.998` (`k32`) / 0.994 (`k64`) over all 32
    bands, ±5% band by band; the mean-of-ratios convention is biased +26…36% above that.
    For the clean probe the two conventions coincide (constant denominator at every band of
    `k64` and every band up to `b = 12` of `k32`), so `r_probe` is directly comparable to
    `sqrt(P)`.
13. **The low band is inflated, not merely retained.** `r_probe = 1.38 ± 0.14` at `b = 1`
    in the `k32` arm — P0's gain-inflation artifact reappearing at the band that carries
    40% of the corpus's power. Every low-band statement must carry it.
14. **`b = 16` at `k = 32` is the sampling Nyquist**, where the matched filter's ± images
    coincide and a pure tone reads `|Z| = 2A`. Its probe bin power is 0.2405 against 0.1257
    elsewhere, so the `b = 16` row is flagged wherever it could matter.
15. **One metric from Appendix B was not produced: phase error.** The retention reading
    is amplitude-only and the brief did not ask for it, so it is recorded here as a gap
    rather than silently omitted.

---

# T2b — the shape-hypothesis control: flatten the corpus, keep the probe

Added 2026-09-16. `scripts/t2b.py` → `runs/t2b/<variant>_<arm>_s<seed>.json`
(**15 trainings**: 3 corpus variants × 2 apertures × 3 seeds, the dose arm at one
aperture only), pooled into `runs/t2b/summary.json`, plus
`runs/t2b/flatten.json` (the filter's own diagnostics, no training). This is the
control the T2 section above asks for in its first "next step".

## Answer, up front

**The decline vanishes.** Flattening the corpus's mean power spectrum — and
changing nothing else, with the probe byte-identical — takes the clean-probe
decline from

| | `k32` (16 h horizon) | `k64` (32 h horizon) |
| :-- | --: | --: |
| arm A (T2's corpus, 617× / 694× dynamic range) | **7.89×** | **5.78×** |
| arm B (flattened, 3.11× / 3.04× dynamic range) | **1.12×** | **0.85×** |

and the trend with `log2 b` from `corr = −0.87 / −0.96` to `−0.30 / +0.29`. Arm B
is a *flat* curve with a 2.0× (k32) / 2.4× (k64) band-to-band spread that is not
monotone in `b`, while the predictability curve `P(b)` that both arms re-measure
is comparably flat (`sqrt(P)` spans 1.37× at k32, 1.08× at k64).

**So T2's decline was a spectral-shape / dynamic-range effect.** The band-wise
prior "calibrated to predictability" is **not supported on this corpus**: where
the corpus's dynamic range is removed, so is the slope, and the residual
structure in arm B tracks the corpus's *band power* (`corr(r_probe, power) =
+0.86` at k64, with `corr(r_probe, sqrt(P)) = −0.08`). A dose check agrees —
the decline shrinks monotonically with the dynamic range:

| corpus | dynamic range (8 probe bands) | decline `r(b=1)/r(b=12)` | `corr(log2 b, r)` |
| :-- | --: | --: | --: |
| A (`α = 0`, T2's) | 617× | 7.89× | −0.87 |
| C (`α = 0.5`, dose) | 81× | 5.37× | −0.94 |
| B (`α = 1`, flat) | 3.11× | 1.12× | −0.30 |

Two things that **do not** change and must travel with the answer:

- **A band-flat deficit survives.** In arm B the model still returns only
  45–80% of `sqrt(P)` on the clean probe — the same at every band. The *level* of
  T2's deficit was not shape-driven; only its *slope* was. That residual is
  uniform across bands and is the clean-probe/OOD or uniform-optimisation-gap
  story (P0's artifact, P3's gap), not a frequency effect.
- **The dose arm does not lie on a clean line.** `α = 0.5` cuts the dynamic range
  by 7.6× and the decline by only 1.5×; the big change happens between 81× and
  3.1×. The response is monotone, not proportional.

## The arms, and the one thing that changes

| arm | `α` | training corpus | probe |
| :-- | --: | :-- | :-- |
| **A** (reproduce T2) | 0 | the corpus as T2 used it | T2's flat clean probe |
| **B** (flattened) | 1 | the same corpus, spectrally flattened | **the identical probe** |
| **C** (dose) | 0.5 | half-flattened; `k32` only | the identical probe |

The probe is `t2.probe_windows` unchanged: `b ∈ {1, 2, 4, 6, 8, 10, 12, 16}`,
`f = 16b`, 256 windows, seed 123, minimum gap exactly 1 cycle/patch (A4). Its
per-band power is the same number in every arm up to contract A9's scale factor
— `0.12566 / 0.12500 / 0.12575` at k32 (A / B / C) and `0.12115 / 0.12229` at k64,
i.e. the probe's **shape** is identical and only its RMS follows that arm's
corpus (A9 ratio 1.0000 in all five runs). Everything else is T2's, imported
rather than re-implemented: same 2 122 / 1 061 non-overlapping windows, same
per-series centre and unit variance, same per-patch demeaning before the matched
filter, same split, same model (hidden 32 / 2L / 4 heads, SGD 1e-2, batch 64),
same 20 000 steps, same 3 seeds, same band grid, same readouts, same fit gate
with persistence and mean-predictor baselines.

## The flattening: method, parameters, flatness

One linear filter on each whole series,

```
g(f) = S(f) ** (−α / 2)          α = 1 (B), 0.5 (C), 0 (A, identity)
```

where `S` is the corpus's mean power spectrum: the mean over the five series of
their whole-record periodograms, interpolated onto a common frequency grid and
smoothed with a **truncated boxcar of half-width `1/64` cycles/sample**. The
series are re-centred and re-scaled to unit variance afterwards (T1's
per-series normalisation), so the filter's overall scale is irrelevant.

Two parameters, both fixed before the run and both tied to the aperture rather
than to the outcome:

- **The smoothing width `±1 b` at `k_ref = 64`** (= `±1/64` cycles/sample). The
  matched filter's mainlobe is `±1 b` wide at *every* `k` (nulls at integer
  `Δb`, contract A4), and a local mean over that width spreads a coherent line
  across the window by the same amount the aperture integrates it into a band
  reading — with matched kernels the two are the same estimator. `k = 64` is the
  finer of the two apertures read here, so this is the narrower of the two
  defensible choices; the price is the residual comb at the `k64` aperture
  measured below, and a narrower window would trade it for a deeper notch at each
  line (untested — the width was fixed from the geometry, not from the result).
- **No floor and no ceiling.** There is no noise floor in this data to protect:
  0.1 MW quantisation against series sds of 147–1 362 MW. The realised gain spans
  `0.25 → 32` over `f ∈ [4e-6, 0.5]` (at `f = 0.01, 0.02, 0.05, 0.1, 0.2, 0.3,
  0.4, 0.5`: `0.25, 0.36, 0.63, 3.46, 8.79, 17.96, 32.21, 23.05`), i.e. a 128×
  span of gain covering the corpus's 617× of power.

One implementation detail is worth recording because getting it wrong changes
the answer at the band edge: the smoothing window is **truncated**, not padded
with the edge value. The raw periodogram of a 13-year record has coherent
components landing in single bins, and two of the five series have a Nyquist bin
3 000× and 6 000× their local continuum (a component worth 7e-5 of the variance,
but 20 000× the local *density*). Replicating that one edge bin inflated the local
estimate at the top of the band by **1 430×** — the gain there by 38× — and the
measured `b = 16` band by 3.2× (less than the gain error only because in the
wrong version that band's reading is 99.7% sidelobe leakage: its `in-band` goes
0.003 → 0.687). The first version of this filter was rejected over exactly that,
before any training was run; every number here is the corrected one. The `k32`
`b = 16` band is still the one place the flattening does not reach — a single
remaining bin spike leaves its local estimate 4.7× high, so the band reads 3.04e-2
against a flat level of 6e-2 — and it is flagged wherever it could matter.

### How flat the result actually is

Max/min band power (T1's band units: `|Z|²/2` of the demeaned patches at the 8
probe bands), before and after:

| arm | `k32` before → after | `k64` before → after | 32-band grid after | `in-band` at the probe bands, after |
| :-- | --: | --: | --: | :-- |
| A | 616.60 → 616.60 | 693.70 → 693.70 | 642.72 / 1 675.31 | 0.047–0.962 / 0.087–0.886 |
| B | **616.60 → 3.11** | **693.70 → 3.04** | 3.11 / 5.35 | 0.618–0.899 / 0.056–0.916 |
| C | 616.60 → 81.04 | 693.70 → 40.70 | 81.04 / 89.37 | 0.422–0.950 / 0.082–0.882 |

At `k32` the 8-band spread is 3.11×, but **1.67× excluding the Nyquist band**,
where the flattened corpus reads 3.04e-2 against a flat level of ~6e-2 (the band
is 2× low and its local estimate is still 4.7× high — the one place the edge
spike survives, above). At `k64` it is 3.04× and that residual is a **harmonic
comb**: the bands that sit on exact diurnal harmonics (`b = 4, 8, 12, 16` → 8 h,
4 h, 2.67 h, 2 h) read 5.4/5.1/5.1/4.3e-2 while the bands between them
(`b = 2, 6, 10`) read 2.4/1.8/2.0e-2. The filter's gain is smooth at the
aperture's resolution, so it cannot notch individual harmonics; at `k32`'s
aperture (`±2` harmonic spacings) the same corpus measures flat to 1.67×, at
`k64`'s aperture (`±0.75` spacings) it does not. This is a stated limitation of a
one-pass smooth equaliser, and it is the *only* structure left in arm B's corpus.

The shape confound T2 could not resolve is therefore gone: the **probe/corpus
per-band power ratio** goes from `0.29 … 180` at `k32` (`0.35 … 239` at `k64`) in
arm A to `1.33 … 4.14` (`2.26 … 6.86`) in arm B. The probe is still above the
corpus at every band (necessarily: A9 matches *total* RMS, and an 8-tone probe
puts 1/8 of its variance in each of 8 bands while a flat corpus puts ~1/16 into
each of 32 overlapping bands), but it is now above it *uniformly*, by a factor
that varies 3× instead of 620×.

The flattening also removes T2's second caveat for free: in arm B, `in-band` is
0.62–0.90 at every `k32` probe band against 0.047–0.962 in arm A, i.e. the
high-band readings are the bands' own content rather than leakage.

### RMS (contract A9) and `P(b)` re-measured

| arm / k | corpus window RMS before | after | probe RMS | A9 ratio |
| :-- | --: | --: | --: | --: |
| A / k32 | 0.99995 | 0.99995 | 0.99995 | 1.0000 |
| A / k64 | 0.98448 | 0.98448 | 0.98448 | 1.0000 |
| B / k32 | 0.99995 | 0.99729 | 0.99729 | 1.0000 |
| B / k64 | 0.98448 | 0.98911 | 0.98911 | 1.0000 |
| C / k32 | 0.99995 | 1.00028 | 1.00028 | 1.0000 |

The flattening is a linear filter and the conditional structure is transformed
coherently rather than scrambled, and the measurement says so: `P_own` is
re-measured on each arm's own corpus and survives. Over the 8 probe bands,
`P_own` spans **0.917 → 0.699** (1.31×) in A/k32 and **0.946 → 0.506** (1.87×) in
B/k32; at k64, **0.850 → 0.645** (1.32×) in A and **0.921 → 0.779** (1.18×) in B.
So arm B is not a corpus that lost its structure — it is a corpus with a flat
spectrum and just as much per-band conditional predictability.

## Arm A reproduces T2, field by field

`--summarize` compares every pooled quantity against `runs/t2_summary.json`:

```
k32: max |arm A − T2| over P_full 1.67e-15, in_band 1.80e-16, P_own 1.11e-16, power 5.55e-17
k64: max |arm A − T2| over P_full 3.33e-15, in_band 5.00e-16, P_own 2.22e-16, power 1.11e-16
worst difference 3.331e-15 -> REPRODUCES T2
```

The training is bit-identical, not merely close: `probe_mean`, `probe_sd`,
`corpus_mean`, `oracle_probe_mean`, `band_var_explained`, `ve_last`, the gate's
`ve_last` / persistence / mean, `spread_b`, the seed sd and `corr(log2 b, r)` all
differ by **exactly 0.0**. The transient reproduces too — arm A's seed-mean
per-band `r_probe` at step 1 000 is `1.006 / 0.501 / 0.339 / 0.244 / 0.258 /
0.229 / 0.266 / 0.420`, T2's published numbers. Arm A is the identity filter by
construction (`α = 0`), and `run_arm` asserts `np.array_equal` against
`t2.raw_windows` before training, so "arm A" is T2's corpus byte for byte and not
a re-implementation.

## Table A — power, predictability, retention, per arm (3 seeds, mean ± sd)

`power` = `|Z|²/2` of that arm's demeaned patches (T1's units); `in-band` = T1's
leakage column; `r_probe` = the clean probe; `r_corpus` = the held-out corpus
windows; `or_probe` = contract 5's oracle at the probe; `probe/corpus` = the
probe's per-band power over the corpus's — the shape-confound ratio, which T2
quoted as 0.29 … 180.

### Arm A (`α = 0`, T2's corpus) — `k32`

| b | T(h) | power | `P_own` | `P_full` | sqrt(P) | in-band | `r_probe` | `r_corpus` | `or_probe` | probe/corpus |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 16.00 | 4.31e-01 | 0.917 | 0.939 | 0.958 | 0.962 | **1.3816 ± 0.1398** | 1.0906 | 2.790 | 0.29 |
| 2 | 8.00 | 6.13e-02 | 0.826 | 0.900 | 0.909 | 0.343 | **0.6364 ± 0.0362** | 1.0243 | 1.175 | 2.05 |
| 4 | 4.00 | 8.68e-03 | 0.749 | 0.881 | 0.866 | 0.145 | **0.3585 ± 0.0454** | 1.1361 | 0.585 | 14.48 |
| 6 | 2.67 | 2.93e-03 | 0.722 | 0.863 | 0.849 | 0.154 | **0.2233 ± 0.0059** | 1.1405 | 0.420 | 42.89 |
| 8 | 2.00 | 1.48e-03 | 0.699 | 0.831 | 0.836 | 0.103 | **0.1851 ± 0.0342** | 1.2934 | 0.219 | 84.92 |
| 10 | 1.60 | 1.04e-03 | 0.712 | 0.845 | 0.844 | 0.064 | **0.1833 ± 0.0039** | 1.2394 | 0.151 | 121.45 |
| 12 | 1.33 | 8.56e-04 | 0.715 | 0.840 | 0.846 | 0.047 | **0.1750 ± 0.0210** | 1.2277 | 0.147 | 146.76 |
| 16 | 1.00 | 7.00e-04 | 0.718 | 0.845 | 0.847 | 0.096 | **0.3194 ± 0.1018** | 3.2666 | 0.317 | 179.62 |

### Arm A — `k64`

| b | T(h) | power | `P_own` | `P_full` | sqrt(P) | in-band | `r_probe` | `r_corpus` | `or_probe` | probe/corpus |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 32.00 | 3.51e-01 | 0.850 | 0.889 | 0.922 | 0.886 | **1.0154 ± 0.0484** | 1.0196 | 1.605 | 0.35 |
| 2 | 16.00 | 1.14e-01 | 0.849 | 0.904 | 0.921 | 0.095 | **0.6154 ± 0.0394** | 1.1735 | 0.766 | 1.06 |
| 4 | 8.00 | 3.02e-02 | 0.823 | 0.880 | 0.907 | 0.651 | **0.5198 ± 0.0130** | 1.1566 | 0.772 | 4.02 |
| 6 | 5.33 | 4.10e-03 | 0.645 | 0.783 | 0.803 | 0.183 | **0.2405 ± 0.0031** | 1.1361 | 0.384 | 29.53 |
| 8 | 4.00 | 3.11e-03 | 0.706 | 0.842 | 0.840 | 0.359 | **0.2214 ± 0.0141** | 1.0939 | 0.440 | 38.95 |
| 10 | 3.20 | 1.25e-03 | 0.698 | 0.760 | 0.835 | 0.087 | **0.1764 ± 0.0289** | 1.0664 | 0.211 | 96.66 |
| 12 | 2.67 | 1.11e-03 | 0.684 | 0.816 | 0.827 | 0.379 | **0.1756 ± 0.0035** | 1.8374 | 0.288 | 108.85 |
| 16 | 2.00 | 5.06e-04 | 0.679 | 0.769 | 0.824 | 0.271 | **0.1567 ± 0.0138** | 1.6009 | 0.159 | 239.39 |

### Arm B (flattened) — `k32`

| b | T(h) | power | `P_own` | `P_full` | sqrt(P) | in-band | `r_probe` | `r_corpus` | `or_probe` | probe/corpus |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 16.00 | 9.45e-02 | 0.946 | 0.948 | 0.973 | 0.899 | **0.4328 ± 0.0390** | 1.0211 | 0.540 | 1.33 |
| 2 | 8.00 | 6.92e-02 | 0.836 | 0.864 | 0.914 | 0.732 | **0.5650 ± 0.0165** | 1.0366 | 0.656 | 1.82 |
| 4 | 4.00 | 7.07e-02 | 0.800 | 0.854 | 0.894 | 0.668 | **0.7112 ± 0.0083** | 1.0509 | 0.683 | 1.78 |
| 6 | 2.67 | 7.13e-02 | 0.790 | 0.822 | 0.889 | 0.655 | **0.5497 ± 0.0025** | 1.0327 | 0.444 | 1.76 |
| 8 | 2.00 | 5.67e-02 | 0.710 | 0.726 | 0.843 | 0.729 | **0.5533 ± 0.0061** | 1.0108 | 0.516 | 2.22 |
| 10 | 1.60 | 6.11e-02 | 0.575 | 0.591 | 0.758 | 0.716 | **0.5180 ± 0.0045** | 1.0444 | 0.479 | 2.06 |
| 12 | 1.33 | 5.86e-02 | 0.506 | 0.528 | 0.711 | 0.618 | **0.3875 ± 0.0090** | 0.9528 | 0.387 | 2.15 |
| 16 | 1.00 | 3.04e-02 | 0.562 | 0.563 | 0.750 | 0.687 | **0.3540 ± 0.0029** | 4.7553 | 0.506 | 4.14 |

### Arm B — `k64`

| b | T(h) | power | `P_own` | `P_full` | sqrt(P) | in-band | `r_probe` | `r_corpus` | `or_probe` | probe/corpus |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 32.00 | 4.58e-02 | 0.909 | 0.913 | 0.953 | 0.852 | **0.5257 ± 0.0159** | 1.0566 | 0.606 | 2.65 |
| 2 | 16.00 | 2.43e-02 | 0.921 | 0.935 | 0.959 | 0.056 | **0.3168 ± 0.0123** | 1.0434 | 0.340 | 4.98 |
| 4 | 8.00 | 5.37e-02 | 0.859 | 0.876 | 0.927 | 0.916 | **0.4829 ± 0.0096** | 0.9839 | 0.537 | 2.26 |
| 6 | 5.33 | 1.77e-02 | 0.793 | 0.858 | 0.890 | 0.242 | **0.3157 ± 0.0232** | 1.0783 | 0.343 | 6.86 |
| 8 | 4.00 | 5.05e-02 | 0.811 | 0.866 | 0.900 | 0.881 | **0.6808 ± 0.0531** | 0.9495 | 0.746 | 2.40 |
| 10 | 3.20 | 2.03e-02 | 0.779 | 0.776 | 0.883 | 0.239 | **0.2807 ± 0.0181** | 0.9315 | 0.334 | 5.98 |
| 12 | 2.67 | 5.08e-02 | 0.832 | 0.844 | 0.912 | 0.875 | **0.6182 ± 0.0130** | 1.0720 | 0.634 | 2.38 |
| 16 | 2.00 | 4.27e-02 | 0.790 | 0.772 | 0.889 | 0.902 | **0.6466 ± 0.0222** | 1.0026 | 0.647 | 2.84 |

### Arm C (dose, `α = 0.5`) — `k32`

| b | T(h) | power | `P_own` | `P_full` | sqrt(P) | in-band | `r_probe` | `r_corpus` | `or_probe` | probe/corpus |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 16.00 | 3.86e-01 | 0.932 | 0.946 | 0.965 | 0.950 | **0.9095 ± 0.0668** | 1.0847 | 1.249 | 0.33 |
| 2 | 8.00 | 1.17e-01 | 0.835 | 0.898 | 0.914 | 0.590 | **0.6674 ± 0.0502** | 1.0296 | 0.830 | 1.08 |
| 4 | 4.00 | 3.48e-02 | 0.777 | 0.871 | 0.882 | 0.468 | **0.5928 ± 0.0433** | 1.0321 | 0.565 | 3.62 |
| 6 | 2.67 | 1.88e-02 | 0.784 | 0.854 | 0.885 | 0.546 | **0.4351 ± 0.0026** | 1.0696 | 0.473 | 6.69 |
| 8 | 2.00 | 9.37e-03 | 0.710 | 0.758 | 0.842 | 0.617 | **0.3508 ± 0.0186** | 1.0366 | 0.392 | 13.41 |
| 10 | 1.60 | 7.06e-03 | 0.649 | 0.655 | 0.805 | 0.526 | **0.2244 ± 0.0213** | 1.0010 | 0.264 | 17.80 |
| 12 | 1.33 | 6.52e-03 | 0.610 | 0.655 | 0.781 | 0.422 | **0.1694 ± 0.0235** | 1.0824 | 0.179 | 19.27 |
| 16 | 1.00 | 4.76e-03 | 0.680 | 0.696 | 0.825 | 0.635 | **0.3571 ± 0.0761** | 2.5634 | 0.370 | 26.39 |

## 1. Is the clean-probe curve flat?

| | A/k32 | A/k64 | B/k32 | B/k64 | C/k32 |
| :-- | --: | --: | --: | --: | --: |
| `spread_b` (max−min, seed mean) | 1.2066 | 0.8587 | **0.3572** | **0.4001** | 0.7401 |
| pooled per-band seed sd | 0.0661 | 0.0258 | 0.0159 | 0.0246 | 0.0447 |
| `spread_b` / seed sd | 18.25 | 33.31 | 22.49 | 16.25 | 16.55 |
| `corr(log2 b, r_probe)` | −0.87 | −0.96 | **−0.30** | **+0.29** | −0.94 |
| decline `r(b=1)/r(b=12)` | 7.89 | 5.78 | **1.12** | **0.85** | 5.37 |
| max/min over the 8 bands | 7.89 | 6.48 | 2.01 | 2.43 | 5.37 |
| mean `r_probe` | 0.4328 | 0.3902 | 0.5089 | 0.4834 | 0.4633 |
| corpus dynamic range (8 bands) | 616.60 | 693.70 | 3.11 | 3.04 | 81.04 |
| `P_own` max/min (8 bands) | 1.31 | 1.32 | 1.87 | 1.18 | 1.53 |

The `spread_b / seed sd` column keeps a large number in arm B (22.5 / 16.3), and
the reading matters: `spread_b` is still the same *statistic* but it is no longer
a *slope*. What is left in arm B is a band-to-band wobble of 0.36 / 0.40 that is
not monotone in `log2 b` (correlations −0.30 / +0.29, i.e. the sign flips between
the two apertures) and is 3× below arm A's `spread_b` — and, at `k64`, it is a
near-perfect read of the corpus's own residual comb (below). The seed sd is
*three times smaller* in arm B than in arm A at `k32` (0.0159 against 0.0661), so
the ratio is large mainly because the noise fell, not because the spread grew.
Agreement with the pre-stated prediction is decided by the decline and by
`corr(log2 b, r)`, not by that ratio.

## 2. Which variable does `r_probe` track inside an arm?

Both candidate drivers are measurable in each arm: the corpus's per-band **power**
(the shape account) and the corpus's per-band **predictability** (the
predictability account). Correlations over the 8 probe bands, seed mean:

| | A/k32 | A/k64 | B/k32 | B/k64 | C/k32 |
| :-- | --: | --: | --: | --: | --: |
| `corr(power, P_own)` (are the two drivers even separable?) | +0.91 | +0.72 | +0.79 | **+0.19** | +0.83 |
| `corr(r_probe, power)` | +0.97 | +0.94 | +0.36 | **+0.86** | +0.86 |
| `corr(r_probe, sqrt(P_own))` | +0.97 | +0.88 | +0.48 | **−0.08** | +0.97 |
| `corr(r_probe, P_own)` | +0.98 | +0.88 | +0.46 | −0.09 | +0.97 |
| `r/sqrt(P)` range | 0.21–1.44 | 0.19–1.10 | 0.45–0.80 | 0.32–0.76 | 0.22–0.94 |
| mean `r/sqrt(P)` | 0.48 | 0.44 | 0.60 | 0.53 | 0.52 |

The first row is the one that decides whether a correlation can discriminate at
all. In arm A the two candidate drivers are themselves correlated (+0.91 / +0.72
— power falls 617× and `P(b)` falls 1.31× *in the same direction*), so both
predictors rank the bands almost the same way and both read ≈ +0.95 with the
retention. **Arm B/k64 is the one arm where they are decoupled (+0.19)**, and
there the retention follows the corpus's residual band power at `+0.86` while its
correlation with `sqrt(P)` is `−0.08`. That is the sharpest single piece of
evidence in this run, and it is a within-arm statement, independent of the
cross-arm comparison.

On the question the brief asks of each arm separately — *does `r_probe` track
`sqrt(P)`?* — the honest answer is **partly**:

- **Arm A**: the *ranking* tracks (`+0.97` at k32) but the *relation* does not:
  `r/sqrt(P)` runs from 0.21 to 1.44, i.e. the ratio itself moves 7× across the
  bands. Rank agreement over a curve that is driven by something else.
- **Arm B**: `r/sqrt(P)` sits in 0.45–0.80 at k32 (a 1.8× range) and 0.32–0.76 at
  k64 — so once the shape is controlled, each band's retention is within a factor
  of ~2 of what its predictability implies, with no systematic band trend. The
  remaining scatter is not explained by `sqrt(P)` at k64 (`corr = −0.08`).
- **Arm C** behaves like arm A (`+0.97`, `r/sqrt(P)` 0.22–0.94).

So the predictability account survives as an *approximate, band-flat* statement
in arm B, and fails as an explanation of the *slope* in arms A and C.

## 3. Fit gate and baselines (contract A8)

| arm / k | `ve_last` (mean ± sd) | persistence | mean predictor | margin | verdict |
| :-- | --: | --: | --: | --: | :-- |
| A / k32 | +0.9161 ± 0.0016 | −0.8995 | +0.1526 | +0.7635 | **PASS** |
| A / k64 | +0.8771 ± 0.0015 | −1.0028 | +0.1316 | +0.7455 | **PASS** |
| B / k32 | +0.7224 ± 0.0012 | −0.9593 | +0.0106 | +0.7119 | **PASS** |
| B / k64 | +0.7475 ± 0.0001 | −1.0193 | +0.0104 | +0.7372 | **PASS** |
| C / k32 | +0.8758 ± 0.0009 | −1.0603 | +0.0839 | +0.7920 | **PASS** |

All five clear the pre-stated gate (≥ 0.02 absolute and ≥ 3× the across-seed sd
above both baselines) by 0.71–0.79, so **no arm is void**. Two things worth
noting: flattening *lowers* the achievable variance explained (0.916 → 0.722 at
k32, 0.877 → 0.748 at k64), because it attenuates the clock lines relative to the
flattened continuum, and it *kills the mean predictor* (+0.15 → +0.01): on a flat
corpus the context mean carries no information about the next patch, while the
clock structure the model still finds does. Persistence stays strongly negative
at every aperture (the 16 h / 32 h lag is not a multiple of 24 h). Per-band
variance explained is positive at every band in every arm (0.54–0.91 in arm B).

## 4. The transient (contract A2)

The shape is in place at the first checkpoint, in every arm — so nothing here is
a training-length artifact, and nothing is a late-emerging effect either.

| arm | step | `ve_last` | mean `r_probe` | `r_b` at `b = 1, 4, 8, 12, 16` |
| :-- | --: | --: | --: | :-- |
| A / k32 (seed mean) | 1 000 | +0.7900 | 0.405 | 1.006, 0.339, 0.258, 0.266, 0.420 |
| A / k32 (seed mean) | 20 000 | +0.9161 | 0.389 | 1.382, 0.358, 0.185, 0.175, 0.319 |
| B / k32 (seed 0) | 1 000 | +0.4926 | 0.471 | 0.482, 0.554, 0.403, 0.315, 0.569 |
| B / k32 (seed 0) | 20 000 | +0.7226 | 0.516 | 0.484, 0.721, 0.545, 0.397, 0.351 |
| B / k64 (seed 0) | 1 000 | +0.4365 | 0.303 | 0.338, 0.384, 0.382, 0.387, 0.316 |
| B / k64 (seed 0) | 20 000 | +0.7477 | 0.476 | 0.544, 0.472, 0.617, 0.621, 0.616 |
| C / k32 (seed 0) | 1 000 | +0.7037 | 0.448 | 0.765, 0.479, 0.252, 0.248, 0.528 |
| C / k32 (seed 0) | 20 000 | +0.8771 | 0.472 | 0.820, 0.648, 0.325, 0.200, 0.354 |

`r_probe` is still rising at 20 000 in arm B (mean 0.47 → 0.52 at k32) and in
arm C, while arm A has saturated — but the *shape* the experiment is about does
not move: arm B/k64's comb is already alternating at step 1 000 (0.384, 0.382,
0.387, 0.316 on the harmonic bands `b = 4, 8, 12, 16` against 0.200, 0.207, 0.215
between them at `b = 2, 6, 10`) and arm B/k32's `r(b=1)/r(b=12)` is 1.53 at step
1 000 against 1.12 at the end. Reporting the last checkpoint as T2 does is the
right comparison; the earlier ones do not change any conclusion here.

## 5. What survives: the model returns ~½ of `sqrt(P)` everywhere

In arm B — a flat corpus, a leakage-free measurement, and a clean probe that is
within 1.3–4.1× of the corpus's own band power at every band — the model still
returns only `r_probe = 0.35–0.71` where the corpus's own ridge on its own corpus
and band would return 0.71–0.97, and the oracle readout of the same frozen
latents gives 0.39–0.68. That deficit is now **flat in `b`** (`r/sqrt(P)` ∈
[0.45, 0.80] at k32, with no band trend), so it is not a frequency prior. It is
the "a perfectly clean tone is not in the corpus" effect that P0's design note
and contract A13 are about, or a uniform optimisation gap of P3's kind; T2b
separates it from the slope but does not explain it, and it is the natural next
question. The oracle reproducing the *shape* in both arms (arm B/k64: 0.606,
0.340, 0.537, 0.343, 0.746, 0.334, 0.634, 0.647 against the head's 0.526, 0.317,
0.483, 0.316, 0.681, 0.281, 0.618, 0.647) says the slope that disappeared was in
the *representation*, not only in the head.

## Pass criteria (stated before the run)

| # | criterion | outcome |
| --: | :-- | :-- |
| 16 | arm A reproduces T2's `k32` and `k64` numbers band by band | **pass** — worst \|Δ\| 3.3e-15; probe, corpus, oracle, gate, spread and correlation fields identical to 0.0 |
| 17 | the probe is byte-identical between arms | **pass** — same construction, seed, frequencies and 256 windows; per-band power equal up to the A9 scale (0.3%) |
| 18 | the probe satisfies A4 (`Δb ≥ 1`) and A9 (RMS match) in every arm | **pass** — minimum gap exactly 1; probe/corpus RMS ratio 1.0000 in all five runs |
| 19 | the flattened corpus is actually flat (≤ 5× over the probe bands) | **pass, with one band flagged** — 3.11× / 3.04×; 1.67× at k32 excluding the Nyquist band, where a single edge bin still inflates the local estimate 4.7× |
| 20 | flattening does not destroy the arm (`P(b)` survives, gate passes) | **pass** — `P_own` 0.51–0.95 (k32) / 0.78–0.92 (k64); all five arms pass A8 with margin ≥ 0.71 |
| 21 | 3 seeds per arm, spread reported beside every reading | **pass** — 15 runs; every table row carries a seed sd |
| 22 | the converged state is reported and the transient logged | **pass** — 6 checkpoints per run; the shape is fixed by step 1 000 in every arm |
| 23 | nothing is tuned to make the decline vanish or persist | **pass** — the flattening exponent and the smoothing width were fixed from the aperture's geometry, and their *achieved* flatness is reported beside every reading |
| 24 | the decline is a predictability effect and survives flattening | **FAIL** — 7.89× → 1.12× (k32) and 5.78× → 0.85× (k64); `corr(log2 b, r)` −0.87 → −0.30 and −0.96 → +0.29 |

## Scripts, reproduce, wall time

| script | role | wall |
| :-- | :-- | --: |
| `scripts/t2b.py --report` → `runs/t2b/flatten.json` | the filter, the corpus flatness and `P(b)` per variant, no training | **80 s** |
| `scripts/t2b.py --variant <v> --arm <k> --seed <s>` → `runs/t2b/<v>_<k>_s<s>.json` | one training (corpus + 20 k steps) | **1 662–1 690 s** (6 concurrent), **860–868 s** (3 concurrent) |
| `scripts/t2b.py --summarize` → `runs/t2b/summary.json` | the tables, the decision, and arm A's check against T2 | <1 s |

```
.venv/bin/python experiments/1_realdata/scripts/t2b.py --report
OMP_NUM_THREADS=2 .venv/bin/python experiments/1_realdata/scripts/t2b.py --variant B --arm k32 --seed 0
# ... one process per (variant, arm, seed); waves of 6 / 6 / 3 = 4 249 s of wall
.venv/bin/python experiments/1_realdata/scripts/t2b.py --summarize
```

CPU only, no downloads, `Salesforce/GiftEvalPretrain` untouched. Wall for the
whole grid: **1 695 s** (wave 1, arms A), **1 683 s** (wave 2, arms B) and
**871 s** (wave 3, the 3 dose runs) = **4 249 s** for 15 trainings of 20 000
steps, plus 80 s for `--report`. Each process used 2 torch threads and
`OMP_NUM_THREADS=2`; running the same numpy-heavy estimator with unrestricted
BLAS threads alongside itself cost ~70× in wall time in a first attempt, which is
why the thread count is pinned in every command above.

## Where this contradicts or refines the brief

16. **The T2 confound was the whole effect, and the shape account wins.** On this
    corpus the model's per-band clean-probe retention tracks the corpus's
    spectral *shape*, not its per-band predictability: flatten the corpus and the
    7.9× / 5.8× decline becomes 1.12× / 0.85×, while `P(b)` is re-measured and
    stays high and nearly flat in every arm. The "prior calibrated to
    predictability" account is **not supported on this corpus**.
17. **The dose is monotone but not proportional.** Cutting the corpus's dynamic
    range 617× → 81× (7.6×) cuts the decline only 7.89× → 5.37× (1.5×); the
    remaining collapse happens between 81× and 3.1×. A "log-dynamic-range" story
    fits the three points better than a linear one — with three points, stated
    rather than fitted.
18. **A band-flat deficit survives, and it is not small.** `r_probe/sqrt(P)` is
    0.45–0.80 in arm B at every band, with no band trend. T2's "4.5–5.3× below
    `sqrt(P)` at the top bands" becomes 1.5–1.8× there once the corpus is flat
    (b = 8 / 10 / 12: 4.5× / 4.6× / 4.8× → 1.5× / 1.5× / 1.8×), i.e. shape
    accounts for about two thirds of it in log terms and a band-independent ~1.6×
    shrink for the rest. Any future clean-probe reading on real data has to
    separate the two; only the *slope* was ever a frequency claim.
19. **The flattening has to be done carefully at the band edge, and one band is
    still not fixed.** Replicating the edge bin of the periodogram (the obvious
    implementation) inflated the top of the estimate 1 430×, because a long
    record's coherent components land in single bins: two of the five series have
    a Nyquist bin 3 000× and 6 000× their local continuum. With a truncated
    window the `k32` `b = 16` band still reads 2× low (3.04e-2 against 6e-2) and
    is the one probe band whose `in-band` is not restored — do not read that band
    alone.
20. **The flattening also removes T2's `in-band` caveat, as a side effect.** In
    arm B the probe-band `in-band` is 0.62–0.90 (`k32`), against 0.047–0.962 in
    arm A; T2's "the high bands are 90%+ leakage" applies to arm A only.
21. **`spread_b / seed sd` stops meaning "there is a systematic slope" once the
    slope is gone.** Arm B reports 22.5 / 16.3 for that ratio — larger than
    T2's 18.3 at `k32` — while its `corr(log2 b, r)` is −0.30 and its decline is
    1.12×. The statistic should be read with the correlation beside it, and the
    arm's seed sd fell 4× at the same time.
22. **Consequence for T3, which T2's section proposes as the decisive
    experiment.** T3 (per-patch phase randomisation) preserves band power exactly
    and moves only predictability — which is the right manipulation, but T2b says
    the *shape* channel is the one that carried T2's signal, so a T3 that is run
    on a red corpus must flatten that corpus too, or it will be measuring the
    same confound from the other side.
