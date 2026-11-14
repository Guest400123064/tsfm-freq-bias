from __future__ import annotations

import math

import torch


def b_of(f, k, ctx):
    r"""Frequency in cycles per patch: ``b = f * k / ctx``."""
    return f * k / ctx


def make_mixture(
    freqs, n_per, ctx, k, seed, amp_jitter=0.0, phase_jitter=0.0, normalize=False
):
    r"""Mixtures of pure tones, ``(n_per, ctx + k)``.

    Frequencies are in cycles per window (length ``ctx``). Every window draws
    its own amplitude (``1``, plus ``amp_jitter`` times a standard normal) and
    phase (uniform on ``[0, 2 pi)``, plus ``phase_jitter`` times a standard
    normal), held fixed over the window. Splitting on ``seed`` is what makes
    a train and a probe corpus. With ``normalize`` the sum is divided by
    ``sqrt(len(freqs))``; without it the RMS grows as ``sqrt(len(freqs))``,
    so a corpus of a different tone count must not be compared to it directly.

    Contract PLAN.md §3.4: any two ``freqs`` must be at least one cycle per
    patch apart (``abs(b_i - b_j) >= 1``).
    """
    freqs = torch.as_tensor(freqs, dtype=torch.float32).flatten()
    if freqs.numel() > 1:
        b = freqs * k / ctx
        bs, _ = torch.sort(b)
        step = bs[1:] - bs[:-1]
        if float(step.min()) < 1.0 - 1e-6:
            i = int(step.argmin())
            d = float(step[i])
            raise AssertionError(
                f"freqs {float(bs[i])} and {float(bs[i + 1])} in cycles/patch "
                f"are {d} apart, need at least 1 (PLAN.md §3.4)"
            )

    gen = torch.Generator().manual_seed(seed)
    n, f = int(n_per), freqs.numel()
    amp = torch.ones((n, f)) + amp_jitter * torch.randn((n, f), generator=gen)
    phase = 2 * math.pi * torch.rand((n, f), generator=gen)
    if phase_jitter:
        phase = phase + phase_jitter * torch.randn((n, f), generator=gen)

    t = torch.arange(ctx + k, dtype=torch.float32)
    x = torch.zeros((n, ctx + k))
    for j, f_j in enumerate(freqs):
        x += amp[:, j, None] * torch.cos(
            2 * math.pi * float(f_j) * t / ctx + phase[:, j, None]
        )
    if normalize:
        x = x / math.sqrt(f)
    return x


def band_noise(x, f_lo, f_hi, rms, ctx, seed):
    r"""Band-limited Gaussian noise shaped like ``x``, RMS ``rms``.

    White noise over the ``x.shape[-1]`` samples, kept only in the FFT bins
    whose frequency lies in ``[f_lo, f_hi]`` cycles per ``ctx`` samples, then
    rescaled so the total noise power is ``rms**2`` however wide the band is.
    ``f_lo = 0`` with ``f_hi = ctx / 2`` keeps every bin and gives white
    noise. The kept bins are ``ceil(f_lo * n / ctx) .. floor(f_hi * n / ctx)``,
    so a band edge rarely lands exactly on the requested frequency.
    """
    n = x.shape[-1]
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(x.shape, generator=gen, dtype=x.dtype)
    lo, hi = math.ceil(f_lo * n / ctx), min(math.floor(f_hi * n / ctx), n // 2)
    assert lo <= hi, f"empty band [{f_lo}, {f_hi}] in {n} samples"
    band = torch.zeros(n // 2 + 1, dtype=x.dtype)
    band[lo : hi + 1] = 1
    y = torch.fft.irfft(torch.fft.rfft(w, dim=-1) * band, n=n, dim=-1)
    return y * (rms / y.square().mean(-1, keepdim=True).sqrt())


def make_coherent(freqs, band, theta, n_per, ctx, k, seed):
    r"""Tones whose last one keeps only fraction ``theta`` of its power as a tone.

    Every frequency in ``freqs`` is a pure tone with mean power
    ``0.5 / len(freqs)`` -- the per-tone power of ``make_mixture(...,
    normalize=True)`` -- and a phase redrawn per window. The last frequency's
    share is split: ``theta`` of it stays a coherent tone at that frequency,
    and ``1 - theta`` becomes incoherent energy confined to ``band`` (cycles
    per window), so total power is the same for every ``theta``. A last
    frequency outside ``band`` therefore leaves the band empty.

    Returns ``(observed, incoherent)``: the window, and the part of it that is
    not a function of the past. ``theta = 1`` reproduces
    ``make_mixture(freqs, ..., normalize=True)``.
    """
    freqs = torch.as_tensor(freqs, dtype=torch.float32).flatten()
    power = 0.5 / freqs.numel()
    gen = torch.Generator().manual_seed(seed)
    n = int(n_per)
    amp = torch.full((n, freqs.numel()), math.sqrt(2 * power))
    amp[:, -1] = math.sqrt(2 * power * theta)
    phase = 2 * math.pi * torch.rand((n, freqs.numel()), generator=gen)

    t = torch.arange(ctx + k, dtype=torch.float32)
    x = torch.zeros((n, ctx + k))
    for j, f_j in enumerate(freqs):
        x += amp[:, j, None] * torch.cos(
            2 * math.pi * float(f_j) * t / ctx + phase[:, j, None]
        )

    incoherent = torch.zeros_like(x)
    if theta < 1:
        rms = math.sqrt((1 - theta) * power)
        incoherent = band_noise(x, band[0], band[1], rms, ctx, seed)
        x = x + incoherent
    return x, incoherent


def make_banded(freqs, bands, thetas, n_per, ctx, k, seed):
    r"""Tones with a per-tone coherent fraction and a per-tone incoherent band.

    ``freqs[j]`` is a pure tone carrying mean power ``0.5 / len(freqs)``, the
    per-tone power of ``make_mixture(..., normalize=True)``. Fraction
    ``thetas[j]`` of that power stays a coherent tone at ``freqs[j]``; the rest
    becomes incoherent energy confined to ``bands[j]`` (cycles per window).
    Total power is therefore identical for every ``thetas``, so the RMS is
    fixed at ``sqrt(0.5)`` and one normalised probe can be RMS-matched to it
    (PLAN.md §3.9).

    Returns ``(observed, incoherent)``: the window, and the part of it that is
    not a function of the past. ``thetas[j] == 1`` for every ``j`` is
    distributionally ``make_mixture(freqs, ..., normalize=True)``.
    """
    freqs = torch.as_tensor(freqs, dtype=torch.float32).flatten()
    bands = [(float(lo), float(hi)) for lo, hi in bands]
    thetas = [float(t) for t in thetas]
    assert len(bands) == freqs.numel() and len(thetas) == freqs.numel()

    power = 0.5 / freqs.numel()
    gen = torch.Generator().manual_seed(seed)
    phase = 2 * math.pi * torch.rand((int(n_per), freqs.numel()), generator=gen)

    t = torch.arange(ctx + k, dtype=torch.float32)
    x = torch.zeros((int(n_per), ctx + k))
    for j, f_j in enumerate(freqs):
        amp = math.sqrt(2 * power * thetas[j])
        x += amp * torch.cos(2 * math.pi * float(f_j) * t / ctx + phase[:, j, None])

    incoherent = torch.zeros_like(x)
    for j, theta in enumerate(thetas):
        if theta < 1:
            incoherent = incoherent + band_noise(
                x, bands[j][0], bands[j][1], math.sqrt((1 - theta) * power), ctx, seed
            )
    return x + incoherent, incoherent


def make_pm_profile(freqs, powers, betas, n_per, ctx, k, seed):
    r"""Phase-diffusing tones with a per-tone power, ``(n_per, ctx + k)``.

    ``make_pm`` with the flat power replaced by ``powers[j]``, the mean square
    of tone ``j``. The envelope is constant, so the window's total power is
    ``sum(powers)`` however the ``betas`` are set -- power and predictability
    are separate knobs. Everything else is ``make_pm``: the phase advances by
    the carrier ``2 pi b_c`` plus ``beta_j * eps`` per patch boundary, so the
    conditional mean of the next patch shrinks the tone by
    ``exp(-beta_j**2 / 2)`` and the fraction of tone ``j``'s power any function
    of the context can recover is ``exp(-beta_j**2)`` (PLAN.md §5, P2b).

    Returns ``(observed, unpredictable)`` with ``unpredictable`` the part of
    each sample no function of the context can know; its mean square on tone
    ``j`` is ``powers[j] * (1 - exp(-beta_j**2))``.
    """
    freqs = torch.as_tensor(freqs, dtype=torch.float32).flatten()
    powers = torch.as_tensor(powers, dtype=torch.float32).flatten()
    betas = torch.as_tensor(betas, dtype=torch.float32).flatten()
    assert powers.numel() == freqs.numel(), "one power per tone"
    assert betas.numel() == freqs.numel(), "one beta per tone"
    assert (ctx + k) % k == 0, f"{ctx + k} samples is not a whole number of patches"
    n, f = int(n_per), freqs.numel()
    n_patch = (ctx + k) // k
    amp = (2 * powers).sqrt().view(1, 1, f)

    gen = torch.Generator().manual_seed(seed)
    step = betas.view(1, 1, f) * torch.randn((n, n_patch, f), generator=gen)
    start = 2 * math.pi * torch.rand((n, 1, f), generator=gen)
    b = (freqs * k / ctx).view(1, 1, f)
    patch = torch.arange(n_patch, dtype=torch.float32).view(1, n_patch, 1)
    phi = start + 2 * math.pi * b * patch + torch.cumsum(step, dim=1)
    psi = phi - step  # the phase the previous patch determines
    shrink = torch.exp(-0.5 * betas**2).view(1, 1, f)

    idx = torch.arange(ctx + k) // k
    u = (torch.arange(ctx + k) % k).to(torch.float32)
    x = torch.zeros((n, ctx + k))
    unpredictable = torch.zeros_like(x)
    for j, f_j in enumerate(freqs):
        arg = 2 * math.pi * float(f_j) * u / ctx  # within a patch; phi adds 2 pi b
        full = torch.cos(arg + phi[:, idx, j])
        mean = shrink[0, 0, j] * torch.cos(arg + psi[:, idx, j])
        x += amp[0, 0, j] * full
        unpredictable += amp[0, 0, j] * (full - mean)
    return x, unpredictable


def make_pm(freqs, betas, n_per, ctx, k, seed):
    r"""Tones whose per-patch phase diffuses, ``(n_per, ctx + k)``.

    ``make_pm_profile`` with flat power: every frequency is a tone of mean
    square ``0.5 / len(freqs)``, so the total power is ``0.5``, the same as
    ``make_mixture(..., normalize=True)`` (PLAN.md §3.9), and is independent of
    ``betas``.

    ``eps_p`` is drawn independently per patch, per tone, per window, so the
    next patch's phase is genuinely not a function of the context -- unlike
    ``make_banded``'s incoherent energy, which the window's own FFT bins
    determine (PLAN.md §4). ``betas`` all zero advances the phase
    deterministically and is ``make_mixture(..., normalize=True)``
    distributionally.
    """
    freqs = torch.as_tensor(freqs, dtype=torch.float32).flatten()
    powers = torch.full((freqs.numel(),), 0.5 / freqs.numel())
    return make_pm_profile(freqs, powers, betas, n_per, ctx, k, seed)


def make_broad(
    n_tones,
    n_per,
    ctx,
    k,
    seed,
    b_lo=0.05,
    b_hi=None,
    normalize=True,
    resample_freqs=True,
):
    r"""Broad random mixtures, ``(n_per, ctx + k)``.

    Every window draws its own ``n_tones`` frequencies independently and
    uniformly over ``[b_lo, b_hi]`` in cycles per patch (``f = b * ctx / k``),
    each with amplitude ``1`` and a uniform phase on ``[0, 2 pi)``. ``b_hi``
    defaults to the per-patch Nyquist ``k / 2``. With ``normalize`` the sum is
    divided by ``sqrt(n_tones)``, so the expected RMS does not grow with the
    tone count and corpora of different sizes stay on the same scale.

    With ``resample_freqs=False`` the ``n_tones`` frequencies are drawn once
    per corpus and reused by every window, which still draws its own phases:
    the corpus becomes one fixed tone set rather than a new one per window.

    The ``b``-spacing contract of ``make_mixture`` does not apply here: the
    tones are random, not a fixed probe frequency set.
    """
    b_hi = k / 2 if b_hi is None else b_hi
    gen = torch.Generator().manual_seed(seed)
    n = int(n_per)
    if resample_freqs:
        b = b_lo + (b_hi - b_lo) * torch.rand((n, n_tones), generator=gen)
    else:
        b0 = b_lo + (b_hi - b_lo) * torch.rand((n_tones,), generator=gen)
        b = b0.expand(n, n_tones)
    phase = 2 * math.pi * torch.rand((n, n_tones), generator=gen)

    t = torch.arange(ctx + k, dtype=torch.float32)
    x = torch.zeros((n, ctx + k))
    for j in range(n_tones):
        f = b[:, j] * ctx / k
        x += torch.cos(2 * math.pi * f[:, None] * t / ctx + phase[:, j, None])
    if normalize:
        x = x / math.sqrt(n_tones)
    return x


def circle_rule(omega, gamma):
    r"""Circle-map message rule, ``z -> z * exp(i (omega + gamma sin(arg z)))``.

    The phase advance depends on the message itself, so ``gamma`` is the
    complexity knob: ``gamma = 0`` is a pure rotation by ``omega`` per patch
    (``omega = gamma = 0`` copies) and larger ``gamma`` bends the coordinate
    circle. The envelope is untouched -- ``|rule(z)| = |z|`` for every ``z``.
    """
    omega, gamma = float(omega), float(gamma)

    def rule(z):
        return z * torch.exp(1j * (omega + gamma * torch.sin(torch.angle(z))))

    return rule


def make_coeff_recurrence(bands, rule, powers, n_per, ctx, k, seed):
    r"""Tones whose per-patch coefficients follow ``rule``, ``(n_per, ctx + k)``.

    Band ``i`` carries a constant envelope ``sqrt(2 * powers[i])`` and a message
    ``m_j``: ``z_j(i) = A_i exp(i m_j)`` and ``z_{j+1} = rule(z_j)``, so the
    per-band power is ``powers[i]`` whatever ``rule`` does. The window is
    ``x_j(u) = sum_i Re[z_j(i) exp(2 pi i b_i u / k)]`` for ``u`` in ``[0, k)``
    and ``j`` in ``[0, n_patch)`` with ``n_patch = ctx // k + 1``, i.e. exactly
    ``ctx + k`` samples; ``bands`` are in cycles per patch and ``rule`` maps a
    complex array ``(n, n_band)`` to the next patch's coefficients.

    ``b = k / 2`` (Nyquist) is refused: there the negative-frequency image does
    not cancel and the matched-filter readback is not exact.
    """
    bands = torch.as_tensor(bands, dtype=torch.float32).flatten()
    powers = torch.as_tensor(powers, dtype=torch.float32).flatten()
    assert powers.numel() == bands.numel(), "one power per band"
    assert (ctx + k) % k == 0, f"{ctx + k} samples is not a whole number of patches"
    assert not bool(((bands - k / 2).abs() < 1e-9).any()), (
        f"band k/2 = {k / 2} (Nyquist) has no exact matched-filter readback"
    )

    n, n_band = int(n_per), bands.numel()
    n_patch = ctx // k + 1
    gen = torch.Generator().manual_seed(seed)
    phi = 2 * math.pi * torch.rand((n, n_band), generator=gen)
    z = torch.zeros((n, n_patch, n_band), dtype=torch.complex64)
    z[:, 0, :] = (2 * powers).sqrt() * torch.exp(1j * phi)
    for j in range(1, n_patch):
        z[:, j, :] = rule(z[:, j - 1, :])

    u = torch.arange(k, dtype=torch.float32)
    ph = torch.exp(2j * math.pi * bands[:, None] * u[None, :] / k)  # (n_band, k)
    wave = torch.real(z[:, :, :, None] * ph).sum(dim=-2)  # (n, n_patch, k)
    return wave.reshape(n, ctx + k)
