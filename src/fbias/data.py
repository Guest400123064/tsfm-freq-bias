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
