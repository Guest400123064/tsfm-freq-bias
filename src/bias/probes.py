from __future__ import annotations

import math

import torch


def matched_amp(x, b, k):
    r"""Complex amplitude at ``b`` cycles/patch: ``(..., k) -> (...,)``.

    Matched filter ``(2/k) * sum_t x[t] * exp(-2j pi b t / k)``; a pure tone of
    amplitude ``A`` and phase ``phi`` at its own ``b`` returns ``A e^{j phi}``
    exactly when ``2b`` is an integer. Elsewhere the negative-frequency image
    leaks in, so a low ``b`` reads a phase-dependent gain; ratios (retention)
    are unaffected.
    """
    t = torch.arange(k, dtype=x.dtype, device=x.device)
    ctype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    w = torch.exp(-2j * math.pi * b * t / k).to(ctype)
    return (2 / k) * torch.sum(x.to(ctype) * w, dim=-1)


def retention(pred, true, b, k):
    r"""Predicted over true amplitude in band ``b``, ``(...,)``. PLAN.md §4."""
    return matched_amp(pred, b, k).abs() / matched_amp(true, b, k).abs()


def phase_error(pred, true, b, k):
    r"""``angle(Z_pred) - angle(Z_true)`` in radians, ``(...,)``. PLAN.md §4."""
    return torch.angle(matched_amp(pred, b, k)) - torch.angle(matched_amp(true, b, k))


def delta_k(pred, true, b, k):
    r"""Fredformer's relative error, ``|Z(pred) - Z(true)| / |Z(true)|``, ``(...,)``.

    Piao et al. 2024, eq. 1-2, read through our patch-level matched filter. It
    is ``|r e^{i dphi} - 1|``, so it is *not* amplitude-only: by the triangle
    inequality ``delta >= |1 - r|``, with equality exactly when the emitted
    phase is locked. Reported next to ``r`` so their figures can be converted;
    ``r`` stays the primary reading because only it is amplitude.
    """
    z_true = matched_amp(true, b, k)
    return (matched_amp(pred, b, k) - z_true).abs() / z_true.abs()


def band_component(x, b, k):
    r"""The part of ``x`` in band ``b``, ``(..., k)``: ``Re(Z e^{2j pi b t / k})``."""
    z = matched_amp(x, b, k)
    ctype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    t = torch.arange(k, dtype=x.dtype, device=x.device)
    return torch.real(z.unsqueeze(-1) * torch.exp(2j * math.pi * b * t / k).to(ctype))


def band_r2(pred, true, b, k):
    r"""Fraction of the band-``b`` waveform's variance the prediction explains."""
    p, t = band_component(pred, b, k), band_component(true, b, k)
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def fit_oracle(z_fit, target_fit, z_eval, target_eval, b, k, ridge=1e-3):
    r"""Upper bound: ridge map from frozen latents to the next patch (PLAN.md §3.5).

    Latents ``(..., Q+1, H)`` and patch targets ``(..., Q+1, k)``. ``z[:, j]``
    predicts patch ``j+1``, so the fit is on ``(z[:, :-1], patches[:, 1:])``.
    The eval call is the same shift. Returns the fitted patches ``pred``
    ``(..., Q, k)`` plus retention, phase error, and band ``r2`` pooled over
    every position.
    """
    assert z_fit.ndim >= 3, "z must be (..., patches, hidden)"

    def design(z):
        x = z[:, :-1].reshape(-1, z.shape[-1])
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        return torch.cat([x, ones], dim=-1)

    xa, ya = design(z_fit), target_fit[:, 1:].reshape(-1, k)
    reg = ridge * torch.eye(xa.shape[1], dtype=xa.dtype, device=xa.device)
    reg[-1, -1] = 0  # the intercept is not shrunk
    w = torch.linalg.solve(xa.mT @ xa + reg, xa.mT @ ya)

    true = target_eval[:, 1:]
    pred = (design(z_eval) @ w).reshape(*z_eval.shape[:-2], true.shape[-2], k)
    return {
        "pred": pred,
        "retention": float(retention(pred, true, b, k).mean()),
        "phase_error": float(phase_error(pred, true, b, k).mean()),
        "r2": band_r2(pred, true, b, k),
    }


def _top2(latents):
    r"""Top-2 principal directions, ``(N, H) -> (H, 2)``."""
    x = latents.reshape(-1, latents.shape[-1]).double()
    x = x - x.mean(dim=0, keepdim=True)
    vh = torch.linalg.svd(x, full_matrices=False).Vh
    return vh[:2].mT


def ring_dim12(latents):
    r"""Variance fraction of the top two principal components. PLAN.md §4."""
    x = latents.reshape(-1, latents.shape[-1]).double()
    x = x - x.mean(dim=0, keepdim=True)
    var = torch.linalg.svdvals(x) ** 2
    return float(var[:2].sum() / var.sum())


def plane_overlap(latents_a, latents_b):
    r"""Grassmann ``mean(cos^2)`` between the top-2 subspaces. PLAN.md §4."""
    s = torch.linalg.svdvals(_top2(latents_a).mT @ _top2(latents_b))
    return float((s**2).mean())
