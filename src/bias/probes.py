import math

import torch


def matched_amp(x, b, k):
    """Return the complex amplitude at b cycles per patch."""
    t = torch.arange(k, dtype=x.dtype, device=x.device)
    ctype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    w = torch.exp(-2j * math.pi * b * t / k).to(ctype)
    return (2 / k) * torch.sum(x.to(ctype) * w, dim=-1)


def retention(pred, true, b, k):
    """Return predicted divided by true amplitude in band b."""
    return matched_amp(pred, b, k).abs() / matched_amp(true, b, k).abs()


def phase_error(pred, true, b, k):
    """Return the signed phase difference in radians."""
    return torch.angle(matched_amp(pred, b, k)) - torch.angle(matched_amp(true, b, k))


def delta_k(pred, true, b, k):
    """Return |Z_pred - Z_true| / |Z_true|, combining amplitude and phase error."""
    z_true = matched_amp(true, b, k)
    return (matched_amp(pred, b, k) - z_true).abs() / z_true.abs()


def band_coefficients(x, bands, k):
    """Recover complex tone coefficients by jointly inverting a cos/sin basis."""
    bands = torch.as_tensor(bands, dtype=x.dtype, device=x.device).flatten()
    assert 2 * len(bands) <= k, "tone count exceeds the patch's real degrees of freedom"
    t = torch.arange(k, dtype=x.dtype, device=x.device)
    angles = 2 * math.pi * bands[:, None] * t / k
    basis = torch.cat([torch.cos(angles), torch.sin(angles)]).mT
    coeff = torch.linalg.solve(basis, x.reshape(-1, k).mT).mT
    return torch.complex(coeff[..., : len(bands)], -coeff[..., len(bands) :]).reshape(
        *x.shape[:-1], len(bands)
    )
