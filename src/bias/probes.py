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


def band_component(x, b, k):
    """Reconstruct the waveform component at b cycles per patch."""
    z = matched_amp(x, b, k)
    ctype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    t = torch.arange(k, dtype=x.dtype, device=x.device)
    return torch.real(z.unsqueeze(-1) * torch.exp(2j * math.pi * b * t / k).to(ctype))


def band_r2(pred, true, b, k):
    """Return variance explained in the reconstructed band waveform."""
    p, t = band_component(pred, b, k), band_component(true, b, k)
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def fit_oracle(z_fit, target_fit, z_eval, ridge=1e-3):
    """Fit a trace-scaled ridge readout from frozen latents to shifted patches."""

    def design(z):
        x = z[:, :-1].reshape(-1, z.shape[-1])
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        return torch.cat([x, ones], dim=-1)

    xa = design(z_fit)
    ya = target_fit[:, 1:].reshape(-1, target_fit.shape[-1])
    gram = xa.mT @ xa
    scale = torch.trace(gram) / gram.shape[0]
    reg = ridge * scale * torch.eye(xa.shape[1], dtype=xa.dtype, device=xa.device)
    reg[-1, -1] = 0
    weights = torch.linalg.solve(gram + reg, xa.mT @ ya)

    k = target_fit.shape[-1]
    return (design(z_eval) @ weights).reshape(z_eval.shape[0], -1, k)
