import math

import torch


def b_of(f, k, ctx):
    """Convert cycles per context window to cycles per patch."""
    return f * k / ctx


def make_mixture(freqs, n_per, ctx, k, seed, normalize=False):
    """Generate independent-phase tone mixtures of shape (n_per, ctx + k)."""
    freqs = torch.as_tensor(freqs, dtype=torch.float32).flatten()
    if freqs.numel() > 1:
        bands, _ = torch.sort(b_of(freqs, k, ctx))
        spacing = bands[1:] - bands[:-1]
        assert float(spacing.min()) >= 1.0 - 1e-6, (
            "tone frequencies must be at least 1 cycle/patch apart"
        )

    gen = torch.Generator().manual_seed(seed)
    phase = 2 * math.pi * torch.rand((int(n_per), freqs.numel()), generator=gen)
    t = torch.arange(ctx + k, dtype=torch.float32)
    x = torch.zeros((int(n_per), ctx + k))
    for j, frequency in enumerate(freqs):
        x += torch.cos(2 * math.pi * frequency * t / ctx + phase[:, j, None])
    return x / math.sqrt(freqs.numel()) if normalize else x


def make_pm_profile(freqs, powers, betas, n_per, ctx, k, seed):
    """Generate constant-power tones with independent per-patch phase diffusion.

    Phase advances by 2πb plus N(0, beta²) at each patch boundary. Returns the
    observed windows and their component that is unpredictable from the past.
    """
    freqs = torch.as_tensor(freqs, dtype=torch.float32).flatten()
    powers = torch.as_tensor(powers, dtype=torch.float32).flatten()
    betas = torch.as_tensor(betas, dtype=torch.float32).flatten()
    assert powers.numel() == freqs.numel()
    assert betas.numel() == freqs.numel()
    assert (ctx + k) % k == 0

    n, n_bands = int(n_per), freqs.numel()
    n_patches = (ctx + k) // k
    amp = (2 * powers).sqrt().view(1, 1, n_bands)
    gen = torch.Generator().manual_seed(seed)
    steps = betas.view(1, 1, n_bands) * torch.randn(
        (n, n_patches, n_bands), generator=gen
    )
    start = 2 * math.pi * torch.rand((n, 1, n_bands), generator=gen)
    bands = b_of(freqs, k, ctx).view(1, 1, n_bands)
    patch = torch.arange(n_patches, dtype=torch.float32).view(1, n_patches, 1)
    phase = start + 2 * math.pi * bands * patch + torch.cumsum(steps, dim=1)
    determined_phase = phase - steps
    shrink = torch.exp(-0.5 * betas**2).view(1, 1, n_bands)

    patch_index = torch.arange(ctx + k) // k
    within_patch = (torch.arange(ctx + k) % k).to(torch.float32)
    x = torch.zeros((n, ctx + k))
    unpredictable = torch.zeros_like(x)
    for j, frequency in enumerate(freqs):
        angle = 2 * math.pi * frequency * within_patch / ctx
        full = torch.cos(angle + phase[:, patch_index, j])
        conditional_mean = shrink[0, 0, j] * torch.cos(
            angle + determined_phase[:, patch_index, j]
        )
        x += amp[0, 0, j] * full
        unpredictable += amp[0, 0, j] * (full - conditional_mean)
    return x, unpredictable


def circle_rule(omega, gamma):
    """Return z -> z exp(i(omega + gamma sin(arg z))), preserving |z|."""
    omega, gamma = float(omega), float(gamma)

    def rule(z):
        return z * torch.exp(1j * (omega + gamma * torch.sin(torch.angle(z))))

    return rule


def make_coeff_recurrence(bands, rule, powers, n_per, ctx, k, seed):
    """Generate tones whose coefficients follow a constant-envelope map."""
    bands = torch.as_tensor(bands, dtype=torch.float32).flatten()
    powers = torch.as_tensor(powers, dtype=torch.float32).flatten()
    assert powers.numel() == bands.numel()
    assert (ctx + k) % k == 0
    assert not bool(((bands - k / 2).abs() < 1e-9).any())

    n, n_bands = int(n_per), bands.numel()
    n_patches = ctx // k + 1
    gen = torch.Generator().manual_seed(seed)
    phase = 2 * math.pi * torch.rand((n, n_bands), generator=gen)
    z = torch.zeros((n, n_patches, n_bands), dtype=torch.complex64)
    z[:, 0, :] = (2 * powers).sqrt() * torch.exp(1j * phase)
    for j in range(1, n_patches):
        z[:, j, :] = rule(z[:, j - 1, :])

    u = torch.arange(k, dtype=torch.float32)
    carrier = torch.exp(2j * math.pi * bands[:, None] * u[None, :] / k)
    wave = torch.real(z[:, :, :, None] * carrier).sum(dim=-2)
    return wave.reshape(n, ctx + k)
