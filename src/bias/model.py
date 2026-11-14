from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from configmixin import ConfigMixin, register_to_config
from x_transformers import Decoder


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_size, use_glu=False):
        super().__init__()

        self.pe = nn.Linear(patch_size, hidden_size)
        self.use_glu = use_glu
        if use_glu:
            self.ff = SwiGLU(hidden_size, hidden_size * 2)
            self.nm = nn.RMSNorm(hidden_size)

    def forward(self, p):
        p = self.pe(torch.atleast_2d(p))
        if not self.use_glu:
            return p
        return self.ff(self.nm(p)) + p


class SimTFM(nn.Module, ConfigMixin):
    config_name = "model.json"

    @register_to_config
    def __init__(
        self,
        *,
        hidden_size=32,
        num_attn_heads=4,
        num_layers=2,
        patch_size=32,
        context_size=512,
        use_glu=True,
        use_rope=True,
        rotary_base_rescale_factor=0.1,
    ):
        super().__init__()
        assert context_size % patch_size == 0

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.context_size = context_size
        self.patch_size = patch_size
        self.num_patches = context_size // patch_size
        self.use_rope = use_rope
        self.rotary_base_rescale_factor = rotary_base_rescale_factor

        self.lm_head = nn.Linear(hidden_size, patch_size)
        self.emb_pos = (
            nn.Embedding(self.num_patches, hidden_size) if not use_rope else None
        )
        self.encoder = PatchEmbedding(patch_size, hidden_size, use_glu=use_glu)
        self.decoder = Decoder(
            dim=hidden_size,
            depth=num_layers,
            heads=num_attn_heads,
            rotary_pos_emb=use_rope,
            rotary_base_rescale_factor=rotary_base_rescale_factor,
            use_rmsnorm=True,
            pre_norm=True,
            ff_glu=True,
            ff_mult=2,
        )

    def pat(self, x):
        r"""``(B, T)`` or ``(B, T, 1)`` -> ``(B, T // patch_size, patch_size)``."""

        x = torch.atleast_2d(x)
        if x.ndim == 3:
            x = x.squeeze(-1)
        assert x.size(1) % self.patch_size == 0, (
            f"{x.size(1)} timesteps is not a multiple of patch_size={self.patch_size}"
        )
        return x.unfold(1, self.patch_size, self.patch_size)

    def map(self, p):
        return self.encoder(p)

    def agg(self, z, mask=None):
        if self.use_rope:
            return self.decoder(z, mask=mask)
        p = torch.arange(z.size(1), device=z.device, dtype=torch.long)
        return self.decoder(z + self.emb_pos(p), mask=mask)

    def forward(self, x, mask=None):
        p = self.pat(x)
        z = self.map(p)
        h = self.agg(z, mask=mask)
        return self.lm_head(h), h, z

    def rollout(self, x, num_patches=1):
        r"""Autoregressively forecast ``num_patches`` patches past the context.

        Each step predicts the patch after the window's last position and
        appends it, dropping the oldest, so the window length is fixed and
        every prediction is made from that same trained role. Assumes
        training windows are ``context_size + patch_size`` timesteps, which
        is what gives the last position a forecasting target; truncating
        the context to ``context_size`` is the caller's job.

        ``x`` ``(B, T)`` or ``(B, T, 1)``. Returns the patches
        ``(B, num_patches, patch_size)`` -- reshape to ``(B, -1)`` for the
        time domain.
        """
        self.eval()
        with torch.no_grad():
            z = self.map(self.pat(x))
            p_hat = []
            for _ in range(num_patches):
                h = self.agg(z)
                p_nxt = self.lm_head(h[:, -1:])  # (B, 1, patch_size)
                p_hat.append(p_nxt)
                z = torch.cat([z[:, 1:], self.map(p_nxt)], dim=1)
            return torch.cat(p_hat, dim=1)
