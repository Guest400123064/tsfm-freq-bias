import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Decoder


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(patch_size, hidden_size)
        self.norm = nn.RMSNorm(hidden_size)
        self.ff = SwiGLU(hidden_size, hidden_size * 2)

    def forward(self, patches):
        z = self.linear(patches)
        return z + self.ff(self.norm(z))


class SimTFM(nn.Module):
    def __init__(
        self,
        *,
        hidden_size=32,
        num_attn_heads=4,
        num_layers=2,
        patch_size=32,
        context_size=512,
        rotary_base_rescale_factor=0.1,
    ):
        super().__init__()
        assert context_size % patch_size == 0

        self.context_size = context_size
        self.patch_size = patch_size
        self.num_patches = context_size // patch_size
        self.lm_head = nn.Linear(hidden_size, patch_size)
        self.encoder = PatchEmbedding(patch_size, hidden_size)
        self.decoder = Decoder(
            dim=hidden_size,
            depth=num_layers,
            heads=num_attn_heads,
            rotary_pos_emb=True,
            rotary_base_rescale_factor=rotary_base_rescale_factor,
            use_rmsnorm=True,
            pre_norm=True,
            ff_glu=True,
            ff_mult=2,
        )

    def pat(self, x):
        """Patch a (B, T) or (B, T, 1) sequence into (B, T / k, k)."""
        x = torch.atleast_2d(x)
        if x.ndim == 3:
            x = x.squeeze(-1)
        assert x.size(1) % self.patch_size == 0
        return x.unfold(1, self.patch_size, self.patch_size)

    def map(self, patches):
        return self.encoder(patches)

    def agg(self, z, mask=None):
        return self.decoder(z, mask=mask)

    def forward(self, x, mask=None):
        z = self.map(self.pat(x))
        h = self.agg(z, mask=mask)
        return self.lm_head(h), h, z

    def predict_next(self, x):
        """Predict the patch after the context and return its observed target."""
        target = self.pat(x)[:, -1]
        pred = self(x[:, : self.context_size])[0][:, -1]
        return pred, target

    def rollout(self, x, num_patches=1):
        """Forecast patches autoregressively, returning (B, num_patches, k)."""
        self.eval()
        with torch.no_grad():
            z = self.map(self.pat(x))
            forecasts = []
            for _ in range(num_patches):
                next_patch = self.lm_head(self.agg(z)[:, -1:])
                forecasts.append(next_patch)
                z = torch.cat([z[:, 1:], self.map(next_patch)], dim=1)
            return torch.cat(forecasts, dim=1)
