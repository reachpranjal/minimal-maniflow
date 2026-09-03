import torch
import torch.nn as nn

from maniflow.config import ModelConfig


def goal_relative_features(
    curr: torch.Tensor, goal_relative_pairs: tuple[tuple[slice, slice], ...]
) -> torch.Tensor:
    return torch.cat([curr[:, g] - curr[:, c] for g, c in goal_relative_pairs], dim=-1)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        i = torch.arange(half, device=t.device, dtype=torch.float32)
        log_max = torch.log(torch.tensor(10000.0, device=t.device))
        freqs = torch.exp(-i * (log_max / (half - 1)))
        x = t[:, None].float() * freqs[None, :]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.film = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return self.norm(x + self.net(x) * (1.0 + scale) + shift)


class VelocityField(nn.Module):
    def __init__(self, act_flat: int, obs_cond_dim: int, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.hidden_dim

        self.act_proj = nn.Linear(act_flat, hidden)
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.time_enc = nn.Sequential(
            SinusoidalEmbedding(cfg.time_dim),
            nn.Linear(cfg.time_dim, hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResBlock(hidden, cond_dim=hidden, dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.norm_out = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, act_flat)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, x_flat: torch.Tensor, t: torch.Tensor, obs_cond: torch.Tensor
    ) -> torch.Tensor:
        cond = self.obs_enc(obs_cond) + self.time_enc(t)
        h = self.act_proj(x_flat)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(self.norm_out(h))
