import numpy as np
import torch
import torch.nn as nn

from maniflow.config import ModelConfig
from maniflow.dataloader.normalize import Normalizer
from maniflow.model.backbone import VelocityField, goal_relative_features


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        goal_relative_pairs: tuple[tuple[slice, slice], ...],
        model_cfg: ModelConfig | None = None,
    ):
        super().__init__()
        model_cfg = model_cfg or ModelConfig()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.goal_relative_pairs = tuple(goal_relative_pairs)

        goal_feature_dim = sum(g.stop - g.start for g, _ in self.goal_relative_pairs)
        act_flat = pred_horizon * action_dim
        obs_flat = obs_horizon * obs_dim
        cond_dim = obs_flat + goal_feature_dim
        self.net = VelocityField(act_flat, cond_dim, model_cfg)

        self.cond_dropout_prob = model_cfg.cond_dropout_prob
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model params: {n_params:,}")

    def _build_cond(
        self, obs: torch.Tensor, drop_mask: torch.Tensor | None
    ) -> torch.Tensor:
        B = obs.shape[0]
        goal_rel = goal_relative_features(obs[:, -1, :], self.goal_relative_pairs)
        obs_cond = torch.cat([obs.reshape(B, -1), goal_rel], dim=-1)

        if drop_mask is None and self.training and self.cond_dropout_prob > 0.0:
            drop_mask = torch.rand(B, device=obs.device) < self.cond_dropout_prob
        if drop_mask is not None:
            obs_cond = torch.where(drop_mask[:, None], self.null_cond, obs_cond)
        return obs_cond

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs: torch.Tensor,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        obs_cond = self._build_cond(obs, drop_mask)
        v = self.net(x_t.reshape(B, -1), t, obs_cond)
        return v.reshape(B, self.pred_horizon, self.action_dim)

    def load_checkpoint(self, state_dict: dict) -> None:
        if not any(k.startswith("net.") for k in state_dict):
            state_dict = {f"net.{k}": v for k, v in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"load_checkpoint: missing={missing} unexpected={unexpected}")


def _clamp(z: torch.Tensor, clamp_std: float | None) -> torch.Tensor:
    if clamp_std is not None and clamp_std > 0:
        return torch.clamp(z, -clamp_std, clamp_std)
    return z


def _guided_velocity(
    policy: FlowMatchingPolicy,
    z: torch.Tensor,
    t: torch.Tensor,
    obs: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    v_cond = policy(z, t, obs)
    if guidance_scale == 1.0:
        return v_cond
    drop_all = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
    v_uncond = policy(z, t, obs, drop_mask=drop_all)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


@torch.inference_mode()
def sample_action(
    policy: FlowMatchingPolicy,
    obs_window: np.ndarray,
    obs_norm: Normalizer,
    action_norm: Normalizer,
    device: torch.device,
    n_steps: int = 20,
    clamp_std: float | None = 4.0,
    guidance_scale: float = 1.0,
) -> np.ndarray:

    policy.eval()
    obs = (
        obs_norm.normalize(torch.from_numpy(obs_window).float()).unsqueeze(0).to(device)
    )
    x = torch.randn(1, policy.pred_horizon, policy.action_dim, device=device)

    x = _clamp(x, clamp_std)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = torch.full((1,), i * dt, device=device)
        v0 = _guided_velocity(policy, x, t0, obs, guidance_scale)
        x = _clamp(x + v0 * dt, clamp_std)

    return action_norm.unnormalize(x.squeeze(0)).cpu().numpy()
