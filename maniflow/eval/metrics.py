from dataclasses import dataclass

import numpy as np
import torch

from maniflow.config import DataConfig
from maniflow.dataloader import Normalizer
from maniflow.model import FlowMatchingPolicy
from maniflow.protocols import Env
from maniflow.sim.rollout import rollout


@dataclass(slots=True)
class EvalMetrics:
    mean_final_dist: float
    mean_return: float
    sr05: float
    sr20: float
    sr50: float

    @classmethod
    def from_episodes(
        cls, final_dist: np.ndarray, episode_return: np.ndarray
    ) -> "EvalMetrics":
        return cls(
            mean_final_dist=float(final_dist.mean()),
            mean_return=float(episode_return.mean()),
            sr05=float((final_dist < 0.05).mean()),
            sr20=float((final_dist < 0.20).mean()),
            sr50=float((final_dist < 0.50).mean()),
        )


def evaluate(
    data_cfg: DataConfig,
    policy: FlowMatchingPolicy,
    env: Env,
    obs_norm: Normalizer,
    action_norm: Normalizer,
    device: torch.device,
    n_episodes: int = 20,
    num_obstacles: int | str = "all",
    n_ode_steps: int = 20,
    guidance_scale: float = 1.0,
    joint_vel_limit_scale: float | None = 1.0,
) -> EvalMetrics:
    final_dist = np.empty(n_episodes)
    episode_return = np.empty(n_episodes)

    for ep in range(n_episodes):
        final_dist[ep], episode_return[ep] = rollout(
            policy=policy,
            env=env,
            obs_norm=obs_norm,
            action_norm=action_norm,
            data_cfg=data_cfg,
            device=device,
            num_obstacles=num_obstacles,
            n_ode_steps=n_ode_steps,
            guidance_scale=guidance_scale,
            joint_vel_limit_scale=joint_vel_limit_scale,
            seed=1000 + ep,
        )

    return EvalMetrics.from_episodes(final_dist, episode_return)
