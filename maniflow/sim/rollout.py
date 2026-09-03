from typing import Callable

import numpy as np
import torch

from maniflow.config import DataConfig
from maniflow.dataloader import Normalizer
from maniflow.model import FlowMatchingPolicy, sample_action
from maniflow.protocols import Env

StepHook = Callable[[Env], bool]

def rollout(
    policy: FlowMatchingPolicy,
    env: Env,
    obs_norm: Normalizer,
    action_norm: Normalizer,
    data_cfg: DataConfig,
    num_obstacles: int | str = "all",
    n_ode_steps: int = 20,
    ode_clamp_std: float | None = None,
    guidance_scale: float = 1.0,
    joint_vel_limit_scale: float | None = 1.0,
    step_hook: StepHook | None = None,
    device: torch.device = "cuda",
    seed: int = 0,
) -> tuple[float, float]:
    obs, _ = env.reset(seed=seed, num_obstacles=num_obstacles)
    obs_buf = np.stack([obs] * data_cfg.obs_horizon)

    total_return = 0.0
    final_dist = float("nan")
    max_decisions = env.cfg.max_episode_steps // data_cfg.exec_action_horizon + 2

    max_step = None
    if joint_vel_limit_scale is not None:
        dt = env.cfg.n_substeps * env.cfg.timestep
        max_step = env.joint_vel_limit * joint_vel_limit_scale * dt
    prev_action: np.ndarray | None = None

    done = False
    for _ in range(max_decisions):
        chunk = sample_action(
            policy,
            obs_buf,
            obs_norm,
            action_norm,
            device,
            n_steps=n_ode_steps,
            clamp_std=ode_clamp_std,
            guidance_scale=guidance_scale,
        )

        for action in chunk[: data_cfg.exec_action_horizon]:
            if max_step is not None and prev_action is not None:
                action = prev_action + np.clip(
                    action - prev_action, -max_step, max_step
                )
            prev_action = action

            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            final_dist = info["dist"]
            obs_buf = np.roll(obs_buf, -1, axis=0)
            obs_buf[-1] = obs

            stopped = step_hook is not None and not step_hook(env)
            done = terminated or truncated or stopped
            if done:
                break
        if done:
            break

    return final_dist, total_return
