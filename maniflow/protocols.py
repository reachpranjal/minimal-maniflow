from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import torch

    from maniflow.config import EnvConfig


class Env(Protocol):
    cfg: "EnvConfig"

    obs_dim: int
    action_dim: int
    goal_relative_pairs: tuple[tuple[slice, slice], ...]
    joint_vel_limit: np.ndarray

    def reset(
        self,
        seed: int | None = None,
        num_obstacles: int | str = "all",
        goal_qpos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]: ...

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]: ...

    def close(self) -> None: ...

    def augment_observation(
        self, obs: "torch.Tensor", std: float
    ) -> "torch.Tensor": ...
