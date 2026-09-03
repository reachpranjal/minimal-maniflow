import mujoco
import numpy as np
import torch

from maniflow.config import EnvConfig

from . import spec
from .kinematics import FrankaPandaArm
from .scene import Scene
from .task import GoalReachingTask


class SingleArmEnv:
    obs_dim = spec.obs_dim
    action_dim = spec.action_dim
    goal_relative_pairs = spec.goal_relative_pairs

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()

        self.panda_arm = FrankaPandaArm(self.cfg)
        self.scene = Scene(self.panda_arm.model, self.cfg)
        self.task = GoalReachingTask(self.cfg)

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._n_active_obs = 0

        self._renderer = None

    @property
    def model(self):
        return self.panda_arm.model

    @property
    def data(self):
        return self.panda_arm.data

    @property
    def joint_low(self) -> np.ndarray:
        return self.panda_arm.joint_low

    @property
    def joint_high(self) -> np.ndarray:
        return self.panda_arm.joint_high

    @property
    def joint_vel_limit(self) -> np.ndarray:
        return self.panda_arm.joint_vel_limit

    @property
    def goal_qpos(self) -> np.ndarray:
        return self.task.goal_qpos

    def get_obstacle_positions(self) -> np.ndarray:
        return self.scene.obstacle_positions()

    def augment_observation(self, obs: torch.Tensor, std: float) -> torch.Tensor:
        span = spec.obs_slices["ee_quat"].stop
        out = obs.clone()
        out[..., :span] += std * torch.randn_like(out[..., :span])
        return out

    def reset(
        self,
        seed: int | None = None,
        num_obstacles: int | str = "all",
        goal_qpos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        collision_fn = lambda qpos: self.panda_arm.is_collision_free(  # noqa: E731
            qpos, self.scene.obstacle_geom_ids
        )  # noqa: E731
        self._n_active_obs = self.scene.reset_world(
            self._rng, collision_fn, num_obstacles
        )

        self.panda_arm.reset_home()

        self.task.sample_goal(self.panda_arm, self._rng, collision_fn, goal_qpos)
        self.scene.set_marker_pos(self.task.goal_ee_pos)
        mujoco.mj_forward(self.panda_arm.model, self.panda_arm.data)

        self._step_count = 0

        state = self.panda_arm.state(self.scene.obstacle_positions())
        obs = self.task.observation(state)

        return obs, {"num_obstacles": self._n_active_obs}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.panda_arm.step(action)
        self._step_count += 1

        state = self.panda_arm.state(self.scene.obstacle_positions())

        reward = self.task.reward(state)
        dist = self.task.distance(state)
        success = self.task.success(state)
        truncated = self._step_count >= self.cfg.max_episode_steps
        obs = self.task.observation(state)

        return obs, reward, success, truncated, {"success": success, "dist": dist}

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.panda_arm.model,
                height=self.cfg.img_height,
                width=self.cfg.img_width,
            )
        self._renderer.update_scene(self.panda_arm.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
