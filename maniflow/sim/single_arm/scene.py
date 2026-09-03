import mujoco
import numpy as np

from maniflow.config import EnvConfig

from . import spec
from .kinematics import home_qpos

_HIDDEN_POS = np.array([0.0, 0.0, -10.0])


class Scene:
    def __init__(self, model: mujoco.MjModel, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.model = model

        self.marker_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker"
        )

        self.obstacle_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            for i in range(spec.n_max_obstacles)
        ]
        self.obstacle_geom_ids = frozenset(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom_{i}"
            )
            for i in range(spec.n_max_obstacles)
        )

    def set_obstacle_pos(self, i: int, pos: np.ndarray) -> None:
        self.model.body_pos[self.obstacle_body_ids[i]] = pos

    def hide_obstacle(self, i: int) -> None:
        self.model.body_pos[self.obstacle_body_ids[i]] = _HIDDEN_POS

    def set_marker_pos(self, pos: np.ndarray) -> None:
        self.model.body_pos[self.marker_id] = pos

    def obstacle_positions(self) -> np.ndarray:
        positions = np.zeros((spec.n_max_obstacles, 3), dtype=np.float64)
        for i, body_id in enumerate(self.obstacle_body_ids):
            pos = self.model.body_pos[body_id]
            if pos[2] > -5.0:
                positions[i] = pos
        return positions

    def reset_world(
        self,
        rng: np.random.Generator,
        is_collision_free_fn: callable,
        num_obstacles: int | str = "all",
        max_retries: int = 20,
    ) -> int:
        if num_obstacles == "all":
            n_active = int(rng.integers(0, spec.n_max_obstacles + 1))
        else:
            n_active = int(np.clip(num_obstacles, 0, spec.n_max_obstacles))

        for i in range(spec.n_max_obstacles):
            self.hide_obstacle(i)

        for i in range(n_active):
            for _ in range(max_retries):
                self.set_obstacle_pos(i, self._sample_pos(rng))
                if is_collision_free_fn(home_qpos):
                    break
            else:
                self.hide_obstacle(i)

        return n_active

    @staticmethod
    def _sample_pos(rng: np.random.Generator) -> np.ndarray:
        r = rng.uniform(0.20, 0.65)
        theta = rng.uniform(-np.pi, np.pi)
        z = rng.uniform(0.10, 0.85)
        return np.array([r * np.cos(theta), r * np.sin(theta), z])
