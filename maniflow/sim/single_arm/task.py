import numpy as np

from maniflow.config import EnvConfig

from .kinematics import FrankaPandaArm, State, home_qpos


class GoalReachingTask:
    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.goal_qpos = home_qpos.copy()
        self.goal_ee_pos = np.zeros(3)
        self.goal_ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def sample_goal(
        self,
        sim: FrankaPandaArm,
        rng: np.random.Generator,
        collision_fn: callable,
        goal_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        if goal_qpos is not None:
            self.goal_qpos = goal_qpos.copy()
        else:
            self.goal_qpos = self._sample_goal_qpos(sim, rng, collision_fn)
        self.goal_ee_pos, self.goal_ee_quat = sim.forward_kinematics(self.goal_qpos)
        return self.goal_qpos

    def _sample_goal_qpos(
        self,
        sim: FrankaPandaArm,
        rng: np.random.Generator,
        collision_fn: callable,
        max_attempts: int = 200,
    ) -> np.ndarray:
        for _ in range(max_attempts):
            q = rng.uniform(sim.joint_low, sim.joint_high)
            if np.linalg.norm(q - home_qpos) > self.cfg.goal_min_dist and collision_fn(
                q
            ):
                return q
        return home_qpos.copy()

    def distance(self, state: State) -> float:
        return float(np.linalg.norm(state.joint_pos - self.goal_qpos))

    def reward(self, state: State) -> float:
        return -self.distance(state)

    def success(self, state: State) -> bool:
        return self.distance(state) < self.cfg.success_thresh

    def observation(self, state: State) -> np.ndarray:
        obs = np.concatenate(
            [
                state.joint_pos,
                state.joint_vel,
                state.ee_pos,
                state.ee_quat,
                self.goal_qpos,
                self.goal_ee_pos,
                self.goal_ee_quat,
                state.obstacle_positions.reshape(-1),
            ]
        )
        return obs.astype(np.float32)
