import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from scipy.interpolate import CubicSpline

from maniflow.config import EnvConfig
from . import spec
from .kinematics import robot_link_names
from .utils import load_panda_model

_hidden_pos = np.array([0.0, 0.0, -10.0])


class _ObstacleChecker(ob.StateValidityChecker):
    def __init__(self, si, model, data, obstacle_geom_ids, robot_link_ids, min_z):
        super().__init__(si)
        self._model = model
        self._data = data
        self._obstacle_geom_ids = obstacle_geom_ids
        self._robot_link_ids = robot_link_ids
        self._min_z = min_z

    def isValid(self, state) -> bool:
        for i in range(spec.num_dof):
            self._data.qpos[i] = state[i]
        self._data.qpos[spec.num_dof :] = 0.0
        mujoco.mj_forward(self._model, self._data)

        for c in range(self._data.ncon):
            ct = self._data.contact[c]
            if (
                ct.geom1 in self._obstacle_geom_ids
                or ct.geom2 in self._obstacle_geom_ids
            ):
                if ct.dist < 0:
                    return False

        for body_id in self._robot_link_ids:
            if self._data.xpos[body_id, 2] < self._min_z:
                return False
        return True


class RRTConnectPlanner:
    def __init__(self, env_cfg: EnvConfig | None = None):
        env_cfg = env_cfg or EnvConfig()
        self.model = load_panda_model(env_cfg)
        self.data = mujoco.MjData(self.model)

        self.joint_low = self.model.jnt_range[: spec.num_dof, 0].copy()
        self.joint_high = self.model.jnt_range[: spec.num_dof, 1].copy()

        self._obstacle_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            for i in range(spec.n_max_obstacles)
        ]
        obstacle_geom_ids = frozenset(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom_{i}"
            )
            for i in range(spec.n_max_obstacles)
        )
        robot_link_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in robot_link_names
        )

        space = ob.RealVectorStateSpace(spec.num_dof)
        bounds = ob.RealVectorBounds(spec.num_dof)
        for i in range(spec.num_dof):
            bounds.setLow(i, float(self.joint_low[i]))
            bounds.setHigh(i, float(self.joint_high[i]))
        space.setBounds(bounds)

        self._si = ob.SpaceInformation(space)
        self._checker = _ObstacleChecker(
            self._si,
            self.model,
            self.data,
            obstacle_geom_ids,
            robot_link_ids,
            env_cfg.min_z_height,
        )
        self._si.setStateValidityChecker(self._checker)
        self._si.setup()
        self._simplifier = og.PathSimplifier(self._si)

    def _sync_obstacles(self, obstacle_positions: np.ndarray) -> None:
        for body_id, pos in zip(self._obstacle_body_ids, obstacle_positions):
            self.model.body_pos[body_id] = pos if np.any(pos != 0) else _hidden_pos

    def plan(
        self,
        start_cfg: np.ndarray,
        goal_cfg: np.ndarray,
        obstacle_positions: np.ndarray,
        n_waypoints: int = 50,
        timeout: float = 5.0,
    ) -> np.ndarray | None:
        self._sync_obstacles(obstacle_positions)

        start = self._si.allocState()
        goal = self._si.allocState()
        for i in range(spec.num_dof):
            start[i] = float(start_cfg[i])
            goal[i] = float(goal_cfg[i])

        pdef = ob.ProblemDefinition(self._si)
        pdef.setStartAndGoalStates(start, goal)

        planner = og.RRTConnect(self._si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        if not planner.solve(timeout):
            return None

        path = pdef.getSolutionPath()

        self._simplifier.simplifyMax(path)
        self._simplifier.smoothBSpline(path)
        path.interpolate(n_waypoints)
        return np.array(
            [
                [path.getState(i)[j] for j in range(spec.num_dof)]
                for i in range(path.getStateCount())
            ],
            dtype=np.float32,
        )

    def smooth(self, waypoints: np.ndarray, n_out: int) -> np.ndarray:
        n_in = len(waypoints)
        cs = CubicSpline(np.linspace(0.0, 1.0, n_in), waypoints, bc_type="clamped")
        smoothed = cs(np.linspace(0.0, 1.0, n_out))
        return np.clip(smoothed, self.joint_low, self.joint_high).astype(np.float32)
