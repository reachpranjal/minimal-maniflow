from dataclasses import dataclass

import mujoco
import numpy as np

from maniflow.config import EnvConfig

from . import spec
from .utils import load_panda_model

home_qpos = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    dtype=np.float64,
)

joint_vel_limit = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])

robot_link_names = (
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
)


@dataclass(slots=True)
class State:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    obstacle_positions: np.ndarray


class FrankaPandaArm:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

        self.model = load_panda_model(self.cfg)
        self.data = mujoco.MjData(self.model)

        self.query_data = mujoco.MjData(self.model)

        self.joint_low = self.model.jnt_range[: spec.num_dof, 0].copy()
        self.joint_high = self.model.jnt_range[: spec.num_dof, 1].copy()
        self.joint_vel_limit = joint_vel_limit.copy()

        self.hand_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hand",
        )

        self.robot_link_ids = tuple(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
            for name in robot_link_names
        )

    def reset_home(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[: spec.num_dof] = home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[: spec.num_dof] = home_qpos

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

    def step(self, action: np.ndarray) -> None:
        action = np.clip(action, self.joint_low, self.joint_high)
        self.data.ctrl[: spec.num_dof] = action
        for _ in range(self.cfg.n_substeps):
            mujoco.mj_step(self.model, self.data)

    def state(self, obstacle_positions: np.ndarray) -> State:
        return State(
            joint_pos=self.data.qpos[: spec.num_dof].copy(),
            joint_vel=self.data.qvel[: spec.num_dof].copy(),
            ee_pos=self.data.xpos[self.hand_id].copy(),
            ee_quat=self.data.xquat[self.hand_id].copy(),
            obstacle_positions=obstacle_positions,
        )

    def _query(self, qpos: np.ndarray) -> mujoco.MjData:
        data = self.query_data
        data.qpos[:] = self.data.qpos
        data.qvel[:] = 0.0
        data.qpos[: spec.num_dof] = qpos

        mujoco.mj_forward(self.model, data)

        return data

    def forward_kinematics(
        self,
        qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self._query(qpos)
        return (
            data.xpos[self.hand_id].copy(),
            data.xquat[self.hand_id].copy(),
        )

    def is_collision_free(
        self,
        qpos: np.ndarray,
        obstacle_geom_ids: frozenset[int],
    ) -> bool:
        data = self._query(qpos)

        for c in range(data.ncon):
            contact = data.contact[c]
            if (
                contact.geom1 in obstacle_geom_ids or contact.geom2 in obstacle_geom_ids
            ) and contact.dist < 0.0:
                return False

        for body_id in self.robot_link_ids:
            if data.xpos[body_id, 2] < self.cfg.min_z_height:
                return False
        return True
