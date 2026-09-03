"""
Observation layout (50-dim float32)
------------------------------------
  joint_pos: 7- current joint angles (rad)
  joint_vel: 7- current joint velocities (rad/s)
  ee_pos: 3- end-effector position [x, y, z] (m)
  ee_quat: 4- EE orientation quaternion [w, x, y, z]
  goal_joint_pos: 7- goal joint angles (rad)
  goal_ee_pos: 3- goal EE position [x, y, z] (m)
  goal_ee_quat: 4- goal EE quaternion [w, x, y, z]
  obstacle_positions: 15- 5 obstacle centres, [x, y, z] each (zeroed if inactive)

Action (7-dim float32)
-----------------------
  Target joint angles for the Panda's built-in PD position controller.
"""

import itertools
from dataclasses import dataclass

from maniflow.config import EnvConfig

_cfg = EnvConfig()
num_dof = _cfg.num_dof
n_max_obstacles = _cfg.n_max_obstacles


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    dim: int


obs_fields: tuple[Field, ...] = (
    Field("joint_pos", num_dof),
    Field("joint_vel", num_dof),
    Field("ee_pos", 3),
    Field("ee_quat", 4),
    Field("goal_joint_pos", num_dof),
    Field("goal_ee_pos", 3),
    Field("goal_ee_quat", 4),
    Field("obstacle_positions", n_max_obstacles * 3),
)


def _build_slices(fields: tuple[Field, ...]) -> dict[str, slice]:
    bounds = list(itertools.accumulate((f.dim for f in fields), initial=0))
    return {f.name: slice(lo, hi) for f, lo, hi in zip(fields, bounds, bounds[1:])}


obs_slices: dict[str, slice] = _build_slices(obs_fields)
obs_dim: int = sum(f.dim for f in obs_fields)
action_dim: int = num_dof

goal_relative_pairs: tuple[tuple[slice, slice], ...] = (
    (obs_slices["goal_joint_pos"], obs_slices["joint_pos"]),
    (obs_slices["goal_ee_pos"], obs_slices["ee_pos"]),
)
