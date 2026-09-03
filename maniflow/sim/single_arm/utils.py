from functools import lru_cache
from pathlib import Path

import mujoco

from maniflow.config import EnvConfig

_panda_xml = (
    Path(__file__).resolve().parent.parent / "assets" / "franka_panda" / "panda.xml"
)

_hidden_pos = (0.0, 0.0, -10.0)


@lru_cache(maxsize=1)
def _panda_xml_path() -> str:
    if not _panda_xml.exists():
        raise FileNotFoundError(
            f"Vendored Panda model not found at {_panda_xml}. "
            "Expected sim/assets/franka_panda/panda.xml."
        )
    return str(_panda_xml)


def _add_cosmetics(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    mat = spec.add_material(
        name="groundplane", texrepeat=[5, 5], texuniform=True, reflectance=0.2
    )
    roles = mat.textures
    roles[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    mat.textures = roles

    head = spec.visual.headlight
    head.diffuse = [0.6, 0.6, 0.6]
    head.ambient = [0.3, 0.3, 0.3]
    head.specular = [0.0, 0.0, 0.0]
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]


def build_scene_spec(env_cfg: EnvConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(_panda_xml_path())
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.timestep = env_cfg.timestep

    _add_cosmetics(spec)

    world = spec.worldbody
    light = world.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    light.name = "ceiling"

    world.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="groundplane",
    )

    for i in range(env_cfg.n_max_obstacles):
        body = world.add_body(name=f"obstacle_{i}", pos=list(_hidden_pos))
        body.add_geom(
            name=f"obstacle_geom_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[env_cfg.obstacle_radius, 0.0, 0.0],
            rgba=[0.85, 0.2, 0.15, 1.0],
            contype=1,
            conaffinity=1,
        )

    marker = world.add_body(name="goal_marker", pos=[0.0, 0.0, 0.5])
    marker.add_geom(
        name="goal_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.04, 0.0, 0.0],
        rgba=[0.0, 1.0, 0.0, 0.4],
        contype=0,
        conaffinity=0,
    )

    return spec


def load_panda_model(env_cfg: EnvConfig) -> mujoco.MjModel:
    return build_scene_spec(env_cfg).compile()
