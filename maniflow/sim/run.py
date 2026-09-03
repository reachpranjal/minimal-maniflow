import argparse

from maniflow.checkpoint import load_policy
from maniflow.config import Config
from maniflow.sim.rollout import rollout
from maniflow.sim.single_arm.env import SingleArmEnv
from maniflow.sim.single_arm.viewer import LiveViewer


def _obstacles_for(ep: int, num_obstacles: int | str, n_max_obstacles: int) -> int:
    if num_obstacles == "all":
        return ep % (n_max_obstacles + 1)
    return num_obstacles


def render(
    cfg: Config,
    ckpt_path: str = "checkpoints/model.pt",
    n_episodes: int = None,
    num_obstacles: int | str = "all",
    orbit: bool = False,
):
    env = SingleArmEnv(cfg.env)

    policy, obs_norm, action_norm = load_policy(cfg, ckpt_path, env, compile=False)

    env.reset(
        seed=0, num_obstacles=_obstacles_for(0, num_obstacles, cfg.env.n_max_obstacles)
    )

    ep = 0
    try:
        with LiveViewer(env, speed=1.0, rotate=orbit, rotate_speed=36) as viewer:
            while ep < n_episodes and viewer.is_running():
                obstacles = _obstacles_for(ep, num_obstacles, cfg.env.n_max_obstacles)

                env.reset(seed=ep, num_obstacles=obstacles)
                viewer.sync()
                viewer.pause(0.8)
                rollout(
                    data_cfg=cfg.data,
                    policy=policy,
                    env=env,
                    obs_norm=obs_norm,
                    action_norm=action_norm,
                    num_obstacles=obstacles,
                    seed=ep,
                    n_ode_steps=cfg.train.n_ode_steps,
                    guidance_scale=cfg.train.guidance_scale,
                    joint_vel_limit_scale=cfg.train.joint_vel_limit_scale,
                    step_hook=viewer.step_hook,
                    device=cfg.device,
                )
                ep += 1
                viewer.pause(1.2)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description="Rollout Policy Closed-Loop")
    ap.add_argument("--ckpt", type=str, default="checkpoints/model.pt")
    ap.add_argument(
        "--n_episodes", type=int, default=500, help="default: loop ~forever"
    )
    ap.add_argument(
        "--num_obstacles",
        type=int,
        default=None,
        help="obstacle count 0-5, or 'all' to cycle through every count",
    )
    ap.add_argument(
        "--enable_orbit",
        action="store_true",
        help="Orbit the spectator camera. Looks cool to me :)",
    )
    args = ap.parse_args()

    n_obstacles = "all" if args.num_obstacles is None else args.num_obstacles

    cfg = Config()

    render(
        cfg=cfg,
        ckpt_path=args.ckpt,
        n_episodes=args.n_episodes,
        num_obstacles=n_obstacles,
        orbit=args.enable_orbit,
    )


if __name__ == "__main__":
    main()
