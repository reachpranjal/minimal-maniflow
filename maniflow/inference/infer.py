import argparse

from maniflow.checkpoint import load_policy
from maniflow.config import Config
from maniflow.eval.metrics import evaluate
from maniflow.sim.single_arm.env import SingleArmEnv


def run_eval(
    cfg: Config,
    ckpt_path: str,
    n_episodes: int = 50,
    num_obstacles: int | str = "all",
) -> dict:
    env = SingleArmEnv(cfg.env)
    policy, obs_norm, action_norm = load_policy(cfg, ckpt_path, env)
    print(f"Episodes: {n_episodes} | num_obstacles={num_obstacles}\n")

    m = evaluate(
        policy=policy,
        env=env,
        obs_norm=obs_norm,
        action_norm=action_norm,
        device=cfg.device,
        data_cfg=cfg.data,
        n_episodes=n_episodes,
        num_obstacles=num_obstacles,
        n_ode_steps=cfg.train.n_ode_steps,
        guidance_scale=cfg.train.guidance_scale,
        joint_vel_limit_scale=cfg.train.joint_vel_limit_scale,
    )
    env.close()
    return m


def main():
    ap = argparse.ArgumentParser(description="Checkpoint Evaluation")
    ap.add_argument("--ckpt", type=str, default="checkpoints/model.pt")
    ap.add_argument("--eval_eps", type=int, default=50)
    ap.add_argument(
        "--num_obstacles",
        type=int,
        default=None,
        help="obstacle count 0-5, or 'all' to cycle through every count",
    )
    args = ap.parse_args()

    n_obstacles = "all" if args.num_obstacles is None else args.num_obstacles

    cfg = Config()

    evals = run_eval(
        cfg, ckpt_path=args.ckpt, n_episodes=args.eval_eps, num_obstacles=n_obstacles
    )

    print(f"SR@0.05 rad (success): {evals.sr05:.1%}")
    print(f"SR@0.20 rad: {evals.sr20:.1%}")
    print(f"SR@0.50 rad: {evals.sr50:.1%}")
    print(f"Mean Final dist: {evals.mean_final_dist:.4f} rad")
    print(f"Mean Return: {evals.mean_return:.3f}")


if __name__ == "__main__":
    main()
