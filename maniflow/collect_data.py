import argparse
import time

import numpy as np
import zarr

from maniflow.config import EnvConfig
from maniflow.sim.single_arm.env import SingleArmEnv
from maniflow.sim.single_arm.expert import RRTConnectPlanner
from maniflow.sim.single_arm.kinematics import home_qpos


def collect(
    cfg: EnvConfig,
    n_episodes: int = 200,
    n_waypoints: int = 150,
    num_obstacles: int | str = "all",
    out_path: str = "demos.zarr",
    seed: int = 0,
    n_settle: int = 30,
) -> None:
    rng = np.random.default_rng(seed)

    env = SingleArmEnv(cfg)
    planner = RRTConnectPlanner(cfg)

    all_obs: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    episode_ends: list[int] = []
    total_steps = 0
    failed = 0

    print(f"Collecting {n_episodes} Episodes ...")
    print(
        f"\nConfig: num_obstacles={num_obstacles} | waypoints={n_waypoints} + {n_settle} settle steps"
    )

    t0 = time.time()

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed, num_obstacles=num_obstacles)
        goal = env.goal_qpos.copy()
        obstacle_positions = env.get_obstacle_positions()

        raw = planner.plan(
            home_qpos,
            goal,
            obstacle_positions,
            n_waypoints=max(10, n_waypoints // 3),
        )
        if raw is None:
            failed += 1
            continue

        traj = planner.smooth(raw, n_waypoints)

        ep_obs: list[np.ndarray] = []
        ep_action: list[np.ndarray] = []
        terminated = truncated = False

        for waypoint in traj:
            ep_obs.append(obs)
            ep_action.append(waypoint)
            obs, _, terminated, truncated, _ = env.step(waypoint)
            if terminated or truncated:
                break

        if not terminated:
            for _ in range(n_settle):
                ep_obs.append(obs)
                ep_action.append(goal)
                obs, _, terminated, truncated, _ = env.step(goal)
                if terminated:
                    break

        all_obs.extend(ep_obs)
        all_action.extend(ep_action)
        total_steps += len(ep_obs)
        episode_ends.append(total_steps)

        if (ep + 1) % 20 == 0:
            print(
                f"\t[{ep + 1:4d}/{n_episodes}]  steps={total_steps}"
                f"\tfailed={failed}  ({time.time() - t0:.1f}s)"
            )

    env.close()
    
    _write_zarr(out_path, all_obs, all_action, episode_ends, env)

    ep_lengths = np.diff(np.concatenate([[0], np.array(episode_ends)]))

    print(f"\nSaved {len(episode_ends)} episodes / {total_steps} steps  ->  {out_path}")
    print(
        f"\nEpisode length: Min={ep_lengths.min()} | Max={ep_lengths.max()} | Mean={ep_lengths.mean():.1f}"
    )
    print(f"\nFailed plans: {failed}")
    print(f"\nElapsed: {time.time() - t0:.1f}s")


def _write_zarr(out_path, all_obs, all_action, episode_ends, env) -> None:
    store = zarr.open(out_path, mode="w")
    store.create_dataset(
        "data/obs",
        data=np.array(all_obs, dtype=np.float32),
        chunks=(1000, env.obs_dim),
        compressor=zarr.Blosc(cname="lz4"),
    )
    store.create_dataset(
        "data/action",
        data=np.array(all_action, dtype=np.float32),
        chunks=(1000, env.action_dim),
        compressor=zarr.Blosc(cname="lz4"),
    )
    store.create_dataset(
        "meta/episode_ends", data=np.array(episode_ends, dtype=np.int64)
    )


def _obstacles_arg(value: str) -> int | str:
    return value if value == "all" else int(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect expert demos via RRT-Connect.")
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--n_waypoints", type=int, default=150)
    ap.add_argument("--num_obstacles", type=_obstacles_arg, default="all")
    ap.add_argument("--out", type=str, default="demos.zarr")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = EnvConfig()

    collect(
        cfg=cfg,
        n_episodes=args.n_episodes,
        n_waypoints=args.n_waypoints,
        num_obstacles=args.num_obstacles,
        out_path=args.out,
        seed=args.seed,
        n_settle=30,
    )


if __name__ == "__main__":
    main()
