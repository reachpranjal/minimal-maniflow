import argparse
from dataclasses import replace

import torch

from maniflow.config import Config, DataConfig, TrainConfig
from maniflow.dataloader.dataloader import load_zarr_data
from maniflow.model import FlowMatchingPolicy
from maniflow.sim.single_arm.env import SingleArmEnv
from maniflow.training.trainer import Trainer


def build_config(args):
    cfg_data = replace(
        DataConfig(),
        zarr_path=args.zarr,
        batch_size=args.batch,
        num_workers=args.num_workers,
    )
    cfg_train = replace(
        TrainConfig(),
        checkpoint_path=args.checkpoint_path,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=args.device,
        state_noise=args.state_noise,
        torch_compile=not args.no_compile,
    )
    return Config(data=cfg_data, train=cfg_train)


def init_scheduler(cfg, optimizer, steps_per_epoch):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr * 5.0,
        epochs=cfg.n_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )


def main():
    torch.set_float32_matmul_precision("high")

    cfg = Config()

    ap = argparse.ArgumentParser(description="Train the flow-matching policy.")

    ap.add_argument("--zarr", type=str, default=cfg.data.zarr_path)
    ap.add_argument("--batch", type=int, default=cfg.data.batch_size)
    ap.add_argument("--num_workers", type=int, default=cfg.data.num_workers)

    ap.add_argument("--no_compile", action="store_true")

    ap.add_argument("--n_epochs", type=int, default=cfg.train.n_epochs)
    ap.add_argument("--lr", type=float, default=cfg.train.lr)
    ap.add_argument("--device", type=str, default=cfg.train.device)
    ap.add_argument("--checkpoint_path", type=str, default=cfg.train.checkpoint_path)
    ap.add_argument("--state_noise", type=float, default=cfg.train.state_noise)

    args = ap.parse_args()

    cfg = build_config(args)

    env = SingleArmEnv(cfg.env)
    action_dim = env.action_dim
    obs_dim = env.obs_dim
    relative_goal = env.goal_relative_pairs

    dataset = load_zarr_data(cfg.data)

    model = FlowMatchingPolicy(
        model_cfg=cfg.model,
        obs_horizon=cfg.data.obs_horizon,
        pred_horizon=cfg.data.pred_horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        goal_relative_pairs=relative_goal,
    ).to(cfg.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = init_scheduler(
        cfg=cfg.train,
        optimizer=optim,
        steps_per_epoch=len(dataset.train_loader),
    )

    trainer = Trainer(
        cfg=cfg,
        dataloader=dataset,
        policy=model,
        optimizer=optim,
        scheduler=scheduler,
        eval_env=env,
    )

    trainer.train()


if __name__ == "__main__":
    main()
