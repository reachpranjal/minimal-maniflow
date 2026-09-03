import os

import torch
import torch.nn as nn

from maniflow.config import Config
from maniflow.dataloader.normalize import Normalizer
from maniflow.model import FlowMatchingPolicy
from maniflow.protocols import Env


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    obs_norm: Normalizer,
    action_norm: Normalizer,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "obs_norm": obs_norm.to_dict(),
            "action_norm": action_norm.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str, device: torch.device
) -> tuple[dict, Normalizer, Normalizer]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_norm = Normalizer.from_dict(ckpt["obs_norm"])
    action_norm = Normalizer.from_dict(ckpt.get("action_norm"))
    return ckpt, obs_norm, action_norm


def load_policy(cfg: Config, checkpoint_path: str, env: Env, compile: bool = True):
    ckpt, obs_norm, action_norm = load_checkpoint(checkpoint_path, cfg.device)
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    model = FlowMatchingPolicy(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        obs_horizon=cfg.data.obs_horizon,
        pred_horizon=cfg.data.pred_horizon,
        goal_relative_pairs=env.goal_relative_pairs,
        model_cfg=cfg.model,
    ).to(cfg.device)
    model.load_checkpoint(ckpt["model"])
    model.eval()

    if compile and cfg.use_torch_compile:
        model = torch.compile(model)

    return model, obs_norm, action_norm
