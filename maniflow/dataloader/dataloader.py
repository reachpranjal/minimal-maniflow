from dataclasses import dataclass, field

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

from maniflow.config import DataConfig
from maniflow.dataloader.normalize import Normalizer


@dataclass(frozen=True)
class DataModule:
    train_loader: DataLoader
    val_loader: DataLoader
    obs_norm: Normalizer
    action_norm: Normalizer
    stats: dict = field(default_factory=dict)


class SingleArmDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        obs: np.ndarray,
        action: np.ndarray,
        episode_ends: np.ndarray,
        obs_norm: Normalizer,
        action_norm: Normalizer,
    ):
        self.obs = obs
        self.action = action
        self.obs_norm = obs_norm
        self.action_norm = action_norm
        self.obs_horizon = cfg.obs_horizon
        self.pred_horizon = cfg.pred_horizon

        ep_starts = np.concatenate([[0], episode_ends[:-1]])
        ep_lengths = episode_ends - ep_starts
        valid_lengths = np.maximum(ep_lengths - cfg.pred_horizon + 1, 0)

        t_locals = np.concatenate([np.arange(v) for v in valid_lengths])
        starts = np.repeat(ep_starts, valid_lengths)
        self._index = np.stack([starts, t_locals], axis=1)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx):
        ep_start, t_local = self._index[idx]
        t_global = ep_start + t_local

        indices = np.clip(
            np.arange(t_global - self.obs_horizon + 1, t_global + 1), ep_start, None
        )
        observation = self.obs[indices]
        action_chunk = self.action[t_global : t_global + self.pred_horizon]

        return (
            self.obs_norm.normalize(torch.from_numpy(observation)),
            self.action_norm.normalize(torch.from_numpy(action_chunk)),
        )


def load_zarr_data(cfg: DataConfig) -> DataModule:
    store = zarr.open(cfg.zarr_path, mode="r")
    obs_all = store["data/obs"][:]
    act_all = store["data/action"][:]
    ep_ends = store["meta/episode_ends"][:]

    num_ep = len(ep_ends)
    num_train_ep = int(num_ep * cfg.train_frac)
    num_val_ep = num_ep - num_train_ep

    train_ends = ep_ends[:num_train_ep]
    val_ends = ep_ends[num_train_ep:]
    train_cutoff = int(train_ends[-1])

    train_obs, train_act = obs_all[:train_cutoff], act_all[:train_cutoff]
    val_obs, val_act = obs_all[train_cutoff:], act_all[train_cutoff:]
    val_ends_local = val_ends - train_cutoff

    obs_norm = Normalizer(train_obs)
    action_norm = Normalizer(train_act)

    train_dataset = SingleArmDataset(
        cfg=cfg,
        obs=train_obs,
        action=train_act,
        episode_ends=train_ends,
        obs_norm=obs_norm,
        action_norm=action_norm,
    )

    val_dataset = SingleArmDataset(
        cfg=cfg,
        obs=val_obs,
        action=val_act,
        episode_ends=val_ends_local,
        obs_norm=obs_norm,
        action_norm=action_norm,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )

    print(f"Dataset: {cfg.zarr_path}")
    print(f"Episodes: {num_ep}  ({num_train_ep} train / {num_val_ep} val)")
    print(f"Samples: {len(train_dataset):,} train / {len(val_dataset):,} val")

    return DataModule(
        train_loader=train_loader,
        val_loader=val_loader,
        obs_norm=obs_norm,
        action_norm=action_norm,
        stats={
            "num_train_ep": num_train_ep,
            "num_val_ep": num_val_ep,
            "num_train_steps": len(train_dataset),
            "num_val_steps": len(val_dataset),
        },
    )


def parse_zarr(zarr_path: str) -> dict:
    store = zarr.open(zarr_path, mode="r")
    return {
        "obs": store["data/obs"][:],
        "action": store["data/action"][:],
        "episode_ends": store["meta/episode_ends"][:],
    }
