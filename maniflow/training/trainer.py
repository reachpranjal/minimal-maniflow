import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from maniflow.checkpoint import save_checkpoint
from maniflow.config import Config
from maniflow.dataloader import DataModule
from maniflow.eval.metrics import evaluate
from maniflow.protocols import Env

from .viz import save_training_plots


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        dataloader: DataModule,
        policy: nn.Module,
        optimizer,
        scheduler,
        eval_env: Env,
    ):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.device = cfg.device
        _set_seed(cfg.data.seed)

        self.train_set = dataloader.train_loader
        self.val_set = dataloader.val_loader
        self.obs_norm = dataloader.obs_norm
        self.action_norm = dataloader.action_norm

        self.policy = policy
        self.model = torch.compile(policy) if cfg.use_torch_compile else policy
        n_params = sum(p.numel() for p in self.policy.parameters())
        self.ema = ExponentialMovingAverage(
            self.policy.parameters(),
            decay=self.train_cfg.ema_decay,
            use_num_updates=False,
        )
        self.eval_policy = deepcopy(self.policy).requires_grad_(False).eval()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self._sched_per_batch = isinstance(scheduler, OneCycleLR)

        self._amp = dict(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.train_cfg.bf16,
        )

        self.eval_env = eval_env

        os.makedirs(self.train_cfg.checkpoint_path, exist_ok=True)
        self.plot_dir = os.path.join(self.train_cfg.checkpoint_path, "plots")
        self.history: dict = defaultdict(list)
        self.history["best_val_epoch"] = None

        self.best_val = float("inf")
        self.best_obsall_sr05 = -1.0
        self.best_obsall_dist = float("inf")
        self.best_obsall_sr50 = -1.0

        print(f"Device {self.device}")
        print(f"Model Params {n_params: }")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        loss_sum = torch.zeros((), device=self.device)
        n = 0
        pbar = tqdm(
            self.train_set,
            desc=f"Epoch {epoch:>4}",
            leave=False,
            unit="batch",
        )
        std = self.train_cfg.state_noise
        for i, (obs, action) in enumerate(pbar):
            obs = obs.to(self.device, non_blocking=True)
            x1 = action.to(self.device, non_blocking=True)

            if std > 0.0:
                obs = self.eval_env.augment_observation(obs, std)

            time = torch.rand(obs.shape[0], device=obs.device)
            x0 = torch.randn_like(x1)
            t = time[:, None, None]
            xt = (1.0 - t) * x0 + t * x1
            ut = x1 - x0

            with torch.autocast(**self._amp):
                vt = self.model(xt, time, obs)
                train_loss = F.mse_loss(vt, ut)

            self.optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.train_cfg.grad_clip
            )
            self.optimizer.step()
            if self._sched_per_batch:
                self.scheduler.step()
            self.ema.update()

            loss_sum += train_loss.detach()
            n += 1

            if i % 50 == 0:
                pbar.set_postfix(loss=f"{train_loss.item():.4f}")

        return (loss_sum / max(n, 1)).item()

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        loss_sum = torch.zeros((), device=self.device)
        n = 0

        for obs, action in self.val_set:
            obs = obs.to(self.device, non_blocking=True)
            x1 = action.to(self.device, non_blocking=True)
            time = torch.rand(obs.shape[0], device=obs.device)
            x0 = torch.randn_like(x1)
            t = time[:, None, None]
            xt = (1.0 - t) * x0 + t * x1
            ut = x1 - x0

            with torch.autocast(**self._amp):
                vt = self.model(xt, time, obs)
                val_loss = F.mse_loss(vt, ut)

            loss_sum += val_loss.detach()
            n += 1
        return (loss_sum / max(n, 1)).item()

    def _ema_policy(self) -> nn.Module:
        self.ema.copy_to(self.eval_policy.parameters())
        return self.eval_policy

    def _save(self, name: str, epoch: int, model: nn.Module) -> None:
        save_checkpoint(
            path=os.path.join(self.train_cfg.checkpoint_path, name),
            epoch=epoch,
            model=model,
            obs_norm=self.obs_norm,
            action_norm=self.action_norm,
        )

    def _eval_obstacles(self, ema_policy: nn.Module, num_obstacles: int | str):
        return evaluate(
            data_cfg=self.cfg.data,
            policy=ema_policy,
            env=self.eval_env,
            obs_norm=self.obs_norm,
            action_norm=self.action_norm,
            n_episodes=self.train_cfg.eval_episodes,
            n_ode_steps=self.train_cfg.n_ode_steps,
            guidance_scale=self.train_cfg.guidance_scale,
            joint_vel_limit_scale=self.train_cfg.joint_vel_limit_scale,
            num_obstacles=num_obstacles,
            device=self.device,
        )

    def _run_env_eval(self, epoch: int):
        ema_policy = self._ema_policy()

        no_obs = self._eval_obstacles(ema_policy, num_obstacles=0)
        all_obs = self._eval_obstacles(ema_policy, num_obstacles="all")

        self.history["eval_epochs"].append(epoch)
        for prefix, m in [("obs0", no_obs), ("obsall", all_obs)]:
            self.history[f"{prefix}_dist"].append(m.mean_final_dist)
            self.history[f"{prefix}_sr05"].append(m.sr05)
            self.history[f"{prefix}_sr20"].append(m.sr20)
            self.history[f"{prefix}_sr50"].append(m.sr50)

        sr05, dist = all_obs.sr05, all_obs.mean_final_dist
        if sr05 > self.best_obsall_sr05 or (
            sr05 == self.best_obsall_sr05 and dist < self.best_obsall_dist
        ):
            self.best_obsall_sr05, self.best_obsall_dist = sr05, dist
            self._save("best.pt", epoch, self.policy)
            self._save("best_ema.pt", epoch, self._ema_policy())

        if all_obs.sr50 > self.best_obsall_sr50:
            self.best_obsall_sr50 = all_obs.sr50
            self._save(f"epoch_{epoch}.pt", epoch, self._ema_policy())

    def train(self):
        pbar = tqdm(range(1, self.train_cfg.n_epochs + 1), desc="Training", unit="ep")

        for epoch in pbar:
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch()

            if not self._sched_per_batch:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            pbar.set_postfix(
                train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.2e}"
            )

            if epoch % self.train_cfg.eval_every == 0:
                self._run_env_eval(epoch)

            save_training_plots(self.plot_dir, self.history)

        self._save("best_ema_final.pt", self.train_cfg.n_epochs, self._ema_policy())

        out = self.train_cfg.checkpoint_path
        tqdm.write(f"\nBest obsall SR (raw): {out}/model.pt")
        tqdm.write(f"Best obsall SR (EMA): {out}/best_ema.pt")
        tqdm.write(f"Final EMA: {out}/best_ema_final.pt")

        tqdm.write(f"Plots saved to: {self.plot_dir}/")
