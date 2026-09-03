from dataclasses import dataclass, field

import torch

@dataclass(frozen=True, slots=True)
class EnvConfig:
    timestep: float = 0.002
    n_substeps: int = 10
    max_episode_steps: int = 200

    success_thresh: float = 0.05
    goal_min_dist: float = 0.5
    min_z_height: float = 0.05

    img_width: int = 84
    img_height: int = 84

    num_dof = 7
    n_max_obstacles: int = 5
    obstacle_radius = 0.08


@dataclass(frozen=True, slots=True)
class DataConfig:
    zarr_path: str = "dataset/data2k.zarr"
    batch_size: int = 256
    obs_horizon: int = 5
    pred_horizon: int = 16
    exec_action_horizon: int = 8
    train_frac: float = 0.8
    seed: int = 0
    num_workers: int = 4


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int = 256
    n_layers: int = 8
    time_dim: int = 128
    dropout: float = 0.1

    cond_dropout_prob: float = 0.2


@dataclass(frozen=True, slots=True)
class TrainConfig:
    checkpoint_path: str = "checkpoints"

    n_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    state_noise: float = 0.1

    n_ode_steps: int = 5
    guidance_scale: float = 1.5
    joint_vel_limit_scale: float | None = 1.0

    eval_every: int = 20
    eval_episodes: int = 10
    device: str = "auto"

    torch_compile: bool = True
    bf16: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def device(self):
        spec = self.train.device
        if spec == "auto":
            spec = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(spec)

    @property
    def use_torch_compile(self) -> bool:
        if not self.train.torch_compile:
            return False
        return hasattr(torch, "compile")
