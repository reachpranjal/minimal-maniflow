from typing import TypeVar

import numpy as np
import torch

Array = TypeVar("Array", np.ndarray, torch.Tensor)

_eps = 1e-8


class Normalizer:
    __slots__ = ("mean", "std")

    def __init__(self, data: np.ndarray):
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = (data.std(axis=0) + _eps).astype(np.float32)

    @classmethod
    def from_stats(cls, mean: np.ndarray, std: np.ndarray) -> "Normalizer":
        obj = cls.__new__(cls)
        obj.mean = np.asarray(mean, dtype=np.float32)
        obj.std = np.asarray(std, dtype=np.float32)
        return obj

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls.from_stats(d["mean"], d["std"])

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def _stats_like(self, x: Array) -> tuple[Array, Array]:
        if isinstance(x, torch.Tensor):
            return (
                torch.from_numpy(self.mean).to(x.device),
                torch.from_numpy(self.std).to(x.device),
            )
        return self.mean, self.std

    def normalize(self, x: Array) -> Array:
        mean, std = self._stats_like(x)
        return (x - mean) / std

    def unnormalize(self, x: Array) -> Array:
        mean, std = self._stats_like(x)
        return x * std + mean
