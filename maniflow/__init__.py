import importlib

__version__ = "0.1.0"

_exports: dict[str, str] = {
    "Config": "maniflow.config:Config",
    "EnvConfig": "maniflow.config:EnvConfig",
    "DataConfig": "maniflow.config:DataConfig",
    "ModelConfig": "maniflow.config:ModelConfig",
    "TrainConfig": "maniflow.config:TrainConfig",
    "Normalizer": "maniflow.dataloader.normalize:Normalizer",
    "load_zarr_data": "maniflow.dataloader.dataloader:load_zarr_data",
    "DataModule": "maniflow.dataloader.dataloader:DataModule",
    "collect": "maniflow.collect_data:collect",
    "RRTConnectPlanner": "maniflow.sim.single_arm.expert:RRTConnectPlanner",
    "SingleArmEnv": "maniflow.sim.single_arm.env:SingleArmEnv",
    "rollout": "maniflow.sim.rollout:rollout",
    "evaluate": "maniflow.eval.metrics:evaluate",
    "EvalMetrics": "maniflow.eval.metrics:EvalMetrics",
    "FlowMatchingPolicy": "maniflow.model.policy:FlowMatchingPolicy",
    "sample_action": "maniflow.model.policy:sample_action",
    "Trainer": "maniflow.training.trainer:Trainer",
}

__all__ = sorted(_exports)


def __getattr__(name: str):
    target = _exports.get(name)
    if target is None:
        raise AttributeError(f"module 'maniflow' has no attribute {name!r}")
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted([*globals(), *_exports])
