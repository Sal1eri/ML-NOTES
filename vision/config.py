from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def flatten_config(cfg: DictConfig) -> argparse.Namespace:
    container = OmegaConf.to_container(cfg, resolve=True)
    experiment: dict[str, Any] = container["experiment"]
    data: dict[str, Any] = container["data"]
    train: dict[str, Any] = container["train"]
    logging: dict[str, Any] = container["logging"]
    flat = {
        "config": cfg.get("_config_name_", ""),
        "raw_config": container,
        "model": experiment["model"],
        "run_name": experiment["run_name"],
        "seed": int(experiment["seed"]),
        "device": experiment["device"],
        "dataset_name": data["dataset_name"],
        "data_dir": data["data_dir"],
        "cache_dir": data["cache_dir"],
        "validation_size": float(data["validation_size"]),
        "num_workers": int(data["num_workers"]),
        "limit_train": int(data["limit_train"]),
        "limit_val": int(data["limit_val"]),
        "limit_test": int(data["limit_test"]),
        "epochs": int(train["epochs"]),
        "batch_size": int(train["batch_size"]),
        "lr": float(train["lr"]),
        "weight_decay": float(train["weight_decay"]),
        "use_amp": bool(train["use_amp"]),
        "output_dir": logging["output_dir"],
        "checkpoint_dir": logging["checkpoint_dir"],
        "log_dir": logging["log_dir"],
        "use_tensorboard": bool(logging["use_tensorboard"]),
        "use_wandb": bool(logging["use_wandb"]),
        "wandb_project": logging["wandb_project"],
        "wandb_entity": logging["wandb_entity"],
        "wandb_mode": logging["wandb_mode"],
        "wandb_group": logging["wandb_group"],
        "wandb_job_type": logging["wandb_job_type"],
        "wandb_tags": list(logging["wandb_tags"]),
        "wandb_watch_freq": int(logging["wandb_watch_freq"]),
        "log_every": int(logging["log_every"]),
        "hist_every": int(logging["hist_every"]),
        "num_visualize": int(logging["num_visualize"]),
        "log_checkpoints_to_wandb": bool(logging["log_checkpoints_to_wandb"]),
    }
    return argparse.Namespace(**flat)


def apply_debug_overrides(cfg: DictConfig) -> DictConfig:
    cfg = cfg.copy()
    cfg.data.limit_train = min_positive(cfg.data.limit_train, 512)
    cfg.data.limit_val = min_positive(cfg.data.limit_val, 128)
    cfg.data.limit_test = min_positive(cfg.data.limit_test, 128)
    cfg.train.epochs = min(int(cfg.train.epochs), 1)
    cfg.logging.wandb_mode = "offline"
    return cfg


def min_positive(current: int, fallback: int) -> int:
    if current and current > 0:
        return min(int(current), fallback)
    return fallback


def save_resolved_yaml(cfg: DictConfig, run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

