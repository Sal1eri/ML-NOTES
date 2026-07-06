from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from vision.data import denormalize

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class ExperimentLogger:
    def __init__(self, args: Namespace, run_dir: Path, class_names: list[str]) -> None:
        self.args = args
        self.run_dir = run_dir
        self.class_names = class_names
        self.tb: SummaryWriter | None = None
        self.wandb_run = None

        if args.use_tensorboard:
            self.tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            self.tb.add_text("config/args", _markdown_dict(vars(args)), 0)
            self.tb.add_text("config/env", _markdown_dict(_tracked_env()), 0)

        if args.use_wandb:
            if wandb is None:
                raise ImportError("wandb is not installed")
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.run_name,
                mode=args.wandb_mode,
                dir=str(run_dir),
                config=vars(args),
                tags=[args.model, "cifar10", "from-scratch"],
            )
            wandb.config.update({"tracked_env": _tracked_env()}, allow_val_change=True)

    def watch_model(self, model: torch.nn.Module) -> None:
        if self.wandb_run is not None:
            wandb.watch(model, log="all", log_freq=self.args.wandb_watch_freq)

    def log_model_graph(self, model: torch.nn.Module, example_images: torch.Tensor) -> None:
        if self.tb is None:
            return
        try:
            self.tb.add_graph(model, example_images)
        except Exception as exc:
            self.tb.add_text("warnings/add_graph", str(exc), 0)

    def log_images(self, tag: str, images: torch.Tensor, labels: torch.Tensor, step: int) -> None:
        images = denormalize(images.detach().cpu())
        if self.tb is not None:
            self.tb.add_image(tag, make_grid(images, nrow=min(8, len(images))), step)
        if self.wandb_run is not None:
            captions = [self.class_names[int(label)] for label in labels]
            wandb.log({tag: [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]}, step=step)

    def log_predictions(
        self,
        tag: str,
        images: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        step: int,
    ) -> None:
        probs = logits.softmax(dim=1).detach().cpu()
        preds = probs.argmax(dim=1)
        images = denormalize(images.detach().cpu())
        if self.wandb_run is not None:
            table = wandb.Table(columns=["image", "target", "prediction", "confidence"])
            for image, target, pred, prob in zip(images, labels.cpu(), preds, probs):
                table.add_data(
                    wandb.Image(image),
                    self.class_names[int(target)],
                    self.class_names[int(pred)],
                    float(prob[int(pred)]),
                )
            wandb.log({tag: table}, step=step)

    def log_step(self, metrics: dict[str, float], step: int, prefix: str) -> None:
        payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
        if self.tb is not None:
            for key, value in payload.items():
                self.tb.add_scalar(key, value, step)
        if self.wandb_run is not None:
            wandb.log(payload, step=step)

    def log_epoch(
        self,
        epoch: int,
        step: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float,
        y_true: list[int],
        y_pred: list[int],
    ) -> None:
        metrics = {"epoch": epoch, "lr": lr}
        metrics.update({f"train/{key}": value for key, value in train_metrics.items()})
        metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
        if self.tb is not None:
            for key, value in metrics.items():
                self.tb.add_scalar(key, value, epoch)
        if self.wandb_run is not None:
            wandb.log(metrics, step=step)
        self.log_confusion_matrix("val/confusion_matrix", y_true, y_pred, step)

    def log_confusion_matrix(self, tag: str, y_true: list[int], y_pred: list[int], step: int) -> None:
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay(matrix, display_labels=self.class_names).plot(
            include_values=False,
            xticks_rotation=45,
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        fig.tight_layout()
        if self.tb is not None:
            self.tb.add_figure(tag, fig, step)
        if self.wandb_run is not None:
            wandb.log({tag: wandb.Image(fig)}, step=step)
            wandb.log(
                {
                    f"{tag}_interactive": wandb.plot.confusion_matrix(
                        y_true=y_true,
                        preds=y_pred,
                        class_names=self.class_names,
                    )
                },
                step=step,
            )
        plt.close(fig)

    def log_histograms(self, model: torch.nn.Module, step: int) -> None:
        for name, param in model.named_parameters():
            if self.tb is not None:
                self.tb.add_histogram(f"weights/{name}", param.detach().cpu(), step)
                if param.grad is not None:
                    self.tb.add_histogram(f"grads/{name}", param.grad.detach().cpu(), step)
            if self.wandb_run is not None and param.grad is not None:
                wandb.log(
                    {
                        f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy()),
                        f"grads/{name}": wandb.Histogram(param.grad.detach().cpu().numpy()),
                    },
                    step=step,
                )

    def log_checkpoint(self, checkpoint_path: Path, metric: float, aliases: list[str]) -> None:
        if self.wandb_run is None:
            return
        artifact = wandb.Artifact(f"{self.args.run_name}-checkpoint", type="model")
        artifact.add_file(str(checkpoint_path))
        artifact.metadata = {"best_val_acc": metric, "model": self.args.model}
        self.wandb_run.log_artifact(artifact, aliases=aliases)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def save_run_config(args: Namespace, run_dir: Path) -> None:
    payload = {"args": vars(args), "env": _tracked_env()}
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _tracked_env() -> dict[str, Any]:
    keys = [
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "DATA_DIR",
        "OUTPUT_DIR",
        "CACHE_DIR",
        "CHECKPOINT_DIR",
        "LOG_DIR",
        "SEED",
        "DEVICE",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {key: os.environ.get(key, "") for key in keys}


def _markdown_dict(values: dict[str, Any]) -> str:
    lines = ["| key | value |", "| --- | --- |"]
    for key, value in sorted(values.items()):
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)
