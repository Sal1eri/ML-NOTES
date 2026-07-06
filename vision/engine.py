from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from vision.logging_utils import ExperimentLogger


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: ExperimentLogger,
    global_step: int,
    use_amp: bool,
    log_every: int,
    hist_every: int,
) -> tuple[dict[str, float], int]:
    model.train()
    scaler = GradScaler(device.type, enabled=use_amp)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_seen += batch_size
        global_step += 1

        if batch_idx % log_every == 0:
            logger.log_step(
                {
                    "loss": loss.item(),
                    "acc": correct / batch_size,
                    "grad_norm": float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                },
                global_step,
                "train_step",
            )
        if hist_every > 0 and global_step % hist_every == 0:
            logger.log_histograms(model, global_step)
        pbar.set_postfix(loss=total_loss / total_seen, acc=total_correct / total_seen)

    return {"loss": total_loss / total_seen, "acc": total_correct / total_seen}, global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, desc: str) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total_seen += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return {
        "metrics": {
            "loss": total_loss / total_seen,
            "acc": total_correct / total_seen,
            "top5_acc": total_top5 / total_seen,
        },
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        },
        path,
    )
