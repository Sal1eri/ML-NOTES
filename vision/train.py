from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch import nn

from vision.config import apply_debug_overrides, flatten_config, save_resolved_yaml
from vision.data import make_dataloaders, take_batch
from vision.engine import evaluate, save_checkpoint, train_one_epoch
from vision.logging_utils import ExperimentLogger, save_run_config
from vision.models import build_model


load_dotenv()


@hydra.main(version_base=None, config_path="../configs/vision", config_name="cnn")
def main(cfg: DictConfig) -> None:
    if cfg.get("debug", False):
        cfg = apply_debug_overrides(cfg)
    args = flatten_config(cfg)
    if not args.run_name:
        args.run_name = f"{args.model}-cifar10-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.cache_dir = str(Path(args.cache_dir) / "huggingface")
    run_dir = Path(args.output_dir) / "vision" / args.run_name
    checkpoint_dir = Path(args.checkpoint_dir) / "vision" / args.run_name
    log_dir = Path(args.log_dir) / "vision" / args.run_name
    for path in [run_dir, checkpoint_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)
    args.checkpoint_dir = str(checkpoint_dir)
    args.log_dir = str(log_dir)

    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)
    args.use_amp = bool(args.use_amp and device.type == "cuda")

    dataloaders = make_dataloaders(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        validation_size=args.validation_size,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
    )
    model = build_model(args.model, num_classes=len(dataloaders.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_run_config(args, run_dir)
    save_resolved_yaml(cfg, run_dir)
    logger = ExperimentLogger(args, log_dir, dataloaders.class_names)
    logger.watch_model(model)

    example_images, example_labels = take_batch(dataloaders.val_loader, args.num_visualize)
    logger.log_images("data/validation_samples", example_images, example_labels, step=0)
    logger.log_model_graph(model, example_images.to(device))

    best_val_acc = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=dataloaders.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            global_step=global_step,
            use_amp=args.use_amp,
            log_every=args.log_every,
            hist_every=args.hist_every,
        )
        val_result = evaluate(model, dataloaders.val_loader, criterion, device, desc=f"val epoch {epoch}")
        val_metrics = val_result["metrics"]
        scheduler.step()
        logger.log_epoch(
            epoch=epoch,
            step=global_step,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=optimizer.param_groups[0]["lr"],
            y_true=val_result["y_true"],
            y_pred=val_result["y_pred"],
        )

        checkpoint_path = checkpoint_dir / "last.pt"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_acc, args)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_path = checkpoint_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_acc, args)
            if args.log_checkpoints_to_wandb:
                logger.log_checkpoint(best_path, best_val_acc, aliases=["best", f"epoch-{epoch}"])

        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} best_val_acc={best_val_acc:.4f}"
        )

    if dataloaders.test_loader is not None:
        test_result = evaluate(model, dataloaders.test_loader, criterion, device, desc="test")
        logger.log_step(test_result["metrics"], step=global_step + 1, prefix="test")
        logger.log_confusion_matrix("test/confusion_matrix", test_result["y_true"], test_result["y_pred"], global_step + 1)
        print(f"test_acc={test_result['metrics']['acc']:.4f} test_loss={test_result['metrics']['loss']:.4f}")

    sample_images, sample_labels = take_batch(dataloaders.val_loader, args.num_visualize)
    with torch.no_grad():
        sample_logits = model(sample_images.to(device)).cpu()
    logger.log_predictions("predictions/validation_table", sample_images, sample_labels, sample_logits, step=global_step + 2)
    logger.close()
    print(f"TensorBoard: tensorboard --logdir {log_dir / 'tensorboard'}")
    print(f"Checkpoints: {checkpoint_dir}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
