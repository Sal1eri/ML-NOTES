from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vision.data import CIFAR10_CLASSES, make_dataloaders
from vision.models import build_model

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Visualize trained CIFAR-10 embeddings with t-SNE.")
    parser.add_argument("--models", nargs="+", choices=["cnn", "vit"], default=["cnn", "vit"])
    parser.add_argument("--cnn-checkpoint", default="checkpoints/vision/cnn-cifar10-20260706-132602/best.pt")
    parser.add_argument("--vit-checkpoint", default="checkpoints/vision/vit-cifar10-20260706-132622/best.pt")
    parser.add_argument("--dataset-name", default="uoft-cs/cifar10")
    parser.add_argument("--cache-dir", default=os.getenv("CACHE_DIR", "./cache"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "./outputs"))
    parser.add_argument("--log-dir", default=os.getenv("LOG_DIR", "./logs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--device", default=os.getenv("DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--use-tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "vision-cifar10"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_name:
        args.run_name = f"tsne-cifar10-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.cache_dir = str(Path(args.cache_dir) / "huggingface")
    output_dir = Path(args.output_dir) / "vision" / args.run_name
    log_dir = Path(args.log_dir) / "vision" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    loaders = make_dataloaders(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        validation_size=0.1,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_train=1,
        limit_val=args.num_samples if args.split == "validation" else 1,
        limit_test=args.num_samples if args.split == "test" else 1,
    )
    loader = loaders.test_loader if args.split == "test" else loaders.val_loader
    if loader is None:
        raise ValueError("Requested test split, but the dataset did not provide one.")

    tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard")) if args.use_tensorboard else None
    wandb_run = init_wandb(args, log_dir) if args.use_wandb else None

    for model_name in args.models:
        checkpoint_path = checkpoint_for_model(args, model_name)
        model = load_model(model_name, checkpoint_path, device)
        features, labels = extract_features(model, loader, device, args.num_samples)
        coords = compute_tsne(features, args)

        npz_path = output_dir / f"{model_name}_tsne_embeddings.npz"
        np.savez_compressed(npz_path, features=features, tsne=coords, labels=labels, class_names=np.array(CIFAR10_CLASSES))

        fig = plot_tsne(coords, labels, model_name, checkpoint_path, args.split)
        png_path = output_dir / f"{model_name}_tsne.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")

        if tb_writer is not None:
            tb_writer.add_figure(f"tsne/{model_name}", fig, global_step=0)
            tb_writer.add_text(f"checkpoint/{model_name}", str(checkpoint_path), global_step=0)
        if wandb_run is not None:
            wandb.log({f"tsne/{model_name}": wandb.Image(fig)})
            artifact = wandb.Artifact(f"{args.run_name}-{model_name}-tsne", type="embedding")
            artifact.add_file(str(npz_path))
            artifact.add_file(str(png_path))
            artifact.metadata = {
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "split": args.split,
                "num_samples": int(len(labels)),
                "perplexity": args.perplexity,
                "pca_dim": args.pca_dim,
            }
            wandb_run.log_artifact(artifact)

        plt.close(fig)
        print(f"{model_name}: saved {png_path}")
        print(f"{model_name}: saved {npz_path}")

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    print(f"TensorBoard: tensorboard --logdir {log_dir / 'tensorboard'}")
    print(f"Outputs: {output_dir}")


def checkpoint_for_model(args: argparse.Namespace, model_name: str) -> Path:
    path = Path(args.cnn_checkpoint if model_name == "cnn" else args.vit_checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {path}")
    return path


def load_model(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name, num_classes=len(CIFAR10_CLASSES)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader, device: torch.device, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    seen = 0
    for images, labels in tqdm(loader, desc="extract embeddings", leave=False):
        images = images.to(device, non_blocking=True)
        features = model.forward_features(images).detach().cpu()
        all_features.append(features)
        all_labels.append(labels.detach().cpu())
        seen += images.shape[0]
        if seen >= num_samples:
            break
    features = torch.cat(all_features, dim=0)[:num_samples].numpy()
    labels = torch.cat(all_labels, dim=0)[:num_samples].numpy()
    return features, labels


def compute_tsne(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    if args.pca_dim > 0 and scaled.shape[1] > args.pca_dim:
        scaled = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(scaled)
    perplexity = min(args.perplexity, max(5.0, (len(scaled) - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    return tsne.fit_transform(scaled)


def plot_tsne(coords: np.ndarray, labels: np.ndarray, model_name: str, checkpoint_path: Path, split: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.72,
            color=cmap(class_id),
            label=class_name,
            linewidths=0,
        )
    ax.set_title(f"{model_name.upper()} CIFAR-10 {split} embeddings before classifier")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.text(
        0.0,
        -0.12,
        f"checkpoint: {checkpoint_path}",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    return fig


def init_wandb(args: argparse.Namespace, log_dir: Path):
    if wandb is None:
        raise ImportError("wandb is not installed")
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.run_name,
        mode=args.wandb_mode,
        dir=str(log_dir),
        config=vars(args),
        tags=["cifar10", "tsne", "embedding"],
    )


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
