from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vision.data import CIFAR10_CLASSES, denormalize, make_dataloaders
from vision.models import build_model


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Find nearest CIFAR-10 images in model embedding space.")
    parser.add_argument("--model", choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--checkpoint", default="")
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
    parser.add_argument("--num-query-classes", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names or ids. Example: --classes airplane cat ship",
    )
    parser.add_argument("--use-tensorboard", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_name:
        args.run_name = f"{args.model}-neighbors-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.cache_dir = str(Path(args.cache_dir) / "huggingface")

    output_dir = Path(args.output_dir) / "vision" / args.run_name
    log_dir = Path(args.log_dir) / "vision" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.model)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    loader = make_dataloaders(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        validation_size=0.1,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_train=1,
        limit_val=1,
        limit_test=args.num_samples,
    ).test_loader
    if loader is None:
        raise ValueError("CIFAR-10 test loader is unavailable.")

    model = load_model(args.model, checkpoint_path, device)
    embeddings, labels, images = extract_embedding_space(model, loader, device, args.num_samples)
    selected_classes = choose_classes(labels, args)
    rows = find_neighbors(embeddings, labels, selected_classes, args.neighbors, args.seed)

    fig = plot_neighbor_grid(rows, images, labels, args.model, checkpoint_path)
    png_path = output_dir / f"{args.model}_nearest_neighbors.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")

    npz_path = output_dir / f"{args.model}_nearest_neighbors.npz"
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        labels=labels,
        selected_classes=np.array(selected_classes),
        rows=np.array([[item["query"], *item["neighbors"]] for item in rows], dtype=np.int64),
        similarities=np.array([item["similarities"] for item in rows], dtype=np.float32),
        class_names=np.array(CIFAR10_CLASSES),
    )

    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        writer.add_figure(f"neighbors/{args.model}", fig, global_step=0)
        writer.add_text("checkpoint", str(checkpoint_path), global_step=0)
        writer.flush()
        writer.close()

    plt.close(fig)
    print(f"saved image: {png_path}")
    print(f"saved data: {npz_path}")
    print(f"TensorBoard: tensorboard --logdir {log_dir / 'tensorboard'}")


def default_checkpoint(model_name: str) -> Path:
    if model_name == "cnn":
        return Path("checkpoints/vision/cnn-cifar10-20260706-132602/best.pt")
    return Path("checkpoints/vision/vit-cifar10-20260706-132622/best.pt")


def load_model(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name, num_classes=len(CIFAR10_CLASSES)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def extract_embedding_space(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_images: list[torch.Tensor] = []
    seen = 0
    for images, labels in tqdm(loader, desc="extract embedding space", leave=False):
        images = images.to(device, non_blocking=True)
        embeddings = model.forward_features(images).detach().cpu()
        all_embeddings.append(embeddings)
        all_labels.append(labels.detach().cpu())
        all_images.append(denormalize(images.detach().cpu()))
        seen += images.shape[0]
        if seen >= num_samples:
            break

    embeddings = torch.cat(all_embeddings, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]
    images = torch.cat(all_images, dim=0)[:num_samples]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy(), labels.numpy(), images.numpy()


def choose_classes(labels: np.ndarray, args: argparse.Namespace) -> list[int]:
    present = sorted(set(int(label) for label in labels.tolist()))
    if args.classes:
        return [parse_class_id(item) for item in args.classes]
    rng = random.Random(args.seed)
    count = min(args.num_query_classes, len(present))
    return rng.sample(present, count)


def parse_class_id(value: str) -> int:
    if value.isdigit():
        class_id = int(value)
    else:
        class_id = CIFAR10_CLASSES.index(value)
    if class_id < 0 or class_id >= len(CIFAR10_CLASSES):
        raise ValueError(f"Invalid class id: {value}")
    return class_id


def find_neighbors(
    embeddings: np.ndarray,
    labels: np.ndarray,
    selected_classes: list[int],
    k: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    similarities = embeddings @ embeddings.T
    rows = []
    for class_id in selected_classes:
        candidates = np.flatnonzero(labels == class_id)
        query_idx = int(rng.choice(candidates))
        scores = similarities[query_idx].copy()
        scores[query_idx] = -np.inf
        nearest = np.argsort(-scores)[:k]
        rows.append(
            {
                "class_id": class_id,
                "query": query_idx,
                "neighbors": nearest.astype(np.int64).tolist(),
                "similarities": scores[nearest].astype(np.float32).tolist(),
            }
        )
    return rows


def plot_neighbor_grid(rows: list[dict], images: np.ndarray, labels: np.ndarray, model_name: str, checkpoint_path: Path):
    columns = 1 + len(rows[0]["neighbors"])
    fig, axes = plt.subplots(len(rows), columns, figsize=(columns * 2.0, len(rows) * 2.2))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(rows):
        indices = [row["query"], *row["neighbors"]]
        similarities = [None, *row["similarities"]]
        for col_idx, (sample_idx, sim) in enumerate(zip(indices, similarities)):
            ax = axes[row_idx, col_idx]
            image = np.transpose(images[sample_idx], (1, 2, 0))
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            class_name = CIFAR10_CLASSES[int(labels[sample_idx])]
            if col_idx == 0:
                title = f"query\n{class_name}"
                ax.set_ylabel(CIFAR10_CLASSES[row["class_id"]], rotation=0, labelpad=34, va="center")
            else:
                title = f"top {col_idx}\n{class_name}\ncos={sim:.3f}"
            ax.set_title(title, fontsize=9)

    fig.suptitle(f"{model_name.upper()} nearest images in embedding space", y=1.02, fontsize=14)
    fig.text(0.01, 0.01, f"checkpoint: {checkpoint_path}", fontsize=8, color="#444444")
    fig.tight_layout()
    return fig


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()

