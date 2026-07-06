from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    class_names: list[str]


class HFCifarDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[index]
        image = _to_pil(row["img"])
        label = int(row["label"])
        return self.transform(image), label


def _to_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path") is not None:
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def load_cifar10_dataset(dataset_name: str, cache_dir: str | None, validation_size: float, seed: int) -> DatasetDict:
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    if "train" not in dataset:
        raise ValueError(f"{dataset_name} did not provide a train split")
    split = dataset["train"].train_test_split(test_size=validation_size, seed=seed, stratify_by_column="label")
    result = DatasetDict(train=split["train"], validation=split["test"])
    if "test" in dataset:
        result["test"] = dataset["test"]
    return result


def maybe_select(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def make_dataloaders(
    dataset_name: str,
    cache_dir: str | None,
    validation_size: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    limit_train: int | None,
    limit_val: int | None,
    limit_test: int | None,
) -> DataBundle:
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_cifar10_dataset(dataset_name, cache_dir, validation_size, seed)
    train_ds = HFCifarDataset(maybe_select(dataset["train"], limit_train), build_transforms(train=True))
    val_ds = HFCifarDataset(maybe_select(dataset["validation"], limit_val), build_transforms(train=False))
    test_loader = None
    if "test" in dataset:
        test_ds = HFCifarDataset(maybe_select(dataset["test"], limit_test), build_transforms(train=False))
        test_loader = _loader(test_ds, batch_size, num_workers, shuffle=False)
    return DataBundle(
        train_loader=_loader(train_ds, batch_size, num_workers, shuffle=True),
        val_loader=_loader(val_ds, batch_size, num_workers, shuffle=False),
        test_loader=test_loader,
        class_names=CIFAR10_CLASSES,
    )


def _loader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def take_batch(loader: DataLoader, max_images: int) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = next(iter(loader))
    return images[:max_images], labels[:max_images]

