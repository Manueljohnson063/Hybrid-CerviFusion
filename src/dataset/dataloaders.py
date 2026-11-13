"""
Dataloader factory for train/val/test splits.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.constants import SPLITS_DIR
from src.dataset.sipakmed_dataset import SipakmedDataset
from src.dataset.transforms import build_transforms


def create_dataloaders(
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    splits_dir: Path | None = None,
) -> dict[str, DataLoader]:
    splits_dir = splits_dir or SPLITS_DIR

    train_dataset = SipakmedDataset(
        splits_dir / "train.csv",
        transform=build_transforms(image_size=image_size, train=True),
    )
    val_dataset = SipakmedDataset(
        splits_dir / "val.csv",
        transform=build_transforms(image_size=image_size, train=False),
    )
    test_dataset = SipakmedDataset(
        splits_dir / "test.csv",
        transform=build_transforms(image_size=image_size, train=False),
    )

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return {
        "train": DataLoader(
            train_dataset,
            shuffle=True,
            **common_loader_args,
        ),
        "val": DataLoader(
            val_dataset,
            shuffle=False,
            **common_loader_args,
        ),
        "test": DataLoader(
            test_dataset,
            shuffle=False,
            **common_loader_args,
        ),
    }