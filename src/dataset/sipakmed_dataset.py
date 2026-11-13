"""PyTorch Dataset for SIPakMed cropped images loaded from split CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.constants import PROJECT_ROOT


class SipakmedDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None) -> None:
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []

        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = PROJECT_ROOT / row["path"]
                class_idx = int(row["class_idx"])
                class_name = row["class_name"]
                self.samples.append((image_path, class_idx, class_name))

        if not self.samples:
            raise ValueError(f"No samples found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_idx, class_name = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_idx, str(image_path), class_name