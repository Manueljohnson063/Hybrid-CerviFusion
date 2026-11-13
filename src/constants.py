"""Shared constants for CerviFusion."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "archive"
SPLITS_DIR = PROJECT_ROOT / "results" / "splits"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = [
    "Dyskeratotic",
    "Koilocytic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)