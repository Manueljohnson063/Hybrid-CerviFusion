"""Evaluate checkpoints/cervifusion/best_model.pt.

Prints accuracy, precision, recall, specificity, F1, and Cohen's kappa,
and saves a confusion-matrix image.

Example:
  python scripts/09_analyze_model.py
  python scripts/09_analyze_model.py --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES, PROJECT_ROOT as ROOT
from src.dataset.dataloaders import create_dataloaders
from src.models.cervifusion import create_cervifusion
from src.training.metrics import collect_predictions, compute_metrics

CHECKPOINT_PATH = ROOT / "checkpoints" / "cervifusion" / "best_model.pt"

def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def training_config(checkpoint: dict) -> dict:
    saved = checkpoint.get("config")
    if isinstance(saved, dict) and saved:
        return saved

    fallback = ROOT / "configs" / "cervifusion_cpu_light.yaml"
    with fallback.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def build_model(config: dict) -> torch.nn.Module:
    return create_cervifusion(
        efficientnet_name=config.get("efficientnet_name", "efficientnet_b0"),
        vit_name=config.get("vit_name", "vit_small_patch16_224"),
        pretrained=False,
        fusion_dim=int(config.get("fusion_dim", 512)),
        num_heads=int(config.get("fusion_heads", 4)),
        dropout=float(config.get("dropout", 0.3)),
    )

def save_confusion_matrix(matrix: np.ndarray, split_name: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"CerviFusion ({split_name}) confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

def specificity_per_class(matrix: np.ndarray) -> np.ndarray:
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    true_negative = matrix.sum() - true_positive - false_positive - false_negative
    denominator = true_negative + false_positive
    return np.divide(
        true_negative,
        denominator,
        out=np.zeros(len(matrix), dtype=float),
        where=denominator != 0,
    )

def print_metrics(split_name: str, metrics: dict, specificity: np.ndarray, kappa: float) -> None:
    print(f"\n{split_name.upper()}")
    print(f" Accuracy:       {metrics['accuracy']:.4f}")
    print(f" Precision:      {metrics['precision_macro']:.4f}")
    print(f" Recall:         {metrics['recall_macro']:.4f}")
    print(f" Specificity:    {specificity.mean():.4f}")
    print(f" F1:             {metrics['f1_macro']:.4f}")
    print(f" Cohen kappa:    {kappa:.4f}")
    print(f" {'class':<28} {'prec':>8} {'recall':>8} {'spec':>8} {'f1':>8}")
    for index, name in enumerate(CLASS_NAMES):
        row = metrics["classification_report"][name]
        print(
            f" {name:<28} {row['precision']:>8.4f} {row['recall']:>8.4f} "
            f"{specificity[index]:>8.4f} {row['f1-score']:>8.4f}"
        )

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CerviFusion best_model.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "analysis" / "cervifusion" / "confusion_matrix.png",
    )
    args = parser.parse_args()

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    config = training_config(checkpoint)

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    splits_dir = Path(config.get("splits_dir", "results/splits"))
    if not splits_dir.is_absolute():
        splits_dir = ROOT / splits_dir
    dataloaders = create_dataloaders(
        image_size=int(config.get("image_size", 224)),
        batch_size=int(config.get("batch_size", 4)),
        num_workers=int(config.get("num_workers", 0)),
        splits_dir=splits_dir,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    y_true, y_pred, _loss = collect_predictions(model, dataloaders[args.split], device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    matrix = np.asarray(metrics["confusion_matrix"])
    print_metrics(
        args.split,
        metrics,
        specificity_per_class(matrix),
        float(cohen_kappa_score(y_true, y_pred)),
    )

    output_path = args.output
    if args.split != "test":
        output_path = output_path.with_name(f"confusion_matrix_{args.split}.png")
    save_confusion_matrix(matrix, args.split, output_path)
    print(f"\nConfusion matrix saved to {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())