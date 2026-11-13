"""
Train CerviFusion hybrid model (Steps 8-13).

EfficientNet (local) + ViT (global) + attention fusion + 5-class classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES, PROJECT_ROOT as ROOT
from src.dataset.dataloaders import create_dataloaders
from src.models.cervifusion import create_cervifusion
from src.training.metrics import collect_predictions, compute_metrics, save_metrics
from src.training.trainer import evaluate, train_one_epoch


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def configure_cpu_threads(num_threads: int | None) -> None:
    if num_threads is None or num_threads < 1:
        return

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        pass


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": CLASS_NAMES,
            "config": config,
        },
        path,
    )


def save_training_plot(history: list[dict], output_path: Path) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Train loss", marker="o")
    axes[0].plot(epochs, val_loss, label="Val loss", marker="o")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train accuracy", marker="o")
    axes[1].plot(epochs, val_acc, label="Val accuracy", marker="o")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CerviFusion hybrid model")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "cervifusion_cpu_light.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = load_config(args.config)

    image_size = int(config["image_size"])
    batch_size = args.batch_size or int(config["batch_size"])
    num_workers = int(config["num_workers"])
    epochs = args.epochs or int(config["epochs"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    pretrained = bool(config["pretrained"])
    efficientnet_pretrained_path = config.get("efficientnet_pretrained_path")
    vit_pretrained_path = config.get("vit_pretrained_path")
    fusion_dim = int(config["fusion_dim"])
    fusion_heads = int(config["fusion_heads"])
    dropout = float(config["dropout"])

    splits_dir = resolve_path(config["splits_dir"])
    checkpoint_dir = resolve_path(config["checkpoint_dir"])
    metrics_dir = resolve_path(config["metrics_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_threads = config.get("cpu_threads")
    if device.type == "cpu":
        configure_cpu_threads(int(cpu_threads) if cpu_threads is not None else None)

    print(f"Device: {device}")
    print(
        "Model: CerviFusion -> "
        f"EN={config['efficientnet_name']}, ViT={config['vit_name']}, "
        f"fusion_dim={fusion_dim}, heads={fusion_heads}"
    )
    print(
        f"Training config -> image_size={image_size}, batch_size={batch_size}, "
        f"epochs={epochs}, lr={learning_rate}"
    )
    if device.type == "cpu" and cpu_threads is not None:
        print(f"CPU threads limited to: {cpu_threads}")

    if efficientnet_pretrained_path or vit_pretrained_path:
        print(
            "Local pretrained weights -> "
            f"EN={efficientnet_pretrained_path}, ViT={vit_pretrained_path}"
        )
    elif pretrained:
        print("Using timm/HF hub pretrained weights for both branches")

    dataloaders = create_dataloaders(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        splits_dir=splits_dir,
    )

    print(
        "Dataset sizes -> "
        f"train={len(dataloaders['train'].dataset)}, "
        f"val={len(dataloaders['val'].dataset)}, "
        f"test={len(dataloaders['test'].dataset)}"
    )

    model = create_cervifusion(
        efficientnet_name=config["efficientnet_name"],
        vit_name=config["vit_name"],
        pretrained=pretrained,
        efficientnet_pretrained_path=efficientnet_pretrained_path,
        vit_pretrained_path=vit_pretrained_path,
        fusion_dim=fusion_dim,
        num_heads=fusion_heads,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_result = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_result = evaluate(model, dataloaders["val"], criterion, device)

        print(
            f"  train loss={train_result.loss:.4f}, acc={train_result.accuracy:.4f} | "
            f"val loss={val_result.loss:.4f}, acc={val_result.accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
            }
        )

        if val_result.accuracy >= best_val_acc:
            best_val_acc = val_result.accuracy
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                best_val_acc,
                config,
            )
            print(f"  Saved new best model (val acc={best_val_acc:.4f})")

    checkpoint_path = checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        print("ERROR: no checkpoint saved.")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for split_name in ["val", "test"]:
        y_true, y_pred, avg_loss = collect_predictions(
            model, dataloaders[split_name], device
        )
        metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
        metrics["loss"] = avg_loss
        metrics["split"] = split_name
        metrics["best_epoch"] = checkpoint["epoch"]
        metrics["best_val_acc"] = checkpoint["best_val_acc"]
        metrics["model"] = "CerviFusion"
        metrics["efficientnet_name"] = config["efficientnet_name"]
        metrics["vit_name"] = config["vit_name"]

        output_path = metrics_dir / f"{split_name}_metrics.json"
        save_metrics(metrics, output_path)
        print(
            f"\n{split_name.upper()} metrics -> "
            f"acc={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
        print(f"Saved metrics to {output_path}")

    history_path = metrics_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    plot_path = metrics_dir / "training_history.png"
    save_training_plot(history, plot_path)
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training plot saved to: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())