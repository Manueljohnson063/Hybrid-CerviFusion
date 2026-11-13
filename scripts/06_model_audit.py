"""Audit CerviFusion predictions and branch reliance on held-out test images.

This script does not modify training or Grad-CAM scripts. It loads the existing
checkpoint-compatible model definitions from 05_grad_cam.py and writes results
under results/model_audit/.

Example:
    python scripts/06_model_audit.py --samples-per-class 5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAD_CAM_SCRIPT = PROJECT_ROOT / "scripts" / "05_grad_cam.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "cervifusion" / "best_model.pt"
DEFAULT_TEST_CSV = PROJECT_ROOT / "results" / "splits" / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "model_audit"


def load_grad_cam_module():
    """Load the standalone model and Grad-CAM implementation without editing it."""
    spec = importlib.util.spec_from_file_location("cervifusion_grad_cam", GRAD_CAM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load: {GRAD_CAM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_balanced_rows(
    test_csv: Path,
    samples_per_class: int,
    seed: int,
) -> list[dict[str, str]]:
    with test_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_class[row["class_name"]].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for class_name in sorted(rows_by_class):
        candidates = rows_by_class[class_name]
        count = min(samples_per_class, len(candidates))
        selected.extend(rng.sample(candidates, count))
    return selected


def confidence_for_class(model, tensor: torch.Tensor, class_idx: int) -> float:
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    return float(probabilities[class_idx].item())


def mask_from_heatmap(heatmap: np.ndarray, image_size: int, threshold: float) -> torch.Tensor:
    binary_heatmap = torch.from_numpy((heatmap >= threshold).astype(np.float32))
    return F.interpolate(
        binary_heatmap.view(1, 1, *binary_heatmap.shape),
        size=(image_size, image_size),
        mode="nearest",
    )


def confidence_with_branch_disabled(
    model,
    tensor: torch.Tensor,
    class_idx: int,
    branch_name: str,
) -> float:
    branch = getattr(model, branch_name)
    hook = branch.register_forward_hook(
        lambda _module, _inputs, output: torch.zeros_like(output)
    )
    try:
        return confidence_for_class(model, tensor, class_idx)
    finally:
        hook.remove()


def build_summary(
    records: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "per_class": {}}
    total = len(records)
    correct = sum(record["correct"] for record in records)
    summary["overall"] = {
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "mean_prediction_confidence": float(
            np.mean([record["prediction_confidence"] for record in records])
        ) if records else 0.0,
        "mean_cam_confidence_drop": float(
            np.mean([record["cam_confidence_drop"] for record in records])
        ) if records else 0.0,
        "mean_random_confidence_drop": float(
            np.mean([record["random_confidence_drop"] for record in records])
        ) if records else 0.0,
        "mean_efficientnet_ablation_drop": float(
            np.mean([record["efficientnet_ablation_drop"] for record in records])
        ) if records else 0.0,
        "mean_vit_ablation_drop": float(
            np.mean([record["vit_ablation_drop"] for record in records])
        ) if records else 0.0,
    }

    for class_name in class_names:
        class_records = [record for record in records if record["true_class"] == class_name]
        class_correct = sum(record["correct"] for record in class_records)
        summary["per_class"][class_name] = {
            "samples": len(class_records),
            "accuracy": class_correct / len(class_records) if class_records else 0.0,
            "mean_prediction_confidence": float(
                np.mean([record["prediction_confidence"] for record in class_records])
            ) if class_records else 0.0,
            "mean_cam_confidence_drop": float(
                np.mean([record["cam_confidence_drop"] for record in class_records])
            ) if class_records else 0.0,
            "mean_random_confidence_drop": float(
                np.mean([record["random_confidence_drop"] for record in class_records])
            ) if class_records else 0.0,
        }
    return summary


def save_confusion_matrix(
    records: list[dict[str, Any]],
    class_names: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        [record["true_class"] for record in records],
        [record["predicted_class"] for record in records],
        labels=class_names,
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Held-out Test Set Audit",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_idx, col_idx in np.ndindex(matrix.shape):
        axis.text(
            col_idx,
            row_idx,
            str(matrix[row_idx, col_idx]),
            ha="center",
            va="center",
            color="white" if matrix[row_idx, col_idx] > threshold else "black",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CerviFusion on a balanced sample of held-out test images."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cam-threshold", type=float, default=0.70)
    args = parser.parse_args()

    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1")
    if not 0.0 < args.cam_threshold < 1.0:
        raise ValueError("--cam-threshold must be between 0 and 1")

    checkpoint_path = resolve_path(str(args.checkpoint))
    test_csv = resolve_path(str(args.test_csv))
    output_dir = resolve_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    grad_cam_module = load_grad_cam_module()
    class_names = grad_cam_module.CLASS_NAMES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, image_size = grad_cam_module.load_checkpoint(checkpoint_path, device)
    selected_rows = select_balanced_rows(test_csv, args.samples_per_class, args.seed)
    grad_cam = grad_cam_module.GradCAM(
        model,
        grad_cam_module.get_efficientnet_target_layer(model),
    )

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"Auditing {len(selected_rows)} held-out images from {test_csv}")
    records: list[dict[str, Any]] = []

    try:
        for index, row in enumerate(selected_rows, start=1):
            image_path = resolve_path(row["path"])
            if not image_path.exists():
                print(f"[{index}/{len(selected_rows)}] Missing: {image_path}")
                continue

            _, tensor = grad_cam_module.load_image_tensor(image_path, image_size, device)
            heatmap, explained_idx, probabilities = grad_cam.generate(tensor)
            predicted_idx = int(probabilities.argmax())
            predicted_confidence = float(probabilities[predicted_idx].item())
            cam_mask = mask_from_heatmap(heatmap, image_size, args.cam_threshold).to(device)
            random_mask = torch.roll(
                cam_mask,
                shifts=(image_size // 2, image_size // 3),
                dims=(2, 3),
            )
            cam_masked_confidence = confidence_for_class(
                model, tensor * (1 - cam_mask), predicted_idx
            )
            random_masked_confidence = confidence_for_class(
                model, tensor * (1 - random_mask), predicted_idx
            )
            efficientnet_disabled_confidence = confidence_with_branch_disabled(
                model, tensor, predicted_idx, "en_branch"
            )
            vit_disabled_confidence = confidence_with_branch_disabled(
                model, tensor, predicted_idx, "vit_branch"
            )

            record = {
                "path": str(image_path),
                "true_class": row["class_name"],
                "true_class_idx": int(row["class_idx"]),
                "predicted_class": class_names[predicted_idx],
                "predicted_class_idx": predicted_idx,
                "correct": predicted_idx == int(row["class_idx"]),
                "prediction_confidence": predicted_confidence,
                "explained_class": class_names[explained_idx],
                "cam_masked_confidence": cam_masked_confidence,
                "random_masked_confidence": random_masked_confidence,
                "cam_confidence_drop": predicted_confidence - cam_masked_confidence,
                "random_confidence_drop": predicted_confidence - random_masked_confidence,
                "efficientnet_disabled_confidence": efficientnet_disabled_confidence,
                "vit_disabled_confidence": vit_disabled_confidence,
                "efficientnet_ablation_drop": predicted_confidence - efficientnet_disabled_confidence,
                "vit_ablation_drop": predicted_confidence - vit_disabled_confidence,
            }
            records.append(record)
            print(
                f"[{index}/{len(selected_rows)}] {row['class_name']} -> "
                f"{record['predicted_class']} ({predicted_confidence:.3f}) | "
                f"CAM drop={record['cam_confidence_drop']:.4f}, "
                f"EffNet drop={record['efficientnet_ablation_drop']:.4f}, "
                f"ViT drop={record['vit_ablation_drop']:.4f}"
            )
    finally:
        grad_cam.close()

    if not records:
        raise RuntimeError("No audit records were created.")

    csv_path = output_dir / "per_image_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = build_summary(records, class_names)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "test_csv": str(test_csv),
            "samples_per_class_requested": args.samples_per_class,
            "cam_threshold": args.cam_threshold,
            "seed": args.seed,
        }
    )
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    confusion_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(records, class_names, confusion_path)

    print(f"\nPer-image results: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
