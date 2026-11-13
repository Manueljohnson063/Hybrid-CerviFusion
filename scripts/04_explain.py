"""
Generate Grad-CAM explainability visualizations (Steps 16-17).

Produces figures with:
    - original Pap-smear cell image
    - predicted class and probabilities
    - Grad-CAM heatmap overlay
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES, PROJECT_ROOT as ROOT, SPLITS_DIR
from src.dataset.transforms import build_transforms
from src.explain.gradcam import (
    GradCAM,
    denormalize_image,
    get_efficientnet_target_layer,
    overlay_heatmap,
)
from src.models.cervifusion import create_cervifusion
from src.models.efficientnet_baseline import create_efficientnet_baseline


def load_model(model_type: str, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_type == "cervifusion":
        config = checkpoint.get("config", {})
        model = create_cervifusion(
            efficientnet_name=config.get("efficientnet_name", "efficientnet_b0"),
            vit_name=config.get("vit_name", "vit_small_patch16_224"),
            pretrained=False,
            fusion_dim=int(config.get("fusion_dim", 512)),
            num_heads=int(config.get("fusion_heads", 4)),
            dropout=float(config.get("dropout", 0.3)),
        )
    elif model_type == "efficientnet":
        model = create_efficientnet_baseline(pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def load_image_tensor(image_path: Path, image_size: int, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    transform = build_transforms(image_size=image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor


def save_explanation_figure(
    output_path: Path,
    original_pil: Image.Image,
    overlay: torch.Tensor | None,
    overlay_np,
    heatmap,
    predicted_class: int,
    probabilities: torch.Tensor,
    true_class: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(original_pil)
    axes[0].set_title("Original cell")
    axes[0].axis("off")

    prob_lines = [
        f"{CLASS_NAMES[i]}: {probabilities[i].item():.2f}"
        for i in range(len(CLASS_NAMES))
    ]
    title = f"Predicted: {CLASS_NAMES[predicted_class]}"
    if true_class is not None:
        title += f"\nTrue: {true_class}"
    title += "\n" + "\n".join(prob_lines)

    display_image = denormalize_image(overlay) if overlay is not None else overlay_np
    axes[1].imshow(display_image)
    axes[1].set_title(title, fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay_np)
    axes[2].set_title("Grad-CAM overlay\n(red/yellow = important regions)")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def explain_one_image(
    model,
    grad_cam: GradCAM,
    image_path: Path,
    output_path: Path,
    image_size: int,
    device: torch.device,
    true_class: str | None = None,
    target_class: int | None = None,
) -> dict:
    original_pil, tensor = load_image_tensor(image_path, image_size, device)
    heatmap, explained_class, probabilities = grad_cam.generate(tensor, target_class)
    predicted_class = int(torch.argmax(probabilities).item())

    display_image = denormalize_image(tensor[0])
    overlay_np = overlay_heatmap(display_image, heatmap)

    save_explanation_figure(
        output_path=output_path,
        original_pil=original_pil,
        overlay=tensor[0],
        overlay_np=overlay_np,
        heatmap=heatmap,
        predicted_class=predicted_class,
        probabilities=probabilities,
        true_class=true_class,
    )

    return {
        "image": str(image_path),
        "predicted_class": CLASS_NAMES[predicted_class],
        "explained_class": CLASS_NAMES[explained_class],
        "probabilities": {CLASS_NAMES[i]: probabilities[i].item() for i in range(len(CLASS_NAMES))},
        "true_class": true_class,
        "output": str(output_path),
    }


def load_test_samples(limit: int) -> list[dict]:
    test_csv = SPLITS_DIR / "test.csv"
    rows: list[dict] = []
    with test_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    random.shuffle(rows)
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations")
    parser.add_argument(
        "--model-type",
        choices=["cervifusion", "efficientnet"],
        default="cervifusion",
        help="Which trained model to explain",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single BMP image path to explain",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random test images when --image is not provided",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "explainability",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.model_type == "cervifusion":
            args.checkpoint = ROOT / "checkpoints" / "cervifusion" / "best_model.pt"
        else:
            args.checkpoint = ROOT / "checkpoints" / "efficientnet_baseline" / "best_model.pt"

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.model_type} from {args.checkpoint}")

    model, _checkpoint = load_model(args.model_type, args.checkpoint, device)
    target_layer = get_efficientnet_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    results: list[dict] = []

    try:
        if args.image is not None:
            image_path = args.image if args.image.is_absolute() else ROOT / args.image
            output_path = args.output_dir / f"{image_path.stem}_gradcam.png"
            result = explain_one_image(
                model,
                grad_cam,
                image_path,
                output_path,
                args.image_size,
                device,
            )
            results.append(result)
        else:
            samples = load_test_samples(args.num_samples)
            for index, row in enumerate(samples, start=1):
                image_path = ROOT / row["path"]
                true_class = row["class_name"]
                output_path = args.output_dir / f"test_{index:02d}_{image_path.stem}_gradcam.png"
                result = explain_one_image(
                    model,
                    grad_cam,
                    image_path,
                    output_path,
                    args.image_size,
                    device,
                    true_class=true_class,
                )
                results.append(result)
                print(
                    f"[{index}] {image_path.name} -> "
                    f"pred={result['predicted_class']}, true={true_class} -> {output_path}"
                )
    finally:
        grad_cam.close()

    print("\nGrad-CAM generation complete.")
    print(f"Saved {len(results)} figure(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())