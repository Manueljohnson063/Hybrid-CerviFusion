
"""
Grad-CAM for one image.

Example:
    python scripts/08_gradcam_image.py --image "archive/im_Dyskeratotic/your_image.bmp"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# ------------------------------------------------------------
# Project paths (script lives in scripts/)
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "cervifusion"
    / "best_model.pt"
)

CLASS_NAMES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


# ------------------------------------------------------------
# Model (notebook-style, no src/)
# ------------------------------------------------------------

class EfficientNetBranch(nn.Module):

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTBranch(nn.Module):

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )

        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AttentionFusion(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.query = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        local_features,
        global_features,
    ):

        branch_tokens = torch.stack(
            [local_features, global_features],
            dim=1,
        )

        query = self.query.expand(
            branch_tokens.size(0),
            -1,
            -1,
        )

        fused, _ = self.attention(
            query,
            branch_tokens,
            branch_tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        fused = self.dropout(
            self.norm(fused + query)
        )

        return fused.squeeze(1)


class CerviFusion(nn.Module):

    def __init__(
        self,
        efficientnet_name: str = "efficientnet_b0",
        vit_name: str = "vit_small_patch16_224",
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.en_branch = EfficientNetBranch(
            efficientnet_name
        )

        self.vit_branch = ViTBranch(
            vit_name
        )

        self.en_projection = nn.Sequential(
            nn.Linear(
                self.en_branch.feature_dim,
                fusion_dim,
            ),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

        self.vit_projection = nn.Sequential(
            nn.Linear(
                self.vit_branch.feature_dim,
                fusion_dim,
            ),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

        self.fusion = AttentionFusion(
            fusion_dim,
            num_heads,
            dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        local_proj = self.en_projection(
            self.en_branch(x)
        )

        global_proj = self.vit_projection(
            self.vit_branch(x)
        )

        fused = self.fusion(
            local_proj,
            global_proj,
        )

        return self.classifier(fused)


def build_cervifusion_from_config(
    config: dict,
) -> CerviFusion:

    return CerviFusion(
        efficientnet_name=config.get(
            "efficientnet_name",
            "efficientnet_b0",
        ),
        vit_name=config.get(
            "vit_name",
            "vit_small_patch16_224",
        ),
        fusion_dim=int(
            config.get("fusion_dim", 512)
        ),
        num_heads=int(
            config.get("num_heads", 4)
        ),
        dropout=float(
            config.get("dropout", 0.3)
        ),
    )


# ------------------------------------------------------------
# Grad-CAM helpers
# ------------------------------------------------------------

class GradCAM:

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):

        self.model = model
        self.activations = None
        self.gradients = None

        self._fh = target_layer.register_forward_hook(
            lambda _m, _i, o: setattr(
                self,
                "activations",
                o.detach(),
            )
        )

        self._bh = target_layer.register_full_backward_hook(
            lambda _m, _gi, go: setattr(
                self,
                "gradients",
                go[0].detach(),
            )
        )

    def close(self):

        self._fh.remove()
        self._bh.remove()

    def generate(
        self,
        input_tensor,
        target_class=None,
    ):

        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        logits = self.model(input_tensor)

        probs = torch.softmax(
            logits,
            dim=1,
        )[0]

        if target_class is None:
            target_class = int(probs.argmax())

        logits[0, target_class].backward()

        weights = self.gradients[0].mean(
            dim=(1, 2),
            keepdim=True,
        )

        cam = F.relu(
            (
                weights
                * self.activations[0]
            ).sum(dim=0)
        )

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return (
            cam.cpu().numpy(),
            target_class,
            probs.detach().cpu(),
        )


def get_efficientnet_target_layer(
    model: CerviFusion,
) -> nn.Module:

    backbone = model.en_branch.backbone

    if hasattr(backbone, "conv_head"):
        return backbone.conv_head

    return backbone.blocks[-1]


def denormalize_image(
    tensor: torch.Tensor,
) -> np.ndarray:

    mean = torch.tensor(
        IMAGENET_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD
    ).view(3, 1, 1)

    img = (
        tensor.cpu().clone() * std + mean
    ).clamp(0, 1).permute(1, 2, 0).numpy()

    return (img * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:

    import matplotlib.cm as cm

    heatmap_resized = np.array(
        Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize(
            (image.shape[1], image.shape[0]),
            Image.BILINEAR,
        )
    ).astype(np.float32) / 255.0

    colored = cm.jet(heatmap_resized)[..., :3]

    colored = (colored * 255).astype(np.uint8)

    return (
        alpha * colored
        + (1 - alpha) * image
    ).astype(np.uint8)


def load_image_tensor(
    image_path: Path,
    image_size: int,
    device: torch.device,
):

    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    return image, tensor


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train first: python scripts/05_train_cervifusion.py"
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = ckpt.get("config", {})

    model = build_cervifusion_from_config(
        config
    )

    model.load_state_dict(
        dict(ckpt["model_state_dict"])
    )

    model.to(device).eval()

    image_size = int(
        config.get("image_size", 224)
    )

    return model, ckpt, image_size


def save_and_show_figure(
    output_path: Path | None,
    original_pil: Image.Image,
    overlay_np: np.ndarray,
    predicted_class: int,
    probabilities: torch.Tensor,
    show: bool,
) -> None:

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
    )

    axes[0].imshow(original_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    prob_lines = "\n".join(
        f"{CLASS_NAMES[i]}: "
        f"{probabilities[i].item():.3f}"
        for i in range(NUM_CLASSES)
    )

    axes[1].imshow(overlay_np)

    axes[1].set_title(
        f"Grad-CAM \n"
        f"Pred: {CLASS_NAMES[predicted_class]}\n"
        f"{prob_lines}",
        fontsize=9,
    )

    axes[1].axis("off")

    axes[2].bar(
        CLASS_NAMES,
        [
            probabilities[i].item()
            for i in range(NUM_CLASSES)
        ],
        color="steelblue",
    )

    axes[2].set_ylim(0, 1)
    axes[2].set_title("Probabilities")

    axes[2].tick_params(
        axis="x",
        rotation=35,
    )

    plt.tight_layout()

    if output_path is not None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description="Grad-CAM for one image (no src/ deps)"
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image path",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="best_model.pt path",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Class index 0-4",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot window",
    )

    args = parser.parse_args()

    image_path = resolve_path(args.image)

    if not image_path.exists():

        print(
            f"ERROR: image not found: {image_path}"
        )

        return 1

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model, ckpt, image_size = load_checkpoint(
        checkpoint_path,
        device,
    )

    image_size = args.image_size or image_size

    print(
        f"Loaded epoch {ckpt.get('epoch', '?')} | "
        f"val acc {ckpt.get('best_val_acc', 0):.4f} | "
        f"size={image_size}"
    )

    original_pil, tensor = load_image_tensor(
        image_path,
        image_size,
        device,
    )

    grad_cam = GradCAM(
        model,
        get_efficientnet_target_layer(model),
    )

    try:

        (
            heatmap,
            explained_class,
            probabilities,
        ) = grad_cam.generate(
            tensor,
            args.target_class,
        )

        pred = int(
            probabilities.argmax()
        )

        overlay_np = overlay_heatmap(
            denormalize_image(tensor[0]),
            heatmap,
        )

        print(f"Image: {image_path}")

        print(
            f"Prediction: {CLASS_NAMES[pred]} "
            f"({probabilities[pred].item():.4f})"
        )

        out = args.output

        if out is None:

            out = (
                PROJECT_ROOT
                / "results"
                / "explainability"
                / f"{image_path.stem}_gradcam.png"
            )

        else:

            out = resolve_path(out)

        save_and_show_figure(
            out,
            original_pil,
            overlay_np,
            pred,
            probabilities,
            args.show,
        )

    finally:

        grad_cam.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())