"""
Grad-CAM for one cervical cell image - standalone (no src/ imports).

Checkpoint (same as training):
    checkpoints/cervifusion/best_model.pt

Example:
    python scripts/08_gradcam_image.py --image "archive/im_Dyskeratotic/im_Dyskeratotic/CROPPED/007_01.bmp" --show
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

# --------------------------------------------------------------------------
# Project paths (script lives in scripts/)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "cervifusion" / "best_model.pt"

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

output_path = PROJECT_ROOT / "results" / "explainability" / "gradcam.png"
class LetterboxResize:
    def __init__(self, size: int) -> None:
        self.size = size
        self.fill = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------
# Model (notebook-style, no src/)
# --------------------------------------------------------------------------
class EfficientNetBranch(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTBranch(nn.Module):
    def __init__(self, model_name: str = "vit_small_patch16_224"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool="token"
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, local_features, global_features):
        branch_tokens = torch.stack([local_features, global_features], dim=1)
        query = self.query.expand(branch_tokens.size(0), -1, -1)
        fused, _ = self.attention(
            query, branch_tokens, branch_tokens,
            need_weights=True, average_attn_weights=True,
        )
        fused = self.dropout(self.norm(fused + query))
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
        self.en_branch = EfficientNetBranch(efficientnet_name)
        self.vit_branch = ViTBranch(vit_name)
        self.en_projection = nn.Sequential(
            nn.Linear(self.en_branch.feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.vit_projection = nn.Sequential(
            nn.Linear(self.vit_branch.feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.fusion = AttentionFusion(fusion_dim, num_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_proj = self.en_projection(self.en_branch(x))
        global_proj = self.vit_projection(self.vit_branch(x))
        fused = self.fusion(local_proj, global_proj)
        return self.classifier(fused)


def build_cervifusion_from_config(config: dict) -> CerviFusion:
    return CerviFusion(
        efficientnet_name=config.get("efficientnet_name", "efficientnet_b0"),
        vit_name=config.get("vit_name", "vit_small_patch16_224"),
        fusion_dim=int(config.get("fusion_dim", 512)),
        num_heads=int(config.get("fusion_heads", 4)),
        dropout=float(config.get("dropout", 0.3)),
    )


# --------------------------------------------------------------------------
# Grad-CAM helpers
# --------------------------------------------------------------------------
def forward_cervifusion_en_only(model: CerviFusion, x: torch.Tensor) -> torch.Tensor:
    local_proj = model.en_projection(model.en_branch(x))
    global_proj = model.vit_projection(model.vit_branch(x)).detach()
    fused = model.fusion(local_proj, global_proj)
    return model.classifier(fused)


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        *,
        method: str = "gradcam",
        grad_source: str = "full",
    ):
        self.model = model
        self.method = method
        self.grad_source = grad_source
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        self._bh = target_layer.register_full_backward_hook(
            lambda m, _gi, go: setattr(self, "gradients", go[0].detach())
        )

    def close(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        with torch.no_grad():
            full_logits = self.model(input_tensor)
            probs = torch.softmax(full_logits, dim=1)[0]
        if target_class is None:
            target_class = int(probs.argmax())
        if self.grad_source == "en-only":
            logits = forward_cervifusion_en_only(self.model, input_tensor)
        else:
            logits = self.model(input_tensor)
        logits[0, target_class].backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        if self.method == "gradcam++":
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            denom = 2 * grad_2 + (activations * grad_3).sum(dim=(1, 2), keepdim=True)
            alpha = grad_2 / (denom + 1e-8)
            weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        else:
            weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), target_class, probs.detach().cpu()


def get_efficientnet_target_layer(model: CerviFusion, layer: str = "head") -> nn.Module:
    backbone = model.en_branch.backbone
    blocks = getattr(backbone, "blocks", None)
    if layer == "high" and blocks is not None and len(blocks) >= 3:
        return blocks[-3]
    if layer == "last" and blocks is not None:
        return blocks[-1]
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    return backbone.blocks[-1]


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu().clone() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    h, w = image.shape[0], image.shape[1]
    tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        tensor, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    colored = cm.jet(heatmap_resized)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * image).astype(np.uint8)


def load_image_tensor(
    image_path: Path,
    image_size: int,
    device: torch.device,
    resize_mode: str = "stretch",
):
    resize = (
        LetterboxResize(image_size)
        if resize_mode == "letterbox"
        else transforms.Resize((image_size, image_size))
    )
    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train first: python scripts/05_train_cervifusion.py"
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model = build_cervifusion_from_config(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    image_size = int(config.get("image_size", 224))
    return model, ckpt, image_size


def save_and_show_figure(
    output_path: Path | None,
    model_input_np: np.ndarray,
    overlay_np: np.ndarray,
    predicted_class: int,
    probabilities: torch.Tensor,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    h, w = model_input_np.shape[0], model_input_np.shape[1]
    axes[0].imshow(model_input_np)
    axes[0].set_title(f"Model input ({h}x{w})")
    axes[0].axis("off")

    prob_lines = "\n".join(
        f"{CLASS_NAMES[i]}: {probabilities[i].item():.3f}" for i in range(NUM_CLASSES)
    )
    axes[1].imshow(overlay_np)
    axes[1].set_title(
        f"Grad-CAM \nPred: {CLASS_NAMES[predicted_class]}\n{prob_lines}",
        fontsize=9,
    )
    axes[1].axis("off")

    axes[2].bar(
        CLASS_NAMES,
        [probabilities[i].item() for i in range(NUM_CLASSES)],
        color="steelblue",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Probabilities")
    axes[2].tick_params(axis="x", rotation=35)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = PROJECT_ROOT / "results" / "explainability" / "gradcam.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM for one image (no src/ deps)")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="best_model.pt path",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--target-class", type=int, default=None, help="Class index 0-4")
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
    )
    parser.add_argument(
        "--cam-layer",
        choices=["high", "head", "last"],
        default="head",
    )
    parser.add_argument(
        "--cam-method",
        choices=["gradcam", "gradcam++"],
        default="gradcam",
    )
    parser.add_argument(
        "--grad-source",
        choices=["full", "en-only"],
        default="full",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window")
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return 1

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_path(args.checkpoint)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model, ckpt, image_size = load_checkpoint(checkpoint_path, device)
    image_size = args.image_size or image_size
    print(
        f"Loaded epoch {ckpt.get('epoch', '?')} | "
        f"val acc {ckpt.get('best_val_acc', 0):.4f} | size={image_size}"
    )

    original_pil, tensor = load_image_tensor(
        image_path, image_size, device, resize_mode=args.resize_mode
    )
    grad_cam = GradCAM(
        model,
        get_efficientnet_target_layer(model, args.cam_layer),
        method=args.cam_method,
        grad_source=args.grad_source,
    )
    print(
        f"Grad-CAM: layer={args.cam_layer}, method={args.cam_method}, "
        f"grad={args.grad_source}, resize={args.resize_mode}"
    )
    try:
        heatmap, explained_class, probabilities = grad_cam.generate(tensor, args.target_class)
        pred = int(probabilities.argmax())
        model_input_np = denormalize_image(tensor[0])
        overlay_np = overlay_heatmap(model_input_np, heatmap)
        cam_h, cam_w = heatmap.shape[0], heatmap.shape[1]

        print(f"Image: {image_path}")
        print(f"Prediction: {CLASS_NAMES[pred]} ({probabilities[pred].item():.4f})")
        print(f"Grad-CAM grid: {cam_h}x{cam_w} upsampled to {model_input_np.shape[0]}x{model_input_np.shape[1]}")

        out = args.output
        if out is None:
            out = PROJECT_ROOT / "results" / "explainability" / f"{image_path.stem}_gradcam.png"
        else:
            out = resolve_path(out)

        save_and_show_figure(out, model_input_np, overlay_np, pred, probabilities, args.show)
    finally:
        grad_cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())