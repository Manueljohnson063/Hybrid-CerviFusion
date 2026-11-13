"""Grad-CAM explainability for CerviFusion and EfficientNet models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Generate Grad-CAM heatmaps from a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """
        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor
            target_class: class index to explain; uses predicted class if None

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            target_class: explained class index
            probabilities: softmax probabilities (num_classes,)
        """
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        if target_class is None:
            target_class = int(torch.argmax(probabilities).item())

        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture layer outputs.")

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.cpu().numpy()
        return heatmap, target_class, probabilities.detach().cpu()


def get_efficientnet_target_layer(model: nn.Module) -> nn.Module:
    """Return the last conv layer used for Grad-CAM."""
    if hasattr(model, "en_branch"):
        backbone = model.en_branch.backbone
    elif hasattr(model, "backbone"):
        backbone = model.backbone
    else:
        backbone = model

    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]
    raise ValueError("Could not find a suitable convolutional layer for Grad-CAM.")


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor to HWC uint8 RGB image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.cpu().clone() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a heatmap on an RGB image using a jet colormap."""
    import matplotlib.cm as cm

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]),
            Image.BILINEAR,
        )
    ).astype(np.float32) / 255.0

    colored = cm.jet(heatmap_resized)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    overlay = (alpha * colored + (1 - alpha) * image).astype(np.uint8)
    return overlay