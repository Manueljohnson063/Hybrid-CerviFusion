"""EfficientNet baseline classifier."""

from __future__ import annotations

import torch.nn as nn

from src.constants import NUM_CLASSES
from src.models.pretrained_loader import create_timm_backbone


def create_efficientnet_baseline(
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    pretrained_path: str | None = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    return create_timm_backbone(
        model_name,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        num_classes=num_classes,
    )