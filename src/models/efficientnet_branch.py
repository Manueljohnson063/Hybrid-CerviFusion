"""EfficientNet feature extractor branch for CerviFusion."""

from __future__ import annotations

import torch.nn as nn

from src.models.pretrained_loader import create_timm_backbone


class EfficientNetBranch(nn.Module):
    """Local feature extractor with classification head removed."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = create_timm_backbone(
            model_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)