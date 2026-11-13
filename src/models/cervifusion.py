"""CerviFusion hybrid model: EfficientNet + ViT + attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import NUM_CLASSES
from src.models.efficientnet_branch import EfficientNetBranch
from src.models.fusion import AttentionFusion
from src.models.vit_branch import ViTBranch


class CerviFusion(nn.Module):
    """
    Parallel hybrid architecture:
        same image -> EfficientNet (local) + ViT (global)
        -> projection -> attention fusion -> classifier
    """

    def __init__(
        self,
        efficientnet_name: str = "efficientnet_b0",
        vit_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        efficientnet_pretrained_path: str | None = None,
        vit_pretrained_path: str | None = None,
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        self.en_branch = EfficientNetBranch(
            efficientnet_name,
            pretrained=pretrained,
            pretrained_path=efficientnet_pretrained_path,
        )
        self.vit_branch = ViTBranch(
            vit_name,
            pretrained=pretrained,
            pretrained_path=vit_pretrained_path,
        )

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

        self.fusion = AttentionFusion(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

    def extract_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_raw = self.en_branch(x)
        global_raw = self.vit_branch(x)
        local_proj = self.en_projection(local_raw)
        global_proj = self.vit_projection(global_raw)
        fused, attention_weights = self.fusion(local_proj, global_proj)
        return local_raw, global_raw, fused, attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, fused, _ = self.extract_features(x)
        return self.classifier(fused)

    def forward_with_details(self, x: torch.Tensor) -> dict:
        local_raw, global_raw, fused, attention_weights = self.extract_features(x)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "local_features": local_raw,
            "global_features": global_raw,
            "fused_features": fused,
            "attention_weights": attention_weights,
        }


def create_cervifusion(
    efficientnet_name: str = "efficientnet_b0",
    vit_name: str = "vit_small_patch16_224",
    pretrained: bool = True,
    efficientnet_pretrained_path: str | None = None,
    vit_pretrained_path: str | None = None,
    fusion_dim: int = 512,
    num_heads: int = 4,
    dropout: float = 0.3,
    num_classes: int = NUM_CLASSES,
) -> CerviFusion:
    return CerviFusion(
        efficientnet_name=efficientnet_name,
        vit_name=vit_name,
        pretrained=pretrained,
        efficientnet_pretrained_path=efficientnet_pretrained_path,
        vit_pretrained_path=vit_pretrained_path,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes,
    )