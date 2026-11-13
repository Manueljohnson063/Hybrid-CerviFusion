"""Attention-based feature fusion for CerviFusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """
    Fuse local (EfficientNet) and global (ViT) feature vectors using
    multi-head attention over a learnable query token.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            local_features: (batch, embed_dim)
            global_features: (batch, embed_dim)

        Returns:
            fused_features: (batch, embed_dim)
            attention_weights: (batch, 1, 2)
        """
        branch_tokens = torch.stack([local_features, global_features], dim=1)
        batch_size = branch_tokens.size(0)
        query = self.query.expand(batch_size, -1, -1)

        fused, attention_weights = self.attention(
            query,
            branch_tokens,
            branch_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        fused = self.norm(fused + query)
        fused = self.dropout(fused)
        return fused.squeeze(1), attention_weights.squeeze(1)