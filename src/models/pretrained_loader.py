"""Load timm backbones from local weight files or the timm/HF hub."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.models import load_checkpoint

from src.constants import PROJECT_ROOT


def resolve_pretrained_path(path: str | Path | None) -> Path | None:
    """Resolve an optional project-relative pretrained weight path."""
    if path is None:
        return None

    text = str(path).strip()
    if not text:
        return None

    resolved = Path(text)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def create_timm_backbone(
    model_name: str,
    *,
    pretrained: bool = False,
    pretrained_path: str | Path | None = None,
    num_classes: int = 0,
    global_pool: str | None = None,
) -> nn.Module:
    """
    Create a timm model and optionally load weights.

    Priority:
        1. Local file at `pretrained_path` (download once from timm if missing)
        2. timm/HF hub when `pretrained=True`
        3. Random initialization
    """
    kwargs: dict = {"num_classes": num_classes}
    if global_pool is not None:
        kwargs["global_pool"] = global_pool

    local_path = resolve_pretrained_path(pretrained_path)
    if local_path is not None:
        if not local_path.exists():
            print(
                f"[INFO] Pretrained weights not found at {local_path}\n"
                f"       Downloading {model_name} from timm (one-time, needs network)..."
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            hub_model = timm.create_model(model_name, pretrained=True, **kwargs)
            torch.save(hub_model.state_dict(), local_path)
            print(f"[INFO] Saved for offline training: {local_path}")
            return hub_model

        model = timm.create_model(model_name, pretrained=False, **kwargs)
        load_checkpoint(model, str(local_path), strict=False)
        print(f"Loaded local pretrained weights: {local_path}")
        return model

    return timm.create_model(model_name, pretrained=pretrained, **kwargs)