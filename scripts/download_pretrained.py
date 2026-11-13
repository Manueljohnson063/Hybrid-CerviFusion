"""Download pretrained models for CerviFusion."""
from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "checkpoints" / "pretrained"

BACKBONES = [
    ("efficientnet_b0", "efficientnet_b0.pth", {"num_classes": 0, "global_pool": "avg"}),
    (
        "vit_small_patch16_224",
        "vit_small_patch16_224.pth",
        {"num_classes": 0, "global_pool": "token"},
    ),
]


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    for model_name, filename, kwargs in BACKBONES:
        out_path = OUT_DIR / filename
        if out_path.exists():
            print(f"Skip (exists): {filename}")
            continue
        print(f"Downloading {model_name} ...")
        model = timm.create_model(model_name, pretrained=True, **kwargs)
        torch.save(model.state_dict(), out_path)
        print(f"  Saved: {out_path}")

    print("\nDone. Run: python scripts/05_train_cervifusion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())