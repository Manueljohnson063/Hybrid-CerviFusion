"""Generate Cellpose-guided, nucleus-centered crops for CerviFusion retraining.

For each row in the original split CSVs, this script:
1. segments cell instances with Cellpose;
2. selects the mask containing the strongest dark blue/purple nucleus-like pixel;
3. creates a padded square crop around that cell; and
4. blurs pixels outside the selected cell mask to suppress neighboring-cell and
   background shortcuts.

The source images and source split files are never modified. New images and
matching split CSVs are written under results/nucleus_crops/ by default.

Examples:
    python scripts/07_generate_nucleus_crops.py --splits test --limit-per-split 5
    python scripts/07_generate_nucleus_crops.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from cellpose import models
from PIL import Image, ImageFilter
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "results" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "nucleus_crops"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def nucleus_score(image_array: np.ndarray) -> np.ndarray:
    """Score dark blue/purple pixels, which are likely nuclear stain pixels."""
    image_float = image_array.astype(np.float32)
    red, green, blue = np.moveaxis(image_float, -1, 0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_purple = np.maximum(blue - red, 0.0) + np.maximum(blue - green, 0.0)
    return (255.0 - luminance) + 0.5 * blue_purple


def choose_target_mask(masks: np.ndarray, image_array: np.ndarray) -> tuple[int, str]:
    """Choose the cell mask most likely to contain the image's target nucleus."""
    labels = np.unique(masks)
    labels = labels[labels != 0]
    if not len(labels):
        return 0, "no_cellpose_mask"

    scores = nucleus_score(image_array)
    best_label = 0
    best_score = float("-inf")
    for label in labels:
        label_mask = masks == label
        area = int(label_mask.sum())
        if area < 16:
            continue
        score = float(scores[label_mask].max())
        if score > best_score:
            best_score = score
            best_label = int(label)

    return best_label, "cellpose_mask" if best_label else "no_valid_mask"


def square_bounds(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    """Return in-image square crop bounds around the selected cell mask."""
    ys, xs = np.where(mask)
    if not len(xs):
        side = min(image_width, image_height)
        left = (image_width - side) // 2
        top = (image_height - side) // 2
        return left, top, left + side, top + side

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    side = max(max_x - min_x + 1, max_y - min_y + 1)
    side = int(np.ceil(side * padding))
    side = max(32, min(side, image_width, image_height))

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(left, image_width - side))
    top = max(0, min(top, image_height - side))
    return left, top, left + side, top + side


def create_crop(
    image: Image.Image,
    masks: np.ndarray,
    target_label: int,
    padding: float,
    blur_radius: float,
) -> tuple[Image.Image, dict[str, int]]:
    """Create a target-cell crop, blurring all pixels outside its mask."""
    width, height = image.size
    if target_label:
        target_mask = masks == target_label
    else:
        target_mask = np.zeros((height, width), dtype=bool)

    left, top, right, bottom = square_bounds(target_mask, width, height, padding)
    original = np.asarray(image)
    if target_label:
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        blended = np.where(target_mask[..., None], original, blurred).astype(np.uint8)
        crop = Image.fromarray(blended).crop((left, top, right, bottom))
        mask_area = int(target_mask.sum())
    else:
        crop = image.crop((left, top, right, bottom))
        mask_area = 0

    return crop, {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "mask_area": mask_area,
    }


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def process_split(
    split_name: str,
    rows: list[dict[str, str]],
    model: models.CellposeModel,
    output_dir: Path,
    padding: float,
    blur_radius: float,
    limit: int | None,
) -> tuple[list[dict[str, str]], Counter[str]]:
    processed_rows: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()

    if limit is not None:
        rows = rows[:limit]

    for index, row in enumerate(rows, start=1):
        source_path = resolve_path(Path(row["path"]))
        if not source_path.exists():
            status_counts["missing_source"] += 1
            print(f"[{split_name} {index}/{len(rows)}] Missing: {source_path}")
            continue

        with Image.open(source_path) as source_image:
            image = source_image.convert("RGB")
        image_array = np.asarray(image)
        masks, _, _ = model.eval(image_array, channels=None)
        target_label, status = choose_target_mask(masks, image_array)
        crop, crop_metadata = create_crop(
            image, masks, target_label, padding, blur_radius
        )

        output_path = (
            output_dir
            / "images"
            / split_name
            / row["class_folder"]
            / f"{source_path.stem}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)

        output_row = dict(row)
        output_row["path"] = output_path.relative_to(PROJECT_ROOT).as_posix()
        output_row["crop_status"] = status
        output_row["crop_mask_area"] = str(crop_metadata["mask_area"])
        output_row["crop_left"] = str(crop_metadata["left"])
        output_row["crop_top"] = str(crop_metadata["top"])
        output_row["crop_right"] = str(crop_metadata["right"])
        output_row["crop_bottom"] = str(crop_metadata["bottom"])
        processed_rows.append(output_row)
        status_counts[status] += 1
        print(
            f"[{split_name} {index}/{len(rows)}] {source_path.name} -> "
            f"{status}, mask_area={crop_metadata['mask_area']}"
        )

    return processed_rows, status_counts


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        raise RuntimeError(f"No processed images for {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Cellpose-guided target-cell crops for retraining."
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--model-type", default="cpsam")
    parser.add_argument("--padding", type=float, default=1.5)
    parser.add_argument("--blur-radius", type=float, default=12.0)
    parser.add_argument("--limit-per-split", type=int, default=None)
    args = parser.parse_args()

    if args.padding < 1.0:
        raise ValueError("--padding must be at least 1.0")
    if args.blur_radius < 0:
        raise ValueError("--blur-radius cannot be negative")

    splits_dir = resolve_path(args.splits_dir)
    output_dir = resolve_path(args.output_dir)
    output_splits_dir = output_dir / "splits"
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu, model_type=args.model_type)
    print(f"Cellpose model: {args.model_type} | GPU: {use_gpu}")

    manifest: dict[str, Any] = {
        "source_splits_dir": str(splits_dir),
        "model_type": args.model_type,
        "padding": args.padding,
        "blur_radius": args.blur_radius,
        "splits": {},
    }
    for split_name in args.splits:
        source_csv = splits_dir / f"{split_name}.csv"
        rows = read_rows(source_csv)
        processed_rows, status_counts = process_split(
            split_name,
            rows,
            model,
            output_dir,
            args.padding,
            args.blur_radius,
            args.limit_per_split,
        )
        output_csv = output_splits_dir / f"{split_name}.csv"
        write_rows(processed_rows, output_csv)
        manifest["splits"][split_name] = {
            "source_rows": len(rows),
            "processed_rows": len(processed_rows),
            "status_counts": dict(status_counts),
            "csv": str(output_csv),
        }

    shutil.copy2(splits_dir / "split_summary.json", output_splits_dir / "source_split_summary.json")
    manifest_path = output_dir / "crop_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nCrop manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
