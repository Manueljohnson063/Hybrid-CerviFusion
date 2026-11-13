"""
Verify the SIPaKMeD cropped cervical-cell dataset.

Scans dataset/main/<class>/<class>/CROPPED/*.bmp and reports:
- image counts per class
- corrupted/unreadable files
- image dimension statistics
- duplicate filenames and content hashes
- slide-level grouping info (for train/test split planning)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "dataset" / "main"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_verification"

CLASSES = [
    ("im_Dyskeratotic", "Dyskeratotic"),
    ("im_Koilocytotic", "Koilocytotic"),
    ("im_Metaplastic", "Metaplastic"),
    ("im_Parabasal", "Parabasal"),
    ("im_Superficial-Intermediate", "Superficial-Intermediate"),
]

SLIDE_PATTERN = re.compile(r"^(\d+)_\d+\.bmp$", re.IGNORECASE)


@dataclass
class ImageRecord:
    path: Path
    class_folder: str
    class_name: str
    slide_id: str | None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    file_size_bytes: int = 0
    content_hash: str | None = None
    error: str | None = None


@dataclass
class VerificationReport:
    total_images: int = 0
    valid_images: int = 0
    corrupted_images: list[dict] = field(default_factory=list)
    per_class_counts: dict[str, int] = field(default_factory=dict)
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    modes: Counter = field(default_factory=Counter)
    duplicate_names: dict[str, list[str]] = field(default_factory=dict)
    duplicate_hashes: dict[str, list[str]] = field(default_factory=dict)
    slides_per_class: dict[str, int] = field(default_factory=dict)
    images_per_slide: dict[str, int] = field(default_factory=dict)


def cropped_dir(class_folder: str) -> Path:
    return ARCHIVE_ROOT / class_folder / class_folder / "CROPPED"


def parse_slide_id(filename: str) -> str | None:
    match = SLIDE_PATTERN.match(filename)
    return match.group(1) if match else None


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_paths() -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []

    for class_folder, class_name in CLASSES:
        directory = cropped_dir(class_folder)
        if not directory.exists():
            print(f"[WARN] Missing folder: {directory}")
            continue

        for path in sorted(directory.glob("*.bmp")):
            records.append((path, class_folder, class_name))

    return records


def verify_image(path: Path, class_folder: str, class_name: str) -> ImageRecord:
    slide_id = parse_slide_id(path.name)
    record = ImageRecord(
        path=path,
        class_folder=class_folder,
        class_name=class_name,
        slide_id=slide_id,
        file_size_bytes=path.stat().st_size,
    )

    try:
        with Image.open(path) as image:
            record.width, record.height = image.size
            record.mode = image.mode
            image.verify()
    except Exception as exc:  # noqa: BLE001 - report all load failures
        record.error = str(exc)
        return record

    try:
        record.content_hash = file_md5(path)
    except Exception as exc:  # noqa: BLE001 - report all load failures
        record.error = f"hash failed: {exc}"

    return record


def build_report(records: list[ImageRecord]) -> VerificationReport:
    report = VerificationReport()
    name_map: dict[str, list[str]] = defaultdict(list)
    hash_map: dict[str, list[str]] = defaultdict(list)
    slide_sets: dict[str, set[str]] = defaultdict(set)
    slide_image_counts: Counter = Counter()

    for record in records:
        report.total_images += 1
        report.per_class_counts[record.class_name] = (
            report.per_class_counts.get(record.class_name, 0) + 1
        )

        rel_path = str(record.path.relative_to(PROJECT_ROOT))
        name_map[record.path.name].append(rel_path)

        if record.error:
            report.corrupted_images.append(
                {
                    "path": rel_path,
                    "class": record.class_name,
                    "error": record.error,
                }
            )
            continue

        report.valid_images += 1
        if record.width is not None and record.height is not None:
            report.widths.append(record.width)
            report.heights.append(record.height)
        if record.mode:
            report.modes[record.mode] += 1
        if record.content_hash:
            hash_map[record.content_hash].append(rel_path)

        if record.slide_id:
            slide_key = f"{record.class_name}:{record.slide_id}"
            slide_sets[record.class_name].add(record.slide_id)
            slide_image_counts[slide_key] += 1

    report.duplicate_names = {
        name: paths for name, paths in name_map.items() if len(paths) > 1
    }
    report.duplicate_hashes = {
        digest: paths for digest, paths in hash_map.items() if len(paths) > 1
    }
    report.slides_per_class = {cls: len(slides) for cls, slides in slide_sets.items()}
    report.images_per_slide = dict(slide_image_counts)
    return report


def print_summary(report: VerificationReport) -> None:
    print("\n" + "=" * 60)
    print("CerviFusion Dataset Verification (CROPPED images)")
    print("=" * 60)
    print(f"Dataset root: {ARCHIVE_ROOT}")
    print(f"Total BMP files found: {report.total_images}")
    print(f"Valid images:          {report.valid_images}")
    print(f"Corrupted/unreadable:  {len(report.corrupted_images)}")

    print("\nPer-class counts:")
    for _, class_name in CLASSES:
        count = report.per_class_counts.get(class_name, 0)
        print(f"  {class_name:28s} {count:5d}")

    if report.widths and report.heights:
        print("\nImage dimensions:")
        print(f"  Width  -> min: {min(report.widths)}, max: {max(report.widths)}, avg: {sum(report.widths) / len(report.widths):.1f}")
        print(f"  Height -> min: {min(report.heights)}, max: {max(report.heights)}, avg: {sum(report.heights) / len(report.heights):.1f}")

    if report.modes:
        print("\nColor modes:")
        for mode, count in report.modes.most_common():
            print(f"  {mode}: {count}")

    print("\nSlide-level grouping (for split planning):")
    for _, class_name in CLASSES:
        slides = report.slides_per_class.get(class_name, 0)
        print(f"  {class_name:28s} {slides:5d} unique slides")

    if report.duplicate_names:
        print(f"\nDuplicate filenames across classes: {len(report.duplicate_names)}")
        for name, paths in list(report.duplicate_names.items())[:5]:
            print(f"  {name} -> {paths}")
        if len(report.duplicate_names) > 5:
            print("  ...")

    if report.duplicate_hashes:
        print(f"\nExact duplicate images (same content): {len(report.duplicate_hashes)}")
        for digest, paths in list(report.duplicate_hashes.items())[:3]:
            print(f"  {digest[:12]}... -> {len(paths)} files")
    else:
        print("\nExact duplicate images (same content): 0")

    if report.corrupted_images:
        print("\nCorrupted files:")
        for item in report.corrupted_images[:10]:
            print(f"  {item['path']} -> {item['error']}")
        if len(report.corrupted_images) > 10:
            print("  ...")
    else:
        print("\nCorrupted files: none")

    print("\nRecommendation:")
    print("  Use CROPPED images for training.")
    print("  Split by slide ID (e.g. 001 from 001_01.bmp) to avoid data leakage.")
    print("=" * 60)


def save_report(report: VerificationReport) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "verification_summary.json"

    payload = {
        "total_images": report.total_images,
        "valid_images": report.valid_images,
        "corrupted_count": len(report.corrupted_images),
        "corrupted_images": report.corrupted_images,
        "per_class_counts": report.per_class_counts,
        "dimension_stats": {
            "width_min": min(report.widths) if report.widths else None,
            "width_max": max(report.widths) if report.widths else None,
            "width_avg": (sum(report.widths) / len(report.widths)) if report.widths else None,
            "height_min": min(report.heights) if report.heights else None,
            "height_max": max(report.heights) if report.heights else None,
            "height_avg": (sum(report.heights) / len(report.heights)) if report.heights else None,
        },
        "color_modes": dict(report.modes),
        "slides_per_class": report.slides_per_class,
        "duplicate_filename_count": len(report.duplicate_names),
        "duplicate_content_count": len(report.duplicate_hashes),
        "duplicate_filenames": report.duplicate_names,
        "duplicate_hashes": report.duplicate_hashes,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    if not ARCHIVE_ROOT.exists():
        print(f"ERROR: dataset folder not found at {ARCHIVE_ROOT}")
        return 1

    image_paths = collect_image_paths()
    if not image_paths:
        print(f"ERROR: no cropped BMP files found under {ARCHIVE_ROOT}")
        return 1

    records: list[ImageRecord] = []
    for path, class_folder, class_name in tqdm(image_paths, desc="Verifying images"):
        records.append(verify_image(path, class_folder, class_name))

    report = build_report(records)
    print_summary(report)

    output_path = save_report(report)
    print(f"\nSaved JSON summary to: {output_path}")
    return 0 if not report.corrupted_images else 2


if __name__ == "__main__":
    sys.exit(main())