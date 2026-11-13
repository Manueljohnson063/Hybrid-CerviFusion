"""
Create slide-aware train/validation/test splits for CerviFusion.

Each (class, slide_id) stays in exactly one split (no slide leakage).
Within each class, whole slides are assigned so **image counts** are near
70% / 15% / 15% (not merely slide counts).

Default ratio: 70% / 15% / 15%.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "dataset" / "main"
SPLITS_DIR = PROJECT_ROOT / "results" / "splits"

CLASSES = [
    ("im_Dyskeratotic", "Dyskeratotic", 0),
    ("im_Koilocytotic", "Koilocytotic", 1),
    ("im_Metaplastic", "Metaplastic", 2),
    ("im_Parabasal", "Parabasal", 3),
    ("im_Superficial-Intermediate", "Superficial-Intermediate", 4),
]

SLIDE_PATTERN = re.compile(r"^(\d+)_\d+\.bmp$", re.IGNORECASE)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
SPLIT_METHOD = "slide_groups_image_balanced"


@dataclass
class Sample:
    path: str
    class_folder: str
    class_name: str
    class_idx: int
    slide_id: str
    filename: str
    split: str = ""


def cropped_dir(class_folder: str) -> Path:
    return ARCHIVE_ROOT / class_folder / class_folder / "CROPPED"


def parse_slide_id(filename: str) -> str | None:
    match = SLIDE_PATTERN.match(filename)
    return match.group(1) if match else None


def collect_samples() -> list[Sample]:
    samples: list[Sample] = []

    for class_folder, class_name, class_idx in CLASSES:
        directory = cropped_dir(class_folder)
        if not directory.exists():
            raise FileNotFoundError(f"Missing folder: {directory}")

        for path in sorted(directory.glob("*.bmp")):
            slide_id = parse_slide_id(path.name)
            if slide_id is None:
                print(f"[WARN] Skipping file without slide pattern: {path}")
                continue

            samples.append(
                Sample(
                    path=str(path.relative_to(PROJECT_ROOT)),
                    class_folder=class_folder,
                    class_name=class_name,
                    class_idx=class_idx,
                    slide_id=slide_id,
                    filename=path.name,
                )
            )

    return samples


def assign_slides_for_class(
    slide_image_counts: dict[str, int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    """
    Assign each slide in one class to train/val/test.
    Greedy by image count: always place the next slide into the split
    that is furthest below its target image quota.
    """
    slides = list(slide_image_counts.keys())
    if not slides:
        return {}
    if len(slides) == 1:
        return {slides[0]: "train"}
    if len(slides) == 2:
        return {slides[0]: "train", slides[1]: "test"}

    rng = random.Random(seed)
    rng.shuffle(slides)

    total = sum(slide_image_counts[s] for s in slides)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * test_ratio,
    }
    filled = {"train": 0.0, "val": 0.0, "test": 0.0}
    assignment: dict[str, str] = {}

    for slide_id in slides:
        count = slide_image_counts[slide_id]
        split = max(("train", "val", "test"), key=lambda name: targets[name] - filled[name])
        assignment[slide_id] = split
        filled[split] += count

    return assignment


def assign_splits(samples: list[Sample]) -> list[Sample]:
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        by_class[sample.class_name][sample.slide_id] += 1

    slide_to_split: dict[tuple[str, str], str] = {}
    for class_idx, (class_name, slide_counts) in enumerate(sorted(by_class.items())):
        seed = RANDOM_SEED + class_idx * 9973
        mapping = assign_slides_for_class(
            dict(slide_counts),
            seed,
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
        )
        for slide_id, split_name in mapping.items():
            slide_to_split[(class_name, slide_id)] = split_name

    for sample in samples:
        sample.split = slide_to_split[(sample.class_name, sample.slide_id)]

    return samples


def write_csv(path: Path, rows: list[Sample]) -> None:
    fieldnames = [
        "path",
        "class_folder",
        "class_name",
        "class_idx",
        "slide_id",
        "filename",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "class_folder": row.class_folder,
                    "class_name": row.class_name,
                    "class_idx": row.class_idx,
                    "slide_id": row.slide_id,
                    "filename": row.filename,
                    "split": row.split,
                }
            )


def check_slide_leakage(samples: list[Sample]) -> dict:
    """Each (class, slide_id) must map to exactly one split."""
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        key = (sample.class_name, sample.slide_id)
        group_splits[key].add(sample.split)

    leaked = {k: sorted(v) for k, v in group_splits.items() if len(v) > 1}
    return {
        "slide_groups_total": len(group_splits),
        "leaked_groups": len(leaked),
        "leakage_ok": len(leaked) == 0,
        "leaked_examples": [
            {"class": k[0], "slide_id": k[1], "splits": v}
            for k, v in list(leaked.items())[:10]
        ],
    }


def summarize(samples: list[Sample]) -> dict:
    total = len(samples)
    split_counts = dict(Counter(s.split for s in samples))
    split_percentages = {
        name: (100.0 * split_counts.get(name, 0) / total if total else 0.0)
        for name in ("train", "val", "test")
    }

    slides_per_split: dict[str, int] = {}
    slide_pct: dict[str, float] = {}
    total_groups = len({(s.class_name, s.slide_id) for s in samples})

    for split_name in ["train", "val", "test"]:
        split_samples = [s for s in samples if s.split == split_name]
        n_slides = len({f"{s.class_name}:{s.slide_id}" for s in split_samples})
        slides_per_split[split_name] = n_slides
        slide_pct[split_name] = 100.0 * n_slides / total_groups if total_groups else 0.0

    summary: dict = {
        "split_method": SPLIT_METHOD,
        "random_seed": RANDOM_SEED,
        "target_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "total_samples": total,
        "total_slide_groups": total_groups,
        "split_counts": split_counts,
        "split_percentages": split_percentages,
        "slides_per_split": slides_per_split,
        "slide_group_percentages": slide_pct,
        "split_class_counts": {},
        "leakage_check": check_slide_leakage(samples),
    }

    for split_name in ["train", "val", "test"]:
        split_samples = [s for s in samples if s.split == split_name]
        summary["split_class_counts"][split_name] = dict(
            Counter(s.class_name for s in split_samples)
        )

    return summary


def _pct_exact(count: int, total: int) -> str:
    if total <= 0:
        return "0.000000%"
    return f"{100.0 * count / total:.6f}%"


def print_summary(summary: dict) -> None:
    total = summary["total_samples"]
    counts = summary["split_counts"]
    targets = summary["target_ratios"]
    leakage = summary["leakage_check"]
    slide_counts = summary["slides_per_split"]

    n_train = counts.get("train", 0)
    n_val = counts.get("val", 0)
    n_test = counts.get("test", 0)
    n_sum = n_train + n_val + n_test

    print("\n" + "=" * 60)
    print("SPLIT RESULT (exact image counts)")
    print("=" * 60)
    print(f"  train : {n_train} images   ({_pct_exact(n_train, total)} of {total})")
    print(f"  val   : {n_val} images   ({_pct_exact(n_val, total)} of {total})")
    print(f"  test  : {n_test} images   ({_pct_exact(n_test, total)} of {total})")
    print(f"  total : {n_sum} images   (must equal {total})")
    print(
        f"  target: train {int(targets['train'] * 100)}% | "
        f"val {int(targets['val'] * 100)}% | test {int(targets['test'] * 100)}%"
    )
    print(f"  method: {summary['split_method']} | seed: {summary['random_seed']}")

    print("\nSlide groups (exact):")
    print(f"  train : {slide_counts.get('train', 0)}")
    print(f"  val   : {slide_counts.get('val', 0)}")
    print(f"  test  : {slide_counts.get('test', 0)}")
    print(f"  total : {summary['total_slide_groups']}")

    print("\nSlide leakage:")
    if leakage["leakage_ok"]:
        print(f"  OK ({leakage['slide_groups_total']} groups, 0 leaked)")
    else:
        print(f"  FAIL ({leakage['leaked_groups']} groups in multiple splits)")
        for ex in leakage["leaked_examples"]:
            print(f"    {ex['class']} slide {ex['slide_id']} -> {ex['splits']}")

    print("\nPer-class image counts (exact):")
    labels = ["Dysk", "Koil", "Meta", "Para", "S-I"]
    print("       " + " ".join(f"{lb:>4s}" for lb in labels))
    for split_name in ["train", "val", "test"]:
        class_counts = summary["split_class_counts"].get(split_name, {})
        parts = [str(class_counts.get(name, 0)) for _, name, _ in CLASSES]
        print(f"  {split_name:5s}: " + " ".join(f"{n:>4s}" for n in parts))

    rel = SPLITS_DIR.relative_to(PROJECT_ROOT)
    print(f"\nWrote CSVs under: {rel}/")
    print("=" * 60)


def main() -> int:
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        print("ERROR: train/val/test ratios must sum to 1.0")
        return 1

    if not ARCHIVE_ROOT.exists():
        print(f"ERROR: archive folder not found at {ARCHIVE_ROOT}")
        return 1

    samples = collect_samples()
    if not samples:
        print("ERROR: no cropped BMP samples found.")
        return 1

    samples = assign_splits(samples)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(SPLITS_DIR / "all_samples.csv", samples)
    write_csv(SPLITS_DIR / "train.csv", [s for s in samples if s.split == "train"])
    write_csv(SPLITS_DIR / "val.csv", [s for s in samples if s.split == "val"])
    write_csv(SPLITS_DIR / "test.csv", [s for s in samples if s.split == "test"])

    summary = summarize(samples)
    summary_path = SPLITS_DIR / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary)
    print(f"JSON summary: {summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())