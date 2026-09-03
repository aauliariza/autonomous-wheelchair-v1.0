#!/usr/bin/env python3
"""Collapse an annotated detection dataset to a single ``obstacle`` class (spec A, C).

CLASS POLICY
------------
The system performs obstacle DETECTION, not object RECOGNITION. Every annotated
object — chair, table, person, door, bed, couch, shelf, wardrobe — becomes:

.. code-block:: text

    class_id 0, class_name "obstacle"

Output is standard YOLO format, one ``.txt`` per image, normalized to ``[0, 1]``:

.. code-block:: text

    0 x_center y_center width height

NO BOUNDING BOX IS EVER FABRICATED (spec section C). This script only TRANSFORMS
annotations you already have. If you have none, it creates the empty directory
structure and a dataset YAML so real annotations can be dropped in later, and
says so explicitly rather than inventing labels.

SUPPORTED INPUTS
----------------
``--format yolo``  an existing YOLO dataset with any number of classes; every
                   class id is rewritten to 0 and duplicates are merged.
``--format coco``  a COCO-style ``instances.json``; every category maps to 0.
``--format scaffold``  create the empty structure only, no annotations.

VALIDITY FILTERING
------------------
Boxes are dropped when they are degenerate (zero width/height), fall outside the
image, or are smaller than ``--min-box-size`` pixels. A dataset silently
containing zero-area boxes trains a detector to emit them.

Usage:
    python datasets/scripts/convert_to_obstacle_dataset.py --format yolo \
        --source datasets/raw_detection --output datasets/obstacle
    python datasets/scripts/convert_to_obstacle_dataset.py --format scaffold --output datasets/obstacle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.io import save_yaml  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("convert_obstacle")

OBSTACLE_CLASS_ID = 0
OBSTACLE_CLASS_NAME = "obstacle"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class ConversionError(Exception):
    """Raised when the source annotations cannot be converted."""


def clamp01(v: float) -> float:
    """Clamp a normalized coordinate into ``[0, 1]``."""
    return max(0.0, min(1.0, v))


def valid_yolo_box(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int, min_box_size: float
) -> bool:
    """Reject degenerate or out-of-frame boxes.

    Args:
        xc, yc, w, h (float): Normalized YOLO box.
        img_w, img_h (int): Image size, for the pixel-size floor.
        min_box_size (float): Minimum side length in pixels.
    """
    if not all(0.0 <= v <= 1.0 for v in (xc, yc)):
        return False
    if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
        return False
    if w * img_w < min_box_size or h * img_h < min_box_size:
        return False
    # The box must retain area after clipping to the frame.
    x1, x2 = xc - w / 2, xc + w / 2
    y1, y2 = yc - h / 2, yc + h / 2
    return min(x2, 1.0) > max(x1, 0.0) and min(y2, 1.0) > max(y1, 0.0)


def convert_yolo(source: Path, output: Path, splits: list[str], min_box_size: float) -> dict[str, int]:
    """Rewrite an existing YOLO dataset so every class id becomes 0."""
    import cv2

    counts: dict[str, int] = {}
    for split in splits:
        src_lbl = source / "labels" / split
        src_img = source / "images" / split
        if not src_lbl.exists():
            LOG.warning("No labels directory for split '%s' at %s; skipping.", split, src_lbl)
            continue

        dst_lbl = output / "labels" / split
        dst_img = output / "images" / split
        dst_lbl.mkdir(parents=True, exist_ok=True)
        dst_img.mkdir(parents=True, exist_ok=True)

        kept = dropped = 0
        for lbl in sorted(src_lbl.glob("*.txt")):
            image = next(
                (src_img / f"{lbl.stem}{ext}" for ext in IMAGE_EXTENSIONS if (src_img / f"{lbl.stem}{ext}").exists()),
                None,
            )
            if image is None:
                LOG.warning("Label %s has no matching image; skipping.", lbl.name)
                continue

            img = cv2.imread(str(image))
            if img is None:
                LOG.warning("Unreadable image %s; skipping.", image.name)
                continue
            ih, iw = img.shape[:2]

            lines: list[str] = []
            for raw in lbl.read_text(encoding="utf-8").splitlines():
                parts = raw.split()
                if len(parts) < 5:
                    continue
                try:
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    dropped += 1
                    continue
                if not valid_yolo_box(xc, yc, w, h, iw, ih, min_box_size):
                    dropped += 1
                    continue
                # Every source class collapses to the single obstacle class.
                lines.append(f"{OBSTACLE_CLASS_ID} {clamp01(xc):.6f} {clamp01(yc):.6f} {clamp01(w):.6f} {clamp01(h):.6f}")
                kept += 1

            (dst_lbl / lbl.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            dst = dst_img / image.name
            if not dst.exists():
                dst.write_bytes(image.read_bytes())

        counts[split] = kept
        LOG.info("Split '%s': kept %d boxes, dropped %d invalid.", split, kept, dropped)
    return counts


def convert_coco(source: Path, output: Path, annotations: Path, split: str, min_box_size: float) -> dict[str, int]:
    """Convert COCO ``instances.json`` boxes to single-class YOLO labels."""
    if not annotations.exists():
        raise ConversionError(
            f"COCO annotation file not found: {annotations.resolve()}\n"
            f"  Recovery: pass --annotations /path/to/instances.json"
        )

    data = json.loads(annotations.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    if not images:
        raise ConversionError(f"{annotations} contains no 'images' entries.")

    dst_lbl = output / "labels" / split
    dst_img = output / "images" / split
    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    per_image: dict[int, list[str]] = {}
    kept = dropped = 0

    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            dropped += 1
            continue
        im = images.get(ann.get("image_id"))
        if im is None:
            dropped += 1
            continue

        iw, ih = float(im["width"]), float(im["height"])
        x, y, w, h = (float(v) for v in ann["bbox"])  # COCO: top-left xywh in pixels
        xc, yc = (x + w / 2) / iw, (y + h / 2) / ih
        wn, hn = w / iw, h / ih

        if not valid_yolo_box(xc, yc, wn, hn, int(iw), int(ih), min_box_size):
            dropped += 1
            continue

        per_image.setdefault(ann["image_id"], []).append(
            f"{OBSTACLE_CLASS_ID} {clamp01(xc):.6f} {clamp01(yc):.6f} {clamp01(wn):.6f} {clamp01(hn):.6f}"
        )
        kept += 1

    for image_id, im in images.items():
        stem = Path(im["file_name"]).stem
        (dst_lbl / f"{stem}.txt").write_text("\n".join(per_image.get(image_id, [])) + "\n", encoding="utf-8")
        src = source / im["file_name"]
        if src.exists():
            dst = dst_img / Path(im["file_name"]).name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    LOG.info("Split '%s': kept %d boxes, dropped %d invalid/crowd.", split, kept, dropped)
    return {split: kept}


def scaffold(output: Path, splits: list[str]) -> dict[str, int]:
    """Create the empty directory structure with no annotations.

    Used when obstacle annotations are not yet available. It creates a place for
    real labels; it does NOT invent any.
    """
    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    LOG.warning(
        "Created an EMPTY obstacle dataset scaffold at %s. It contains NO annotations, "
        "and none were fabricated. Add your own images to images/<split>/ and matching "
        "YOLO .txt labels (class id 0) to labels/<split>/, then re-run verification.",
        output,
    )
    return dict.fromkeys(splits, 0)


def write_yaml(output: Path, counts: dict[str, int], config_path: Path) -> Path:
    """Write the nc=1 obstacle dataset YAML."""
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {OBSTACLE_CLASS_ID: OBSTACLE_CLASS_NAME},
        "box_counts": counts,
    }
    if (output / "images" / "test").exists():
        data["test"] = "images/test"
    return save_yaml(data, config_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Collapse a detection dataset to the single 'obstacle' class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--format", choices=["yolo", "coco", "scaffold"], required=True, help="Source annotation format")
    ap.add_argument("--source", type=Path, default=None, help="Source dataset root (not needed for scaffold)")
    ap.add_argument("--output", type=Path, default=Path("datasets/obstacle"), help="Output dataset root")
    ap.add_argument("--annotations", type=Path, default=None, help="COCO instances.json (--format coco)")
    ap.add_argument("--split", default="train", help="Split name for --format coco")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits for yolo/scaffold")
    ap.add_argument("--min-box-size", type=float, default=4.0, help="Minimum box side in pixels")
    ap.add_argument("--config-out", type=Path, default=Path("configs/data/obstacle.yaml"), help="Dataset YAML to write")
    args = ap.parse_args(argv)

    try:
        if args.format == "scaffold":
            counts = scaffold(args.output, args.splits)
        elif args.format == "yolo":
            if args.source is None:
                raise ConversionError("--source is required for --format yolo.")
            if not args.source.exists():
                raise ConversionError(f"Source dataset not found: {args.source.resolve()}")
            counts = convert_yolo(args.source, args.output, args.splits, args.min_box_size)
        else:
            if args.source is None or args.annotations is None:
                raise ConversionError("--source and --annotations are required for --format coco.")
            counts = convert_coco(args.source, args.output, args.annotations, args.split, args.min_box_size)

        cfg = write_yaml(args.output, counts, args.config_out)
        LOG.info("Obstacle dataset YAML written to %s (nc=1, names={0: obstacle})", cfg)
        total = sum(counts.values())
        if total == 0:
            LOG.warning("Dataset currently contains 0 boxes. Training a detector on it will not work.")
            LOG.warning("The navigation pipeline meanwhile runs with detection.class_agnostic: true, which")
            LOG.warning("relabels every COCO detection as 'obstacle' and needs no annotations at all.")
    except ConversionError as e:
        LOG.error("%s", e)
        return 1
    except (OSError, ValueError, KeyError) as e:
        LOG.error("Conversion failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
