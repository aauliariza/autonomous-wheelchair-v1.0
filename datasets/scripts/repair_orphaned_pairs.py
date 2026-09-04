#!/usr/bin/env python3
"""Remove orphaned image/depth files left by the pre-fix prepare_sunrgbd.py.

BACKGROUND
----------
An earlier version of ``prepare_sunrgbd.py`` did not check ``cv2.imwrite()``'s
return value. When a depth PNG write silently failed (return False, no
exception) while the paired RGB write succeeded, the result was a JPG in
``images/<split>/`` with no matching PNG in ``depth/<split>/`` -- invisible
until Ultralytics tried to train on it and reported "No labels found".

``prepare_sunrgbd.py`` itself is now fixed (it verifies every write and
removes any partial pair as it happens), but that fix only protects a FRESH
conversion. It does not repair a dataset that was already converted with the
old code. This script finds and removes the resulting orphans from an
EXISTING ``datasets/sunrgbd/`` output, without re-running the full,
multi-hour conversion of the raw archive.

WHAT COUNTS AS AN ORPHAN
-------------------------
* An image in ``images/<split>/`` with no file of the same stem in
  ``depth/<split>/``.
* A depth file in ``depth/<split>/`` with no file of the same stem in
  ``images/<split>/`` (the symmetric case; rarer, but a genuine leftover if it
  occurs).

NOTHING IS EVER FABRICATED (spec section C): this script only DELETES
half-pairs. It never invents a replacement image or depth map.

Usage:
    # Dry run first -- always. Reports what WOULD be removed.
    python datasets/scripts/repair_orphaned_pairs.py --data datasets/sunrgbd

    # Apply the removal once you've reviewed the dry-run report.
    python datasets/scripts/repair_orphaned_pairs.py --data datasets/sunrgbd --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.io import save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("repair_orphaned_pairs")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_orphans(dataset_root: Path, split: str) -> dict[str, list[str]]:
    """Find image-without-depth and depth-without-image orphans for one split."""
    img_dir = dataset_root / "images" / split
    dep_dir = dataset_root / "depth" / split

    if not img_dir.exists() or not dep_dir.exists():
        return {"orphan_images": [], "orphan_depths": []}

    img_stems = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    dep_stems = {p.stem: p for p in dep_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}

    orphan_images = sorted(set(img_stems) - set(dep_stems))
    orphan_depths = sorted(set(dep_stems) - set(img_stems))

    return {
        "orphan_images": [str(img_stems[s]) for s in orphan_images],
        "orphan_depths": [str(dep_stems[s]) for s in orphan_depths],
        "total_images": len(img_stems),
        "total_depths": len(dep_stems),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Find and remove orphaned image/depth files from a prepared SUN RGB-D dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", type=Path, default=Path("datasets/sunrgbd"), help="Prepared dataset root")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to check")
    ap.add_argument("--apply", action="store_true", help="Actually delete orphans (default: dry run only)")
    ap.add_argument("--report", type=Path, default=None, help="Write a JSON report of what was found/removed")
    args = ap.parse_args(argv)

    if not args.data.exists():
        LOG.error("Dataset root not found: %s", args.data.resolve())
        return 1

    report: dict[str, dict] = {}
    total_orphan_images = 0
    total_orphan_depths = 0

    for split in args.splits:
        if not (args.data / "images" / split).exists():
            continue
        result = find_orphans(args.data, split)
        report[split] = result

        n_img, n_dep = len(result["orphan_images"]), len(result["orphan_depths"])
        total_orphan_images += n_img
        total_orphan_depths += n_dep

        LOG.info(
            "Split '%s': %d image(s), %d depth file(s) on disk -> %d orphan image(s), %d orphan depth(s)",
            split,
            result.get("total_images", 0),
            result.get("total_depths", 0),
            n_img,
            n_dep,
        )

        if n_img:
            LOG.warning("  Sample orphan images (no matching depth): %s", result["orphan_images"][:3])
        if n_dep:
            LOG.warning("  Sample orphan depths (no matching image): %s", result["orphan_depths"][:3])

    total = total_orphan_images + total_orphan_depths
    print()
    if total == 0:
        LOG.info("No orphaned files found. Dataset is internally consistent.")
    else:
        LOG.warning(
            "Found %d orphaned file(s) total (%d images, %d depths).", total, total_orphan_images, total_orphan_depths
        )

        if not args.apply:
            LOG.warning("DRY RUN -- nothing deleted. Re-run with --apply to remove them.")
        else:
            removed = 0
            for result in report.values():
                for p in result["orphan_images"] + result["orphan_depths"]:
                    Path(p).unlink(missing_ok=True)
                    removed += 1
            LOG.info("Removed %d orphaned file(s).", removed)
            LOG.info("Next: python datasets/scripts/verify_dataset.py --data configs/data/sunrgbd.yaml")

    if args.report:
        save_json(report, args.report)
        LOG.info("Report written to %s", args.report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
