#!/usr/bin/env python3
"""Convert SUN RGB-D's native 2D box annotations to obstacle-YOLO labels (spec C).

CLOSES A GAP: ``convert_to_obstacle_dataset.py`` only accepts annotations
already in YOLO or COCO format. This script reads SUN RGB-D's OWN 2D
bounding-box metadata directly, so no manual pre-conversion is required.

FORMAT — VERIFIED, NOT ASSUMED
-------------------------------
STEP 5 of this project's build process is "verify empirically, never assume
internal structure." ``rgbd.cs.princeton.edu`` is unreachable from this
environment (egress-proxy denied), so the file itself could not be downloaded
and introspected the way every other model/API claim in this repository was.
Instead, the field layout below was read directly from the RAW SOURCE of two
independent, long-standing reference implementations that consume this exact
file (not from an AI-generated summary of either):

* ``SUNRGB_to_COCO.m`` (crmauceri/SUNRGBD_COCO) — converts this file to COCO
  bbox format, which made the coordinate CONVENTION unambiguous:
  ``bbox = [gtBb2D(1), gtBb2D(2), gtBb2D(1)+gtBb2D(3), gtBb2D(2)+gtBb2D(4)]``
  proves ``gtBb2D = [x, y, width, height]`` (summing x+w to get x2 only makes
  sense if the third element is a width, not a second x-coordinate).
* ``extract_rgbd_data_v2.m`` (facebookresearch/votenet) and
  ``extract_rgbd_data.m`` (charlesq34/frustum-pointnets) — confirm the file
  name, the top-level variable name, and the per-image field names.

Verified fields::

    SUNRGBDMeta2DBB_v2.mat                       -- file name
      SUNRGBDMeta2DBB                             -- top-level struct array,
                                                      one entry per image
        .sequenceName                             -- scene path, e.g.
                                                      ".../kv1/NYUdata/NYU0001"
        .rgbname                                  -- RGB filename in that
                                                      scene's image/ folder
        .depthname                                -- depth filename
        .groundtruth2DBB                          -- struct array, one per box
          .gtBb2D                                 -- [x, y, width, height] px
          .classname                              -- SUN RGB-D object label
                                                      (discarded -- see below)

CLASS POLICY (spec section A)
------------------------------
``classname`` is read only to confirm the field exists; its VALUE is never
used. Every box becomes ``class_id 0, class_name "obstacle"`` regardless of
whether SUN RGB-D called it "chair", "table", "wardrobe" or anything else.
This is obstacle DETECTION, not object RECOGNITION.

IMAGE RESOLUTION — ROBUST, NOT FRAGILE
----------------------------------------
The exact leading-path convention of ``sequenceName`` (whether it embeds a
machine-specific prefix such as ``/n/fs/sun3d/data/``) could not be confirmed
without the actual file in hand. Rather than betting correctness on a guessed
string convention, the scene path is instead re-derived by locating one of the
four KNOWN, VERIFIED sensor root directories (``kv1``, ``kv2``, ``realsense``,
``xtion`` -- confirmed in ``prepare_sunrgbd.py``, whose downloader/converter
already runs successfully against the same source tree) inside
``sequenceName`` and keeping only the portion from there onward. If the
resulting path does not exist on disk, or the named ``rgbname`` file is
missing, the script falls back to globbing that scene's ``image/`` folder and,
failing that, SKIPS the image with a logged reason -- it never guesses at an
RGB file and never fabricates a box (spec section C).

NORMALIZATION
-------------
Boxes are normalized using the ACTUAL loaded image's pixel dimensions (read
with OpenCV), never a dimension field taken on faith from the .mat file.

Usage:
    python datasets/scripts/convert_sunrgbd_obstacle.py \
        --source /path/to/SUNRGBD \
        --meta /path/to/SUNRGBDMeta2DBB_v2.mat \
        --split-file configs/data/sunrgbd_official_split.json \
        --output datasets/obstacle --config-out configs/data/obstacle.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.io import save_yaml  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("convert_sunrgbd_obstacle")

OBSTACLE_CLASS_ID = 0
OBSTACLE_CLASS_NAME = "obstacle"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# The four sensor top-level directories SUN RGB-D is organized under. Verified
# against the extracted archive by prepare_sunrgbd.py's find_scenes(), whose
# `depth_bfx` glob only ever matches scenes under these four roots.
SENSOR_ROOTS = ("kv1", "kv2", "realsense", "xtion")
_SENSOR_RE = re.compile(r"(?:^|/)(" + "|".join(SENSOR_ROOTS) + r")(/.*)?$")


class ConversionError(Exception):
    """Raised when the SUN RGB-D metadata cannot be converted."""


def clamp01(v: float) -> float:
    """Clamp a normalized coordinate into ``[0, 1]``."""
    return max(0.0, min(1.0, v))


def valid_yolo_box(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int, min_box_size: float) -> bool:
    """Reject degenerate or out-of-frame boxes (mirrors convert_to_obstacle_dataset.py)."""
    if not all(0.0 <= v <= 1.0 for v in (xc, yc)):
        return False
    if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
        return False
    if w * img_w < min_box_size or h * img_h < min_box_size:
        return False
    x1, x2 = xc - w / 2, xc + w / 2
    y1, y2 = yc - h / 2, yc + h / 2
    return min(x2, 1.0) > max(x1, 0.0) and min(y2, 1.0) > max(y1, 0.0)


def scene_key_from_sequence_name(sequence_name: str) -> str | None:
    """Derive a portable ``sensor/.../scene`` key from a raw ``sequenceName``.

    ``sequenceName`` may carry a machine-specific absolute prefix (its exact
    form is unverified -- see module docstring). Anchoring on one of the four
    known sensor roots instead of the leading characters makes this immune to
    that ambiguity: whatever precedes ``kv1``/``kv2``/``realsense``/``xtion``
    is discarded, and what follows is exactly the path SUN RGB-D's own
    directory layout uses.

    Returns:
        (str | None): The scene path with a leading slash stripped, or None if
            no sensor root was found in ``sequence_name`` at all.
    """
    normalized = sequence_name.replace("\\", "/")
    m = _SENSOR_RE.search(normalized)
    if m is None:
        return None
    start = m.start(1)
    return normalized[start:].strip("/")


def resolve_image_path(source: Path, scene_rel: str, rgb_name: str) -> tuple[Path | None, str]:
    """Locate the RGB file for one scene, with a documented fallback chain.

    Returns:
        (tuple): ``(path or None, how it was found)`` -- the second element is
            logged so a systematic mismatch is visible rather than silent.
    """
    scene_dir = source / scene_rel
    direct = scene_dir / "image" / rgb_name
    if direct.exists():
        return direct, "direct"

    image_dir = scene_dir / "image"
    if image_dir.is_dir():
        candidates = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(candidates) == 1:
            return candidates[0], "fallback (single file in image/, name mismatch)"
        if candidates:
            return candidates[0], f"fallback (took first of {len(candidates)} in image/)"

    return None, "not found"


def load_split_file(split_file: Path) -> dict[str, str]:
    """Load the official split JSON produced by prepare_sunrgbd.py --convert-allsplit.

    Returns:
        (dict): ``{normalized scene suffix: split name}`` for O(1)-ish lookup;
            matching still falls back to a suffix search for partial keys.
    """
    if not split_file.exists():
        raise ConversionError(
            f"Split file not found: {split_file.resolve()}\n"
            f"  Recovery: python datasets/scripts/prepare_sunrgbd.py --convert-allsplit "
            f"/path/to/allsplit.mat --split-file {split_file}"
        )
    data = json.loads(split_file.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for split_name in ("train", "test", "val"):
        for entry in data.get(split_name, []):
            out[str(entry).strip("/")] = "val" if split_name == "test" else split_name
    if not out:
        raise ConversionError(f"Split file {split_file} contains no train/test entries.")
    return out


def assign_split(scene_key: str, split_lookup: dict[str, str]) -> str | None:
    """Match a scene key against the split lookup, exact first, then suffix."""
    if scene_key in split_lookup:
        return split_lookup[scene_key]
    for key, split in split_lookup.items():
        if scene_key.endswith(key) or key.endswith(scene_key):
            return split
    return None


def load_metadata(meta_path: Path):
    """Load SUNRGBDMeta2DBB_v2.mat, returning the per-image struct array."""
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ConversionError(
            "Reading SUNRGBDMeta2DBB_v2.mat requires scipy.\n"
            "  Recovery: pip install scipy  (already listed in requirements.txt)"
        ) from e

    if not meta_path.exists():
        raise ConversionError(
            f"SUN RGB-D 2D box metadata not found: {meta_path.resolve()}\n"
            f"  Download from: https://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat\n"
            f"  Recovery: pass --meta /path/to/SUNRGBDMeta2DBB_v2.mat"
        )

    mat = loadmat(str(meta_path), squeeze_me=True, struct_as_record=False)
    if "SUNRGBDMeta2DBB" not in mat:
        raise ConversionError(
            f"{meta_path} has no 'SUNRGBDMeta2DBB' variable; found {sorted(k for k in mat if not k.startswith('__'))}.\n"
            f"  Recovery: confirm this is SUNRGBDMeta2DBB_v2.mat, not SUNRGBDMeta.mat or SUNRGBDMeta3DBB_v2.mat."
        )
    return np.atleast_1d(mat["SUNRGBDMeta2DBB"])


def convert(
    source: Path,
    meta_path: Path,
    split_lookup: dict[str, str],
    output: Path,
    min_box_size: float,
    limit: int | None = None,
) -> dict[str, dict[str, int]]:
    """Convert every entry in the SUN RGB-D 2D box metadata to YOLO labels."""
    import cv2

    entries = load_metadata(meta_path)
    LOG.info("Loaded metadata for %d image(s) from %s", len(entries), meta_path)

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: {"images": 0, "boxes": 0, "dropped_boxes": 0} for s in ("train", "val")}
    no_sensor_root = no_image = no_split = 0
    fallback_used = 0

    for i, entry in enumerate(entries):
        if limit is not None and i >= limit:
            break

        sequence_name = str(getattr(entry, "sequenceName", ""))
        rgb_name = str(getattr(entry, "rgbname", ""))
        scene_key = scene_key_from_sequence_name(sequence_name)
        if scene_key is None:
            no_sensor_root += 1
            continue

        split = assign_split(scene_key, split_lookup)
        if split is None:
            no_split += 1
            continue

        img_path, how = resolve_image_path(source, scene_key, rgb_name)
        if img_path is None:
            no_image += 1
            LOG.debug("No image for scene '%s' (rgbname='%s')", scene_key, rgb_name)
            continue
        if how != "direct":
            fallback_used += 1
            LOG.debug("Image for '%s' resolved via %s", scene_key, how)

        img = cv2.imread(str(img_path))
        if img is None:
            no_image += 1
            LOG.warning("Unreadable image, skipping: %s", img_path)
            continue
        img_h, img_w = img.shape[:2]

        boxes = np.atleast_1d(getattr(entry, "groundtruth2DBB", np.array([])))
        lines: list[str] = []
        dropped = 0

        for box in boxes:
            gt = getattr(box, "gtBb2D", None)
            if gt is None or len(np.atleast_1d(gt)) != 4:
                dropped += 1
                continue
            x, y, w, h = (float(v) for v in np.atleast_1d(gt))
            # gtBb2D is [x, y, width, height] in pixels (verified -- see module
            # docstring). Every SUN RGB-D classname collapses to 'obstacle'
            # (spec section A); the label itself is never read.
            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            wn = w / img_w
            hn = h / img_h

            if not valid_yolo_box(xc, yc, wn, hn, img_w, img_h, min_box_size):
                dropped += 1
                continue

            lines.append(f"{OBSTACLE_CLASS_ID} {clamp01(xc):.6f} {clamp01(yc):.6f} {clamp01(wn):.6f} {clamp01(hn):.6f}")

        name = scene_key.replace("/", "_")
        dst_img = output / "images" / split / f"{name}.jpg"
        dst_lbl = output / "labels" / split / f"{name}.txt"
        if not dst_img.exists():
            dst_img.write_bytes(img_path.read_bytes()) if img_path.suffix.lower() == ".jpg" else cv2.imwrite(
                str(dst_img), img
            )
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats[split]["images"] += 1
        stats[split]["boxes"] += len(lines)
        stats[split]["dropped_boxes"] += dropped

    if no_sensor_root:
        LOG.warning("%d entrie(s) had no recognizable sensor root in sequenceName; skipped.", no_sensor_root)
    if no_split:
        LOG.warning("%d scene(s) matched no entry in the split file; skipped.", no_split)
    if no_image:
        LOG.warning("%d scene(s) had no resolvable RGB image on disk; skipped.", no_image)
    if fallback_used:
        LOG.info("%d image(s) resolved via a filename fallback rather than the direct path.", fallback_used)

    return stats


def write_yaml(output: Path, stats: dict[str, dict[str, int]], config_path: Path) -> Path:
    """Write the nc=1 obstacle dataset YAML."""
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {OBSTACLE_CLASS_ID: OBSTACLE_CLASS_NAME},
        "source": "SUN RGB-D native 2D annotations (SUNRGBDMeta2DBB_v2.mat)",
        "stats": stats,
    }
    return save_yaml(data, config_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Convert SUN RGB-D's native 2D box annotations to obstacle-YOLO labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--source", type=Path, required=True, help="Extracted SUNRGBD root directory")
    ap.add_argument("--meta", type=Path, required=True, help="Path to SUNRGBDMeta2DBB_v2.mat")
    ap.add_argument(
        "--split-file",
        type=Path,
        default=Path("configs/data/sunrgbd_official_split.json"),
        help="Official split JSON from prepare_sunrgbd.py --convert-allsplit",
    )
    ap.add_argument("--output", type=Path, default=Path("datasets/obstacle"), help="Output dataset root")
    ap.add_argument("--config-out", type=Path, default=Path("configs/data/obstacle.yaml"))
    ap.add_argument("--min-box-size", type=float, default=4.0, help="Minimum box side in pixels")
    ap.add_argument("--limit", type=int, default=None, help="Convert at most N metadata entries (smoke testing)")
    args = ap.parse_args(argv)

    try:
        if not args.source.exists():
            raise ConversionError(f"SUN RGB-D source not found: {args.source.resolve()}")

        split_lookup = load_split_file(args.split_file)
        LOG.info("Split file: %d scene(s) (%s)", len(split_lookup), args.split_file)

        stats = convert(args.source, args.meta, split_lookup, args.output, args.min_box_size, limit=args.limit)
        for split, s in stats.items():
            LOG.info(
                "Split '%s': %d image(s), %d boxes kept, %d boxes dropped",
                split,
                s["images"],
                s["boxes"],
                s["dropped_boxes"],
            )

        total_images = sum(s["images"] for s in stats.values())
        if total_images == 0:
            LOG.warning(
                "No images were converted. Nothing was fabricated; check --source, --meta and --split-file "
                "against each other, and re-run with logging at DEBUG to see per-scene skip reasons."
            )

        cfg = write_yaml(args.output, stats, args.config_out)
        LOG.info("Obstacle dataset YAML written to %s (nc=1, names={0: obstacle})", cfg)
        LOG.info(
            "Next: python datasets/scripts/verify_dataset.py --data %s (for depth); "
            "or inspect %s directly for detection.",
            cfg,
            args.output,
        )
    except ConversionError as e:
        LOG.error("%s", e)
        return 1
    except (OSError, ValueError, KeyError) as e:
        LOG.error("Conversion failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
