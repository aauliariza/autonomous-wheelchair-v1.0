#!/usr/bin/env python3
"""Prepare SUN RGB-D for YOLO26-Depth training (spec section C).

WHY THIS SCRIPT EXISTS
----------------------
Ultralytics already ships ``depth-sunrgbd.yaml`` with a working downloader and
the correct depth decode. This script does NOT reimplement that. It exists for
one substantive reason: the shipped config selects its validation set as a
**random 1090-scene sample (seed 0)**, ignoring SUN RGB-D's official train/test
partition.

That matters for publication:

* SUN RGB-D contains multiple frames of the same physical scene. A random split
  can place near-duplicate views of one room in both train and val, inflating
  every metric through scene-level leakage.
* Results computed on a private random split are not comparable to any published
  SUN RGB-D number.

``--split official`` (the DEFAULT) uses the official partition distributed with
the SUN RGB-D toolbox. ``--split ultralytics`` reproduces the shipped random
split for direct comparability with Ultralytics-reported figures.

DEPTH DECODING
--------------
SUN RGB-D stores refined depth in ``depth_bfx/`` as uint16 PNGs with a 3-bit
rotation applied. The decode, verified against the Ultralytics converter, is:

.. code-block:: python

    depth_mm = (d >> 3) | (d << 13)
    depth_m  = depth_mm / 1000.0

Output is written as uint16 millimetre PNGs (``depth_scale: 1000``), the format
the Ultralytics depth dataloader expects.

OUTPUT LAYOUT (spec section C)
------------------------------
.. code-block:: text

    datasets/sunrgbd/
        images/{train,val,test}/
        depth/{train,val,test}/

Usage:
    python datasets/scripts/prepare_sunrgbd.py --source /path/to/SUNRGBD --output datasets/sunrgbd
    python datasets/scripts/prepare_sunrgbd.py --source /path/to/SUNRGBD --split ultralytics
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.io import save_yaml  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("prepare_sunrgbd")

# Sensor saturation. SUN RGB-D uses structured-light and early ToF sensors whose
# returns beyond ~10 m are unreliable; the Ultralytics converter clips here too.
MAX_DEPTH_M = 10.0
DEPTH_SCALE = 1000  # PNG value 1000 == 1 metre


class DatasetError(Exception):
    """Raised when the SUN RGB-D source tree is missing or malformed."""


def decode_sunrgbd_depth(raw: np.ndarray) -> np.ndarray:
    """Decode a SUN RGB-D ``depth_bfx`` PNG to metres.

    SUN RGB-D applies a 3-bit rotation to pack depth into uint16. Reading the
    raw value as millimetres directly — a common mistake — yields depths that
    are wrong by orders of magnitude for most pixels.

    Args:
        raw (np.ndarray): uint16 array as read with ``cv2.IMREAD_ANYDEPTH``.

    Returns:
        (np.ndarray): float32 depth in metres, 0 where invalid, clipped to 10 m.
    """
    d = raw.astype(np.uint16)
    depth_mm = (d >> 3) | (d << 13)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0] = 0.0
    return np.clip(depth_m, 0.0, MAX_DEPTH_M)


def find_scenes(source: Path) -> list[Path]:
    """Find every SUN RGB-D scene directory (those containing ``depth_bfx``)."""
    if not source.exists():
        raise DatasetError(
            f"SUN RGB-D source not found: {source.resolve()}\n"
            f"  Expected the extracted SUNRGBD directory (from SUNRGBD.zip).\n"
            f"  Download: https://rgbd.cs.princeton.edu/data/SUNRGBD.zip\n"
            f"  Recovery: pass --source /path/to/SUNRGBD"
        )
    scenes = sorted(p.parent for p in source.rglob("depth_bfx"))
    if not scenes:
        raise DatasetError(
            f"No scenes with a 'depth_bfx' directory under {source.resolve()}.\n"
            f"  Recovery: point --source at the extracted SUNRGBD root, which should "
            f"contain kv1/, kv2/, realsense/ and xtion/ subdirectories."
        )
    return scenes


def scene_name(scene: Path, source: Path) -> str:
    """Stable, filesystem-safe identifier for a scene."""
    return "_".join(scene.relative_to(source).parts)


def convert_allsplit(mat_path: Path, output: Path) -> Path:
    """Convert the SUN RGB-D toolbox ``allsplit.mat`` into the JSON this script reads.

    ``allsplit.mat`` (SUNRGBDtoolbox/traintestSUNRGBD/) stores two cell arrays of
    absolute scene paths, ``alltrain`` and ``alltest``. They are rooted at the
    author's machine (``/n/fs/sun3d/data/SUNRGBD/...``), so only the portion from
    ``SUNRGBD/`` onward is portable; that suffix is what gets matched against the
    discovered scene tree.

    Requires scipy, which is already a project dependency.
    """
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise DatasetError(
            "Reading allsplit.mat requires scipy, which is not installed.\n"
            "  Recovery: pip install scipy  (it is already listed in requirements.txt)"
        ) from e

    if not Path(mat_path).exists():
        raise DatasetError(
            f"allsplit.mat not found: {Path(mat_path).resolve()}\n"
            f"  Obtain it from https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip\n"
            f"  (SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat)"
        )

    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    def entries(*keys: str) -> list[str]:
        for k in keys:
            if k in mat:
                raw = mat[k]
                items = [str(x) for x in np.atleast_1d(raw)]
                out = []
                for it in items:
                    it = it.strip().rstrip("/")
                    # Keep only the portable tail beginning at SUNRGBD/.
                    idx = it.find("SUNRGBD/")
                    out.append(it[idx + len("SUNRGBD/") :] if idx >= 0 else it)
                return [x for x in out if x]
        raise DatasetError(f"allsplit.mat has none of the expected keys {keys}; found {sorted(mat)}.")

    data = {"train": entries("alltrain", "trainvalsplit", "train"), "test": entries("alltest", "test")}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOG.info("Wrote official split: %d train, %d test -> %s", len(data["train"]), len(data["test"]), output)
    return output


def load_official_split(split_file: Path | None, scenes: list[Path], source: Path) -> dict[str, set[str]]:
    """Load the official SUN RGB-D train/test partition.

    The official split ships with the SUN RGB-D toolbox as ``allsplit.mat``,
    which lists absolute scene paths for train and test. Because that file is a
    MATLAB struct, this function accepts a pre-extracted JSON or plain-text
    listing so the pipeline has no MATLAB dependency:

    * JSON: ``{"train": [...], "test": [...]}`` of scene names or path suffixes.
    * Text: one scene path per line, with train/test given as two separate files.

    Raises:
        DatasetError: If the file is missing, so the caller must make an explicit
            choice rather than silently falling back to a random split.
    """
    if split_file is None or not Path(split_file).exists():
        raise DatasetError(
            "Official split file not found.\n"
            "  --split official requires the SUN RGB-D toolbox partition (allsplit.mat),\n"
            "  converted to JSON as {\"train\": [...], \"test\": [...]} and passed via --split-file.\n"
            "  Obtain it from https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip\n"
            "    (SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat)\n"
            "  Convert with:\n"
            "    python datasets/scripts/prepare_sunrgbd.py --convert-allsplit /path/to/allsplit.mat \\\n"
            "        --split-file configs/data/sunrgbd_official_split.json\n"
            "  Recovery: supply --split-file, or choose --split ultralytics to reproduce the\n"
            "  Ultralytics random split instead (NOT recommended for publication)."
        )

    path = Path(split_file)
    names = {scene_name(s, source): s for s in scenes}

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        train_raw = data.get("train", [])
        test_raw = data.get("test", data.get("val", []))
    else:
        raise DatasetError(f"Unsupported split file format '{path.suffix}'. Provide a .json file.")

    def match(entries: list[str]) -> set[str]:
        """Match split entries to discovered scenes by suffix."""
        out: set[str] = set()
        for raw in entries:
            key = str(raw).strip().strip("/").replace("/", "_")
            if key in names:
                out.add(key)
                continue
            # allsplit.mat stores absolute paths; match on the trailing segments.
            hits = [n for n in names if n.endswith(key) or key.endswith(n)]
            if len(hits) == 1:
                out.add(hits[0])
        return out

    train, test = match(train_raw), match(test_raw)
    if not train or not test:
        raise DatasetError(
            f"Official split matched {len(train)} train and {len(test)} test scenes against "
            f"{len(scenes)} discovered scenes — at least one side is empty.\n"
            f"  Recovery: verify --split-file entries correspond to the --source tree."
        )
    overlap = train & test
    if overlap:
        raise DatasetError(f"Official split is inconsistent: {len(overlap)} scenes appear in BOTH train and test.")
    return {"train": train, "val": test}


def ultralytics_split(scenes: list[Path], source: Path, num_val: int = 1090, seed: int = 0) -> dict[str, set[str]]:
    """Reproduce the shipped Ultralytics random split (seed 0, 1090 val scenes).

    Provided for comparability with Ultralytics-reported numbers only. It carries
    the scene-level leakage risk described in the module docstring.
    """
    names = [scene_name(s, source) for s in scenes]
    val = set(random.Random(seed).sample(names, k=min(num_val, len(names))))
    train = set(names) - val
    LOG.warning(
        "Using the ULTRALYTICS RANDOM split (seed %d, %d val scenes). This ignores the official "
        "SUN RGB-D partition and may leak near-duplicate views of the same room across train/val. "
        "Prefer --split official for published results.",
        seed, len(val),
    )
    return {"train": train, "val": val}


def convert(
    source: Path,
    output: Path,
    splits: dict[str, set[str]],
    scenes: list[Path],
    limit: int | None = None,
) -> dict[str, int]:
    """Convert scenes into the images/depth layout, returning per-split counts."""
    import cv2

    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "depth" / split).mkdir(parents=True, exist_ok=True)

    lookup = {scene_name(s, source): s for s in scenes}
    counts = dict.fromkeys(splits, 0)
    skipped = 0

    for split, members in splits.items():
        for i, name in enumerate(sorted(members)):
            if limit is not None and i >= limit:
                break
            scene = lookup.get(name)
            if scene is None:
                continue

            depth_files = sorted((scene / "depth_bfx").glob("*.png"))
            image_files = sorted((scene / "image").glob("*.jpg")) + sorted((scene / "image").glob("*.png"))
            if not depth_files or not image_files:
                skipped += 1
                continue

            raw = cv2.imread(str(depth_files[0]), cv2.IMREAD_ANYDEPTH)
            if raw is None:
                LOG.warning("Unreadable depth file, skipping scene: %s", depth_files[0])
                skipped += 1
                continue

            depth_m = decode_sunrgbd_depth(raw)
            rgb = cv2.imread(str(image_files[0]), cv2.IMREAD_COLOR)
            if rgb is None:
                LOG.warning("Unreadable RGB file, skipping scene: %s", image_files[0])
                skipped += 1
                continue

            # Depth and RGB must be pixel-aligned; SUN RGB-D pairs already are,
            # but a sensor mismatch would silently corrupt every metric.
            if rgb.shape[:2] != depth_m.shape[:2]:
                depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            png = np.clip(depth_m * DEPTH_SCALE, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(output / "images" / split / f"{name}.jpg"), rgb)
            cv2.imwrite(str(output / "depth" / split / f"{name}.png"), png)
            counts[split] += 1

    if skipped:
        LOG.warning("Skipped %d scene(s) with missing or unreadable files.", skipped)
    return counts


def write_data_yaml(output: Path, counts: dict[str, int], split_mode: str, config_path: Path) -> Path:
    """Write the Ultralytics-compatible dataset YAML."""
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "depth"},
        "channels": 3,
        "depth_scale": DEPTH_SCALE,
        "max_depth": MAX_DEPTH_M,
        "split_mode": split_mode,
        "counts": counts,
    }
    if (output / "images" / "test").exists() and counts.get("test"):
        data["test"] = "images/test"
    return save_yaml(data, config_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Prepare SUN RGB-D for YOLO26-Depth training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--source", type=Path, default=None, help="Extracted SUNRGBD root directory")
    ap.add_argument(
        "--convert-allsplit",
        type=Path,
        default=None,
        help="Convert the toolbox allsplit.mat to the JSON --split-file format, then exit",
    )
    ap.add_argument("--output", type=Path, default=Path("datasets/sunrgbd"), help="Output dataset root")
    ap.add_argument(
        "--split",
        choices=["official", "ultralytics"],
        default="official",
        help="Split strategy. 'official' avoids scene-level leakage and is required for publication.",
    )
    ap.add_argument("--split-file", type=Path, default=None, help="JSON with the official train/test scene lists")
    ap.add_argument("--config-out", type=Path, default=Path("configs/data/sunrgbd.yaml"), help="Dataset YAML to write")
    ap.add_argument("--limit", type=int, default=None, help="Convert at most N scenes per split (smoke testing)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --split ultralytics")
    args = ap.parse_args(argv)

    try:
        if args.convert_allsplit is not None:
            out = args.split_file or Path("configs/data/sunrgbd_official_split.json")
            convert_allsplit(args.convert_allsplit, out)
            LOG.info("Next: re-run with --source /path/to/SUNRGBD --split-file %s", out)
            return 0

        if args.source is None:
            LOG.error("--source is required (or use --convert-allsplit). See --help.")
            return 1

        scenes = find_scenes(args.source)
        LOG.info("Discovered %d scenes under %s", len(scenes), args.source)

        if args.split == "official":
            splits = load_official_split(args.split_file, scenes, args.source)
        else:
            splits = ultralytics_split(scenes, args.source, seed=args.seed)

        LOG.info("Split '%s': %s", args.split, {k: len(v) for k, v in splits.items()})

        counts = convert(args.source, args.output, splits, scenes, limit=args.limit)
        LOG.info("Converted: %s", counts)

        cfg = write_data_yaml(args.output, counts, args.split, args.config_out)
        LOG.info("Dataset config written to %s", cfg)
        LOG.info("Next: python datasets/scripts/verify_dataset.py --data %s", cfg)
    except DatasetError as e:
        LOG.error("%s", e)
        return 1
    except (OSError, ValueError) as e:
        LOG.error("Dataset preparation failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
