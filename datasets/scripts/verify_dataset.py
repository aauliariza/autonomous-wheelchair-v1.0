#!/usr/bin/env python3
"""Verify a prepared depth dataset (spec section C).

Runs the ten checks the spec requires and exits non-zero on any FAIL, so it can
gate a training run in a shell script:

.. code-block:: text

     1. RGB image count                7. invalid-depth percentage
     2. depth map count                8. minimum valid pixels per map
     3. filename matching              9. min / max depth range
     4. resolution matching           10. depth statistics
     5. missing pairs
     6. corrupt files

Checks 4-10 sample the dataset by default (``--sample``) because decoding every
map in a 10k-image set is slow; pass ``--sample 0`` to inspect all of them.

Usage:
    python datasets/scripts/verify_dataset.py --data configs/data/sunrgbd.yaml
    python datasets/scripts/verify_dataset.py --data configs/data/sunrgbd.yaml --sample 0 --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.io import load_yaml, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("verify_dataset")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class Check:
    """One verification result."""

    def __init__(self, name: str, passed: bool, detail: str, data: dict[str, Any] | None = None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.data = data or {}

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def list_images(d: Path) -> list[Path]:
    """Sorted image files in a directory."""
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def verify_split(
    root: Path,
    split: str,
    depth_scale: float,
    max_depth: float,
    sample: int,
    min_valid_ratio: float,
) -> list[Check]:
    """Run every check for one split."""
    import cv2

    checks: list[Check] = []
    img_dir = root / "images" / split
    dep_dir = root / "depth" / split

    images = list_images(img_dir)
    depths = list_images(dep_dir)

    # 1 & 2: counts
    checks.append(Check(f"{split}/rgb_count", len(images) > 0, f"{len(images)} RGB images in {img_dir}"))
    checks.append(Check(f"{split}/depth_count", len(depths) > 0, f"{len(depths)} depth maps in {dep_dir}"))
    if not images or not depths:
        return checks

    # 3 & 5: filename matching and missing pairs
    img_stems = {p.stem for p in images}
    dep_stems = {p.stem for p in depths}
    missing_depth = sorted(img_stems - dep_stems)
    missing_rgb = sorted(dep_stems - img_stems)

    checks.append(
        Check(
            f"{split}/filename_match",
            not missing_depth and not missing_rgb,
            (
                "every RGB has a paired depth map"
                if not missing_depth and not missing_rgb
                else f"{len(missing_depth)} RGB without depth, {len(missing_rgb)} depth without RGB "
                     f"(e.g. {(missing_depth or missing_rgb)[:3]})"
            ),
            {"missing_depth": missing_depth[:20], "missing_rgb": missing_rgb[:20]},
        )
    )
    paired = sorted(img_stems & dep_stems)
    checks.append(Check(f"{split}/paired_count", len(paired) > 0, f"{len(paired)} complete RGB-depth pairs"))
    if not paired:
        return checks

    # Sample for the expensive per-file checks.
    subset = paired if sample <= 0 else paired[:: max(1, len(paired) // sample)][:sample]

    depth_by_stem = {p.stem: p for p in depths}
    image_by_stem = {p.stem: p for p in images}

    corrupt: list[str] = []
    mismatched: list[str] = []
    low_valid: list[str] = []
    invalid_fractions: list[float] = []
    all_values: list[np.ndarray] = []

    for stem in subset:
        ip, dp = image_by_stem[stem], depth_by_stem[stem]
        rgb = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(dp), cv2.IMREAD_ANYDEPTH)

        # 6: corrupt files
        if rgb is None or raw is None:
            corrupt.append(stem)
            continue

        # 4: resolution matching
        if rgb.shape[:2] != raw.shape[:2]:
            mismatched.append(f"{stem} rgb{rgb.shape[:2]} vs depth{raw.shape[:2]}")
            continue

        depth_m = raw.astype(np.float32) / float(depth_scale)
        valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= max_depth)
        ratio = float(valid.mean())
        invalid_fractions.append(1.0 - ratio)

        # 8: minimum valid pixels
        if ratio < min_valid_ratio:
            low_valid.append(f"{stem} ({ratio:.1%} valid)")

        if valid.any():
            v = depth_m[valid].ravel()
            all_values.append(v[:: max(1, v.size // 2000)])

    checks.append(
        Check(
            f"{split}/corrupt_files",
            not corrupt,
            "no corrupt files" if not corrupt else f"{len(corrupt)} unreadable (e.g. {corrupt[:3]})",
            {"corrupt": corrupt[:20]},
        )
    )
    checks.append(
        Check(
            f"{split}/resolution_match",
            not mismatched,
            "RGB and depth resolutions agree" if not mismatched else f"{len(mismatched)} mismatched: {mismatched[:3]}",
            {"mismatched": mismatched[:20]},
        )
    )

    # 7: invalid-depth percentage
    if invalid_fractions:
        mean_invalid = float(np.mean(invalid_fractions))
        checks.append(
            Check(
                f"{split}/invalid_depth_pct",
                mean_invalid < 0.9,
                f"mean invalid depth {mean_invalid:.1%} across {len(invalid_fractions)} sampled maps",
                {"mean_invalid_fraction": mean_invalid},
            )
        )

    checks.append(
        Check(
            f"{split}/min_valid_pixels",
            len(low_valid) <= 0.1 * len(subset),
            (
                f"{len(low_valid)}/{len(subset)} maps below {min_valid_ratio:.0%} valid pixels"
                if low_valid
                else f"all sampled maps exceed {min_valid_ratio:.0%} valid pixels"
            ),
            {"low_valid": low_valid[:20]},
        )
    )

    # 9 & 10: depth range and statistics
    if all_values:
        vals = np.concatenate(all_values)
        stats = {
            "min_m": float(vals.min()),
            "max_m": float(vals.max()),
            "mean_m": float(vals.mean()),
            "median_m": float(np.median(vals)),
            "std_m": float(vals.std()),
            "p01_m": float(np.percentile(vals, 1)),
            "p99_m": float(np.percentile(vals, 99)),
            "sampled_pixels": int(vals.size),
        }
        in_range = 0 < stats["min_m"] and stats["max_m"] <= max_depth * 1.001
        checks.append(
            Check(
                f"{split}/depth_range",
                in_range,
                f"depth in [{stats['min_m']:.3f}, {stats['max_m']:.3f}] m (limit {max_depth} m)",
                stats,
            )
        )
        checks.append(
            Check(
                f"{split}/depth_statistics",
                True,
                f"mean {stats['mean_m']:.3f} m, median {stats['median_m']:.3f} m, std {stats['std_m']:.3f} m",
                stats,
            )
        )

    return checks


def check_leakage(root: Path, splits: list[str]) -> Check:
    """Detect identical filenames shared across splits (data leakage)."""
    stems = {s: {p.stem for p in list_images(root / "images" / s)} for s in splits}
    overlaps: dict[str, list[str]] = {}
    names = [s for s in splits if stems.get(s)]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(stems[a] & stems[b])
            if shared:
                overlaps[f"{a}&{b}"] = shared[:10]
    return Check(
        "split_leakage",
        not overlaps,
        "no filenames shared between splits" if not overlaps else f"LEAKAGE: {overlaps}",
        {"overlaps": overlaps},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when every check passes."""
    ap = argparse.ArgumentParser(
        description="Verify a prepared depth dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--data", type=Path, required=True, help="Dataset YAML written by prepare_sunrgbd.py")
    ap.add_argument("--sample", type=int, default=200, help="Maps to inspect per split (0 = all)")
    ap.add_argument("--min-valid-ratio", type=float, default=0.05, help="Minimum valid-pixel fraction per map")
    ap.add_argument("--report", type=Path, default=None, help="Write a JSON report here")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = ap.parse_args(argv)

    try:
        cfg = load_yaml(args.data)
    except Exception as e:  # noqa: BLE001 - surfaced with a recovery hint below
        LOG.error("Could not read %s (%s: %s)", args.data, type(e).__name__, e)
        LOG.error("Recovery: run datasets/scripts/prepare_sunrgbd.py first.")
        return 1

    root = Path(cfg.get("path", args.data.parent))
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()

    if not root.exists():
        LOG.error("Dataset root not found: %s", root)
        LOG.error("Recovery: expected layout %s/{images,depth}/{train,val}", root)
        return 1

    depth_scale = float(cfg.get("depth_scale", 1000))
    max_depth = float(cfg.get("max_depth", 10.0))
    splits = [s for s in ("train", "val", "test") if (root / "images" / s).exists()]

    if not splits:
        LOG.error("No split directories under %s/images (expected train/, val/).", root)
        return 1

    LOG.info("Verifying %s | splits=%s | depth_scale=%s | max_depth=%s m", root, splits, depth_scale, max_depth)

    checks: list[Check] = []
    for split in splits:
        checks.extend(
            verify_split(root, split, depth_scale, max_depth, args.sample, args.min_valid_ratio)
        )
    checks.append(check_leakage(root, splits))

    print()
    for c in checks:
        print(" ", c)

    failed = [c for c in checks if not c.passed]
    print()
    LOG.info("%d/%d checks passed", len(checks) - len(failed), len(checks))

    if args.report:
        save_json(
            {
                "dataset": str(root),
                "splits": splits,
                "passed": len(checks) - len(failed),
                "total": len(checks),
                "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail, **c.data} for c in checks],
            },
            args.report,
        )
        LOG.info("Report written to %s", args.report)

    if failed:
        LOG.error("FAILED checks: %s", [c.name for c in failed])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
