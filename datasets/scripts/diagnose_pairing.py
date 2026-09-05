#!/usr/bin/env python3
"""Print exactly why Ultralytics does or does not find a depth companion.

WHY THIS EXISTS
---------------
``training/common.py:verify_depth_pairing`` applies Ultralytics' own resolution
rule and passed on a dataset that Ultralytics then rejected image by image. Two
runs cannot both be right about the same files, so this reports the raw facts —
which code is checked out, which YAML is read, what the resolver returns, and
what the filesystem says about that exact path — instead of inferring from a
truncated training log.

It only READS. Nothing is created, moved or deleted.

Usage:
    python datasets/scripts/diagnose_pairing.py --data configs/data/sunrgbd.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def git_head(repo: Path) -> str:
    """Return the checked-out commit, so a missing 'git pull' is visible."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "--oneline", "-1"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or f"(git said: {out.stderr.strip()})"
    except (OSError, subprocess.SubprocessError) as e:
        return f"(unavailable: {e})"


def describe_png(path: Path) -> str:
    """Report the fields verify_image_depth asserts on a PNG depth map."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            ok = im.format == "PNG" and im.mode in {"I", "I;16"}
            return f"format={im.format} mode={im.mode} size={im.size} accepted={ok}"
    except (OSError, ValueError) as e:
        return f"unreadable: {type(e).__name__}: {e}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Diagnose RGB-depth pairing.")
    ap.add_argument("--data", type=Path, default=repo / "configs/data/sunrgbd.yaml")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args(argv)

    print(f"cwd          : {Path.cwd()}")
    print(f"repo         : {repo}")
    print(f"git HEAD     : {git_head(repo)}")
    print(f"data yaml    : {args.data.resolve()}  exists={args.data.is_file()}")
    if not args.data.is_file():
        print("  -> that YAML does not exist; pass the right --data path.")
        return 1

    from training.common import ultralytics_depth_path  # noqa: E402
    from utils.io import load_yaml  # noqa: E402

    cfg = load_yaml(args.data)
    root = Path(cfg.get("path", args.data.parent))
    print(f"path: entry  : {cfg.get('path')!r}  -> {root}  is_dir={root.is_dir()}")
    print(f"depth_scale  : {cfg.get('depth_scale')!r}")

    for split in ("train", "val"):
        img_dir, dep_dir = root / "images" / split, root / "depth" / split
        print(f"\n--- {split} ---")
        print(f"images dir   : {img_dir}  is_dir={img_dir.is_dir()}")
        print(f"depth  dir   : {dep_dir}  is_dir={dep_dir.is_dir()}")
        if not img_dir.is_dir():
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        depths = sorted(dep_dir.iterdir()) if dep_dir.is_dir() else []
        counts: dict[str, int] = {}
        for d in depths:
            counts[d.suffix] = counts.get(d.suffix, 0) + 1
        print(f"image count  : {len(images)}")
        print(f"depth entries: {len(depths)}  by suffix: {counts or 'NONE'}")

        cache = root / "depth" / f"{split}.cache"
        print(f"stale cache  : {cache}  exists={cache.is_file()}"
              + (f"  size={cache.stat().st_size}B" if cache.is_file() else ""))

        resolved = [(p, ultralytics_depth_path(p)) for p in images[: args.samples]]
        for img, dep in resolved:
            print(f"\n  image      : {img.name}")
            if dep is None:
                want = img.with_suffix(".png").name
                print(f"  resolver   : None  (neither {want} nor its .npy is a file)")
                cand = (dep_dir / want)
                print(f"  os.path.isfile({cand}) = {os.path.isfile(cand)}")
            else:
                print(f"  resolver   : {dep.name}")
                print(f"  isfile     : {os.path.isfile(dep)}")
                if dep.suffix.lower() == ".png":
                    print(f"  png fields : {describe_png(dep)}")

        missing = sum(1 for p in images if ultralytics_depth_path(p) is None)
        print(f"\n  unpaired   : {missing} of {len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
