#!/usr/bin/env python3
"""Profile an obstacle-detection label set to explain its detection ceiling.

``convert_sunrgbd_obstacle.py`` keeps EVERY annotated object (its docstring says
``classname`` is read but never used) and drops only boxes under
``--min-box-size`` pixels. SUN RGB-D annotations are object annotations, not
curated navigation obstacles, so the resulting class mixes a wardrobe with a
picture on a wall. That mixture — not the detector — is usually what caps mAP:
the label is "something a human boxed", which has no consistent visual
signature, and a correct detection of an unlabelled object still scores as a
false positive.

This reports what is actually in the labels, so the ceiling can be reasoned
about from data:

* boxes per image, and the box-area distribution
* how many boxes are tiny, which mAP50-95 punishes hardest
* how many sit high in the frame — wall-mounted things a wheelchair drives
  under, which are not floor-path obstacles
* how many boxes would survive some candidate filters

Read-only. It changes nothing.

Usage:
    python datasets/scripts/analyze_obstacle_labels.py --labels datasets/obstacle/labels/train
"""

from __future__ import annotations

import argparse
from pathlib import Path


def read_boxes(labels_dir: Path) -> tuple[list[tuple[float, float, float, float]], int]:
    """Return every YOLO box as (xc, yc, w, h) plus the label-file count."""
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Label directory not found: {labels_dir.resolve()}\n"
            f"  Recovery: python datasets/scripts/convert_sunrgbd_obstacle.py ... --output datasets/obstacle"
        )
    boxes: list[tuple[float, float, float, float]] = []
    files = sorted(labels_dir.glob("*.txt"))
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                boxes.append(tuple(float(v) for v in parts[1:5]))  # type: ignore[arg-type]
            except ValueError:
                continue
    return boxes, len(files)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, so no numpy dependency is needed."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def profile(boxes: list[tuple[float, float, float, float]], n_files: int) -> dict[str, float]:
    """Summarize box geometry and the share caught by candidate filters."""
    areas = [w * h for _, _, w, h in boxes]
    n = len(boxes)
    if n == 0:
        return {"boxes": 0, "images": n_files}
    # A box whose bottom edge stays in the top 40% of the frame is above the
    # floor plane: wall art, windows, ceiling lamps. A wheelchair passes under it.
    high = sum(1 for _, yc, _, h in [(b[0], b[1], b[2], b[3]) for b in boxes] if yc + h / 2 < 0.40)
    return {
        "images": n_files,
        "boxes": n,
        "per_image": n / max(n_files, 1),
        "area_p10": percentile(areas, 10),
        "area_p50": percentile(areas, 50),
        "area_p90": percentile(areas, 90),
        "tiny_lt_0p1pct": sum(1 for a in areas if a < 0.001) / n,
        "small_lt_1pct": sum(1 for a in areas if a < 0.01) / n,
        "high_in_frame": high / n,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Profile obstacle detection labels.")
    ap.add_argument("--labels", type=Path, required=True, help="labels/<split> directory")
    args = ap.parse_args(argv)

    try:
        boxes, n_files = read_boxes(args.labels)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    p = profile(boxes, n_files)
    if not p.get("boxes"):
        print(f"No boxes found in {args.labels}.")
        return 1

    print(f"label files          : {p['images']}")
    print(f"boxes                : {p['boxes']}  ({p['per_image']:.2f} per image)")
    print("box area (fraction of the image):")
    print(f"  p10 {p['area_p10']:.5f}   p50 {p['area_p50']:.5f}   p90 {p['area_p90']:.5f}")
    print(f"boxes under 0.1% of the image : {p['tiny_lt_0p1pct'] * 100:5.1f}%   (mAP50-95 punishes these hardest)")
    print(f"boxes under 1%   of the image : {p['small_lt_1pct'] * 100:5.1f}%")
    print(f"boxes entirely in the top 40% : {p['high_in_frame'] * 100:5.1f}%   (wall/ceiling: a wheelchair drives under)")
    print()
    print("These are OBJECT annotations, not curated navigation obstacles. A high")
    print("share of tiny or high-in-frame boxes caps mAP and, worse, teaches the")
    print("detector to report things that never block the floor path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
