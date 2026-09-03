#!/usr/bin/env python3
"""Camera intrinsic calibration (spec section U).

Estimates ``fx, fy, cx, cy`` and the distortion coefficients from checkerboard or
ChArUco images and writes them to ``configs/camera.yaml``.

WHY THIS MATTERS HERE
---------------------
Without real intrinsics, Euclidean obstacle distance is not merely approximate --
it is geometrically undefined. The pipeline refuses to report Euclidean distance
from placeholder values, and this script is what turns ``calibrated: false`` into
``true``.

QUALITY GUIDANCE
----------------
The RMS reprojection error is reported and stored. Below ~0.5 px is good; above
~1.0 px usually means too few views, a poorly-flat board, or an inaccurate
``--square-size``. Capture 15-30 views covering the whole frame, including the
corners (where the distortion model is least constrained), at varied angles and
distances.

Usage:
    python calibration/camera_calibration.py --images "calib/*.jpg" --pattern-size 9 6 \
        --square-size 0.025 --output configs/camera.yaml
    python calibration/camera_calibration.py --capture 0 --pattern-size 9 6 --square-size 0.025
"""

from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import save_yaml  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("camera_calibration")


class CalibrationError(Exception):
    """Raised when calibration cannot be completed."""


def find_checkerboard_corners(images: list[Path], pattern_size: tuple[int, int], square_size: float):
    """Detect checkerboard corners across a set of images.

    Returns:
        (tuple): ``(object_points, image_points, image_size, used_files)``.
    """
    import cv2

    cols, rows = pattern_size

    # Board coordinates in metres: Z=0 plane, scaled by the real square size.
    # A wrong --square-size scales the whole reconstruction, so intrinsics are
    # right but any derived metric distance is proportionally wrong.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    obj_points, img_points, used = [], [], []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            LOG.warning("Unreadable image, skipping: %s", path)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            # Mixing resolutions would fit one focal length to two sensors.
            LOG.warning("Skipping %s: size %s differs from %s", path, (gray.shape[1], gray.shape[0]), image_size)
            continue

        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            LOG.debug("No pattern found in %s", path.name)
            continue

        # Sub-pixel refinement materially improves the fit.
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(refined)
        used.append(path)

    return obj_points, img_points, image_size, used


def capture_images(camera: int, pattern_size: tuple[int, int], out_dir: Path, target: int = 20) -> list[Path]:
    """Interactively capture calibration frames from a live camera."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise CalibrationError(f"Could not open camera {camera}.")

    cols, rows = pattern_size
    saved: list[Path] = []
    LOG.info("SPACE saves a frame when the pattern is detected; q quits. Target: %d views.", target)

    try:
        while len(saved) < target:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(
                display,
                f"saved {len(saved)}/{target}  pattern={'YES' if found else 'no'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow("calibration capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and found:
                path = out_dir / f"calib_{len(saved):03d}.jpg"
                cv2.imwrite(str(path), frame)
                saved.append(path)
                LOG.info("Saved %s", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved


def calibrate(obj_points, img_points, image_size) -> dict:
    """Run OpenCV's intrinsic calibration and compute per-view reprojection error."""
    import cv2

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    # Per-view error localises a bad capture; a single outlier view often
    # dominates the RMS and is worth removing and re-running.
    per_view = []
    for i, (op, ip) in enumerate(zip(obj_points, img_points, strict=True)):
        projected, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
        # Corner arrays come back as (N,1,2) or (N,2) depending on the OpenCV
        # version, and cv2.norm rejects a channel-count mismatch. Reducing both
        # to a flat (N,2) float64 array makes the error version-independent.
        a = np.asarray(ip, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        per_view.append(float(np.sqrt(((a - b) ** 2).sum(axis=1)).mean()))

    return {
        "rms": float(rms),
        "K": K,
        "dist": dist.ravel().tolist(),
        "per_view_error": per_view,
        "mean_view_error": float(np.mean(per_view)),
        "worst_view_error": float(np.max(per_view)),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from checkerboard images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images", default=None, help="Glob of calibration images, e.g. 'calib/*.jpg'")
    ap.add_argument("--capture", type=int, default=None, help="Capture from this camera index first")
    ap.add_argument("--capture-dir", type=Path, default=Path("outputs/calibration"))
    ap.add_argument("--pattern-size", type=int, nargs=2, default=[9, 6], help="INNER corners (cols rows), not squares")
    ap.add_argument("--square-size", type=float, default=0.025, help="Square side in METRES")
    ap.add_argument("--min-images", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("configs/camera.yaml"))
    args = ap.parse_args(argv)

    try:
        pattern = (args.pattern_size[0], args.pattern_size[1])

        if args.capture is not None:
            images = capture_images(args.capture, pattern, args.capture_dir)
        elif args.images:
            images = [Path(p) for p in sorted(glob.glob(args.images))]
        else:
            raise CalibrationError("Provide either --images '<glob>' or --capture <camera index>.")

        if not images:
            raise CalibrationError(
                f"No calibration images found.\n"
                f"  Searched: {args.images}\n"
                f"  Recovery: capture 15-30 checkerboard views, or check the glob pattern."
            )

        LOG.info("Detecting a %dx%d inner-corner pattern in %d image(s)...", pattern[0], pattern[1], len(images))
        obj_points, img_points, image_size, used = find_checkerboard_corners(images, pattern, args.square_size)

        if len(used) < args.min_images:
            raise CalibrationError(
                f"Pattern found in only {len(used)}/{len(images)} image(s); at least {args.min_images} are needed.\n"
                f"  Recovery: check --pattern-size counts INNER corners (a 10x7 board has 9x6 inner corners),\n"
                f"  ensure the whole board is visible and in focus, and avoid extreme glare."
            )

        LOG.info("Calibrating from %d view(s) at %dx%d...", len(used), image_size[0], image_size[1])
        result = calibrate(obj_points, img_points, image_size)

        K = result["K"]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        LOG.info("RMS reprojection error: %.4f px", result["rms"])
        if result["rms"] > 1.0:
            LOG.warning(
                "RMS above 1.0 px indicates a poor calibration. Worst view: %.4f px. "
                "Recovery: drop outlier views, add coverage near the frame corners, "
                "and verify --square-size is the real square side in metres.",
                result["worst_view_error"],
            )

        data = {
            "calibrated": True,
            "allow_uncalibrated": False,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion": result["dist"],
            "calibration_meta": {
                "date": datetime.now(timezone.utc).isoformat(),
                "pattern": "checkerboard",
                "pattern_size": list(pattern),
                "square_size_m": args.square_size,
                "num_images": len(used),
                "reprojection_error_px": result["rms"],
                "mean_view_error_px": result["mean_view_error"],
                "worst_view_error_px": result["worst_view_error"],
                "camera_model": "pinhole + Brown-Conrady",
            },
        }
        save_yaml(data, args.output)

        print()
        print(f"fx = {fx:.4f}   fy = {fy:.4f}")
        print(f"cx = {cx:.4f}   cy = {cy:.4f}")
        print(f"distortion = {[round(d, 6) for d in result['dist']]}")
        print(f"RMS reprojection error = {result['rms']:.4f} px")
        hfov = np.degrees(2 * np.arctan(image_size[0] / (2 * fx)))
        print(f"horizontal FOV = {hfov:.2f} deg")
        print()
        LOG.info("Intrinsics written to %s (calibrated: true)", args.output)
    except CalibrationError as e:
        LOG.error("%s", e)
        return 1
    except (OSError, ValueError, ImportError) as e:
        LOG.error("Calibration failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
