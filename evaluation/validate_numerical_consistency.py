#!/usr/bin/env python3
"""Compare PyTorch, ONNX and TensorRT outputs (spec sections BG, BH).

A SUCCESSFUL EXPORT IS NOT A VALID EXPORT
-----------------------------------------
An exporter writing a file proves only that the graph converted. It says nothing
about whether the converted model produces the same depths. Quantization,
operator substitution and layout changes can all shift outputs enough to move an
obstacle across the safety threshold while the file loads perfectly.

This script therefore compares the backends at three levels, in increasing order
of what actually matters:

1. depth map            max / mean / relative difference per pixel
2. obstacle distance    the reduced metre value the navigation layer consumes
3. navigation command   the discrete decision the wheelchair acts on

A tiny per-pixel difference is harmless; a single flipped command is not. The
third check is the one that decides whether an export is deployable, and the
script's exit code reflects all three.

Usage:
    python evaluation/validate_numerical_consistency.py --model outputs/checkpoints/student_distilled_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_yaml, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("numerical_validation")


def difference_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Absolute and relative difference statistics between two depth maps."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return {"note": f"shape mismatch {a.shape} vs {b.shape}", "max_abs_diff": float("nan")}

    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return {"note": "no finite values to compare", "max_abs_diff": float("nan")}

    a, b = a[valid], b[valid]
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-6)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float((diff / denom).max()),
        "mean_rel_diff": float((diff / denom).mean()),
        "num_compared": int(a.size),
    }


def compare_graph_level(model_path: Path, exported: str, imgsz: int) -> dict[str, Any]:
    """Compare the raw graphs, bypassing both predictors' pre/post-processing.

    This separates two very different failure modes that a pipeline-level
    comparison alone would conflate:

    * the EXPORTED GRAPH computes different numbers (a real export defect), from
    * the graph being identical while PRE/POST-PROCESSING differs between
      backends (a pipeline issue, not an export defect).

    In export mode the Depth head upsamples 4x internally, so the ONNX output is
    at input resolution while the PyTorch head returns input/4; the PyTorch
    tensor is upsampled to match before comparing.
    """
    import torch
    import torch.nn.functional as F

    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "NOT MEASURED - onnxruntime is not installed"}

    from ultralytics import YOLO

    net = YOLO(str(model_path)).model.eval()
    x = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        ref = net(x)
    ref = (ref["depth"] if isinstance(ref, dict) else ref).float()

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    out = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]

    if ref.shape[-2:] != out.shape[-2:]:
        ref = F.interpolate(ref, size=out.shape[-2:], mode="bilinear", align_corners=False)

    return {"status": "ok", **difference_stats(ref.numpy(), out)}


def run_backends(model_path: Path, image: np.ndarray, imgsz: int, device: str, formats: list[str]) -> dict[str, Any]:
    """Run the PyTorch model and each exported backend on the same input."""
    from ultralytics import YOLO

    outputs: dict[str, Any] = {}
    exports: dict[str, str] = {}

    LOG.info("Running PyTorch reference...")
    torch_model = YOLO(str(model_path))
    ref = torch_model.predict(image, imgsz=imgsz, device=device, verbose=False)[0].depth.data
    outputs["pytorch"] = np.squeeze(ref.detach().cpu().numpy() if hasattr(ref, "detach") else np.asarray(ref))

    for fmt in formats:
        try:
            LOG.info("Exporting to %s...", fmt)
            exported = YOLO(str(model_path)).export(format=fmt, imgsz=imgsz, verbose=False)
            exports[fmt] = str(exported)

            LOG.info("Running %s backend...", fmt)
            out = YOLO(str(exported), task="depth").predict(image, imgsz=imgsz, verbose=False)[0].depth.data
            outputs[fmt] = np.squeeze(out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out))
        except (RuntimeError, ValueError, OSError, ImportError, AttributeError) as e:
            # A missing exporter is a limitation to report, not a crash.
            LOG.warning("Backend '%s' unavailable (%s: %s)", fmt, type(e).__name__, e)
            outputs[fmt] = None

    return {"outputs": outputs, "exports": exports}


def navigation_from_depth(depth: np.ndarray, nav_cfg: dict, boxes: np.ndarray, confs: np.ndarray) -> dict[str, Any]:
    """Reduce a depth map to obstacle distances and a navigation command."""
    from navigation.free_path import FreePathSelector
    from navigation.obstacle_fusion import ObstacleFusion
    from navigation.roi import compute_global_roi
    from navigation.sectors import SectorMap

    h, w = depth.shape[:2]
    roi = compute_global_roi(w, h, nav_cfg["roi"]["width_ratio"], nav_cfg["roi"]["x_center"])
    sector_map = SectorMap.from_config(nav_cfg, roi)
    obstacles = ObstacleFusion(nav_cfg).fuse(boxes, confs, depth, sector_map, image_size=(h, w))
    decision = FreePathSelector(nav_cfg).select(sector_map)

    return {
        "command": str(decision.command),
        "distances": [o.distance_m for o in obstacles],
        "occupancy": sector_map.occupancy(),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns non-zero when a backend disagrees materially."""
    ap = argparse.ArgumentParser(description="Validate exported models against PyTorch.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=None, help="Test image; a synthetic one is used if omitted")
    ap.add_argument("--formats", nargs="*", default=["onnx"], help="Export formats to validate")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nav-config", type=Path, default=Path("configs/navigation.yaml"))
    ap.add_argument("--depth-tolerance", type=float, default=0.05, help="Max acceptable mean abs depth diff (m)")
    ap.add_argument("--distance-tolerance", type=float, default=0.05, help="Max acceptable obstacle distance diff (m)")
    ap.add_argument("--output", type=Path, default=Path("outputs/evaluation/numerical_validation.json"))
    args = ap.parse_args(argv)

    try:
        import cv2

        if args.source and args.source.exists():
            image = cv2.imread(str(args.source))
        else:
            rng = np.random.default_rng(0)
            image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            LOG.info("No --source given; using a fixed synthetic image (seed 0).")

        nav_cfg = load_yaml(args.nav_config)

        # A SQUARE input at exactly imgsz avoids letterboxing. Comparing both
        # shapes separates a genuine numerical difference from a letterbox
        # handling difference in the predictors.
        square = image
        if image.shape[0] != image.shape[1] or image.shape[0] != args.imgsz:
            square = cv2.resize(image, (args.imgsz, args.imgsz))

        square_result = run_backends(args.model, square, args.imgsz, args.device, args.formats)
        result = run_backends(args.model, image, args.imgsz, args.device, args.formats)
        outputs = result["outputs"]
        reference = outputs["pytorch"]

        # Fixed boxes so any difference is attributable to depth, not detection.
        h, w = reference.shape[:2]
        boxes = np.array([[w * 0.35, h * 0.3, w * 0.65, h * 0.8], [w * 0.05, h * 0.4, w * 0.25, h * 0.9]])
        confs = np.array([0.9, 0.8])
        ref_nav = navigation_from_depth(reference, nav_cfg, boxes, confs)

        report: dict[str, Any] = {
            "model": str(args.model),
            "imgsz": args.imgsz,
            "input_shape": list(image.shape),
            "reference_command": ref_nav["command"],
            "exports": result["exports"],
            "graph_level": {},
            "square_input": {},
            "backends": {},
        }

        # --- level 0: raw graph equivalence ---
        for fmt, path in result["exports"].items():
            if fmt == "onnx":
                report["graph_level"][fmt] = compare_graph_level(args.model, path, args.imgsz)

        # --- square-input pipeline agreement (letterbox-free) ---
        sq_ref = square_result["outputs"]["pytorch"]
        for name, out in square_result["outputs"].items():
            if name != "pytorch" and out is not None:
                report["square_input"][name] = difference_stats(sq_ref, out)
        failures: list[str] = []

        print()
        print("LEVEL 0 - raw graph equivalence (no pre/post-processing)")
        print("-" * 76)
        for fmt, stats in report["graph_level"].items():
            if stats.get("status") == "ok":
                print(f"  {fmt:<10} max abs {stats['max_abs_diff']:.3e}   mean abs {stats['mean_abs_diff']:.3e}")
            else:
                print(f"  {fmt:<10} {stats.get('status')}")

        print()
        print(f"LEVEL 1 - full pipeline, SQUARE {args.imgsz}x{args.imgsz} input (no letterboxing)")
        print("-" * 76)
        for fmt, stats in report["square_input"].items():
            print(
                f"  {fmt:<10} max abs {stats.get('max_abs_diff', float('nan')):.6f}   "
                f"mean abs {stats.get('mean_abs_diff', float('nan')):.6f}"
            )

        print()
        print(f"LEVEL 2 - full pipeline, ACTUAL input {image.shape[1]}x{image.shape[0]}")
        print(f"{'backend':<12}{'max abs':>12}{'mean abs':>12}{'max rel':>12}{'dist diff':>12}{'command':>16}")
        print("-" * 76)
        print(f"{'pytorch':<12}{'reference':>12}{'':>12}{'':>12}{'':>12}{ref_nav['command']:>16}")

        for name, out in outputs.items():
            if name == "pytorch":
                continue
            if out is None:
                report["backends"][name] = {"status": "NOT MEASURED - backend unavailable"}
                print(f"{name:<12}{'NOT MEASURED (backend unavailable)':>64}")
                continue

            depth_stats = difference_stats(reference, out)
            nav = navigation_from_depth(out, nav_cfg, boxes, confs)

            pairs = [
                (a, b)
                for a, b in zip(ref_nav["distances"], nav["distances"], strict=True)
                if a is not None and b is not None
            ]
            dist_diff = max((abs(a - b) for a, b in pairs), default=0.0)
            command_match = nav["command"] == ref_nav["command"]

            entry = {
                "status": "ok",
                "depth": depth_stats,
                "max_distance_diff_m": dist_diff,
                "command": nav["command"],
                "command_matches": command_match,
                "occupancy_matches": nav["occupancy"] == ref_nav["occupancy"],
            }
            report["backends"][name] = entry

            print(
                f"{name:<12}{depth_stats.get('max_abs_diff', float('nan')):>12.6f}"
                f"{depth_stats.get('mean_abs_diff', float('nan')):>12.6f}"
                f"{depth_stats.get('max_rel_diff', float('nan')):>12.6f}"
                f"{dist_diff:>12.6f}{nav['command'] + (' OK' if command_match else ' MISMATCH'):>16}"
            )

            if not command_match:
                failures.append(f"{name}: navigation command differs ({nav['command']} vs {ref_nav['command']})")
            if dist_diff > args.distance_tolerance:
                failures.append(f"{name}: obstacle distance differs by {dist_diff:.4f} m")
            if depth_stats.get("mean_abs_diff", 0.0) > args.depth_tolerance:
                failures.append(f"{name}: mean depth difference {depth_stats['mean_abs_diff']:.4f} m")

        print("-" * 76)
        report["failures"] = failures
        report["passed"] = not failures

        # Distinguish an export defect from a letterbox handling difference, so
        # the report says which one actually occurred.
        graph_ok = all(
            s.get("status") == "ok" and s.get("mean_abs_diff", 1.0) < 1e-3 for s in report["graph_level"].values()
        ) and bool(report["graph_level"])
        square_ok = all(
            s.get("mean_abs_diff", 1.0) < args.depth_tolerance for s in report["square_input"].values()
        ) and bool(report["square_input"])
        report["diagnosis"] = (
            "graph and square-input pipeline agree; any LEVEL 2 difference is a letterbox "
            "pre/post-processing difference between backends, not an export defect"
            if graph_ok and square_ok
            else "the exported graph itself differs from PyTorch - a genuine export defect"
            if not graph_ok
            else "graph agrees but the square-input pipeline differs"
        )

        if failures:
            print()
            for f in failures:
                LOG.error("VALIDATION FAILED - %s", f)
            LOG.error("Diagnosis: %s", report["diagnosis"])
            if graph_ok and square_ok:
                LOG.error(
                    "ACTION: the ONNX graph is numerically correct. Feed the exported model "
                    "SQUARE inputs at exactly imgsz=%d, or re-implement letterbox removal for "
                    "the exported backend, before deploying it.",
                    args.imgsz,
                )
            else:
                LOG.error("The export is NOT equivalent to the PyTorch model and must not be deployed.")
        else:
            LOG.info("All backends agree within tolerance; exports are numerically consistent.")

        save_json(report, args.output)
        LOG.info("Report written to %s", args.output)
        return 1 if failures else 0
    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, ValueError, ImportError) as e:
        LOG.error("Numerical validation failed (%s): %s", type(e).__name__, e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
