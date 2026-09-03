#!/usr/bin/env python3
"""Print measured architecture facts for a YOLO26-Depth model.

Every value is PROBED from the loaded network — layer indices come from
``head.f``, channel widths and spatial sizes from a real forward pass. Nothing is
read from documentation. Use this to regenerate the tables in
``docs/architecture.md`` after an Ultralytics upgrade.

Usage:
    python scripts/inspect_model.py
    python scripts/inspect_model.py --models yolo26n-depth.pt yolo26x-depth.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("inspect_model")


def inspect(name: str, imgsz: int, device: str) -> dict:
    """Probe one model and return its measured facts."""
    from models.feature_hooks import describe_feature_layers, resolve_depth_feature_layers
    from models.model_utils import get_calibration, load_depth_model, model_complexity

    net, source = load_depth_model(name, device=device)
    layers = resolve_depth_feature_layers(net)
    shapes = describe_feature_layers(net, layers, imgsz=imgsz, device=device)
    cal_a, cal_b = get_calibration(net)
    complexity = model_complexity(net, imgsz=imgsz, device=device)

    import torch

    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, imgsz, imgsz, device=device))
    out_shape = tuple(out.shape) if torch.is_tensor(out) else None

    return {
        "source": source,
        "parameters": complexity["parameters"],
        "gflops": complexity["gflops"],
        "model_size_mb": complexity["model_size_mb"],
        "kd_layers": layers,
        "feature_shapes": {str(k): list(v) for k, v in shapes.items()},
        "feature_channels": [shapes[i][0] for i in layers],
        "calibration": {"cal_a": cal_a, "cal_b": cal_b},
        "output_shape_at_imgsz": list(out_shape) if out_shape else None,
        "imgsz": imgsz,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Probe YOLO26-Depth architecture facts.")
    ap.add_argument("--models", nargs="+", default=["yolo26n-depth.pt", "yolo26x-depth.pt"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, default=Path("outputs/model_inspection.json"))
    args = ap.parse_args(argv)

    import ultralytics

    report: dict = {"ultralytics_version": ultralytics.__version__, "models": {}}
    LOG.info("Ultralytics %s | probing at imgsz=%d on %s", ultralytics.__version__, args.imgsz, args.device)

    for name in args.models:
        try:
            info = inspect(name, args.imgsz, args.device)
        except Exception as e:  # noqa: BLE001 - report and continue to the next model
            LOG.error("Could not inspect %s (%s: %s)", name, type(e).__name__, e)
            continue

        report["models"][name] = info
        print()
        print("=" * 68)
        print(name)
        print("=" * 68)
        print(f"  parameters       : {info['parameters']:,}")
        print(f"  GFLOPs @{args.imgsz:<4}     : {info['gflops'] if info['gflops'] is not None else 'NOT MEASURED'}")
        print(f"  size (fp32)      : {info['model_size_mb']:.2f} MB")
        print(f"  KD tap layers    : {info['kd_layers']}   (read from head.f)")
        for i in info["kd_layers"]:
            c, h, w = info["feature_shapes"][str(i)]
            print(f"    layer[{i:>2}]      : {c:>4} ch  {h}x{w}")
        print(f"  calibration      : cal_a={info['calibration']['cal_a']:.4f} cal_b={info['calibration']['cal_b']:.4f}")
        print(f"  output shape     : {info['output_shape_at_imgsz']}   (input/4)")

    names = list(report["models"])
    if len(names) == 2:
        a, b = (report["models"][n] for n in names)
        big, small = (a, b) if a["parameters"] >= b["parameters"] else (b, a)
        print()
        print(f"teacher/student parameter ratio: {big['parameters'] / small['parameters']:.2f}x")
        same = all(
            big["feature_shapes"][str(i)][1:] == small["feature_shapes"][str(j)][1:]
            for i, j in zip(big["kd_layers"], small["kd_layers"], strict=True)
        )
        print(
            f"tap spatial dimensions identical: {same}  "
            f"({'channel projection only' if same else 'spatial resampling also required'})"
        )

    save_json(report, args.output)
    LOG.info("Inspection written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
