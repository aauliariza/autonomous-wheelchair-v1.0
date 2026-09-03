#!/usr/bin/env bash
# End-to-end demo requiring no dataset and no trained checkpoint (README STEP 17-19).
#
# Uses the official pretrained weights and the 8-image depth8 fixture that ships
# with Ultralytics, so it runs on a clean checkout in a couple of minutes on CPU.
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
IMGSZ="${IMGSZ:-320}"
OUT="${OUT:-outputs/demo}"
mkdir -p "$OUT"

echo "=== Measured architecture facts ==="
python scripts/inspect_model.py --imgsz "$IMGSZ" --device "$DEVICE" --output "$OUT/model_inspection.json"

echo
echo "=== Fetching the depth8 fixture (8 SUN RGB-D images, ~1.3 MB) ==="
python - <<'PY'
from pathlib import Path
from ultralytics.utils.downloads import download
if not Path("datasets/depth8-png").exists():
    download(["https://github.com/ultralytics/assets/releases/download/v0.0.0/depth8-png.zip"],
             dir=Path("datasets"), unzip=True, delete=True)
print("fixture ready at datasets/depth8-png")
PY

echo
echo "=== Dataset verification ==="
python - <<'PY'
import os, yaml
os.makedirs("configs/data", exist_ok=True)
yaml.safe_dump({"path": os.path.abspath("datasets/depth8-png"),
                "train": "images/train", "val": "images/val", "nc": 1,
                "names": {0: "depth"}, "channels": 3,
                "depth_scale": 1000, "max_depth": 10.0},
               open("configs/data/demo_depth8.yaml", "w"))
PY
python datasets/scripts/verify_dataset.py --data configs/data/demo_depth8.yaml --sample 0

echo
echo "=== Depth evaluation (metric vs aligned) ==="
python evaluation/evaluate_depth.py --model yolo26n-depth.pt --data configs/data/demo_depth8.yaml \
    --imgsz "$IMGSZ" --device "$DEVICE" --output "$OUT/depth_metrics.json"

echo
echo "=== Image inference with the navigation HUD ==="
IMG=$(find datasets/depth8-png/images/val -type f | head -1)
python inference/predict_image.py --source "$IMG" --model yolo26n-depth.pt \
    --device "$DEVICE" --save-depth --output "$OUT"

echo
echo "=== Latency benchmark ==="
python evaluation/benchmark_latency.py --imgsz "$IMGSZ" --device "$DEVICE" \
    --runs 10 --warmup 3 --output "$OUT/latency.json"

echo
echo "Demo complete. Outputs in $OUT"
