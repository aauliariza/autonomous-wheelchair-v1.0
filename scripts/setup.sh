#!/usr/bin/env bash
# Environment setup and installation verification (README STEP 2-4).
#
# Does NOT install torch: pick the right CUDA build for your machine from
# https://pytorch.org/get-started/locally/ first, then run this.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Python ==="
python --version

echo
echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo
echo "=== Verifying the installation ==="
python - <<'PY'
import sys
ok = True
try:
    import torch
    print(f"  torch        {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
except ImportError:
    print("  torch        MISSING -- install from https://pytorch.org/get-started/locally/")
    ok = False

try:
    import ultralytics
    v = ultralytics.__version__
    major, minor = (int(x) for x in v.split(".")[:2])
    # YOLO26 model configs and the depth task arrived in the 8.4.x line.
    if (major, minor) < (8, 4):
        print(f"  ultralytics  {v}  TOO OLD -- YOLO26-Depth needs >= 8.4.0")
        ok = False
    else:
        print(f"  ultralytics  {v}")
except ImportError:
    print("  ultralytics  MISSING")
    ok = False

for mod in ("cv2", "numpy", "yaml", "optuna", "matplotlib"):
    try:
        __import__(mod)
        print(f"  {mod:<12} ok")
    except ImportError:
        print(f"  {mod:<12} MISSING")
        ok = False

# Confirm the depth task is actually registered, not merely that the package imports.
try:
    from ultralytics.cfg import TASKS
    if "depth" in TASKS:
        print("  depth task   registered")
    else:
        print(f"  depth task   NOT FOUND in {TASKS}")
        ok = False
except ImportError as e:
    print(f"  depth task   could not check ({e})")
    ok = False

sys.exit(0 if ok else 1)
PY

echo
echo "=== Running unit tests ==="
python -m pytest tests/ -q

echo
echo "Setup complete. Next: python scripts/inspect_model.py"
