#!/usr/bin/env bash
# Evaluate every trained checkpoint and build the ablation table (README STEP 14-23).
set -euo pipefail

cd "$(dirname "$0")/.."

DATA="${DATA:-configs/data/sunrgbd.yaml}"
DEVICE="${DEVICE:-0}"
OUT="${OUT:-outputs/evaluation}"
mkdir -p "$OUT"

evaluate() {
    local name="$1" ckpt="$2"
    if [ ! -f "$ckpt" ]; then
        echo "  SKIP $name: $ckpt not found (train it first)"
        return 0
    fi
    echo
    echo "=== $name ==="
    python evaluation/evaluate_depth.py --model "$ckpt" --data "$DATA" \
        --device "$DEVICE" --output "$OUT/${name}_depth.json"
    python evaluation/evaluate_distance.py --model "$ckpt" --data "$DATA" \
        --device "$DEVICE" --output "$OUT/${name}_distance.json"
    python evaluation/benchmark_latency.py --depth-model "$ckpt" \
        --device "$DEVICE" --output "$OUT/${name}_latency.json"
}

evaluate teacher            outputs/checkpoints/teacher_best.pt
evaluate student_baseline   outputs/checkpoints/student_baseline_best.pt
evaluate student_distilled  outputs/checkpoints/student_distilled_best.pt

echo
echo "=== Ablation table ==="
python evaluation/ablation.py --experiments outputs/experiments --output docs/ablation

echo
echo "Evaluation complete. Results in $OUT and docs/ablation.md"
echo "Cells reading NOT MEASURED correspond to runs that were not performed."
