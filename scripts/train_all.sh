#!/usr/bin/env bash
# Full experiment sequence: teacher -> baseline -> ablation B-E (README STEP 7-13).
#
# Long-running (many GPU-hours). Each stage must succeed before the next starts,
# so a failure stops the run instead of silently producing empty result rows.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA="${DATA:-configs/data/sunrgbd.yaml}"
DEVICE="${DEVICE:-0}"

if [ ! -f "$DATA" ]; then
    echo "ERROR: dataset config not found: $DATA"
    echo "  Run: python datasets/scripts/prepare_sunrgbd.py --source /path/to/SUNRGBD"
    exit 1
fi

echo "=== Verifying the dataset before spending GPU time ==="
python datasets/scripts/verify_dataset.py --data "$DATA"

echo
echo "=== [1/6] Teacher (Experiment F) ==="
python training/train_teacher.py --config configs/teacher.yaml --set "data.yaml=$DATA" "train.device=$DEVICE"

echo
echo "=== [2/6] Student baseline (Experiment A) ==="
python training/train_student_baseline.py --config configs/student.yaml --set "data.yaml=$DATA" "train.device=$DEVICE"

i=3
for exp in B C D E; do
    echo
    echo "=== [$i/6] Distillation, Experiment $exp ==="
    python training/train_distillation.py --config configs/distillation.yaml \
        --experiment "$exp" --set "data.yaml=$DATA" "train.device=$DEVICE"
    i=$((i + 1))
done

echo
echo "All experiments complete. Next: bash scripts/evaluate_all.sh"
