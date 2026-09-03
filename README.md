# Autonomous Indoor Smart Wheelchair Navigation

**Knowledge-Distilled Lightweight Metric Monocular Depth Estimation and
Obstacle-Aware Free-Path Selection**

A reproducible research framework: a YOLO26x-Depth teacher distills a
YOLO26n-Depth student for metric monocular depth, which is fused with YOLO26n
obstacle detection to estimate obstacle distances in metres and select a free
path across five sectors, with majority-vote temporal smoothing and a fail-safe
safety layer.

> ### Status
>
> The framework is complete and runnable end to end. **No full-scale training run
> has been performed**: SUN RGB-D was unreachable from the development
> environment (egress proxy denies `rgbd.cs.princeton.edu`) and no GPU was
> available. Per the project's no-fabrication rule, every unmeasured result reads
> `NOT MEASURED` in [docs/experiments.md](docs/experiments.md), alongside the
> measurements that *were* made on real data.

---

## Architecture

```
RGB Camera
    |
    +------------------------------+
    |                              |
    v                              v
YOLO26n Object Detection      YOLO26n-Depth Student
(all classes -> obstacle)             ^
    |                                 | Knowledge Distillation
    |                          YOLO26x-Depth Teacher (frozen)
    |                                 |
    +---------------+-----------------+
                    v
          Obstacle-Depth Fusion
                    v
       Obstacle Distance Estimation      (inner-60% bbox ROI, median)
                    v
          60% Global Navigation ROI
                    v
          5-Sector Free-Path      FL | L | CTR | R | FR
                    v
          Majority-Vote Hysteresis (N = 3)
                    v
    FORWARD / TURN_LEFT / TURN_RIGHT / STOP
```

**One class only.** Every detected object is labelled `obstacle`. The system
performs obstacle *detection*, never object *recognition*: a chair, a person and
a wardrobe are all simply things not to hit.

## Three things that make this correct rather than merely working

1. **Metric evaluation, not aligned.** Monocular benchmarks rescale predictions
   by `median(gt)/median(pred)`. Measured here, that inflates δ1 from **0.146 to
   0.714** on identical predictions — a 4.9× difference. Obstacle distance is
   evaluated with `align="none"`; both modes are reported side by side and never
   conflated. See [docs/depth_to_distance.md](docs/depth_to_distance.md).
2. **Axial depth ≠ Euclidean distance.** At the corner of a 640×480 60° frame the
   slant range exceeds axial depth by **23.3%** — 0.23 m at a 1.0 m threshold,
   enough to flip STOP into FORWARD. Both are computed and reported separately.
3. **Conservative under uncertainty.** An obstacle whose distance cannot be
   measured *blocks*; it is never treated as clear. A single STOP cannot be
   outvoted by the majority filter. `EMERGENCY_STOP` latches until explicitly
   reset.

## Installation

### STEP 1 — Clone

```bash
git clone https://github.com/aauliariza/autonomous-wheelchair-v1.0.git
cd autonomous-wheelchair-v1.0
```

### STEP 2 — Create an environment

```bash
conda env create -f environment.yml && conda activate wheelchair
# or
python -m venv .venv && source .venv/bin/activate
```

### STEP 3 — Install dependencies

```bash
# For a specific CUDA build, install torch FIRST from pytorch.org, then:
pip install -r requirements.txt
```

YOLO26-Depth requires **ultralytics >= 8.4.0**. All results here used 8.4.138.

### STEP 4 — Verify the installation

```bash
python -m pytest tests/ -q          # 188 tests, no dataset or checkpoint needed
python scripts/inspect_model.py     # downloads weights and prints measured facts
```

`inspect_model.py` should report the teacher at 57,022,241 parameters, the
student at 6,355,617, and KD tap layers `[16, 19, 22]`.

## Dataset

### STEP 5 — Prepare SUN RGB-D

Download `SUNRGBD.zip` (~6.5 GB) from
<https://rgbd.cs.princeton.edu/data/SUNRGBD.zip> and extract it.

```bash
# Recommended: the OFFICIAL split (avoids scene-level leakage)
python datasets/scripts/prepare_sunrgbd.py \
    --convert-allsplit /path/to/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat \
    --split-file configs/data/sunrgbd_official_split.json

python datasets/scripts/prepare_sunrgbd.py \
    --source /path/to/SUNRGBD \
    --split official --split-file configs/data/sunrgbd_official_split.json \
    --output datasets/sunrgbd --config-out configs/data/sunrgbd.yaml
```

> **Why not the shipped `depth-sunrgbd.yaml`?** Ultralytics selects its
> validation set as a random seed-0 sample of 1090 scenes, ignoring SUN RGB-D's
> official partition. SUN RGB-D contains multiple frames of the same room, so a
> random split can place near-duplicate views in both train and val, inflating
> every metric. Use `--split ultralytics` only to reproduce Ultralytics-reported
> numbers.

### STEP 6 — Verify the dataset

```bash
python datasets/scripts/verify_dataset.py --data configs/data/sunrgbd.yaml --report outputs/dataset_report.json
```

Runs all ten required checks plus cross-split leakage detection, and exits
non-zero on failure so it can gate training.

Optional, for a fine-tuned `nc=1` detector (the pipeline works without it).
SUN RGB-D's own 2D box annotations need no external labels at all — download
`SUNRGBDMeta2DBB_v2.mat` alongside the extracted dataset:

```bash
python datasets/scripts/convert_sunrgbd_obstacle.py \
    --source /path/to/SUNRGBD --meta /path/to/SUNRGBDMeta2DBB_v2.mat \
    --split-file configs/data/sunrgbd_official_split.json --output datasets/obstacle
```

Or convert an externally annotated YOLO/COCO dataset:

```bash
python datasets/scripts/convert_to_obstacle_dataset.py --format yolo \
    --source /path/to/annotated --output datasets/obstacle
```

## Training

### STEP 7-8 — Teacher

```bash
python training/train_teacher.py --config configs/teacher.yaml
python evaluation/evaluate_depth.py --model outputs/checkpoints/teacher_best.pt --data configs/data/sunrgbd.yaml
```

### STEP 9-10 — Student baseline (Experiment A)

```bash
python training/train_student_baseline.py --config configs/student.yaml
python evaluation/evaluate_depth.py --model outputs/checkpoints/student_baseline_best.pt --data configs/data/sunrgbd.yaml
```

### STEP 11 — Hyperparameter search

```bash
python tuning/optuna_study.py --config configs/optuna.yaml       # resumable
python tuning/analyze_trials.py --storage sqlite:///outputs/optuna/kd_depth_search.db --plots
```

### STEP 12 — Distilled student

```bash
python training/train_distillation.py --config configs/distillation.yaml
# or the tuned configuration
python training/train_distillation.py --config outputs/optuna/best_config.yaml
```

### STEP 13 — Ablation study

```bash
for exp in B C D E; do
    python training/train_distillation.py --config configs/distillation.yaml --experiment $exp
done
python evaluation/ablation.py --experiments outputs/experiments --output docs/ablation
```

Ablation rows are **configuration changes, not code changes**.

## Evaluation

### STEP 14 — Depth metrics

```bash
python evaluation/evaluate_depth.py --model outputs/checkpoints/student_distilled_best.pt --data configs/data/sunrgbd.yaml
```

Reports MODE 1 (metric) and MODE 2 (aligned) side by side.

### STEP 15 — Camera calibration

```bash
python calibration/camera_calibration.py --capture 0 --pattern-size 9 6 --square-size 0.025
# or from existing images
python calibration/camera_calibration.py --images "calib/*.jpg" --pattern-size 9 6 --square-size 0.025
```

`--pattern-size` counts **inner corners** (a 10×7 board has 9×6). `--square-size`
is in **metres** — a wrong value scales every derived distance.

### STEP 16 — Obstacle distance

```bash
python evaluation/evaluate_distance.py --model outputs/checkpoints/student_distilled_best.pt --data configs/data/sunrgbd.yaml
```

### STEP 17-19 — Inference

```bash
python inference/predict_image.py --source data/test.jpg --save-depth
python inference/predict_video.py --source data/test.mp4 --show-depth
python inference/webcam.py --camera 0
```

### STEP 20-23 — Pipeline, tests, benchmark, export

```bash
python evaluation/evaluate_navigation.py --predictions outputs/predictions/run_telemetry.csv --ground-truth data/nav_gt.csv
python -m pytest tests/ -q
python evaluation/benchmark_latency.py --depth-model outputs/checkpoints/student_distilled_best.pt
python evaluation/validate_numerical_consistency.py --model outputs/checkpoints/student_distilled_best.pt --formats onnx
```

> **Export caveat (measured).** The ONNX *graph* matches PyTorch to 7×10⁻⁶, but
> the full pipeline diverges by 2.28 m on **letterboxed non-square** input — a
> pre/post-processing difference, not an export defect. Feed square inputs at
> exactly `imgsz`, or re-implement letterbox removal, before deploying ONNX.

## Configuration

Every parameter lives in `configs/`; none is hard-coded.

| File | Controls |
|---|---|
| `teacher.yaml` | YOLO26x-Depth training |
| `student.yaml` | YOLO26n-Depth baseline (Experiment A) |
| `distillation.yaml` | KD terms, λ weights, ablation toggles |
| `detection.yaml` | Single-class obstacle detector |
| `navigation.yaml` | ROI, sectors, safety distance, hysteresis, fail-safes |
| `camera.yaml` | Intrinsics (placeholders until calibrated) |
| `optuna.yaml` | Search space and composite objective |

Override anything from the CLI:

```bash
python training/train_distillation.py --config configs/distillation.yaml \
    --set train.epochs=50 kd.roi.alpha=4.0 train.device=0
```

## Key parameters

| Parameter | Default | Note |
|---|---|---|
| `roi.width_ratio` | 0.60 | Central 60% of horizontal FOV |
| `bbox_roi.inner_ratio` | 0.60 | Inner 60% of each box, **independent** of the above |
| `safety.safety_distance_m` | **1.0** | Research parameter — see warning below |
| `safety.distance_mode` | `axial` | Correct for forward clearance |
| `hysteresis.window` | 3 | N = 3 majority vote |
| `depth_stats.min_valid_ratio` | 0.30 | Below this, distance is INVALID and blocks |
| `seed` | 42 | Python, NumPy, PyTorch, CUDA |

> ### Safety warning
>
> `safety_distance_m: 1.0` is a **research parameter carried over from the
> prototype**. It has **not** been validated against any real wheelchair. Before
> any physical deployment it must be re-derived from that chair's braking
> distance, maximum speed, control latency and occupant comfort.
>
> This software emits a **command**. It never drives a motor. Any real system
> requires an independent motor-controller safety layer (current limiting, speed
> limiting, a watchdog, and a hardware emergency stop) that does not depend on
> anything decided here.

## Reproducibility

- Seed 42 across Python, NumPy, PyTorch and CUDA
- `deterministic: true` for bit-reproducible runs (10–30% slower; see `utils/seed.py`)
- Every run writes `experiment_metadata.json` with git commit, library versions,
  GPU name, OS, seed and config hash
- 188 unit tests run on synthetic data — no dataset or checkpoint required

## Documentation

| Document | Contents |
|---|---|
| [architecture.md](docs/architecture.md) | Verified model facts, integration hazards, module map |
| [knowledge_distillation.md](docs/knowledge_distillation.md) | All five KD terms and their rationale |
| [depth_to_distance.md](docs/depth_to_distance.md) | The three ways to get distance wrong |
| [experiments.md](docs/experiments.md) | Result tables and everything actually measured |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Checkpoint not found` | The error lists every path searched. Train first, or pass `--model`. |
| `Dataset config not found` | Run `prepare_sunrgbd.py`; the message shows the expected command. |
| Student loss decreases but nothing learns | `.pt` files load with `requires_grad=False`. Use `StudentDepthModel`, which calls `unfreeze()`. |
| `cannot pickle '_thread.lock'` | A KD criterion stored on the model. Keep it in the module registry — see `training/kd_trainer.py`. |
| δ1 looks implausibly good | You are reading MODE 2 (aligned). MODE 1 is the metric number. |
| Euclidean distance refused | `configs/camera.yaml` still has `calibrated: false`. Run the calibration script. |
| CUDA requested but unavailable | Falls back to CPU with a warning. Pass `--device cpu` to silence it. |
| ONNX depths differ from PyTorch | Letterboxing. Use square inputs at exactly `imgsz`. |

## Citation

```bibtex
@software{autonomous_wheelchair_2026,
  title  = {Knowledge-Distilled Lightweight Metric Monocular Depth Estimation and
            Obstacle-Aware Free-Path Selection for Autonomous Indoor Smart
            Wheelchair Navigation},
  year   = {2026},
  url    = {https://github.com/aauliariza/autonomous-wheelchair-v1.0}
}
```

Please also cite Ultralytics YOLO26 and the SUN RGB-D dataset (Song et al., CVPR
2015).

## License

AGPL-3.0, inherited from the Ultralytics dependency. See [LICENSE](LICENSE).
