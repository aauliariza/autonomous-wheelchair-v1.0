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

### STEP 6b — Obstacle detection dataset (optional)

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

## Knowledge distillation

The student is trained against five distillation terms plus a mandatory
ground-truth term. Every term lives in its own module and is switched on or off
from `configs/distillation.yaml`, which is what makes the ablation a
configuration change rather than a code change.

| Term | Module | `lambda` | What it transfers |
|---|---|---|---|
| Ground truth | `distillation/losses.py` | 1.0 | Ultralytics' own depth loss against the SUN RGB-D label |
| Depth output | `distillation/depth_kd.py` | 0.5 | The teacher's per-pixel depth, on valid pixels only |
| Feature | `distillation/feature_kd.py` | 0.2 | Intermediate activations at layers `[16, 19, 22]` |
| Boundary | `distillation/boundary_kd.py` | 0.1 | Depth gradients — where an obstacle ends and floor begins |
| Relative depth | `distillation/relative_kd.py` | 0.1 | Ordinal structure: which of two pixels is nearer |
| Obstacle ROI | `distillation/roi_kd.py` | 0.2 | The same residual, weighted `alpha` inside obstacle boxes |

`L_total` is their weighted sum. Four properties are worth knowing before you
change anything:

- **The ground-truth term cannot be disabled.** `DistillationLoss.__init__`
  raises if `lambda_gt <= 0`. A student supervised only by the teacher inherits
  the teacher's systematic errors with nothing to correct them.
- **Teacher and student have different channel widths**, so `models/projection.py`
  aligns them with a 1x1 convolution before the feature distance. Spatial
  dimensions already match at all three levels, so nothing is resampled.
- **Depth, boundary and relative terms work in log space by default.** A 0.1 m
  error at 0.5 m matters far more to a wheelchair than the same error at 8 m; a
  linear-space loss weights them identically.
- **The ROI term weights obstacle pixels by `alpha`, background by 1 — not 0.**
  Driving the background to zero would let the student drift on the floor plane,
  and the floor is what free path is measured against.

Every `lambda` above is the starting point written in the spec, **not a tuned
optimum**. STEP 11 searches for better values. Full derivations and the masking
policy are in [docs/knowledge_distillation.md](docs/knowledge_distillation.md).

### Running one KD term at a time

The presets in STEP 13 are **cumulative** (each row adds a term to the one
before). To measure a term *in isolation* instead — ground truth plus that term
and nothing else — disable the other four with `--set`:

```bash
# 1. Depth output only
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_depth_only \
    kd.feature.enabled=false kd.boundary.enabled=false kd.relative.enabled=false kd.roi.enabled=false

# 2. Feature only
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_feature_only \
    kd.depth.enabled=false kd.boundary.enabled=false kd.relative.enabled=false kd.roi.enabled=false

# 3. Boundary only
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_boundary_only \
    kd.depth.enabled=false kd.feature.enabled=false kd.relative.enabled=false kd.roi.enabled=false

# 4. Relative depth only
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_relative_only \
    kd.depth.enabled=false kd.feature.enabled=false kd.boundary.enabled=false kd.roi.enabled=false

# 5. Obstacle ROI only
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_roi_only \
    kd.depth.enabled=false kd.feature.enabled=false kd.boundary.enabled=false kd.relative.enabled=false
```

Three things to know before running these:

- **`experiment.name` is not optional here.** Unlike `--experiment B`, which
  appends its tag to the run name automatically, `--set` does not rename
  anything: leave it out and all five runs write to the same directory and
  overwrite each other.
- **The ground-truth term cannot be isolated away.** `kd.gt.enabled=false` is
  refused (spec section O), so every row above is *ground truth + one term*, and
  the honest comparison baseline for them is Experiment A, not zero.
- **Isolation and the cumulative presets answer different questions.** Isolation
  says what one term contributes alone; the presets say what it adds on top of
  the terms already present. A term can look weak alone and still help in
  combination. Report which one you ran.

Verify a run used the terms you intended — the trainer logs them at startup:

```
KD terms active: {'active_terms': ['gt', 'boundary'], 'lambdas': {'gt': 1.0, 'boundary': 0.1}}
```

## Training

### STEP 6c — Obstacle detector (optional)

Fine-tunes YOLO26n down to a single class, `obstacle`. Needs the dataset from
STEP 6b; `train_detection.py` refuses to start unless it declares `nc=1`.

```bash
python training/train_detection.py --config configs/detection.yaml
```

Skip this and the pipeline still runs: `configs/navigation.yaml` sets
`detection.class_agnostic: true`, which relabels every COCO detection from the
stock `yolo26n.pt` as `obstacle` with no retraining. Fine-tuning is the
higher-accuracy path once real indoor annotations exist. The detector also
supplies the boxes for the ROI term above, so train it before Experiment E if
you want that term learning from tuned boxes.

### STEP 7-8 — Teacher

Before fine-tuning, measure the released checkpoint **zero-shot** on your split.
If fine-tuning does not beat this, it is hurting rather than helping, and the
fix is a gentler recipe — not more epochs:

```bash
python evaluation/evaluate_depth.py --model yolo26x-depth.pt --data configs/data/sunrgbd.yaml
```

After any run, diagnose the curve before trusting the checkpoint:

```bash
python scripts/analyze_training.py --results outputs/experiments/<run>/results.csv
```

It reports which epoch produced `best.pt`, the val/train loss ratio over time,
and how many epochs were spent after the best one. A ratio that widens while
train loss keeps falling means regularize, not train longer.

The teacher is trained first and to convergence, because every KD experiment
then distils from the same frozen checkpoint — that is what keeps the ablation
rows comparable. Ultralytics fits the log-affine calibration (`cal_a`/`cal_b`)
on the validation split afterwards and writes it into the checkpoint; without
it the SILog training loss is scale-invariant and the output is not metric.

```bash
python training/train_teacher.py --config configs/teacher.yaml
python evaluation/evaluate_depth.py --model outputs/checkpoints/teacher_best.pt --data configs/data/sunrgbd.yaml
```

### STEP 9-10 — Student baseline (Experiment A)

The same student architecture trained on ground truth alone, with no teacher.
This row exists to establish the teacher-student gap **before** any distillation,
so the later KD rows have something to be measured against. Skipping it leaves
the ablation table unable to show that KD helped at all.

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

Experiment E: the student trained with the ground-truth term plus all five KD
terms described above, against the frozen teacher from STEP 7. The teacher runs
under `torch.no_grad()` and is never updated.

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

Each preset disables the KD terms that row excludes (`EXPERIMENT_PRESETS` in
`training/train_distillation.py`):

| Row | Command | KD terms active |
|---|---|---|
| A | `train_student_baseline.py` | none — ground truth only |
| B | `--experiment B` | depth output |
| C | `--experiment C` | + feature |
| D | `--experiment D` | + boundary |
| E | `--experiment E` | + relative + obstacle ROI (complete) |
| F | `train_teacher.py` | teacher, the distillation source |

A and F are not in the loop above because they are produced by STEP 9 and
STEP 7; `evaluation/ablation.py` collects all six rows from `outputs/experiments`.
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
# predict_video.py writes <video stem>_telemetry.csv automatically — no flag needed
python evaluation/evaluate_navigation.py --predictions outputs/predictions/test_telemetry.csv --ground-truth data/nav_gt.csv
python -m pytest tests/ -q
python evaluation/benchmark_latency.py --depth-model outputs/checkpoints/student_distilled_best.pt
python evaluation/validate_numerical_consistency.py --model outputs/checkpoints/student_distilled_best.pt --formats onnx
```

`--ground-truth` is a CSV **you label by hand**, with columns `frame_id` and
`gt_command` (`FORWARD`, `TURN_LEFT`, `TURN_RIGHT` or `STOP`); it is joined to
the telemetry on `frame_id`. Without it the navigation metrics cannot be
computed — there is no automatic ground truth for a free-path decision.

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
| Unsure whether RGB and depth actually pair up | `python datasets/scripts/diagnose_pairing.py --data configs/data/sunrgbd.yaml` prints the checked-out commit, per-split counts by suffix, and for sample images the exact path Ultralytics resolves plus `os.path.isfile`. Read-only. |
| `ignoring image with missing depth map ....npy` then `No labels found` | RGB images have no paired depth map. Ultralytics probes `depth/<split>/<stem>.png` first and only names `.npy` as the fallback, so the `.npy` in that message is a symptom, not the format. The trainers now catch this in under a second and print the per-split counts, the suffixes actually present, and the fix: `repair_orphaned_pairs.py` (drops half-pairs) or a re-run of `prepare_sunrgbd.py` (restores every scene). |
| Student loss decreases but nothing learns | `.pt` files load with `requires_grad=False`. Use `StudentDepthModel`, which calls `unfreeze()`. |
| `cannot pickle '_thread.lock'` | A KD criterion stored on the model. Keep it in the module registry — see `training/kd_trainer.py`. |
| δ1 looks implausibly good | You are reading MODE 2 (aligned). MODE 1 is the metric number. |
| Euclidean distance refused | `configs/camera.yaml` still has `calibrated: false`. Run the calibration script. |
| CUDA requested but unavailable | Falls back to CPU with a warning. Pass `--device cpu` to silence it. |
| ONNX depths differ from PyTorch | Letterboxing. Use square inputs at exactly `imgsz`. |

## Complete runbook

Every command in this repository, in the order you run them. Each is explained
in its own STEP above; this section exists so the whole pipeline can be read —
or copied — from one place. Paths assume you run from the repository root.

```bash
# ---------- Setup (STEP 1-4) ----------
git clone https://github.com/aauliariza/autonomous-wheelchair-v1.0.git
cd autonomous-wheelchair-v1.0
conda env create -f environment.yml && conda activate wheelchair   # or python -m venv .venv
pip install -r requirements.txt
python scripts/inspect_model.py --model yolo26n-depth.pt           # verify the install

# ---------- Depth dataset (STEP 5-6) ----------
python datasets/scripts/prepare_sunrgbd.py \
    --convert-allsplit /path/to/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat \
    --split-file configs/data/sunrgbd_official_split.json
python datasets/scripts/prepare_sunrgbd.py \
    --source /path/to/SUNRGBD \
    --split official --split-file configs/data/sunrgbd_official_split.json \
    --output datasets/sunrgbd --config-out configs/data/sunrgbd.yaml
python datasets/scripts/verify_dataset.py --data configs/data/sunrgbd.yaml \
    --report outputs/dataset_report.json

# ---------- Obstacle dataset + detector, optional (STEP 6b-6c) ----------
python datasets/scripts/convert_sunrgbd_obstacle.py \
    --source /path/to/SUNRGBD --meta /path/to/SUNRGBDMeta2DBB_v2.mat \
    --split-file configs/data/sunrgbd_official_split.json --output datasets/obstacle
python training/train_detection.py --config configs/detection.yaml

# ---------- Teacher, Experiment F (STEP 7-8) ----------
python training/train_teacher.py --config configs/teacher.yaml
python evaluation/evaluate_depth.py --model outputs/checkpoints/teacher_best.pt \
    --data configs/data/sunrgbd.yaml

# ---------- Student baseline, Experiment A (STEP 9-10) ----------
python training/train_student_baseline.py --config configs/student.yaml
python evaluation/evaluate_depth.py --model outputs/checkpoints/student_baseline_best.pt \
    --data configs/data/sunrgbd.yaml

# ---------- Hyperparameter search (STEP 11) ----------
python tuning/optuna_study.py --config configs/optuna.yaml
python tuning/analyze_trials.py \
    --storage sqlite:///outputs/optuna/kd_depth_search.db --plots

# ---------- Distilled student, Experiment E (STEP 12) ----------
python training/train_distillation.py --config configs/distillation.yaml
# or, with the tuned lambdas found above:
python training/train_distillation.py --config outputs/optuna/best_config.yaml

# ---------- Cumulative ablation B-E (STEP 13) ----------
for exp in B C D E; do
    python training/train_distillation.py --config configs/distillation.yaml --experiment $exp
done

# ---------- One KD term at a time (isolation) ----------
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_depth_only \
    kd.feature.enabled=false kd.boundary.enabled=false kd.relative.enabled=false kd.roi.enabled=false
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_feature_only \
    kd.depth.enabled=false kd.boundary.enabled=false kd.relative.enabled=false kd.roi.enabled=false
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_boundary_only \
    kd.depth.enabled=false kd.feature.enabled=false kd.relative.enabled=false kd.roi.enabled=false
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_relative_only \
    kd.depth.enabled=false kd.feature.enabled=false kd.boundary.enabled=false kd.roi.enabled=false
python training/train_distillation.py --config configs/distillation.yaml \
    --set experiment.name=kd_roi_only \
    kd.depth.enabled=false kd.feature.enabled=false kd.boundary.enabled=false kd.relative.enabled=false

# ---------- Collect the ablation table ----------
python evaluation/ablation.py --experiments outputs/experiments --output docs/ablation

# ---------- Evaluation (STEP 14-16) ----------
python evaluation/evaluate_depth.py --model outputs/checkpoints/student_distilled_best.pt \
    --data configs/data/sunrgbd.yaml
python calibration/camera_calibration.py --capture 0 --pattern-size 9 6 --square-size 0.025
python evaluation/evaluate_distance.py --model outputs/checkpoints/student_distilled_best.pt \
    --data configs/data/sunrgbd.yaml

# ---------- Inference (STEP 17-19) ----------
python inference/predict_image.py --source data/test.jpg --save-depth
python inference/predict_video.py --source data/test.mp4 --show-depth
python inference/webcam.py --camera 0

# ---------- Navigation, tests, benchmark, export (STEP 20-23) ----------
python evaluation/evaluate_navigation.py \
    --predictions outputs/predictions/test_telemetry.csv --ground-truth data/nav_gt.csv
python -m pytest tests/ -q
python evaluation/benchmark_latency.py \
    --depth-model outputs/checkpoints/student_distilled_best.pt
python evaluation/validate_numerical_consistency.py \
    --model outputs/checkpoints/student_distilled_best.pt --formats onnx

# ---------- Diagnostics, when something is wrong ----------
python datasets/scripts/diagnose_pairing.py --data configs/data/sunrgbd.yaml
python datasets/scripts/repair_orphaned_pairs.py --data datasets/sunrgbd            # dry run
python datasets/scripts/repair_orphaned_pairs.py --data datasets/sunrgbd --apply
```

Add `--set train.device=cpu train.epochs=1 train.batch=2` to any training
command for a fast smoke test before committing a GPU to a full run. Note that
`--set` is accepted by the four `training/*.py` scripts only; the tuning and
evaluation scripts take explicit flags instead.

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
