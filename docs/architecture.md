# System Architecture

## Overview

```
                          RGB Camera
                              |
                      preprocessing (letterbox, normalize)
                              |
              +---------------+---------------+
              |                               |
              v                               v
     YOLO26n Detection                YOLO26n-Depth Student
     (all classes -> obstacle)        (metric depth, H/4 x W/4)
              |                               ^
              |                               | Knowledge Distillation
              |                               | (training only)
              |                               |
              |                        YOLO26x-Depth Teacher
              |                            (frozen)
              |                               |
              +---------------+---------------+
                              |
                     Obstacle-Depth Fusion
                              |
                  Obstacle Distance Estimation
                   (inner-60% bbox ROI, median)
                              |
                     60% Global Navigation ROI
                              |
                    5-Sector Occupancy Mapping
                     FL | L | CTR | R | FR
                              |
                      Safety Threshold (1.0 m)
                              |
                      Free-Path Selection
                              |
                   Majority-Vote Hysteresis (N=3)
                              |
                       Safety Monitor Override
                              |
            FORWARD / TURN_LEFT / TURN_RIGHT / STOP / EMERGENCY_STOP
                              |
                     [ motor controller ]
                  independent safety layer
                    (outside this software)
```

The teacher participates only in training. At deployment the pipeline is
YOLO26n + YOLO26n-Depth.

## Verified environment

Every architectural fact below was measured in this repository, not taken from
documentation. Reproduce with `python scripts/inspect_model.py`.

| Item | Value | How verified |
|---|---|---|
| Ultralytics | 8.4.138 | `ultralytics.__version__` |
| PyTorch | 2.14.0 | `torch.__version__` |
| Depth task module | `ultralytics/models/yolo/depth/` | source read |
| Depth head class | `ultralytics.nn.modules.head.Depth` | source read |
| Model config | `cfg/models/26/yolo26-depth.yaml` | file present |

## Model facts (measured)

| | Teacher `yolo26x-depth` | Student `yolo26n-depth` |
|---|---|---|
| Parameters | 57,022,241 | 6,355,617 |
| Ratio | 8.97x | 1x |
| KD tap layers (`head.f`) | `[16, 19, 22]` | `[16, 19, 22]` |
| Tap channels | 384 / 768 / 768 | 64 / 128 / 256 |
| Tap spatial @640 | 80² / 40² / 20² | 80² / 40² / 20² |
| Head calibration `cal_b` | −0.3167 | −0.1938 |

Detector `yolo26n.pt`: 2,572,280 parameters, COCO-80 pretrained.

Spatial dimensions are **identical** between teacher and student at every tap, so
feature distillation needs channel projection only — no spatial resampling.

## The Depth head

Read from `ultralytics/nn/modules/head.py`:

```python
out = self.head(fused)  # (B, 1, H/4, W/4)
depth = torch.exp(out.clamp(-4.0, 5.0))  # -> [0.018, 148.4] m

if self.training:
    return {"depth": depth}  # dict, UNCALIBRATED
depth = depth.pow(self.cal_a) * self.cal_b.exp()  # eval only
if self.export:
    depth = F.interpolate(depth, scale_factor=4.0)  # export only
return depth
```

Four consequences the pipeline is built around:

1. **The head is internally log-depth, but Ultralytics already applies `exp()`.**
   No code in this repository exponentiates raw head activations.
2. **Output resolution is input/4.** Verified: 640×640 in → `(1,1,160,160)` out.
3. **Calibration applies in eval mode only.** Training returns a dict without it.
   Teacher (always eval) and student (train mode) therefore live in *different
   scale spaces* during KD. `teacher_space` in `configs/distillation.yaml`
   selects which space the KD terms operate in.
4. **Released checkpoints are not scale-identity.** Both carry a fitted `cal_b`,
   so "raw network output = metres" is false for them.

## Where metric scale comes from

The Ultralytics training loss is `DepthLoss26` = SILog + multi-scale gradient
matching, with `dlam: 1.0` meaning **fully scale-invariant**. The loss constrains
relative structure, not absolute scale.

Absolute scale is supplied afterwards by a two-parameter log-affine calibration
`d' = exp(a·log d + b)` fitted on the validation split by
`ultralytics/models/yolo/depth/calibrate.py`, under a "calibrate only if it
helps" cross-validated policy.

This is why obstacle distance must be evaluated with `align="none"`. See
[depth_to_distance.md](depth_to_distance.md).

## Extension, not modification

No Ultralytics source file is patched. The KD objective is injected through one
documented extension point:

```
BaseModel.loss(batch, preds)  ->  self.criterion(preds, batch)
```

`KDDepthModel` subclasses `DepthModel` and overrides `loss()`, so the stock
`DepthTrainer` — dataloaders, EMA, AMP, DDP, validation, post-training
calibration — runs unchanged. Loss column names are derived automatically from
the criterion's returned dict keys.

### Three integration hazards found by running it

| Hazard | Symptom | Resolution |
|---|---|---|
| Criterion stored on the model | `TypeError: cannot pickle '_thread.lock'` at the first checkpoint save | Criterion held in a module-level registry |
| Closure-local model class | `Can't pickle local object ... KDDepthModel` | Class defined at module scope |
| Forward hooks on the student | `Can't pickle local object ... hook` | Student features captured in an overridden `_predict_once`; the teacher still uses hooks since it is never serialized |

A fourth, subtler one: `.pt` checkpoints load with `requires_grad=False` on every
parameter. A KD loop that does not re-enable them computes a valid loss, runs
`backward()` without error, and trains nothing. `StudentDepthModel.assert_trainable()`
guards it.

## Module map

| Directory | Responsibility |
|---|---|
| `configs/` | Every tunable parameter; no hyperparameter is hard-coded |
| `datasets/scripts/` | SUN RGB-D preparation, verification, obstacle conversion |
| `models/` | Teacher/student wrappers, feature taps, projections, complexity |
| `distillation/` | The five KD terms and the combined objective |
| `training/` | Teacher, baseline, KD and detector training entry points |
| `tuning/` | Optuna search space, composite objective, trial analysis |
| `evaluation/` | Depth, distance, navigation, latency, ablation, export validation |
| `navigation/` | ROI, sectors, fusion, distance, free path, hysteresis, safety |
| `inference/` | Unified pipeline and the image/video/webcam CLIs |
| `calibration/` | Intrinsic calibration and camera geometry |
| `visualization/` | Depth colorization and the navigation HUD |
| `tests/` | 188 unit tests on synthetic data; no dataset or checkpoint needed |

## Real-time budget

Measured on CPU (Intel Xeon @ 2.80 GHz, 4 cores, no GPU), `imgsz=320`:

| Stage | mean (ms) | P95 | P99 |
|---|---|---|---|
| detection | 28.51 | 29.11 | 29.12 |
| depth | 50.09 | 51.59 | 51.68 |
| fusion + distance | 0.03 | 0.04 | 0.04 |
| free-path selection | 0.03 | 0.05 | 0.06 |
| **total** | **78.74** | **80.47** | **80.86** |

12.7 FPS on CPU. GPU figures are **NOT MEASURED** — no GPU was available in the
environment where this was developed. Navigation logic costs 0.06 ms, so latency
is entirely dominated by the two networks.

P99 rather than the mean governs `temporal_safety.max_inference_latency_s`: a
frame exceeding that budget forces a STOP.
