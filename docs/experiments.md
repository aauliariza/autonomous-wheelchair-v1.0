# Experimental Results

## Status of this document

**No full-scale training run has been performed.** Every result table below is a
template whose cells read `NOT MEASURED`.

Two hard constraints in the development environment prevented full training, and
per spec section BK no value has been invented to fill the gap:

1. **SUN RGB-D was unreachable.** `rgbd.cs.princeton.edu` is denied by the
   environment's egress proxy (`CONNECT` returns 403, confirmed proxy-side). The
   6.5 GB archive could not be downloaded.
2. **No GPU.** CPU-only, 4 cores, 15 GB RAM. Fine-tuning the 57 M-parameter
   teacher on ~9 k images is not feasible by orders of magnitude.

What **was** measured — on real data, in this environment — is reported in the
"Verified measurements" section, clearly separated from the empty tables.

To fill the tables, run the commands in the README on GPU hardware with SUN RGB-D
prepared. `evaluation/ablation.py` regenerates them automatically from each run's
`metrics.json`.

---

## Table 1 — Depth accuracy (MODE 1: metric, `align=none`)

| Exp | Model | KD terms | AbsRel | SqRel | RMSE | RMSElog | δ1 | δ2 | δ3 | SILog |
|---|---|---|---|---|---|---|---|---|---|---|
| A | YOLO26n-Depth | none (baseline) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| B | YOLO26n-Depth | + output | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| C | YOLO26n-Depth | + feature | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| D | YOLO26n-Depth | + boundary | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| E | YOLO26n-Depth | complete proposed | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| F | YOLO26x-Depth | teacher | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

Report MODE 2 (aligned) in a **separate** column set. Do not mix the two: on this
project's own measurements alignment inflated δ1 by 4.9×.

## Table 2 — Complexity and latency

| Exp | Params | GFLOPs | Size (MB) | Latency mean (ms) | P95 | P99 | FPS | Peak GPU mem |
|---|---|---|---|---|---|---|---|---|
| A/B/C/D/E | 6,355,617 | 8.142 @320 | 24.32 | NOT MEASURED (GPU) | — | — | — | NOT MEASURED |
| F (teacher) | 57,022,241 | 52.419 @320 | 217.85 | NOT MEASURED (GPU) | — | — | — | NOT MEASURED |

Parameters, GFLOPs and size **are** measured (`scripts/inspect_model.py`). GPU
latency is not — CPU figures appear below.

## Table 3 — Obstacle distance accuracy

| Exp | MAE (m) | RMSE (m) | MAPE (%) | Median AE (m) | ±10% | ±20% | ±30% |
|---|---|---|---|---|---|---|---|
| A–F | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

Ground truth: median of valid GT depth in the inner-60% bbox ROI, axial depth,
same robust reduction as the prediction.

## Table 4 — Navigation and safety

| Exp | Command acc. | **Unsafe rate** | Over-cautious | FORWARD | TURN_L | TURN_R | STOP |
|---|---|---|---|---|---|---|---|
| A–F | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

**Unsafe rate is the primary metric, not accuracy.** Errors are asymmetric: saying
FORWARD when the truth is STOP can injure the occupant; saying STOP when the truth
is FORWARD is merely inconvenient. A model at 95% accuracy whose errors are all
unsafe is worse than one at 85% whose errors are all over-cautious.

Requires ground-truth per-frame commands, which must be human-annotated.

## Table 5 — Training configuration

| Field | Value |
|---|---|
| Dataset | SUN RGB-D (official split) |
| Teacher | YOLO26x-Depth, 57,022,241 params |
| Student | YOLO26n-Depth, 6,355,617 params |
| Epochs / batch / imgsz | see `configs/*.yaml` |
| Optimizer / LR | AdamW / 1e-3 (student), 5e-4 (teacher) |
| Seed | 42 |
| Depth loss | SILog (`dlog=1.0`) + gradient (`dgrad=0.5`), `dlam=1.0` |
| Actual run | NOT PERFORMED |

---

# Verified measurements

Everything below was measured in this session on real data. No value is estimated.

## Model complexity (measured)

| | Teacher `yolo26x-depth` | Student `yolo26n-depth` | Detector `yolo26n` |
|---|---|---|---|
| Parameters | 57,022,241 | 6,355,617 | 2,572,280 |
| GFLOPs @320 | 52.419 | 8.142 | 1.529 |
| Size fp32 (MB) | 217.85 | 24.32 | 9.89 |
| Compression | 1× | **8.97× fewer params** | — |

## Pipeline latency (measured, CPU)

Intel Xeon @ 2.80 GHz, 4 cores, no GPU, `imgsz=320`, 8 runs after 3 warm-up:

| Stage | mean (ms) | median | P95 | P99 | max |
|---|---|---|---|---|---|
| detection | 28.51 | 28.33 | 29.11 | 29.12 | 29.13 |
| depth | 50.09 | 50.07 | 51.59 | 51.68 | 51.70 |
| fusion + distance | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 |
| free-path selection | 0.03 | 0.02 | 0.05 | 0.06 | 0.06 |
| **total** | **78.74** | 78.74 | 80.47 | 80.86 | 80.96 |

**12.70 FPS on CPU.** Navigation logic costs 0.06 ms — 0.08% of the budget — so
latency is entirely dominated by the two networks. GPU figures: NOT MEASURED.

## The alignment gap (measured)

Pretrained `yolo26n-depth.pt`, `depth8` val split (4 SUN RGB-D images, `imgsz=320`):

| metric | MODE 1 (metric) | MODE 2 (aligned) | gap |
|---|---|---|---|
| abs_rel | 1.1062 | 0.1942 | −0.9119 |
| sq_rel | 3.5858 | 0.1333 | −3.4525 |
| rmse | 2.1276 | 0.4586 | −1.6690 |
| rmse_log | 0.7093 | 0.2241 | −0.4852 |
| **delta1** | **0.1455** | **0.7136** | **+0.5681** |
| delta2 | 0.3659 | 0.9184 | +0.5525 |
| delta3 | 0.5145 | 0.9884 | +0.4740 |
| silog | 19.5342 | 19.7492 | +0.2149 |

Median alignment inflates δ1 by **4.9×** on identical predictions. This is the
single most important methodological result in this repository: it is why
obstacle distance must be evaluated with `align="none"`.

> Caveat: 4 images on an out-of-domain pretrained checkpoint. The *magnitude* is
> not a benchmark figure; the *phenomenon* is what matters and is reproducible.

## Obstacle distance on the pretrained student (measured)

36 evaluation regions over the `depth8` val split, axial mode, uncalibrated for
this data:

| metric | value |
|---|---|
| MAE | 1.9809 m |
| RMSE | 2.5044 m |
| MAPE | 105.11% |
| within ±10% / ±20% / ±30% | 2.8% / 13.9% / 22.2% |
| mean signed error | +1.5044 m |
| **overestimated fraction** | **72.2%** |

The bias direction is the safety-relevant finding: the model **overestimates**
distance in 72% of regions, and overestimation is what causes collisions. A
model tuned on this dataset with proper calibration should be re-measured.

## Export numerical validation (measured)

`yolo26n-depth.pt` → ONNX, `imgsz=320`:

| Level | comparison | max abs diff | verdict |
|---|---|---|---|
| 0 | raw graph vs PyTorch | 7.39×10⁻⁶ | exact |
| 1 | pipeline, square 320×320 | 5.30×10⁻⁵ | exact |
| 2 | pipeline, 640×480 letterboxed | 2.77 m (mean 2.28 m) | **diverges** |

**Diagnosis:** the exported graph is numerically correct. The divergence is
letterbox pre/post-processing differing between backends, not an export defect.
The obstacle-distance consequence is **2.18 m** — more than twice the 1.0 m
safety threshold.

**Action before deploying ONNX:** feed square inputs at exactly `imgsz`, or
re-implement letterbox removal for the exported backend. A file-exists check
would have missed this entirely (spec section BG).

## Camera calibration (validated)

Synthetic checkerboard rendered under known intrinsics, 18 views:

| parameter | true | recovered | error |
|---|---|---|---|
| fx | 600.00 | 590.80 | 1.53% |
| fy | 600.00 | 591.46 | 1.42% |
| cx | 320.00 | 320.75 | 0.23% |
| cy | 240.00 | 243.24 | 1.35% |
| RMS reprojection | — | 0.4563 px | good (< 0.5) |

## KD training (smoke-verified)

2 epochs on the `depth8` fixture (8 SUN RGB-D images), all six terms active,
CPU, `imgsz=320`:

```
Epoch  gt_loss  depth_loss  feature_loss  boundary_loss  relative_loss  roi_loss
1/2     0.8418      0.6595       0.02597        0.01124         0.6263    0.6062
```

Training, validation, checkpointing, auto-calibration (`scale-only`, `b=−0.6279`)
and export all completed. The exported student loads and predicts under stock
Ultralytics with no dependency on this repository.

This proves the pipeline **runs**. It is not a research result — 8 images and 2
epochs support no claim about KD effectiveness.

## Test suite

188 unit tests, all passing, on synthetic data. `ruff check` and `ruff format`
clean across 67 files.

---

## Reproducibility

Every run writes `outputs/experiment_metadata.json` capturing git commit, Python,
PyTorch, Ultralytics and CUDA versions, GPU name, OS, seed and a config hash.
Environment as developed:

| | |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.14.0+cu130 |
| Ultralytics | 8.4.138 |
| NumPy | 2.4.6 |
| CUDA available | False (CPU-only) |
| Seed | 42 |
