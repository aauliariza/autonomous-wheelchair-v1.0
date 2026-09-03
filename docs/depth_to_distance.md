# From Depth Map to Obstacle Distance

This is the most error-prone part of the system. Three distinct mistakes are
possible, and each one silently produces a plausible number.

## Mistake 1 — treating depth as distance

A depth map stores **axial depth**: the distance along the camera's optical axis
(the Z coordinate). It is **not** the Euclidean distance from the camera centre
to the surface point. The two coincide only at the principal point.

Given intrinsics `fx, fy, cx, cy` and axial depth `Z` at pixel `(u, v)`:

```
X = (u − cx)/fx · Z
Y = (v − cy)/fy · Z
Z = Z

d_euclidean = √(X² + Y² + Z²) = Z·√(1 + ((u−cx)/fx)² + ((v−cy)/fy)²)
```

### Measured magnitude

640×480, `fx = fy = 554.26` (60° HFOV), axial depth 1.000 m:

| pixel | Euclidean | error |
|---|---|---|
| principal point (320, 240) | 1.0000 m | 0.0% |
| mid-edge (0, 240) | 1.1547 m | +15.5% |
| corner (0, 0) | 1.2332 m | **+23.3%** |

At a 1.0 m safety threshold, 0.23 m is easily enough to flip a STOP into a
FORWARD.

### What this project does

Both quantities are computed and reported **separately** and never interchanged.
`navigation.safety.distance_mode` selects which one gates the safety threshold:

- **`axial`** (default) — correct for forward clearance. A wheelchair advancing
  along its optical axis is limited by Z, not by slant range.
- **`euclidean`** — true range to the surface point; reported alongside regardless.

Euclidean output is **refused** when intrinsics are placeholders
(`calibrated: false`), rather than silently substituting axial depth.

## Mistake 2 — reading depth from one pixel, or the whole box

A single pixel is not a measurement: depth maps are noisiest exactly at object
boundaries. The full bounding box is no better — its edges straddle the
background, so depth sampled there mixes the obstacle with the wall behind it and
biases the estimate *far*, which is the dangerous direction.

The pipeline uses the **inner 60%** of each box (spec section V):

```
x1_inner = x1 + 0.20·w      x2_inner = x2 − 0.20·w
y1_inner = y1 + 0.20·h      y2_inner = y2 − 0.20·h
```

then reduces it robustly:

1. drop invalid depths — `0`, `NaN`, `Inf`, `≤ 0`, and outside `[0.1, 10.0]` m
2. clip the 5th/95th percentiles
3. take the **median** (tolerates up to 50% contamination; a box always contains
   some background)
4. report dispersion as `MAD × 1.4826` — MAD rather than standard deviation
   because a bimodal patch (obstacle + background) inflates σ and would
   understate confidence in the median
5. if the valid fraction is below `min_valid_ratio` (default 0.30), report
   **INVALID**

An INVALID distance is **not** "no obstacle". The safety layer treats it as
blocking.

> Small boxes are protected by `min_size_px`: a distant obstacle a few pixels
> wide would otherwise inset to nothing and be reported as having no valid depth
> — precisely the far-obstacle case that must not silently disappear.

### Two ROIs, deliberately separate

| | purpose | default |
|---|---|---|
| `GLOBAL_NAVIGATION_ROI` | which obstacles are *relevant* to the path | central 60% of width |
| `BBOX_DEPTH_ROI` | which pixels are *read* for one obstacle's distance | central 60% of each box |

Both default to 60% but answer different questions and are configured
independently. Conflating them is a correctness bug, not a style choice.

## Mistake 3 — evaluating with scale alignment

Monocular depth benchmarks conventionally rescale each prediction by
`median(gt)/median(pred)` before scoring, because most monocular models are
scale-*ambiguous*: they recover structure, not metres.

**This research is about absolute metric distance, so that protocol hides exactly
the error that matters.**

### Measured on this project

A pure 2× scale error — perfect structure, wrong scale — scores:

```
align="none"    delta1 = 0.0000     (correctly punished)
align="median"  delta1 = 1.0000     (completely invisible)
```

And on the real pretrained student over the `depth8` split:

```
                MODE 1 (metric)   MODE 2 (aligned)
abs_rel                  1.1062             0.1942
rmse                     2.1276             0.4586
delta1                   0.1455             0.7136
```

Median alignment inflates δ1 by **4.9×** on identical predictions.

`DepthEvaluator` computes both in one pass and reports the gap as a result in its
own right. **Never quote an aligned figure as evidence of metric accuracy.**

> Note: Ultralytics' own `DepthValidator` uses `align="median"` by default. Its
> reported δ1 during training is therefore scale-invariant and is *not* the metric
> accuracy this project targets.

## Where metric scale actually comes from

The training loss (`DepthLoss26` = SILog + gradient matching, `dlam: 1.0`) is
**fully scale-invariant**. It constrains relative structure only.

Absolute scale comes from a two-parameter log-affine calibration
`d' = exp(a·log d + b)` fitted on the validation split after training, under a
cross-validated "calibrate only if it helps" policy.

Three practical consequences:

1. Calibration is **dataset-dependent**. A model calibrated on SUN RGB-D is not
   calibrated for your camera and room.
2. To push the network itself toward metric output, lower `dlam` (e.g. 0.85) so
   the loss stops being fully scale-invariant.
3. `teacher_space: calibrated` lets KD inject the teacher's metric scale into the
   student, which the scale-invariant GT loss cannot supply.

## Ground truth for distance evaluation

Stated explicitly, as spec section AK requires:

> Ground-truth obstacle distance is the **median of valid ground-truth depth
> within the same inner-60% bounding-box ROI**, reduced by the **same** robust
> statistics as the prediction, in the **same** distance mode.

Prediction and ground truth therefore differ *only* in their depth source
(network vs sensor). Any other choice would confound distance error with a
difference in the reduction itself. This definition is written into every
`distance_metrics.json`.

## Camera calibration

Run `calibration/camera_calibration.py` on 15–30 checkerboard views covering the
whole frame, including the corners where the distortion model is least
constrained. RMS reprojection error below ~0.5 px is good; above ~1.0 px usually
means too few views, a non-flat board, or a wrong `--square-size`.

`--square-size` must be the real square side **in metres**. A wrong value leaves
the intrinsics self-consistent but scales every derived metric distance
proportionally — a silent, systematic error.

Validated against a synthetic checkerboard rendered under known intrinsics
(`fx = fy = 600`, `cx = 320`, `cy = 240`): recovered `fx = 590.80`, `cx = 320.75`,
`cy = 243.24` at 0.4563 px RMS.
