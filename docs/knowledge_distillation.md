# Knowledge Distillation

## Objective

```
L_total = λ_gt       · L_GT
        + λ_depth    · L_depthKD
        + λ_feature  · L_feature
        + λ_boundary · L_boundary
        + λ_relative · L_relative
        + λ_roi      · L_roiKD
```

`L_GT` is **mandatory**. `DistillationLoss.__init__` raises if `λ_gt ≤ 0`. If the
teacher were the only supervision, the student would inherit the teacher's
systematic errors with no signal able to correct them, and would be bounded above
by teacher accuracy by construction.

Starting weights in `configs/distillation.yaml`:

| term | λ | rationale |
|---|---|---|
| `gt` | 1.0 | reference scale for the others |
| `depth` | 0.5 | strongest KD signal, but must not outweigh ground truth |
| `feature` | 0.2 | dense and high-magnitude; needs a smaller weight |
| `boundary` | 0.1 | a gradient term, naturally larger than the value term |
| `relative` | 0.1 | ordinal, complementary rather than primary |
| `roi` | 0.2 | overlaps `depth`; weighted to emphasise, not dominate |

**These are a starting point, not an optimum.** Search them with
`tuning/optuna_study.py` (spec section P).

## The scale-space hazard

The Depth head applies its log-affine calibration **only in eval mode**:

| mode | returns | calibration |
|---|---|---|
| `train()` | `dict{"depth"}` | **not applied** |
| `eval()` | tensor | **applied** |

The teacher always runs in eval mode, the student in train mode. A naive
teacher→student comparison therefore spans two different scale spaces. Measured
on the released checkpoints: student `cal_b = −0.1938`, teacher `cal_b = −0.3167`
— different from each other and both non-zero.

`teacher_space` selects the resolution:

- **`calibrated`** (default) — the teacher keeps its fitted metric scale, so the
  student is pulled toward metric output. This is desirable here precisely
  because the ground-truth SILog loss is scale-invariant and supplies no absolute
  scale of its own.
- **`raw`** — teacher calibration is reset to identity, distilling pure relative
  structure.

## Term 1 — Depth output (spec I, J)

```
L_depthKD = mean over valid pixels of  ρ( log(D_S) − log(D_T) )
```

`ρ ∈ {L1, SmoothL1, BerHu}`, configurable. Log space is the default because it
equalizes *relative* error across the depth range: a 0.1 m error at 0.5 m matters
far more to a wheelchair than the same error at 8 m, and a linear-space loss
weights them identically.

The head is internally log-depth but **Ultralytics already applies `exp()`**, so
this module never touches raw activations — it re-takes `log()` of the returned
metric depth.

## Term 2 — Feature (spec K)

```
L_feature = Σᵢ d( Pᵢ(F_Tᵢ), F_Sᵢ )
```

Tap layers `[16, 19, 22]` are read from `head.f` by introspection, never guessed.
Measured widths: teacher 384/768/768, student 64/128/256, with **identical
spatial dimensions**, so `Pᵢ` is a 1×1 convolution and no resampling occurs.

A 1×1 projection is used deliberately rather than a deeper adapter: a powerful
adapter could mask a poor feature match and make the term measure the adapter's
capacity instead of the student's agreement with the teacher.

`normalize: true` L2-normalizes each channel before the distance. The teacher is
~9× larger with a different activation scale; without normalization a handful of
high-magnitude channels dominate the gradient and the term degenerates into scale
matching.

Projections are **auxiliary**: they are optimized alongside the student but are
not part of the deployed model, so they affect neither its parameter count nor
its latency. They must be in the optimizer — a frozen random projection would
make the term measure distance to noise.

> Available option: both models' `head.proj` already map every level to 256
> channels, giving a projection-free comparison point
> (`use_head_projection: true`). `ProjectionBank` inserts `nn.Identity` where
> widths already match, costing zero parameters.

## Term 3 — Boundary (spec L)

```
Gx = ∂ₓD,  Gy = ∂_yD,  G = √(Gx² + Gy²)
L_boundary = SmoothL1(G_S, G_T)
```

Free-path selection depends on where an obstacle *ends* and free floor begins. A
student matching the teacher's average depth but smearing the discontinuity at a
table edge reports a plausible mean distance while placing the boundary in the
wrong sector — the error mode most likely to produce an unsafe FORWARD.

A finite difference is only valid where **both** contributing pixels are valid;
otherwise the "edge" is an artifact of a data hole rather than scene structure.
The mask is eroded accordingly. `√(·+ε)` keeps the gradient defined on flat
floors and walls, which cover most of an indoor frame.

## Term 4 — Relative depth (spec M)

```
L_relative = mean over sampled pairs of  softplus( −r_ij · (log D_S(j) − log D_S(i)) )
```

with `r_ij ∈ {−1, 0, +1}` the teacher's ordinal label.

Obstacle avoidance is fundamentally comparative, and ordinal structure survives
scale error, so this keeps the student's *ranking* correct even where absolute
scale drifts.

Exhaustive pairing over a 160×160 map would be ~3.3×10⁸ pairs per image. Pairs
are **randomly sampled** from valid pixels with a `num_pairs` budget (spec
section M explicitly requires bounded sampling), which fixes cost regardless of
resolution. Pairs whose teacher log-ratio falls below `tolerance` are labelled
"equal" and excluded: forcing an order on pixels the teacher considers coplanar
injects pure noise.

## Term 5 — Obstacle ROI (spec N)

```
w(p) = α  if p ∈ obstacle ROI, else 1
L_roiKD = Σ w(p)·ρ(p) / Σ w(p)
```

A uniform pixelwise loss optimizes for floor, walls and ceiling, which occupy
most of an indoor frame and are irrelevant to clearance. The pixels that decide
FORWARD versus STOP are on obstacles.

Background is weighted 1, **not 0**, on purpose: driving the background term to
zero would let the student drift arbitrarily on the floor plane, and the floor is
what free path is measured against. `α < 1` is rejected outright.

Boxes come from YOLO26n with every class collapsed to `obstacle`, using the same
inner-60% ROI convention as `navigation.distance`, so the term emphasizes exactly
the pixels the deployed distance estimator reads. Detections are cached after the
first epoch — they depend only on the RGB input, not on the student's weights, so
a 100-epoch run otherwise pays 99 redundant detector passes per image.

## Masking policy

Every term is evaluated **only** where ground truth is valid. In SUN RGB-D,
invalid depth means "the sensor returned nothing" — not "the surface is at 0 m".
Distilling there teaches the student to reproduce the teacher's hallucinations in
exactly the regions (glass, specular surfaces, far walls) where the teacher is
least reliable, and those regions are disproportionately the dangerous ones for a
wheelchair.

Masked reductions divide by the valid count and return a graph-connected zero
when nothing is valid, so an all-invalid batch contributes nothing rather than
producing NaN.

## Ablation (spec Q)

Row selection is a configuration change, never a code change:

| Exp | Configuration | Command |
|---|---|---|
| A | baseline, no KD | `train_student_baseline.py --config configs/student.yaml` |
| B | + output KD | `train_distillation.py --experiment B` |
| C | + feature KD | `train_distillation.py --experiment C` |
| D | + boundary KD | `train_distillation.py --experiment D` |
| E | complete proposed KD | `train_distillation.py --experiment E` |
| F | teacher | `train_teacher.py --config configs/teacher.yaml` |

For the comparison to be valid, A–E must share schedule, augmentation, optimizer
and seed; only the loss may differ.

## Verified behaviour

A 2-epoch run on the `depth8` fixture, all six terms active:

```
Epoch  gt_loss  depth_loss  feature_loss  boundary_loss  relative_loss  roi_loss
1/2     0.8418      0.6595       0.02597        0.01124         0.6263    0.6062
```

All terms finite and decreasing; checkpointing, validation, auto-calibration and
export completed. The exported student loads under stock Ultralytics with no
dependency on this repository.

Full-scale SUN RGB-D KD results are **NOT MEASURED** — see
[experiments.md](experiments.md).
