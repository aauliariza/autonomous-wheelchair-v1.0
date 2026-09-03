# Datasets

**Nothing in this directory is committed to git.** Datasets are large and often
license-restricted; `.gitignore` excludes them. This file explains how to obtain
and prepare them.

## Layout produced by the preparation scripts

```
datasets/
  sunrgbd/                     # depth training data
    images/{train,val,test}/   # RGB, *.jpg
    depth/{train,val,test}/    # paired uint16 PNG, millimetres
  obstacle/                    # detection data (optional)
    images/{train,val}/
    labels/{train,val}/        # YOLO format, class 0 = obstacle
```

Depth maps are resolved from image paths by swapping `/images/` for `/depth/`,
which is the convention the Ultralytics depth dataloader expects. Filenames must
match exactly apart from the extension.

## SUN RGB-D — depth

10,335 RGB-D indoor scenes from four sensors (Kinect v1/v2, Xtion, RealSense).
Reliable depth to roughly 10 m.

1. Download `SUNRGBD.zip` (~6.5 GB) from
   <https://rgbd.cs.princeton.edu/data/SUNRGBD.zip> and extract it.
2. Download `SUNRGBDtoolbox.zip` for the official split
   (`traintestSUNRGBD/allsplit.mat`).
3. Convert the split, then the data:

```bash
python datasets/scripts/prepare_sunrgbd.py \
    --convert-allsplit /path/to/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat \
    --split-file configs/data/sunrgbd_official_split.json

python datasets/scripts/prepare_sunrgbd.py \
    --source /path/to/SUNRGBD --split official \
    --split-file configs/data/sunrgbd_official_split.json \
    --output datasets/sunrgbd --config-out configs/data/sunrgbd.yaml
```

Expect roughly 14 GB after conversion, plus the 6.5 GB archive during it.

### Why the official split matters

Ultralytics' shipped `depth-sunrgbd.yaml` selects its validation set as a
**random seed-0 sample of 1090 scenes**, ignoring SUN RGB-D's official
partition. SUN RGB-D contains multiple frames of the same physical room, so a
random split can place near-duplicate views in both train and val. That inflates
every metric and produces numbers comparable to no published baseline.

`--split ultralytics` reproduces that split when direct comparison with
Ultralytics-reported figures is the goal.

### Depth encoding

SUN RGB-D stores refined depth in `depth_bfx/` as uint16 with a 3-bit rotation:

```python
depth_mm = (raw >> 3) | (raw << 13)
depth_m  = depth_mm / 1000.0        # then clipped at 10 m
```

Reading the raw value as millimetres — a common mistake — is wrong by orders of
magnitude for most pixels. The decode is unit-tested (`tests/test_dataset.py`).

Output is uint16 millimetre PNG (`depth_scale: 1000`), where code `0` means "no
sensor return", **not** "0 m".

## Verification

```bash
python datasets/scripts/verify_dataset.py --data configs/data/sunrgbd.yaml --report outputs/dataset_report.json
```

Ten checks plus cross-split leakage detection: file counts, filename matching,
missing pairs, resolution agreement, corrupt files, invalid-depth percentage,
minimum valid pixels, depth range and depth statistics. Exits non-zero on
failure, so it can gate a training run.

## Obstacle detection data

The navigation pipeline needs **no** detection dataset: `navigation.yaml` sets
`detection.class_agnostic: true`, which relabels every COCO detection from the
stock `yolo26n.pt` as `obstacle`.

With real annotations, a fine-tuned single-class detector performs better:

```bash
python datasets/scripts/convert_to_obstacle_dataset.py --format yolo \
    --source /path/to/annotated --output datasets/obstacle
python datasets/scripts/convert_to_obstacle_dataset.py --format coco \
    --source /path/to/images --annotations instances.json --output datasets/obstacle
```

Every class collapses to id 0 (`obstacle`) and degenerate boxes are dropped.
**No bounding box is ever fabricated.** With no annotations available:

```bash
python datasets/scripts/convert_to_obstacle_dataset.py --format scaffold --output datasets/obstacle
```

creates the empty structure to drop real labels into later.

### Using SUN RGB-D's own 2D annotations

SUN RGB-D ships native 2D bounding boxes (146,617 of them across the dataset),
so it does not need external annotations at all. Download
`SUNRGBDMeta2DBB_v2.mat` (from the SUNRGBDtoolbox release on
<https://rgbd.cs.princeton.edu/>) alongside the extracted `SUNRGBD.zip`
`--source` tree, then:

```bash
python datasets/scripts/convert_sunrgbd_obstacle.py \
    --source /path/to/SUNRGBD \
    --meta /path/to/SUNRGBDMeta2DBB_v2.mat \
    --split-file configs/data/sunrgbd_official_split.json \
    --output datasets/obstacle
```

`--split-file` reuses the same official split JSON produced by
`prepare_sunrgbd.py --convert-allsplit` (see above), so the detection dataset
and the depth dataset partition scenes identically — no leakage between the
two.

Every SUN RGB-D object class collapses to `obstacle` (spec section A): the
field is read only to confirm it exists, never to decide the output label.
Degenerate and sub-`--min-box-size` boxes are dropped the same way as the
generic converter above.

> SUN RGB-D's 2D annotations are *object* annotations, not curated obstacle
> labels: not every annotated object is a navigation obstacle (a picture on a
> wall is not), and coverage varies by scene. Inspect a sample of converted
> labels before training a detector on them at scale.

**Format note.** `SUNRGBDMeta2DBB_v2.mat`'s field layout (`SUNRGBDMeta2DBB(i)
.groundtruth2DBB(j).gtBb2D` = `[x, y, width, height]` pixels, `.classname`)
was verified against the raw source of two independent, long-maintained
reference implementations that consume this exact file
([facebookresearch/votenet](https://github.com/facebookresearch/votenet),
[charlesq34/frustum-pointnets](https://github.com/charlesq34/frustum-pointnets))
plus an explicit SUN RGB-D → COCO converter
([crmauceri/SUNRGBD_COCO](https://github.com/crmauceri/SUNRGBD_COCO)) whose
box arithmetic (`x2 = gtBb2D(1) + gtBb2D(3)`) is what pins down the `[x,y,w,h]`
convention — not from an assumption. `rgbd.cs.princeton.edu` itself is
unreachable from some sandboxed environments (egress-proxy denied), so the
converter never trusts a single guessed path convention for locating each
scene's RGB file: it anchors on the four known sensor root directories
(`kv1`, `kv2`, `realsense`, `xtion`) and falls back to a logged,
non-fabricating filename search — see the module docstring in
`convert_sunrgbd_obstacle.py` for the full reasoning.

## Smoke-test fixture

`depth8` — 8 SUN RGB-D images (1.3 MB) shipping with Ultralytics — is enough to
exercise the whole pipeline without any download:

```bash
bash scripts/run_demo.sh
```

The unit tests need neither this nor any checkpoint: they run on synthetic
tensors.

## Alternative depth datasets

Ultralytics ships configs for NYU Depth V2, ARKitScenes, DIODE, Hypersim, KITTI,
TartanAir and Virtual KITTI 2. Point `data.yaml` at one of them to train on it.
For indoor wheelchair navigation, SUN RGB-D, NYU Depth V2 and ARKitScenes are the
relevant ones — KITTI and Virtual KITTI 2 are outdoor driving data with an 80 m
range and a completely different depth distribution.
