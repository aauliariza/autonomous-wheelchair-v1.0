"""Shared training helpers: config -> Ultralytics arguments, and run bookkeeping.

Keeping this translation in one place means every experiment (teacher, baseline,
KD, detection) consumes YAML identically, so an ablation differs only by its
config file — which is what makes the study reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.checkpoint import save_experiment_metadata
from utils.io import save_yaml
from utils.logger import get_logger
from utils.seed import seed_everything

LOG = get_logger("training")

# Keys forwarded verbatim to Ultralytics' trainer from the config's `train` block.
PASSTHROUGH_TRAIN_KEYS = (
    "epochs",
    "batch",
    "imgsz",
    "device",
    "workers",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "cos_lr",
    "patience",
    "amp",
    "freeze",
    "val",
    "plots",
    "save_period",
    "single_cls",
    "dropout",
    "label_smoothing",
    "nbs",
    "fraction",
    "close_mosaic",
    "resume",
    "pretrained",
    "profile",
    "deterministic",
    "seed",
)

# Depth-loss gains and augmentation keys, also forwarded verbatim.
PASSTHROUGH_LOSS_KEYS = ("dlog", "dgrad", "dlam", "box", "cls", "dfl")
PASSTHROUGH_AUGMENT_KEYS = (
    "mosaic",
    "mixup",
    "copy_paste",
    "fliplr",
    "flipud",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "erasing",
    "auto_augment",
)


def build_ultralytics_args(config: dict[str, Any], data_yaml: str | Path, **overrides: Any) -> dict[str, Any]:
    """Translate a project config into Ultralytics ``model.train()`` keyword args."""
    args: dict[str, Any] = {"data": str(data_yaml)}

    for block, keys in (
        ("train", PASSTHROUGH_TRAIN_KEYS),
        ("loss", PASSTHROUGH_LOSS_KEYS),
        ("augment", PASSTHROUGH_AUGMENT_KEYS),
    ):
        section = config.get(block, {}) or {}
        for k in keys:
            if k in section and section[k] is not None:
                args[k] = section[k]

    exp = config.get("experiment", {}) or {}
    args.setdefault("project", str(Path(exp.get("output_dir", "outputs/experiments"))))
    args.setdefault("name", exp.get("name", "run"))
    args.setdefault("seed", exp.get("seed", 42))
    args.setdefault("deterministic", exp.get("deterministic", False))
    args.setdefault("exist_ok", True)

    args.update({k: v for k, v in overrides.items() if v is not None})
    return args


def resolve_data_yaml(config: dict[str, Any]) -> Path:
    """Resolve the dataset YAML, failing with the expected preparation command."""
    data_cfg = config.get("data", {}) or {}
    ref = data_cfg.get("yaml")
    if ref is None:
        raise FileNotFoundError("config has no data.yaml entry; set it to your prepared dataset YAML.")

    path = Path(ref)
    if path.exists():
        return path

    # Bare Ultralytics dataset names (depth8.yaml, nyu-depth.yaml) resolve internally.
    if path.parent == Path(".") and not path.is_absolute():
        return path

    raise FileNotFoundError(
        f"Dataset config not found: {path.resolve()}\n"
        f"  Recovery: prepare the dataset first:\n"
        f"    python datasets/scripts/prepare_sunrgbd.py --source /path/to/SUNRGBD --config-out {path}\n"
        f"  then verify it:\n"
        f"    python datasets/scripts/verify_dataset.py --data {path}"
    )


def setup_experiment(config: dict[str, Any], config_path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Seed RNGs, create the run directory, and snapshot config + provenance.

    Returns:
        (Path): The run directory holding ``config.yaml`` and metadata.
    """
    exp = config.get("experiment", {}) or {}
    seed = int(exp.get("seed", 42))
    deterministic = bool(exp.get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    run_dir = Path(exp.get("output_dir", "outputs/experiments")) / exp.get("name", "run")
    run_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(config, run_dir / "config.yaml")
    save_experiment_metadata(
        path=run_dir / "experiment_metadata.json",
        config=config,
        seed=seed,
        extra={"source_config": str(config_path), "experiment_tag": exp.get("tag"), **(extra or {})},
    )
    LOG.info("Experiment '%s' | seed=%d deterministic=%s | run dir: %s", exp.get("name"), seed, deterministic, run_dir)
    return run_dir


def export_best_checkpoint(run_dir: Path, destination: str | Path) -> Path | None:
    """Copy a run's ``best.pt`` into ``outputs/checkpoints/`` under a stable name.

    Downstream stages reference stable names (``teacher_best.pt``,
    ``student_distilled_best.pt``) rather than Ultralytics' incrementing run
    directories, so configs do not need editing between runs.
    """
    import shutil

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        LOG.warning("No best.pt found in %s; nothing exported.", run_dir / "weights")
        return None

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    LOG.info("Exported best checkpoint -> %s", dest)
    return dest


# Extensions Ultralytics' DepthDataset accepts for a depth target, in the order
# it probes them (ultralytics/data/dataset.py: DepthDataset._depth_path_for).
DEPTH_SUFFIXES = (".png", ".npy")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _stems(directory: Path, suffixes: tuple[str, ...]) -> set[str]:
    """Return the filename stems in ``directory`` whose suffix is in ``suffixes``."""
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.iterdir() if p.suffix.lower() in suffixes}


def verify_depth_pairing(data_yaml: str | Path, splits: tuple[str, ...] = ("train", "val")) -> dict[str, int]:
    """Fail fast when RGB images have no paired depth map.

    Ultralytics' ``DepthDataset`` silently SKIPS an image whose depth companion is
    missing, then aborts with "No labels found" only after scanning the whole
    split. On SUN RGB-D that wastes minutes per attempt and hides the real cause,
    so this reproduces the same pairing rule (``images/<split>/x.jpg`` ->
    ``depth/<split>/x.{png,npy}``) up front, in one directory listing per split.

    Args:
        data_yaml (str | Path): Prepared dataset YAML with a ``path`` entry.
        splits (tuple): Split subdirectories to check.

    Returns:
        (dict): Paired-image count per split that exists.

    Raises:
        FileNotFoundError: The dataset root or a split's ``images/`` is absent.
        ValueError: A split has unpaired images, naming the recovery command.
    """
    from utils.io import load_yaml

    data_yaml = Path(data_yaml)
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {root.resolve()} (from 'path:' in {data_yaml})\n"
            f"  Recovery: python datasets/scripts/prepare_sunrgbd.py --source /path/to/SUNRGBD "
            f"--output {root} --config-out {data_yaml}"
        )

    paired: dict[str, int] = {}
    for split in splits:
        img_dir, dep_dir = root / "images" / split, root / "depth" / split
        if not img_dir.is_dir():
            if split == "train":
                raise FileNotFoundError(f"Missing image split: {img_dir.resolve()}")
            continue

        img_stems = _stems(img_dir, IMAGE_SUFFIXES)
        dep_stems = _stems(dep_dir, DEPTH_SUFFIXES)
        missing = sorted(img_stems - dep_stems)
        paired[split] = len(img_stems) - len(missing)

        if not missing:
            LOG.info("Split '%s': %d RGB-depth pairs OK", split, paired[split])
            continue

        # Report what the depth directory ACTUALLY holds: an empty directory, a
        # wrong extension and a name mismatch each need a different fix, and the
        # counts alone cannot tell them apart.
        present = sorted({p.suffix.lower() for p in dep_dir.iterdir()}) if dep_dir.is_dir() else []
        raise ValueError(
            f"Split '{split}': {len(missing)} of {len(img_stems)} RGB images have no depth map "
            f"in {dep_dir.resolve()} ({paired[split]} usable pairs).\n"
            f"  Ultralytics probes {' then '.join(DEPTH_SUFFIXES)}; suffixes present there: "
            f"{present or 'NONE (directory empty or absent)'}\n"
            f"  First missing stems: {', '.join(missing[:3])}\n"
            f"  Recovery, in order:\n"
            f"    1. Drop the half-pairs (fast, keeps the rest):\n"
            f"       python datasets/scripts/repair_orphaned_pairs.py --data {root}\n"
            f"       python datasets/scripts/repair_orphaned_pairs.py --data {root} --apply\n"
            f"    2. Or re-convert from the raw archive (slow, restores every scene):\n"
            f"       python datasets/scripts/prepare_sunrgbd.py --source /path/to/SUNRGBD "
            f"--output {root} --config-out {data_yaml}\n"
            f"    3. Then confirm: python datasets/scripts/verify_dataset.py --data {data_yaml}"
        )

    if not paired.get("train"):
        raise ValueError(
            f"No usable RGB-depth pairs in {root / 'images' / 'train'}. Training cannot start.\n"
            f"  Recovery: python datasets/scripts/verify_dataset.py --data {data_yaml}"
        )
    return paired
