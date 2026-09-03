"""Checkpoint discovery, provenance and experiment metadata (spec sections AX, BI).

Large checkpoints are never committed to git (see .gitignore); this module only
locates them and records how they were produced.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import config_hash, save_json


class CheckpointError(Exception):
    """Raised when a checkpoint cannot be found or is unusable."""


# Directories searched, in order, when resolving a bare checkpoint name.
DEFAULT_SEARCH_DIRS = (
    Path("outputs/checkpoints"),
    Path("outputs"),
    Path("."),
)


def resolve_checkpoint(name: str | Path, search_dirs: tuple[Path, ...] | None = None) -> Path:
    """Resolve a checkpoint path, reporting every location searched on failure.

    Ultralytics model *names* (e.g. ``yolo26n-depth.pt``) that are not present on
    disk are returned unchanged so that Ultralytics can auto-download them.

    Raises:
        CheckpointError: If a path-like reference does not exist anywhere.
    """
    p = Path(name)
    if p.exists():
        return p

    dirs = search_dirs or DEFAULT_SEARCH_DIRS
    searched = [str(p.resolve())]
    for d in dirs:
        cand = d / p.name
        searched.append(str(cand.resolve()))
        if cand.exists():
            return cand

    # A bare official model name is downloadable by Ultralytics; let it try.
    if p.parent == Path(".") and p.name.startswith("yolo") and p.suffix == ".pt":
        return p

    raise CheckpointError(
        "Checkpoint not found: " + str(name) + "\n  Searched:\n    " + "\n    ".join(searched) + "\n"
        "  Recovery: train one first (see README STEP 7/9/12), or pass an explicit --model path."
    )


def git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown' outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def git_is_dirty() -> bool:
    """True when the working tree has uncommitted changes (provenance caveat)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def collect_environment_metadata(config: dict[str, Any] | None = None, seed: int | None = None) -> dict[str, Any]:
    """Capture the full reproducibility record required by spec section BI.

    Every field is *probed*, never assumed; anything unavailable is reported as
    ``"unavailable"`` rather than guessed.
    """
    meta: dict[str, Any] = {
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unavailable",
        "seed": seed,
    }

    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        meta["cuda_version"] = torch.version.cuda or "cpu-build"
        meta["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        meta["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        meta["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        meta["torch_version"] = "unavailable"

    try:
        import ultralytics

        meta["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        meta["ultralytics_version"] = "unavailable"

    try:
        import numpy

        meta["numpy_version"] = numpy.__version__
    except ImportError:
        meta["numpy_version"] = "unavailable"

    if config is not None:
        meta["config_hash"] = config_hash(config)

    return meta


def save_experiment_metadata(
    path: str | Path = "outputs/experiment_metadata.json",
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the reproducibility record to JSON (spec section BI)."""
    meta = collect_environment_metadata(config=config, seed=seed)
    if extra:
        meta.update(extra)
    return save_json(meta, path)
