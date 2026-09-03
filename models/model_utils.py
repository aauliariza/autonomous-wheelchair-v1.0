"""Device selection, model loading, complexity and latency measurement.

Spec sections AO (complexity), AN (latency), AV (error handling), AW (GPU/CPU).

Every fact this module reports about a model is PROBED at runtime — parameter
counts, GFLOPs, feature shapes and calibration buffers are read from the loaded
network, never hard-coded from documentation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded, with the paths that were searched."""


def select_device(device: str | int | None = None, verbose: bool = True) -> torch.device:
    """Resolve a CLI device string to a torch.device, falling back to CPU safely.

    Accepts ``0``, ``"0"``, ``"cuda:0"``, ``"cpu"``, ``"mps"`` or ``None`` (auto).
    A CUDA request on a machine without CUDA is downgraded to CPU with a warning
    rather than raising, so the pipeline stays runnable on laptops (spec AV/AW).
    """
    from utils.logger import get_logger

    log = get_logger()

    if device is None or str(device).strip() == "":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    text = str(device).strip().lower()

    if text == "cpu":
        return torch.device("cpu")

    if text == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    # Numeric ("0") or explicit ("cuda:0"). Only the first index is used here;
    # multi-GPU is delegated to the Ultralytics trainer via its own device arg.
    index = text.replace("cuda:", "").split(",")[0]
    if not torch.cuda.is_available():
        if verbose:
            log.warning(
                f"CUDA device '{device}' requested but torch.cuda.is_available() is False; "
                f"falling back to CPU. Recovery: install a CUDA build of torch, or pass --device cpu."
            )
        return torch.device("cpu")

    try:
        idx = int(index)
    except ValueError as e:
        raise ModelLoadError(
            f"Could not parse device '{device}'. Expected 'cpu', 'mps', an index like '0', or 'cuda:0'."
        ) from e

    if idx >= torch.cuda.device_count():
        log.warning(
            f"CUDA device {idx} requested but only {torch.cuda.device_count()} device(s) present; using cuda:0."
        )
        idx = 0
    return torch.device(f"cuda:{idx}")


def get_depth_head(model: nn.Module) -> nn.Module:
    """Return the ``Depth`` head of a YOLO26-Depth network.

    The head is identified by its calibration buffers (``cal_a``/``cal_b``) rather
    than by class name or index, mirroring how Ultralytics' own calibrate.py
    locates it. Raises if the model is not a depth model.
    """
    m = model.module if hasattr(model, "module") else model
    seq = getattr(m, "model", None)
    if seq is not None and not isinstance(seq, nn.Sequential):
        seq = getattr(seq, "model", None)
    head = seq[-1] if isinstance(seq, nn.Sequential) else m
    if not hasattr(head, "cal_a"):
        raise ModelLoadError(
            f"No Depth head with calibration buffers found on {type(model).__name__}. "
            f"Recovery: confirm the checkpoint is a *-depth.pt YOLO26-Depth model, not a detection model."
        )
    return head


def get_calibration(model: nn.Module) -> tuple[float, float]:
    """Read the head's log-affine calibration ``(cal_a, cal_b)``.

    Released checkpoints are NOT scale-identity (measured: yolo26n-depth
    cal_b=-0.1938, yolo26x-depth cal_b=-0.3167), so this must be inspected rather
    than assumed to be (1.0, 0.0).
    """
    head = get_depth_head(model)
    return float(head.cal_a), float(head.cal_b)


def denormalize_depth_space(depth: torch.Tensor, cal_a: float, cal_b: float) -> torch.Tensor:
    """Apply the log-affine calibration ``d' = d^a * exp(b)`` to a depth tensor.

    Used to put a training-mode output (which the head leaves uncalibrated) into
    the same space as an eval-mode output.
    """
    return depth.clamp(min=1e-6).pow(cal_a) * float(torch.exp(torch.tensor(cal_b)))


def load_depth_model(
    weights: str | Path,
    fallback: str | Path | None = None,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> tuple[nn.Module, str]:
    """Load a YOLO26-Depth network, optionally falling back to an official checkpoint.

    Returns:
        (tuple): ``(torch module, resolved source string)``. The source is recorded
            in experiment metadata so results are traceable to a specific checkpoint.

    Raises:
        ModelLoadError: If neither the primary nor the fallback can be loaded.
    """
    from ultralytics import YOLO

    from utils.logger import get_logger

    log = get_logger()
    attempts: list[str] = []

    for candidate in (weights, fallback):
        if candidate is None:
            continue
        p = Path(candidate)
        # A bare official name (yolo26x-depth.pt) is downloadable even if absent locally.
        is_official = p.parent == Path(".") and p.name.startswith("yolo")
        if not p.exists() and not is_official:
            attempts.append(f"{p.resolve()} (missing)")
            continue
        try:
            net = YOLO(str(candidate)).model
            net = net.to(device)
            if candidate is not weights:
                log.warning(f"Primary weights '{weights}' unavailable; using fallback '{candidate}'.")
            return net, str(candidate)
        except (FileNotFoundError, RuntimeError, KeyError, AttributeError) as e:
            attempts.append(f"{candidate} ({type(e).__name__}: {e})")

    raise ModelLoadError(
        "Could not load a depth model.\n  Attempted:\n    " + "\n    ".join(attempts) + "\n"
        "  Recovery: train a checkpoint (README STEP 7/9/12), or set model.fallback_weights "
        "to an official name such as 'yolo26n-depth.pt' so it can be downloaded."
    )


def model_complexity(model: nn.Module, imgsz: int = 640, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Measure parameters, GFLOPs and model size (spec section AO).

    GFLOPs are measured with ``thop`` on a real forward pass. If thop is absent
    the field is reported as ``None`` and the caller must print "NOT MEASURED" —
    a fabricated FLOP count is never returned.
    """
    model = model.to(device).eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())

    # Size on disk is dominated by fp32 parameters + buffers.
    size_mb = (total + buffers) * 4 / (1024**2)

    gflops: float | None = None
    try:
        from thop import profile

        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = 2.0 * macs / 1e9  # 1 MAC = 2 FLOPs
    except (ImportError, RuntimeError, TypeError):
        gflops = None

    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "buffers": buffers,
        "model_size_mb": round(size_mb, 3),
        "gflops": round(gflops, 3) if gflops is not None else None,
        "imgsz": imgsz,
    }


@torch.inference_mode()
def measure_latency(
    model: nn.Module,
    imgsz: int = 640,
    device: torch.device | str = "cpu",
    runs: int = 50,
    warmup: int = 10,
) -> dict[str, float]:
    """Measure single-image inference latency (spec section AN).

    Reports mean/median/P95/P99/std in milliseconds — never FPS alone. Model
    loading is excluded: only the forward pass is timed, after ``warmup``
    iterations so lazy CUDA kernel compilation is not counted (spec section BF).
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)

    for _ in range(max(warmup, 0)):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(max(runs, 1)):
        t0 = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(samples, dtype=torch.float64)
    return {
        "mean_ms": float(t.mean()),
        "median_ms": float(t.median()),
        "p95_ms": float(t.quantile(0.95)),
        "p99_ms": float(t.quantile(0.99)),
        "std_ms": float(t.std()) if len(samples) > 1 else 0.0,
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
        "fps": float(1000.0 / t.mean()),
        "runs": len(samples),
        "device": str(device),
        "imgsz": imgsz,
    }


def peak_memory_mb(device: torch.device | str = "cpu") -> float | None:
    """Peak CUDA memory in MB since the last reset, or None on CPU."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)
