"""Reproducibility helpers (spec section AS).

Seeding covers Python's ``random``, NumPy, PyTorch CPU and all CUDA devices.

Determinism trade-off
---------------------
``deterministic=True`` sets ``torch.use_deterministic_algorithms(True)`` and
``cudnn.deterministic=True`` while disabling ``cudnn.benchmark``. This makes runs
bit-reproducible on identical hardware but is typically **10-30% slower**, because
cuDNN can no longer autotune convolution algorithms per input shape, and because
some fast non-deterministic kernels (notably scatter/atomic reductions) are
replaced by slower deterministic ones.

``deterministic=False`` keeps ``cudnn.benchmark=True``, which is faster for fixed
input shapes but selects algorithms by runtime autotuning, so results vary slightly
between runs and between machines.

For published ablation studies prefer ``deterministic=True`` so that a reported
delta between two KD variants cannot be an artifact of kernel scheduling.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed every RNG used by the pipeline.

    Args:
        seed (int): Base seed. The project default is 42 (spec section AS).
        deterministic (bool): Enable deterministic kernels. See module docstring
            for the speed trade-off.

    Returns:
        (int): The seed that was applied, for logging into experiment metadata.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # CUBLAS workspace config is required by torch for deterministic matmul on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError):
            # warn_only is unavailable on very old torch; determinism is then best-effort.
            torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Give every DataLoader worker a distinct but reproducible seed.

    Without this, forked workers share the parent's NumPy state and can emit
    identical augmentation streams.
    """
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
