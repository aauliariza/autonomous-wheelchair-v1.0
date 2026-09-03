"""Frozen YOLO26x-Depth teacher (spec section F).

The teacher is frozen three independent ways, because any one alone is
insufficient:

1. ``requires_grad_(False)`` on every parameter — no gradient is accumulated.
2. ``.eval()`` — BatchNorm running statistics stop updating. Without this the
   teacher's own normalization would drift during student training even with
   zero gradients, silently changing the distillation target.
3. ``torch.no_grad()`` around the forward — no autograd graph is built, which
   also cuts teacher activation memory.

Output-space note (VERIFIED in the session audit): the Depth head applies its
log-affine calibration ONLY in eval mode. Because the teacher always runs in
eval mode, its output is calibrated. ``space="raw"`` temporarily neutralizes the
calibration so pure relative structure can be distilled instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .feature_hooks import FeatureExtractor, resolve_depth_feature_layers
from .model_utils import get_calibration, get_depth_head, load_depth_model


class TeacherDepthModel:
    """Frozen teacher providing depth targets and intermediate features for KD.

    Args:
        weights (str | Path): Preferred checkpoint (a fine-tuned teacher).
        fallback (str | Path, optional): Used when ``weights`` is absent, e.g.
            the official ``yolo26x-depth.pt``.
        layers (list[int], optional): KD tap indices. Resolved from the head by
            introspection when omitted.
        device (str | torch.device): Compute device.
        space (str): ``"calibrated"`` keeps the fitted metric scale (default);
            ``"raw"`` forces calibration to identity for the KD forward.

    Examples:
        >>> teacher = TeacherDepthModel("yolo26x-depth.pt", device="cpu")  # doctest: +SKIP
        >>> depth, feats = teacher.forward_with_features(images)           # doctest: +SKIP
    """

    def __init__(
        self,
        weights: str | Path,
        fallback: str | Path | None = None,
        layers: list[int] | None = None,
        device: str | torch.device = "cpu",
        space: str = "calibrated",
    ):
        if space not in ("calibrated", "raw"):
            raise ValueError(f"space must be 'calibrated' or 'raw', got '{space}'.")

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model, self.source = load_depth_model(weights, fallback=fallback, device=self.device)
        self.space = space

        self.freeze()

        self.layers = layers if layers is not None else resolve_depth_feature_layers(self.model)
        self.cal_a, self.cal_b = get_calibration(self.model)
        self._extractor: FeatureExtractor | None = None

        if space == "raw":
            head = get_depth_head(self.model)
            head.cal_a = torch.ones_like(head.cal_a)
            head.cal_b = torch.zeros_like(head.cal_b)

    def freeze(self) -> None:
        """Apply all three freezing mechanisms."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @property
    def is_frozen(self) -> bool:
        """True when no parameter requires grad and the module is in eval mode."""
        return not any(p.requires_grad for p in self.model.parameters()) and not self.model.training

    def train(self, mode: bool = True) -> None:
        """Refuse to leave eval mode.

        Ultralytics trainers call ``.train()`` on every module they own; this
        guard makes it impossible to un-freeze the teacher by accident.
        """
        del mode
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict depth. Always eval + no_grad, so calibration is applied.

        Returns:
            (torch.Tensor): ``(B, 1, H/4, W/4)`` depth in metres.
        """
        self.model.eval()
        out = self.model(x.to(self.device))
        depth = out["depth"] if isinstance(out, dict) else out
        if isinstance(depth, (tuple, list)):
            depth = depth[0]
        return depth if depth.ndim == 4 else depth.unsqueeze(1)

    @torch.no_grad()
    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Predict depth and capture KD features in a single forward pass.

        Running one pass rather than two halves teacher compute, which dominates
        KD training cost (the teacher is ~9x the student).

        Returns:
            (tuple): ``(depth (B,1,H/4,W/4), [F_T per level])`` — all detached.
        """
        if self._extractor is None:
            self._extractor = FeatureExtractor(self.model, self.layers, detach=True).attach()
        self._extractor.clear()
        depth = self.forward(x)
        return depth, self._extractor.ordered()

    def feature_shapes(self, imgsz: int = 640) -> list[tuple[int, ...]]:
        """Measured ``(C, H, W)`` per KD level, for sizing the projection bank."""
        with FeatureExtractor(self.model, self.layers, detach=True) as fx, torch.no_grad():
            self.model(torch.zeros(1, 3, imgsz, imgsz, device=self.device))
            return [tuple(f.shape[1:]) for f in fx.ordered()]

    def feature_channels(self, imgsz: int = 640) -> list[int]:
        """Measured channel count per KD level."""
        return [s[0] for s in self.feature_shapes(imgsz)]

    def close(self) -> None:
        """Remove persistent hooks."""
        if self._extractor is not None:
            self._extractor.remove()
            self._extractor = None

    def info(self) -> dict[str, Any]:
        """Provenance and freezing state for experiment metadata."""
        return {
            "role": "teacher",
            "source": self.source,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "frozen": self.is_frozen,
            "kd_layers": self.layers,
            "calibration": {"cal_a": self.cal_a, "cal_b": self.cal_b},
            "output_space": self.space,
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return f"TeacherDepthModel(source={self.source!r}, params={n:,}, frozen={self.is_frozen}, layers={self.layers})"
