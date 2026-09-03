"""Trainable YOLO26n-Depth student (spec section G).

CRITICAL, VERIFIED IN THE SESSION AUDIT
---------------------------------------
Ultralytics loads ``.pt`` checkpoints with ``requires_grad=False`` on EVERY
parameter (measured: 271/271 params on yolo26n-depth.pt, 463/463 on
yolo26x-depth.pt). Its own trainer re-enables them, but a custom KD loop does
not get that for free. A first KD prototype in this session produced a valid
loss, ran ``backward()`` without error, and left the student with **zero
gradients** — a silent no-op that looks like a converged run.

``StudentDepthModel.__init__`` therefore calls ``unfreeze()`` and
``assert_trainable()`` verifies it, so this failure mode cannot recur unnoticed.

Output-space note: in train mode the head returns ``dict{"depth"}`` WITHOUT
calibration; in eval mode it returns a tensor WITH calibration. ``forward()``
normalizes both to a plain ``(B,1,H/4,W/4)`` tensor and reports which space the
value is in via ``last_output_calibrated``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .feature_hooks import FeatureExtractor, resolve_depth_feature_layers
from .model_utils import get_calibration, load_depth_model


class StudentDepthModel:
    """Trainable student wrapper exposing depth plus KD features.

    Args:
        weights (str | Path): Checkpoint or architecture YAML to start from.
        fallback (str | Path, optional): Used when ``weights`` is absent.
        layers (list[int], optional): KD tap indices; introspected when omitted.
        device (str | torch.device): Compute device.
        freeze_backbone_layers (int, optional): Freeze layers ``[0, n)`` for
            partial fine-tuning. ``None`` trains everything.
    """

    def __init__(
        self,
        weights: str | Path,
        fallback: str | Path | None = None,
        layers: list[int] | None = None,
        device: str | torch.device = "cpu",
        freeze_backbone_layers: int | None = None,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model, self.source = load_depth_model(weights, fallback=fallback, device=self.device)

        self.unfreeze()
        if freeze_backbone_layers:
            self.freeze_backbone(freeze_backbone_layers)

        self.layers = layers if layers is not None else resolve_depth_feature_layers(self.model)
        self.cal_a, self.cal_b = get_calibration(self.model)
        self._extractor: FeatureExtractor | None = None
        self.last_output_calibrated: bool = False

    def unfreeze(self) -> int:
        """Enable gradients on every parameter. Returns the count enabled."""
        n = 0
        for p in self.model.parameters():
            p.requires_grad_(True)
            n += 1
        return n

    def freeze_backbone(self, num_layers: int) -> list[int]:
        """Freeze layers ``[0, num_layers)`` for partial fine-tuning."""
        frozen: list[int] = []
        seq = self.model.model
        for i, layer in enumerate(seq):
            if i >= num_layers:
                break
            for p in layer.parameters():
                p.requires_grad_(False)
            frozen.append(i)
        return frozen

    def assert_trainable(self) -> None:
        """Fail loudly if no parameter can receive a gradient.

        Guards the exact silent failure documented in this module's docstring.
        """
        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        if trainable == 0:
            raise RuntimeError(
                "Student has ZERO trainable parameters — backward() would be a silent no-op.\n"
                "  Cause: Ultralytics loads .pt checkpoints with requires_grad=False on all parameters.\n"
                "  Recovery: call StudentDepthModel.unfreeze() (done automatically in __init__), "
                "or set student.unfreeze_on_load: true in configs/distillation.yaml."
            )

    @property
    def num_trainable(self) -> int:
        """Number of parameter tensors currently requiring grad."""
        return sum(1 for p in self.model.parameters() if p.requires_grad)

    def train(self, mode: bool = True) -> None:
        """Set train/eval mode on the underlying network."""
        self.model.train(mode)

    def eval(self) -> None:
        """Switch to eval mode (enables head calibration on the output)."""
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict depth, normalizing the train/eval output difference.

        Returns:
            (torch.Tensor): ``(B, 1, H/4, W/4)``. In train mode the value is
                UNCALIBRATED; ``last_output_calibrated`` records which it was.
        """
        out = self.model(x.to(self.device))
        self.last_output_calibrated = not self.model.training
        depth = out["depth"] if isinstance(out, dict) else out
        if isinstance(depth, (tuple, list)):
            depth = depth[0]
        return depth if depth.ndim == 4 else depth.unsqueeze(1)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Predict depth and capture KD features in one pass.

        Features stay ATTACHED to the graph (unlike the teacher's) so gradients
        from the feature KD term reach the student's backbone.
        """
        if self._extractor is None:
            self._extractor = FeatureExtractor(self.model, self.layers, detach=False).attach()
        self._extractor.clear()
        depth = self.forward(x)
        return depth, self._extractor.ordered()

    def feature_shapes(self, imgsz: int = 640) -> list[tuple[int, ...]]:
        """Measured ``(C, H, W)`` per KD level."""
        was_training = self.model.training
        self.model.eval()
        with FeatureExtractor(self.model, self.layers, detach=True) as fx, torch.no_grad():
            self.model(torch.zeros(1, 3, imgsz, imgsz, device=self.device))
            shapes = [tuple(f.shape[1:]) for f in fx.ordered()]
        self.model.train(was_training)
        return shapes

    def feature_channels(self, imgsz: int = 640) -> list[int]:
        """Measured channel count per KD level."""
        return [s[0] for s in self.feature_shapes(imgsz)]

    def parameters(self):
        """Underlying parameters, for the optimizer."""
        return self.model.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Underlying state dict, for checkpointing."""
        return self.model.state_dict()

    def close(self) -> None:
        """Remove persistent hooks."""
        if self._extractor is not None:
            self._extractor.remove()
            self._extractor = None

    def info(self) -> dict[str, Any]:
        """Provenance and trainability for experiment metadata."""
        return {
            "role": "student",
            "source": self.source,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_tensors": self.num_trainable,
            "kd_layers": self.layers,
            "calibration": {"cal_a": self.cal_a, "cal_b": self.cal_b},
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return f"StudentDepthModel(source={self.source!r}, params={n:,}, trainable_tensors={self.num_trainable})"
