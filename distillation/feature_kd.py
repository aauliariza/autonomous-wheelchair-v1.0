"""Feature-level distillation (spec section K).

.. math::
    L_{feature} = \\sum_i \; d\\big(P_i(F_{T_i}),\; F_{S_i}\\big)

Intermediate features carry the scene structure the teacher uses to arrive at
its depth, so matching them transfers more than the output alone: the student is
told *how* to see, not only what to conclude.

Normalization
-------------
Teacher and student activation magnitudes differ substantially (the teacher is
~9x larger and trained to a different scale). With ``normalize=True`` each
feature is L2-normalized per channel before the distance, so the loss measures
the DIRECTION of the feature response rather than its magnitude. Without it, a
handful of high-magnitude teacher channels dominate the gradient and the term
degenerates into a scale-matching penalty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureKDLoss(nn.Module):
    """Multi-level feature matching between projected teacher and student maps.

    Args:
        loss_type (str): ``mse`` | ``smooth_l1`` | ``cosine``. ``cosine`` is
            inherently scale-invariant and is a good choice when the two networks'
            activation ranges differ a lot.
        normalize (bool): Per-channel L2 normalization before the distance.
        beta (float): SmoothL1 transition point.
        level_weights (list[float], optional): Per-level weights. Defaults to
            equal weighting across all levels.
    """

    VALID = ("mse", "smooth_l1", "cosine")

    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = True,
        beta: float = 0.1,
        level_weights: list[float] | None = None,
    ):
        super().__init__()
        if loss_type not in self.VALID:
            raise ValueError(f"Unknown feature loss '{loss_type}'. Available: {self.VALID}.")
        self.loss_type = loss_type
        self.normalize = normalize
        self.beta = beta
        self.level_weights = level_weights

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        """L2-normalize each channel over its spatial extent."""
        b, c = x.shape[:2]
        flat = x.reshape(b, c, -1)
        return F.normalize(flat, p=2, dim=2, eps=1e-6).reshape_as(x)

    def _distance(self, ft: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            ft, fs = self._norm(ft), self._norm(fs)

        if self.loss_type == "mse":
            return F.mse_loss(fs, ft)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(fs, ft, beta=self.beta)

        # Cosine: 1 - similarity over the channel axis, averaged spatially.
        b, c = fs.shape[:2]
        a = fs.reshape(b, c, -1)
        t = ft.reshape(b, c, -1)
        return (1.0 - F.cosine_similarity(a, t, dim=1, eps=1e-6)).mean()

    def forward(self, teacher_feats: list[torch.Tensor], student_feats: list[torch.Tensor]) -> torch.Tensor:
        """Sum the per-level feature distances.

        Args:
            teacher_feats (list[torch.Tensor]): Projected teacher features.
            student_feats (list[torch.Tensor]): Student features, same shapes.

        Returns:
            (torch.Tensor): Scalar loss.
        """
        if len(teacher_feats) != len(student_feats):
            raise ValueError(f"Level mismatch: {len(teacher_feats)} teacher vs {len(student_feats)} student features.")
        if not teacher_feats:
            raise ValueError("No features supplied to FeatureKDLoss.")

        weights = self.level_weights or [1.0] * len(teacher_feats)
        if len(weights) != len(teacher_feats):
            raise ValueError(f"level_weights has {len(weights)} entries for {len(teacher_feats)} levels.")

        total = None
        for w, ft, fs in zip(weights, teacher_feats, student_feats):
            if ft.shape != fs.shape:
                raise ValueError(
                    f"Feature shape mismatch {tuple(ft.shape)} vs {tuple(fs.shape)}. "
                    f"Recovery: the ProjectionBank must align channels before this loss."
                )
            term = w * self._distance(ft.detach(), fs)
            total = term if total is None else total + term
        return total
