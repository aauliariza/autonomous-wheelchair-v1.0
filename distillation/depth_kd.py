"""Depth-output distillation (spec sections I and J).

.. math::
    L_{depthKD} = \\frac{1}{|M|} \\sum_{p \\in M} \\rho\\big(D_S(p) - D_T(p)\\big)

where :math:`M` is the valid ground-truth mask and :math:`\\rho` is L1,
SmoothL1 or BerHu.

Log-space option (spec section J)
---------------------------------
The YOLO26 Depth head is internally a log-depth predictor
(``depth = exp(out.clamp(-4, 5))``, verified in ultralytics 8.4.138), but
Ultralytics ALREADY applies the exponential. This module therefore never touches
raw head activations and never calls ``exp()``; when ``log_space=True`` it simply
re-takes ``log()`` of the returned metric depth.

Distilling in log space equalizes relative error across the depth range: a 0.1 m
error at 0.5 m matters far more to a wheelchair than the same error at 8 m, and a
linear-space loss would weight them identically.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .losses import align_to, get_depth_loss_fn, valid_depth_mask


class DepthKDLoss(nn.Module):
    """Pixelwise teacher-to-student depth distillation.

    Args:
        loss_type (str): ``l1`` | ``smooth_l1`` | ``berhu``.
        beta (float): SmoothL1 transition point.
        berhu_threshold (float): BerHu adaptive cutoff fraction.
        log_space (bool): Compare ``log(D)`` instead of ``D``.
        epsilon (float): Floor added before ``log`` for numerical safety.
        mask_invalid_gt (bool): Restrict the loss to valid GT pixels.
        max_depth (float, optional): Eigen-protocol upper bound.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        beta: float = 0.1,
        berhu_threshold: float = 0.2,
        log_space: bool = True,
        epsilon: float = 1e-6,
        mask_invalid_gt: bool = True,
        max_depth: float | None = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.berhu_threshold = berhu_threshold
        self.log_space = log_space
        self.epsilon = epsilon
        self.mask_invalid_gt = mask_invalid_gt
        self.max_depth = max_depth
        self._fn = get_depth_loss_fn(loss_type)

    def _kwargs(self) -> dict[str, Any]:
        if self.loss_type == "smooth_l1":
            return {"beta": self.beta}
        if self.loss_type == "berhu":
            return {"threshold": self.berhu_threshold}
        return {}

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
        gt_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the output-level KD loss.

        Args:
            student_depth (torch.Tensor): ``(B,1,h,w)`` student prediction.
            teacher_depth (torch.Tensor): ``(B,1,h,w)`` frozen teacher prediction.
            gt_depth (torch.Tensor, optional): ``(B,1,H,W)`` ground truth, used
                only to build the validity mask.

        Returns:
            (torch.Tensor): Scalar loss.
        """
        # Everything is compared at the STUDENT's resolution: upsampling the
        # student to full size would interpolate its own predictions and blur the
        # very boundaries the boundary term is trying to sharpen.
        teacher = align_to(teacher_depth, student_depth).to(student_depth.dtype)

        mask: torch.Tensor | None = None
        if self.mask_invalid_gt and gt_depth is not None:
            gt = align_to(gt_depth, student_depth, mode="nearest")
            mask = valid_depth_mask(gt, max_depth=self.max_depth)

        # The teacher can only be trusted where it produced a finite positive
        # depth, independently of the GT mask.
        teacher_ok = torch.isfinite(teacher) & (teacher > self.epsilon)
        mask = teacher_ok if mask is None else (mask & teacher_ok)

        s, t = student_depth, teacher
        if self.log_space:
            s = torch.log(s.clamp(min=self.epsilon))
            t = torch.log(t.clamp(min=self.epsilon))

        return self._fn(s, t, mask, **self._kwargs())
