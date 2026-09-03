"""Obstacle-aware / ROI-aware distillation (spec section N).

Weights the depth-KD residual higher inside obstacle bounding boxes:

.. math::
    w(p) = \\begin{cases} \\alpha & p \\in \\text{obstacle ROI} \\\\ 1 & \\text{otherwise} \\end{cases}
.. math::
    L_{roiKD} = \\frac{\\sum_p w(p)\\,\\rho\\big(D_S(p) - D_T(p)\\big)}{\\sum_p w(p)}

Rationale: a uniform pixelwise loss optimizes for the floor, walls and ceiling,
which occupy most of an indoor frame and are irrelevant to obstacle clearance.
The pixels that actually decide FORWARD versus STOP are the ones on obstacles,
so they are weighted up by ``alpha``.

Background is weighted 1 rather than 0 on purpose: driving the background term to
zero would let the student drift arbitrarily on the floor plane, and the floor is
what "free path" is measured against.

Boxes come from the YOLO26n detector with every class collapsed to ``obstacle``
(spec section A). The inner-60% ROI convention matches
``navigation.distance``, so the term emphasizes exactly the pixels the deployed
distance estimator will read.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import align_to, get_depth_loss_fn, valid_depth_mask


def boxes_to_mask(
    boxes: list[torch.Tensor],
    shape: tuple[int, int],
    image_size: tuple[int, int] | None = None,
    inner_ratio: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rasterize per-image xyxy boxes into a binary obstacle mask.

    Args:
        boxes (list[torch.Tensor]): One ``(N, 4)`` xyxy tensor per image. Empty
            tensors are allowed and produce an all-zero mask for that image.
        shape (tuple[int, int]): Target ``(H, W)`` of the mask.
        image_size (tuple[int, int], optional): ``(H, W)`` the boxes were
            measured in. When it differs from ``shape``, boxes are rescaled —
            necessary because the depth head predicts at input/4.
        inner_ratio (float): Keep the central fraction of each box (0.6 = the
            inner 60% ROI). 1.0 uses the full box.
        device (torch.device | str): Device for the returned mask.

    Returns:
        (torch.Tensor): ``(B, 1, H, W)`` float mask, 1.0 inside an obstacle ROI.
    """
    h, w = shape
    mask = torch.zeros(len(boxes), 1, h, w, device=device)

    sy, sx = 1.0, 1.0
    if image_size is not None and tuple(image_size) != (h, w):
        sy = h / float(image_size[0])
        sx = w / float(image_size[1])

    inset = (1.0 - float(inner_ratio)) / 2.0

    for bi, bx in enumerate(boxes):
        if bx is None or len(bx) == 0:
            continue
        for box in bx:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            if inset > 0:
                bw, bh = x2 - x1, y2 - y1
                x1, x2 = x1 + inset * bw, x2 - inset * bw
                y1, y2 = y1 + inset * bh, y2 - inset * bh

            # Clip to the mask, and round outward so a sub-pixel box still marks
            # at least one pixel instead of vanishing.
            xi1 = max(0, min(w - 1, int(x1)))
            yi1 = max(0, min(h - 1, int(y1)))
            xi2 = max(xi1 + 1, min(w, int(x2) + 1))
            yi2 = max(yi1 + 1, min(h, int(y2) + 1))
            mask[bi, 0, yi1:yi2, xi1:xi2] = 1.0

    return mask


class ROIKDLoss(nn.Module):
    """Obstacle-weighted depth distillation.

    Args:
        alpha (float): Weight applied to obstacle pixels (background stays 1.0).
        loss_type (str): ``l1`` | ``smooth_l1`` | ``berhu``.
        beta (float): SmoothL1 transition point.
        log_space (bool): Compare log-depth.
        epsilon (float): Floor before ``log``.
        mask_invalid_gt (bool): Restrict to valid GT pixels.
        max_depth (float, optional): Eigen-protocol upper bound.
    """

    def __init__(
        self,
        alpha: float = 3.0,
        loss_type: str = "smooth_l1",
        beta: float = 0.1,
        log_space: bool = True,
        epsilon: float = 1e-6,
        mask_invalid_gt: bool = True,
        max_depth: float | None = None,
    ):
        super().__init__()
        if alpha < 1.0:
            raise ValueError(f"alpha must be >= 1.0 (obstacles are never de-emphasised), got {alpha}.")
        self.alpha = alpha
        self.loss_type = loss_type
        self.beta = beta
        self.log_space = log_space
        self.epsilon = epsilon
        self.mask_invalid_gt = mask_invalid_gt
        self.max_depth = max_depth
        self._fn = get_depth_loss_fn(loss_type)

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
        obstacle_mask: torch.Tensor | None = None,
        gt_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the obstacle-weighted KD loss.

        Args:
            student_depth (torch.Tensor): ``(B,1,h,w)``.
            teacher_depth (torch.Tensor): ``(B,1,h,w)``.
            obstacle_mask (torch.Tensor, optional): ``(B,1,h,w)`` from
                ``boxes_to_mask``. With no detections this reduces to the plain
                masked depth-KD loss.
            gt_depth (torch.Tensor, optional): Used for the validity mask.
        """
        teacher = align_to(teacher_depth, student_depth).to(student_depth.dtype)

        valid: torch.Tensor | None = None
        if self.mask_invalid_gt and gt_depth is not None:
            gt = align_to(gt_depth, student_depth, mode="nearest")
            valid = valid_depth_mask(gt, max_depth=self.max_depth)
        teacher_ok = torch.isfinite(teacher) & (teacher > self.epsilon)
        valid = teacher_ok if valid is None else (valid & teacher_ok)

        s, t = student_depth, teacher
        if self.log_space:
            s = torch.log(s.clamp(min=self.epsilon))
            t = torch.log(t.clamp(min=self.epsilon))

        if obstacle_mask is None:
            return self._fn(s, t, valid, **self._kwargs())

        om = align_to(obstacle_mask.float(), student_depth, mode="nearest")
        weights = 1.0 + (self.alpha - 1.0) * om.clamp(0.0, 1.0)
        weights = weights * valid.to(weights.dtype)

        residual = self._residual(s, t)
        denom = weights.sum()
        if denom <= 0:
            return (student_depth * 0.0).sum()
        return (residual * weights).sum() / denom

    def _kwargs(self) -> dict:
        if self.loss_type == "smooth_l1":
            return {"beta": self.beta}
        if self.loss_type == "berhu":
            return {"threshold": 0.2}
        return {}

    def _residual(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-pixel (unreduced) residual, so ROI weights apply pointwise."""
        import torch.nn.functional as F

        if self.loss_type == "l1":
            return (s - t).abs()
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(s, t, reduction="none", beta=self.beta)
        # BerHu, unreduced
        diff = s - t
        absdiff = diff.abs()
        c = torch.clamp(0.2 * absdiff.max().detach(), min=1e-6)
        return torch.where(absdiff <= c, absdiff, (diff.pow(2) + c.pow(2)) / (2.0 * c))
