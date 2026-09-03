"""Structural / boundary distillation (spec section L).

.. math::
    G_x = \\partial_x D, \\quad G_y = \\partial_y D, \\quad G = \\sqrt{G_x^2 + G_y^2}
.. math::
    L_{boundary} = \\mathrm{SmoothL1}\\big(G_S,\; G_T\\big)

Why this matters for a wheelchair specifically: the free-path decision depends on
where an obstacle ENDS and free floor begins. A student that matches the
teacher's average depth but smears the discontinuity at a table edge will report
a plausible mean distance while placing the boundary in the wrong sector — the
error mode most likely to produce an unsafe FORWARD.

Gradients are taken in log space by default, which makes the term scale-invariant:
it then measures relative depth structure rather than absolute metric change, so a
near doorway and a far doorway contribute comparable edge signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import align_to, smooth_l1_loss, valid_depth_mask


def depth_gradient(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order forward differences along x and y.

    Both outputs are padded back to the input's spatial size (replicating the
    last row/column) so gradients, masks and depth maps stay index-aligned and
    can share a single validity mask.

    Args:
        depth (torch.Tensor): ``(B, 1, H, W)``.

    Returns:
        (tuple): ``(Gx, Gy)``, each ``(B, 1, H, W)``.
    """
    if depth.ndim != 4:
        raise ValueError(f"Expected a 4D (B,1,H,W) tensor, got shape {tuple(depth.shape)}.")

    gx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    gy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    gx = torch.cat([gx, gx[:, :, :, -1:]], dim=3)
    gy = torch.cat([gy, gy[:, :, -1:, :]], dim=2)
    return gx, gy


def gradient_magnitude(depth: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Edge magnitude ``sqrt(Gx^2 + Gy^2)``.

    ``eps`` keeps the square root differentiable where the gradient is exactly
    zero — common on flat floors and walls, which cover most of an indoor frame.
    Without it those pixels produce NaN gradients.
    """
    gx, gy = depth_gradient(depth)
    return torch.sqrt(gx.pow(2) + gy.pow(2) + eps)


class BoundaryKDLoss(nn.Module):
    """Match the student's depth-edge structure to the teacher's.

    Args:
        loss_type (str): ``smooth_l1`` | ``l1``.
        beta (float): SmoothL1 transition point.
        log_space (bool): Differentiate ``log(D)`` for scale invariance.
        epsilon (float): Floor before ``log``.
        mask_invalid_gt (bool): Restrict to valid GT pixels.
        max_depth (float, optional): Eigen-protocol upper bound.
        use_magnitude (bool): Compare the combined magnitude ``G``. When False,
            ``Gx`` and ``Gy`` are matched separately, which preserves edge
            ORIENTATION as well as strength.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        beta: float = 0.1,
        log_space: bool = True,
        epsilon: float = 1e-6,
        mask_invalid_gt: bool = True,
        max_depth: float | None = None,
        use_magnitude: bool = True,
    ):
        super().__init__()
        if loss_type not in ("smooth_l1", "l1"):
            raise ValueError(f"boundary loss_type must be 'smooth_l1' or 'l1', got '{loss_type}'.")
        self.loss_type = loss_type
        self.beta = beta
        self.log_space = log_space
        self.epsilon = epsilon
        self.mask_invalid_gt = mask_invalid_gt
        self.max_depth = max_depth
        self.use_magnitude = use_magnitude

    def _compare(self, s: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.loss_type == "l1":
            from .losses import l1_loss

            return l1_loss(s, t, mask)
        return smooth_l1_loss(s, t, mask, beta=self.beta)

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
        gt_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the boundary KD loss."""
        teacher = align_to(teacher_depth, student_depth).to(student_depth.dtype)

        s, t = student_depth, teacher
        if self.log_space:
            s = torch.log(s.clamp(min=self.epsilon))
            t = torch.log(t.clamp(min=self.epsilon))

        mask: torch.Tensor | None = None
        if self.mask_invalid_gt and gt_depth is not None:
            gt = align_to(gt_depth, student_depth, mode="nearest")
            valid = valid_depth_mask(gt, max_depth=self.max_depth)
            # A finite difference is only meaningful when BOTH contributing
            # pixels are valid; otherwise the "edge" is an artifact of the hole
            # boundary rather than real scene structure.
            vx = valid[:, :, :, 1:] & valid[:, :, :, :-1]
            vy = valid[:, :, 1:, :] & valid[:, :, :-1, :]
            vx = torch.cat([vx, vx[:, :, :, -1:]], dim=3)
            vy = torch.cat([vy, vy[:, :, -1:, :]], dim=2)
            mask = vx & vy

        if self.use_magnitude:
            return self._compare(gradient_magnitude(s), gradient_magnitude(t), mask)

        sgx, sgy = depth_gradient(s)
        tgx, tgy = depth_gradient(t)
        return self._compare(sgx, tgx, mask) + self._compare(sgy, tgy, mask)
