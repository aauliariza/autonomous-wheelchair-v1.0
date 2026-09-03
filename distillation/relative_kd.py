"""Relative-depth distillation (spec section M).

Preserves ordinal depth relationships between sampled pixel pairs: if the
teacher says point *i* is nearer than point *j*, the student should agree.

.. math::
    L_{relative} = \\frac{1}{N} \\sum_{(i,j)}
        \\log\\Big(1 + \\exp\\big(-\\,r_{ij}\\,(\\log D_S(j) - \\log D_S(i))\\big)\\Big)

with :math:`r_{ij} \\in \\{-1, 0, +1\\}` the teacher's ordinal label.

Why ordinal supervision helps here: obstacle avoidance is fundamentally a
comparison ("is that nearer than my safety threshold, and nearer than the thing
beside it?"). Ordinal structure survives scale error, so this term keeps the
student's *ranking* correct even where its absolute metric scale drifts.

Sampling (required by spec section M)
-------------------------------------
An exhaustive pairing over a 160x160 map would be ~3.3e8 pairs per image. Pairs
are therefore RANDOMLY SAMPLED from valid pixels only, with a configurable budget
(``num_pairs``), which bounds both memory and compute per step regardless of
resolution.

Pairs whose teacher log-ratio falls below ``tolerance`` are labelled 0 ("equal")
and excluded from the ranking loss: forcing an order on pixels the teacher itself
considers coplanar would inject pure noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import align_to, valid_depth_mask


class RelativeDepthKDLoss(nn.Module):
    """Pairwise ordinal depth-ranking distillation.

    Args:
        num_pairs (int): Sampled pairs per image.
        margin (float): Required log-space separation for the ranking loss.
        tolerance (float): ``|log(D_T(i)/D_T(j))|`` below this counts as equal.
        loss_type (str): ``ranking`` (soft logistic) | ``pairwise_l1`` (matches
            the log-ratio magnitude, not just its sign).
        temperature (float): Scales the logit; higher values sharpen the decision
            boundary. Exposed for the Optuna search (spec section P).
        epsilon (float): Floor before ``log``.
        max_depth (float, optional): Eigen-protocol upper bound.
        generator (torch.Generator, optional): For reproducible sampling.
    """

    def __init__(
        self,
        num_pairs: int = 4096,
        margin: float = 0.0,
        tolerance: float = 0.03,
        loss_type: str = "ranking",
        temperature: float = 1.0,
        epsilon: float = 1e-6,
        max_depth: float | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if loss_type not in ("ranking", "pairwise_l1"):
            raise ValueError(f"relative loss_type must be 'ranking' or 'pairwise_l1', got '{loss_type}'.")
        if num_pairs < 1:
            raise ValueError(f"num_pairs must be >= 1, got {num_pairs}.")
        self.num_pairs = num_pairs
        self.margin = margin
        self.tolerance = tolerance
        self.loss_type = loss_type
        self.temperature = temperature
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.generator = generator

    def _sample_indices(self, valid_idx: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw two independent index sets from the valid pixels.

        Sampling WITH replacement is deliberate: it costs one cheap randint per
        side and, at a 4096 budget against tens of thousands of valid pixels, the
        collision rate is negligible. Self-pairs (i == j) are filtered by the
        caller via the tolerance test, since their log-ratio is exactly 0.
        """
        n = valid_idx.numel()
        k = min(self.num_pairs, n)
        i = torch.randint(0, n, (k,), device=device, generator=self.generator)
        j = torch.randint(0, n, (k,), device=device, generator=self.generator)
        return valid_idx[i], valid_idx[j]

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
        gt_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the relative-depth KD loss over sampled pairs."""
        teacher = align_to(teacher_depth, student_depth).to(student_depth.dtype)
        device = student_depth.device
        b = student_depth.shape[0]

        base_mask = torch.isfinite(teacher) & (teacher > self.epsilon)
        if gt_depth is not None:
            gt = align_to(gt_depth, student_depth, mode="nearest")
            base_mask = base_mask & valid_depth_mask(gt, max_depth=self.max_depth)

        log_s = torch.log(student_depth.clamp(min=self.epsilon))
        log_t = torch.log(teacher.clamp(min=self.epsilon))

        total = None
        counted = 0

        for bi in range(b):
            valid_idx = base_mask[bi].reshape(-1).nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() < 2:
                continue

            ia, ib = self._sample_indices(valid_idx, device)
            s_flat = log_s[bi].reshape(-1)
            t_flat = log_t[bi].reshape(-1)

            dt = t_flat[ib] - t_flat[ia]  # teacher log-ratio
            ds = s_flat[ib] - s_flat[ia]  # student log-ratio

            if self.loss_type == "pairwise_l1":
                pair_loss = (ds - dt).abs().mean()
            else:
                # Keep only pairs the teacher orders confidently.
                keep = dt.abs() > self.tolerance
                if not keep.any():
                    continue
                sign = torch.sign(dt[keep])
                # softplus(-r * (ds - margin*r)) — a smooth 0/1 ranking penalty.
                logit = self.temperature * (ds[keep] - self.margin * sign)
                pair_loss = F.softplus(-sign * logit).mean()

            total = pair_loss if total is None else total + pair_loss
            counted += 1

        if total is None or counted == 0:
            # No usable pairs in this batch: return a graph-connected zero.
            return (student_depth * 0.0).sum()
        return total / counted
