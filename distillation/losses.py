"""Shared loss primitives and the total KD objective (spec sections I, O).

Masking policy
--------------
Every KD term is evaluated ONLY on pixels where the ground-truth depth is valid
(spec section I). Invalid ground truth in SUN RGB-D means "the sensor returned
nothing", not "the surface is at 0 m" — distilling there would teach the student
to reproduce the teacher's hallucinations in exactly the regions (glass,
specular, far walls) where the teacher is least reliable, and those regions are
disproportionately the dangerous ones for a wheelchair.

Numerical policy
----------------
Masked reductions divide by the number of valid elements, never by the tensor
size, and return a *connected* zero when nothing is valid so that ``backward()``
stays well-defined without a NaN.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# Depth values at or below this (metres) are treated as sensor invalids.
MIN_VALID_DEPTH = 1e-3


def valid_depth_mask(
    depth: torch.Tensor,
    min_depth: float = MIN_VALID_DEPTH,
    max_depth: float | None = None,
) -> torch.Tensor:
    """Boolean mask of usable ground-truth depth pixels.

    Rejects the four invalid cases named in spec section C: ``0``, ``NaN``,
    ``Inf`` and ``<= 0``. ``max_depth`` additionally applies the Eigen protocol's
    upper bound (10 m for SUN RGB-D, beyond the sensors' reliable range).
    """
    mask = torch.isfinite(depth) & (depth > min_depth)
    if max_depth is not None:
        mask = mask & (depth < max_depth)
    return mask


def masked_reduce(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of ``values`` over ``mask``, safe when the mask is empty.

    Returns a zero that is still attached to the graph when no element is valid,
    so an all-invalid batch contributes nothing instead of producing NaN.
    """
    if mask is None:
        return values.mean()
    m = mask.to(values.dtype)
    denom = m.sum()
    if denom <= 0:
        return (values * 0.0).sum()
    return (values * m).sum() / denom


def l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked mean absolute error."""
    return masked_reduce((pred - target).abs(), mask)


def smooth_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, beta: float = 0.1
) -> torch.Tensor:
    """Masked Huber/SmoothL1 loss.

    Quadratic below ``beta`` and linear above it, so large residuals — typically
    depth discontinuities and sensor noise — do not dominate the gradient the way
    they do under a plain L2.
    """
    return masked_reduce(F.smooth_l1_loss(pred, target, reduction="none", beta=beta), mask)


def berhu_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, threshold: float = 0.2
) -> torch.Tensor:
    """Masked reverse-Huber (BerHu) loss.

    The inverse of Huber: L1 for small residuals, L2 for large ones. Standard in
    depth regression (Laina et al., 2016) because it keeps fine-grained gradient
    signal on the many small errors while still punishing gross outliers. The
    cutoff ``c`` is set adaptively to ``threshold * max|residual|`` per batch.
    """
    diff = pred - target
    absdiff = diff.abs()

    if mask is not None and mask.any():
        c = threshold * absdiff[mask].max().detach()
    else:
        c = threshold * absdiff.max().detach()
    c = torch.clamp(c, min=1e-6)

    # L1 branch where |d| <= c; L2 branch (scaled to stay continuous at c) above.
    loss = torch.where(absdiff <= c, absdiff, (diff.pow(2) + c.pow(2)) / (2.0 * c))
    return masked_reduce(loss, mask)


DEPTH_LOSS_FUNCTIONS = {
    "l1": l1_loss,
    "smooth_l1": smooth_l1_loss,
    "berhu": berhu_loss,
}


def get_depth_loss_fn(name: str):
    """Look up a pixelwise depth loss by config name."""
    key = str(name).lower()
    if key not in DEPTH_LOSS_FUNCTIONS:
        raise ValueError(f"Unknown depth loss '{name}'. Available: {sorted(DEPTH_LOSS_FUNCTIONS)}.")
    return DEPTH_LOSS_FUNCTIONS[key]


def align_to(src: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Resize ``src`` to ``ref``'s spatial size if needed.

    The Depth head predicts at input/4 while ground truth is at full resolution,
    so this is exercised on every batch. ``nearest`` must be used for ground
    truth and masks: interpolating a depth map across an invalid region would
    manufacture plausible-looking depth that was never measured.
    """
    if src.shape[-2:] == ref.shape[-2:]:
        return src
    if mode == "nearest":
        return F.interpolate(src, size=ref.shape[-2:], mode="nearest")
    return F.interpolate(src.float(), size=ref.shape[-2:], mode=mode, align_corners=True)


class DistillationLoss(torch.nn.Module):
    """Weighted sum of the ground-truth and knowledge-distillation terms.

    .. math::
        L_{total} = \\lambda_{gt} L_{GT}
                  + \\lambda_{depth} L_{depthKD}
                  + \\lambda_{feature} L_{feature}
                  + \\lambda_{boundary} L_{boundary}
                  + \\lambda_{relative} L_{relative}
                  + \\lambda_{roi} L_{roiKD}

    The ground-truth term is MANDATORY (spec section O): the teacher must never
    become the only source of supervision, or the student inherits the teacher's
    systematic errors with no correction signal. ``__init__`` raises if
    ``lambda_gt <= 0``.

    Every individual term can be switched off from YAML, which is what makes the
    ablation study (spec section Q) a configuration change rather than a code
    change.
    """

    TERMS = ("gt", "depth", "feature", "boundary", "relative", "roi")

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        kd = config.get("kd", config)

        self.lambdas: dict[str, float] = {}
        self.enabled: dict[str, bool] = {}
        for term in self.TERMS:
            spec = kd.get(term, {}) or {}
            self.lambdas[term] = float(spec.get("lambda", 0.0))
            self.enabled[term] = bool(spec.get("enabled", False))

        if not self.enabled["gt"] or self.lambdas["gt"] <= 0:
            raise ValueError(
                "Ground-truth supervision is mandatory (spec section O): set kd.gt.enabled=true "
                "and kd.gt.lambda > 0. The teacher must not be the only supervision source."
            )

        self.last_terms: dict[str, float] = {}

    def forward(self, terms: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Combine per-term losses into the total objective.

        Args:
            terms (dict): Mapping of term name -> scalar loss tensor. Missing or
                disabled terms are skipped.

        Returns:
            (tuple): ``(total_loss, detached per-term dict for logging)``.
        """
        total: torch.Tensor | None = None
        report: dict[str, float] = {}

        for name in self.TERMS:
            if not self.enabled[name] or name not in terms:
                continue
            value = terms[name]
            if value is None:
                continue
            weighted = self.lambdas[name] * value
            total = weighted if total is None else total + weighted
            report[f"{name}_raw"] = float(value.detach())
            report[f"{name}_weighted"] = float(weighted.detach())

        if total is None:
            raise RuntimeError("No active loss terms; at minimum the GT term must be supplied.")

        report["total"] = float(total.detach())
        self.last_terms = report
        return total, report

    def active_terms(self) -> list[str]:
        """Names of the currently enabled terms, for the experiment record."""
        return [t for t in self.TERMS if self.enabled[t]]

    def describe(self) -> dict[str, Any]:
        """Configuration summary written into experiment metadata."""
        return {
            "active_terms": self.active_terms(),
            "lambdas": {t: self.lambdas[t] for t in self.TERMS if self.enabled[t]},
        }
