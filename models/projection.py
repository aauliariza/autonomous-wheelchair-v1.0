"""Channel projections for feature-level KD (spec section K).

Teacher and student produce feature maps with the same spatial size but
different channel counts, so a distance between them requires an alignment
layer:

    L_feature = sum_i  distance( P_i(F_Ti), F_Si )

``ProjectionBank`` owns those ``P_i``. They are auxiliary training-only
parameters: they are optimized alongside the student but are NOT part of the
deployed model, so the student's parameter count and latency are unaffected.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """1x1 convolution aligning one feature map's channels to a target width.

    A 1x1 convolution is used rather than a deeper adapter so the term measures
    how well the student's features already match the teacher, instead of
    letting a powerful adapter mask a poor match.

    Args:
        in_channels (int): Source channels.
        out_channels (int): Target channels.
        norm (bool): Insert BatchNorm after the convolution. Stabilizes training
            when teacher and student activation scales differ widely.
        activation (bool): Append ReLU. Off by default — feature KD matches
            signed activations, and a ReLU would discard the negative half.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: bool = True, activation: bool = False):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` to ``out_channels``."""
        return self.block(x)


class ProjectionBank(nn.Module):
    """One ``ProjectionLayer`` per KD level, plus optional spatial alignment.

    Channel widths are supplied by *measured* feature shapes (see
    ``models.feature_hooks.describe_feature_layers``), never by hard-coded
    architecture constants.

    Args:
        teacher_channels (list[int]): Measured teacher channels per level.
        student_channels (list[int]): Measured student channels per level.
        direction (str): ``"teacher_to_student"`` projects the teacher down into
            the student's width (spec section K). ``"student_to_teacher"`` lifts
            the student instead, which preserves the teacher's full feature
            information at the cost of more auxiliary parameters.
        norm (bool): BatchNorm inside each projection.

    Examples:
        >>> bank = ProjectionBank([384, 768, 768], [64, 128, 256])
        >>> t = [torch.randn(2, c, 40, 40) for c in (384, 768, 768)]
        >>> s = [torch.randn(2, c, 40, 40) for c in (64, 128, 256)]
        >>> pt, ps = bank(t, s)
        >>> [x.shape[1] for x in pt]
        [64, 128, 256]
    """

    VALID_DIRECTIONS = ("teacher_to_student", "student_to_teacher")

    def __init__(
        self,
        teacher_channels: list[int],
        student_channels: list[int],
        direction: str = "teacher_to_student",
        norm: bool = True,
    ):
        super().__init__()
        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {self.VALID_DIRECTIONS}, got '{direction}'.")
        if len(teacher_channels) != len(student_channels):
            raise ValueError(
                f"Level count mismatch: {len(teacher_channels)} teacher vs {len(student_channels)} student. "
                f"Recovery: kd.feature.teacher_layers and student_layers must have equal length."
            )

        self.direction = direction
        self.teacher_channels = list(teacher_channels)
        self.student_channels = list(student_channels)

        if direction == "teacher_to_student":
            pairs = zip(teacher_channels, student_channels)
        else:
            pairs = zip(student_channels, teacher_channels)

        # An identity is used where widths already match, so the projection-free
        # KD point (both heads' proj outputs are 256-ch) costs no parameters.
        self.projections = nn.ModuleList(
            [nn.Identity() if cin == cout else ProjectionLayer(cin, cout, norm=norm) for cin, cout in pairs]
        )

    @staticmethod
    def _match_spatial(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Bilinearly resize ``src`` to ``ref``'s spatial size when they differ.

        A no-op for YOLO26n/x-depth, whose levels already align; kept so the bank
        still works if a future backbone changes stride.
        """
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return nn.functional.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self, teacher_feats: list[torch.Tensor], student_feats: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Bring teacher and student features into a common space.

        Returns:
            (tuple): ``(aligned_teacher, aligned_student)`` — same length, and
                elementwise identical shapes, ready for a distance function.
        """
        if len(teacher_feats) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} teacher features, got {len(teacher_feats)}.")
        if len(student_feats) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} student features, got {len(student_feats)}.")

        out_t: list[torch.Tensor] = []
        out_s: list[torch.Tensor] = []
        for proj, ft, fs in zip(self.projections, teacher_feats, student_feats):
            if self.direction == "teacher_to_student":
                # Teacher is frozen: detach so no gradient can reach it, while the
                # projection itself still trains.
                out_t.append(self._match_spatial(proj(ft.detach()), fs))
                out_s.append(fs)
            else:
                out_t.append(ft.detach())
                out_s.append(self._match_spatial(proj(fs), ft))
        return out_t, out_s

    @property
    def num_parameters(self) -> int:
        """Auxiliary parameter count — excluded from the deployed student."""
        return sum(p.numel() for p in self.parameters())
