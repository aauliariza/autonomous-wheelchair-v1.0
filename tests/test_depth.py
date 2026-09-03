"""KD loss and depth-metric tests (spec section AU items 1, 2, 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from distillation.boundary_kd import BoundaryKDLoss, depth_gradient, gradient_magnitude
from distillation.depth_kd import DepthKDLoss
from distillation.feature_kd import FeatureKDLoss
from distillation.losses import DistillationLoss, berhu_loss, l1_loss, smooth_l1_loss, valid_depth_mask
from distillation.relative_kd import RelativeDepthKDLoss
from distillation.roi_kd import ROIKDLoss, boxes_to_mask
from evaluation.metrics import DepthEvaluator, compute_depth_metrics
from models.projection import ProjectionBank


class TestValidMask:
    """Every invalid depth case named in spec section C must be rejected."""

    def test_rejects_all_invalid_kinds(self) -> None:
        """0, NaN, Inf and negative values are all excluded."""
        d = torch.tensor([[[[1.0, 0.0, float("nan"), float("inf"), -1.0, 2.0]]]])
        mask = valid_depth_mask(d)
        assert mask.flatten().tolist() == [True, False, False, False, False, True]

    def test_max_depth_bound(self) -> None:
        """The Eigen protocol's upper bound excludes far ground truth."""
        d = torch.tensor([[[[1.0, 5.0, 20.0]]]])
        assert valid_depth_mask(d, max_depth=10.0).flatten().tolist() == [True, True, False]


class TestLossPrimitives:
    """Masked reductions must be finite, non-negative and empty-safe."""

    @pytest.mark.parametrize("fn", [l1_loss, smooth_l1_loss, berhu_loss])
    def test_perfect_prediction_is_zero(self, fn) -> None:
        """A perfect prediction yields zero loss."""
        x = torch.rand(2, 1, 8, 8) + 0.5
        assert fn(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("fn", [l1_loss, smooth_l1_loss, berhu_loss])
    def test_empty_mask_is_safe(self, fn) -> None:
        """An all-invalid mask returns 0 rather than NaN."""
        p, t = torch.rand(2, 1, 8, 8) + 0.5, torch.rand(2, 1, 8, 8) + 0.5
        out = fn(p, t, torch.zeros(2, 1, 8, 8, dtype=torch.bool))
        assert torch.isfinite(out) and out.item() == 0.0

    @pytest.mark.parametrize("fn", [l1_loss, smooth_l1_loss, berhu_loss])
    def test_non_negative(self, fn) -> None:
        """Losses are non-negative."""
        p, t = torch.rand(4, 1, 8, 8) + 0.5, torch.rand(4, 1, 8, 8) + 0.5
        assert fn(p, t).item() >= 0.0


class TestKDLosses:
    """Every KD term must be finite, differentiable and mask-aware."""

    def test_shapes_and_finiteness(self, student_pred, teacher_pred, gt_depth) -> None:
        """All terms return finite scalars for mismatched-resolution inputs."""
        for loss in (DepthKDLoss(), BoundaryKDLoss(), RelativeDepthKDLoss(num_pairs=128), ROIKDLoss()):
            out = (
                loss(student_pred, teacher_pred, gt_depth)
                if not isinstance(loss, ROIKDLoss)
                else loss(student_pred, teacher_pred, None, gt_depth)
            )
            assert out.ndim == 0, f"{type(loss).__name__} must return a scalar"
            assert torch.isfinite(out), f"{type(loss).__name__} produced a non-finite loss"

    def test_gradients_reach_student(self, student_pred, teacher_pred, gt_depth) -> None:
        """Backward populates finite student gradients."""
        loss = DepthKDLoss()(student_pred, teacher_pred, gt_depth)
        loss.backward()
        assert student_pred.grad is not None
        assert torch.isfinite(student_pred.grad).all()

    def test_teacher_receives_no_gradient(self, student_pred, teacher_pred, gt_depth) -> None:
        """The frozen teacher must never accumulate gradient."""
        teacher = teacher_pred.clone().requires_grad_(False)
        DepthKDLoss()(student_pred, teacher, gt_depth).backward()
        assert teacher.grad is None

    def test_all_invalid_gt_returns_zero(self, student_pred, teacher_pred) -> None:
        """An all-invalid ground truth contributes nothing instead of NaN."""
        gt = torch.zeros(2, 1, 160, 160)
        for loss in (DepthKDLoss(), BoundaryKDLoss(), RelativeDepthKDLoss(num_pairs=64)):
            out = loss(student_pred, teacher_pred, gt)
            assert torch.isfinite(out) and out.item() == pytest.approx(0.0, abs=1e-6)

    def test_near_zero_depth_is_finite(self) -> None:
        """Near-zero depth cannot produce NaN through the log transform."""
        tiny = torch.full((1, 1, 8, 8), 1e-9)
        out = DepthKDLoss(log_space=True)(tiny, tiny, torch.ones(1, 1, 8, 8))
        assert torch.isfinite(out)

    @pytest.mark.parametrize("loss_type", ["l1", "smooth_l1", "berhu"])
    def test_depth_loss_types(self, loss_type, student_pred, teacher_pred, gt_depth) -> None:
        """Every configurable depth loss type works."""
        assert torch.isfinite(DepthKDLoss(loss_type=loss_type)(student_pred, teacher_pred, gt_depth))

    @pytest.mark.parametrize("loss_type", ["mse", "smooth_l1", "cosine"])
    def test_feature_loss_types(self, loss_type) -> None:
        """Every configurable feature loss type works on aligned features."""
        t = [torch.randn(2, c, 8, 8) for c in (64, 128)]
        s = [torch.randn(2, c, 8, 8, requires_grad=True) for c in (64, 128)]
        assert torch.isfinite(FeatureKDLoss(loss_type=loss_type)(t, s))

    def test_feature_shape_mismatch_raises(self) -> None:
        """Unaligned feature channels are a configuration error, not silent."""
        with pytest.raises(ValueError, match="shape mismatch"):
            FeatureKDLoss()([torch.randn(1, 64, 8, 8)], [torch.randn(1, 32, 8, 8)])

    def test_roi_weighting_emphasises_obstacles(self) -> None:
        """A large error inside the ROI is weighted above the same error outside."""
        student = torch.full((1, 1, 20, 20), 1.0)
        teacher = torch.full((1, 1, 20, 20), 1.0)
        teacher[:, :, 5:10, 5:10] = 3.0  # error only inside the ROI

        mask = torch.zeros(1, 1, 20, 20)
        mask[:, :, 5:10, 5:10] = 1.0
        gt = torch.ones(1, 1, 20, 20)

        low = ROIKDLoss(alpha=1.0)(student, teacher, mask, gt)
        high = ROIKDLoss(alpha=5.0)(student, teacher, mask, gt)
        assert high > low

    def test_roi_alpha_below_one_rejected(self) -> None:
        """Obstacles may never be de-emphasised relative to background."""
        with pytest.raises(ValueError, match="alpha must be"):
            ROIKDLoss(alpha=0.5)


class TestBoundaryGradients:
    """Depth-gradient computation must be shape-preserving and NaN-free."""

    def test_gradient_shape_preserved(self) -> None:
        """Gradients are padded back to the input size for mask alignment."""
        d = torch.rand(2, 1, 16, 16)
        gx, gy = depth_gradient(d)
        assert gx.shape == d.shape and gy.shape == d.shape

    def test_detects_a_step_edge(self) -> None:
        """A vertical step produces a non-zero horizontal gradient at the edge."""
        d = torch.zeros(1, 1, 8, 8)
        d[:, :, :, 4:] = 2.0
        assert gradient_magnitude(d)[0, 0, 0, 3].item() > 1.0

    def test_flat_region_is_finite(self) -> None:
        """A perfectly flat region cannot yield NaN through the sqrt."""
        g = gradient_magnitude(torch.ones(1, 1, 8, 8))
        assert torch.isfinite(g).all()

    def test_rejects_wrong_rank(self) -> None:
        """A non-4D tensor is rejected with a clear message."""
        with pytest.raises(ValueError, match="4D"):
            depth_gradient(torch.rand(8, 8))


class TestProjectionBank:
    """Projections must align real measured channel widths."""

    def test_aligns_teacher_to_student(self) -> None:
        """Teacher features are projected into the student's channel widths."""
        bank = ProjectionBank([384, 768, 768], [64, 128, 256])
        t = [torch.randn(2, c, s, s) for c, s in zip((384, 768, 768), (40, 20, 10), strict=True)]
        s = [torch.randn(2, c, sz, sz) for c, sz in zip((64, 128, 256), (40, 20, 10), strict=True)]
        pt, ps = bank(t, s)
        assert [x.shape[1] for x in pt] == [64, 128, 256]
        assert all(a.shape == b.shape for a, b in zip(pt, ps, strict=True))

    def test_matched_widths_cost_no_parameters(self) -> None:
        """Identity is used when widths already match (the head.proj KD point)."""
        assert ProjectionBank([256] * 3, [256] * 3).num_parameters == 0

    def test_level_count_mismatch_raises(self) -> None:
        """Unequal level counts are a config error."""
        with pytest.raises(ValueError, match="Level count mismatch"):
            ProjectionBank([64, 128], [64])

    def test_invalid_direction_raises(self) -> None:
        """Only the two documented directions are accepted."""
        with pytest.raises(ValueError, match="direction must be"):
            ProjectionBank([64], [64], direction="sideways")


class TestDistillationCombiner:
    """The total objective must enforce mandatory GT supervision."""

    def _cfg(self, gt_enabled=True, gt_lambda=1.0):
        return {
            "kd": {
                "gt": {"enabled": gt_enabled, "lambda": gt_lambda},
                "depth": {"enabled": True, "lambda": 0.5},
            }
        }

    def test_weighted_sum(self) -> None:
        """Terms are combined with their configured lambdas."""
        combiner = DistillationLoss(self._cfg())
        total, report = combiner({"gt": torch.tensor(2.0), "depth": torch.tensor(4.0)})
        assert total.item() == pytest.approx(1.0 * 2.0 + 0.5 * 4.0)
        assert report["gt_weighted"] == pytest.approx(2.0)
        assert report["depth_weighted"] == pytest.approx(2.0)

    def test_gt_is_mandatory(self) -> None:
        """The teacher must never be the only supervision source (spec O)."""
        with pytest.raises(ValueError, match="mandatory"):
            DistillationLoss(self._cfg(gt_enabled=False))
        with pytest.raises(ValueError, match="mandatory"):
            DistillationLoss(self._cfg(gt_lambda=0.0))

    def test_disabled_terms_are_skipped(self) -> None:
        """A disabled term contributes nothing even if a value is supplied."""
        cfg = self._cfg()
        cfg["kd"]["depth"]["enabled"] = False
        total, _ = DistillationLoss(cfg)({"gt": torch.tensor(2.0), "depth": torch.tensor(100.0)})
        assert total.item() == pytest.approx(2.0)


class TestBoxesToMask:
    """Bounding-box rasterization must clip and scale correctly."""

    def test_scales_between_resolutions(self) -> None:
        """Boxes rescale when the depth map is smaller than the image."""
        m = boxes_to_mask([torch.tensor([[10.0, 10.0, 50.0, 50.0]])], (32, 32), image_size=(64, 64))
        assert m.shape == (1, 1, 32, 32) and m.sum() > 0

    def test_empty_boxes_give_empty_mask(self) -> None:
        """An image with no detections yields an all-zero mask."""
        assert boxes_to_mask([torch.empty(0, 4)], (16, 16)).sum() == 0

    def test_out_of_bounds_box_is_clipped(self) -> None:
        """A box extending past the frame is clipped, not wrapped."""
        m = boxes_to_mask([torch.tensor([[-50.0, -50.0, 500.0, 500.0]])], (16, 16))
        assert m.sum() == 16 * 16


class TestDepthMetrics:
    """Metric and aligned evaluation modes must behave as documented."""

    def test_perfect_prediction(self) -> None:
        """A perfect prediction scores delta1 = 1 and zero error."""
        d = np.full((32, 32), 2.0)
        m = compute_depth_metrics(d, d)
        assert m["delta1"] == pytest.approx(1.0)
        assert m["abs_rel"] == pytest.approx(0.0, abs=1e-9)

    def test_alignment_hides_pure_scale_error(self) -> None:
        """This is the core spec S claim: alignment masks absolute-scale error."""
        gt = np.random.RandomState(0).uniform(0.5, 5.0, (64, 64))
        pred = gt * 2.0  # pure 2x scale error, structure perfect

        metric = compute_depth_metrics(pred, gt, align="none")
        aligned = compute_depth_metrics(pred, gt, align="median")

        assert metric["delta1"] == pytest.approx(0.0, abs=1e-6), "metric mode must punish scale error"
        assert aligned["delta1"] == pytest.approx(1.0, abs=1e-6), "aligned mode is blind to scale error"

    def test_too_few_valid_pixels_returns_none(self) -> None:
        """Images below the valid-pixel floor are skipped, not scored."""
        gt = np.zeros((32, 32))
        gt[0, 0] = 2.0  # a single valid pixel
        assert compute_depth_metrics(np.ones((32, 32)), gt) is None

    def test_shape_mismatch_raises(self) -> None:
        """Comparing different resolutions is a caller error."""
        with pytest.raises(ValueError, match="does not match"):
            compute_depth_metrics(np.ones((16, 16)), np.ones((32, 32)))

    def test_non_finite_predictions_are_scored_not_dropped(self) -> None:
        """NaN predictions are clamped to a bound so they cannot improve a score."""
        gt = np.full((32, 32), 2.0)
        pred = np.full((32, 32), np.nan)
        m = compute_depth_metrics(pred, gt)
        assert m is not None and np.isfinite(m["abs_rel"])

    def test_evaluator_reports_both_modes(self) -> None:
        """The evaluator accumulates both protocols and their gap."""
        ev = DepthEvaluator(max_depth=10.0)
        gt = np.random.RandomState(1).uniform(0.5, 5.0, (32, 32))
        assert ev.update(gt * 1.5, gt)
        out = ev.compute()
        assert out["num_images"] == 1
        assert "metric" in out and "aligned" in out and "alignment_gap" in out
        assert out["aligned"]["delta1"] > out["metric"]["delta1"]
