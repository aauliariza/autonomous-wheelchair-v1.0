"""Knowledge-distillation trainer for YOLO26n-Depth (spec sections H-O).

INTEGRATION STRATEGY — extension, never modification
----------------------------------------------------
Ultralytics source is not patched. The whole KD objective is injected through one
documented extension point: ``BaseModel.loss(batch, preds)`` calls
``self.criterion(preds, batch)``, and ``init_criterion()`` decides what that
criterion is. ``KDDepthModel`` subclasses ``DepthModel`` and returns a
``KDCriterion``, so the standard ``DepthTrainer`` — with its dataloaders,
scheduler, EMA, DDP, AMP, validation and automatic post-training calibration —
runs completely unchanged.

Loss component names are picked up automatically: the trainer derives
``loss_names`` from the keys of the dict the criterion returns, so every KD term
appears as its own column in the progress bar and ``results.csv``.

THREE AUDIT FINDINGS THIS FILE ENCODES
--------------------------------------
1. Ultralytics loads ``.pt`` checkpoints with ``requires_grad=False`` on every
   parameter. Ultralytics' own trainer re-enables them, but the projection bank
   we add is a separate module and the student's state is re-verified anyway via
   ``assert_trainable()``.
2. The Depth head applies calibration ONLY in eval mode. The teacher (always
   eval) returns calibrated depth while the student (train mode) does not, so the
   two live in different scale spaces. ``teacher_space`` selects which space the
   KD terms operate in; see configs/distillation.yaml.
3. Feature layer indices come from ``head.f`` by introspection, and projection
   widths from a real forward pass — never from hard-coded constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.nn.tasks import DepthModel

from distillation.boundary_kd import BoundaryKDLoss
from distillation.depth_kd import DepthKDLoss
from distillation.feature_kd import FeatureKDLoss
from distillation.losses import DistillationLoss
from distillation.relative_kd import RelativeDepthKDLoss
from distillation.roi_kd import ROIKDLoss, boxes_to_mask
from models.projection import ProjectionBank
from models.teacher import TeacherDepthModel
from utils.logger import get_logger

LOG = get_logger("kd_trainer")


class KDCriterion:
    """Computes ``L_total`` for one batch (spec section O).

    Wraps the stock ``DepthLoss26`` for the mandatory ground-truth term and adds
    the enabled KD terms on top.

    Args:
        model (nn.Module): The student network (also the feature-hook target).
        kd_config (dict): The ``configs/distillation.yaml`` mapping.
        teacher (TeacherDepthModel): Frozen teacher.
        projections (ProjectionBank, optional): Channel aligners for feature KD.
        detector (Any, optional): YOLO26n detector for the ROI term.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        kd_config: dict[str, Any],
        teacher: TeacherDepthModel,
        projections: ProjectionBank | None = None,
        detector: Any | None = None,
    ):
        from ultralytics.utils.loss import DepthLoss26

        self.model = model
        self.teacher = teacher
        self.projections = projections
        self.detector = detector
        self.config = kd_config

        kd = kd_config.get("kd", {}) or {}
        data = kd_config.get("data", {}) or {}
        self.max_depth = data.get("max_depth")

        # Mandatory ground-truth supervision (Ultralytics SILog + gradient loss).
        self.gt_criterion = DepthLoss26(model)
        self.combiner = DistillationLoss(kd_config)

        self.depth_kd = self._build_depth_kd(kd.get("depth", {}) or {})
        self.feature_kd = self._build_feature_kd(kd.get("feature", {}) or {})
        self.boundary_kd = self._build_boundary_kd(kd.get("boundary", {}) or {})
        self.relative_kd = self._build_relative_kd(kd.get("relative", {}) or {})
        self.roi_kd = self._build_roi_kd(kd.get("roi", {}) or {})

        feat_cfg = kd.get("feature", {}) or {}
        self.student_layers = list(feat_cfg.get("student_layers", []) or [])
        if self.feature_kd is not None and self.student_layers:
            LOG.info("Student feature capture enabled on layers %s", self.student_layers)

        roi_cfg = kd.get("roi", {}) or {}
        self.roi_conf = float(roi_cfg.get("conf", 0.25))
        self.roi_max_det = int(roi_cfg.get("max_det", 50))
        self.roi_inner_ratio = float(roi_cfg.get("inner_ratio", 0.6))
        self.cache_detections = bool(roi_cfg.get("cache_detections", True))
        self._detection_cache: dict[str, list[torch.Tensor]] = {}
        self._warned_no_features = False

        # Loss column names, in a stable order for results.csv.
        self.loss_names = tuple(f"{t}_loss" for t in self.combiner.active_terms())
        LOG.info("KD terms active: %s", self.combiner.describe())

    # ---------------- builders ----------------

    def _build_depth_kd(self, c: dict[str, Any]) -> DepthKDLoss | None:
        if not c.get("enabled"):
            return None
        return DepthKDLoss(
            loss_type=c.get("loss_type", "smooth_l1"),
            beta=float(c.get("beta", 0.1)),
            berhu_threshold=float(c.get("berhu_threshold", 0.2)),
            log_space=bool(c.get("log_space", True)),
            epsilon=float(c.get("epsilon", 1e-6)),
            mask_invalid_gt=bool(c.get("mask_invalid_gt", True)),
            max_depth=self.max_depth,
        )

    def _build_feature_kd(self, c: dict[str, Any]) -> FeatureKDLoss | None:
        if not c.get("enabled"):
            return None
        return FeatureKDLoss(
            loss_type=c.get("loss_type", "mse"),
            normalize=bool(c.get("normalize", True)),
        )

    def _build_boundary_kd(self, c: dict[str, Any]) -> BoundaryKDLoss | None:
        if not c.get("enabled"):
            return None
        return BoundaryKDLoss(
            loss_type=c.get("loss_type", "smooth_l1"),
            beta=float(c.get("beta", 0.1)),
            log_space=bool(c.get("log_space", True)),
            mask_invalid_gt=bool(c.get("mask_invalid_gt", True)),
            max_depth=self.max_depth,
        )

    def _build_relative_kd(self, c: dict[str, Any]) -> RelativeDepthKDLoss | None:
        if not c.get("enabled"):
            return None
        return RelativeDepthKDLoss(
            num_pairs=int(c.get("num_pairs", 4096)),
            margin=float(c.get("margin", 0.0)),
            tolerance=float(c.get("tolerance", 0.03)),
            loss_type=c.get("loss_type", "ranking"),
            temperature=float(c.get("temperature", 1.0)),
            max_depth=self.max_depth,
        )

    def _build_roi_kd(self, c: dict[str, Any]) -> ROIKDLoss | None:
        if not c.get("enabled"):
            return None
        return ROIKDLoss(
            alpha=float(c.get("alpha", 3.0)),
            loss_type=c.get("loss_type", "smooth_l1"),
            log_space=bool(c.get("log_space", True)),
            mask_invalid_gt=bool(c.get("mask_invalid_gt", True)),
            max_depth=self.max_depth,
        )

    # ---------------- helpers ----------------

    def _captured_student_features(self) -> list[torch.Tensor] | None:
        """Return the hooked student activations, or None when unavailable.

        During TRAINING, ``BaseModel.loss()`` runs the forward itself, so the
        hooks always fire and missing activations indicate a real bug -- which is
        raised.

        During VALIDATION the validator computes ``preds`` first and passes them
        in, so ``loss()`` never runs a forward and no activations are captured.
        That is expected, not an error: the feature term is simply skipped for
        the val-loss figure. Depth metrics (delta1/abs_rel/rmse/silog) are
        unaffected, since they are computed from predictions, not from the loss.
        """
        if not self.student_layers:
            return None
        captured = _KD_FEATURES.get(id(self.model), {})
        missing = [i for i in self.student_layers if i not in captured]
        if not missing:
            return [captured[i] for i in self.student_layers]

        if self.model.training:
            raise RuntimeError(
                f"No student activations captured for layer(s) {missing} during TRAINING. "
                f"The forward hooks did not fire, so the feature KD term would silently be a no-op. "
                f"Recovery: confirm KDCriterion is constructed before the forward pass "
                f"(Ultralytics builds the criterion inside BaseModel.loss() before calling forward)."
            )

        if not self._warned_no_features:
            LOG.info(
                "Validation passes precomputed predictions, so student feature hooks do not fire; "
                "the feature KD term is omitted from the reported val loss. Depth metrics are unaffected."
            )
            self._warned_no_features = True
        return None

    def _detect_obstacles(self, images: torch.Tensor, keys: list[str] | None) -> list[torch.Tensor]:
        """Run the obstacle detector for the ROI term, caching per image.

        Boxes depend only on the RGB input, not on the student's weights, so
        after the first epoch they are reused. On a 100-epoch run this removes 99
        detector passes per image.
        """
        if self.detector is None:
            return [torch.empty(0, 4) for _ in range(images.shape[0])]

        if self.cache_detections and keys and all(k in self._detection_cache for k in keys):
            return [self._detection_cache[k] for k in keys]

        boxes: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                results = self.detector.predict(images, conf=self.roi_conf, max_det=self.roi_max_det, verbose=False)
            for r in results:
                b = r.boxes.xyxy.detach().cpu() if r.boxes is not None else torch.empty(0, 4)
                boxes.append(b)
        except (RuntimeError, AttributeError, ValueError) as e:
            # The ROI term degrades to plain depth KD rather than killing the run.
            LOG.warning(
                "Obstacle detection failed (%s: %s); ROI term falls back to uniform weighting.", type(e).__name__, e
            )
            boxes = [torch.empty(0, 4) for _ in range(images.shape[0])]

        if self.cache_detections and keys and len(keys) == len(boxes):
            for k, b in zip(keys, boxes, strict=True):
                self._detection_cache[k] = b
        return boxes

    # ---------------- main entry ----------------

    def __call__(self, preds: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total loss for one batch.

        Returns:
            (tuple): ``(loss * batch_size, {name: detached scalar})`` — the
                convention Ultralytics' trainer expects.
        """
        images = batch["img"]
        student_depth = preds["depth"] if isinstance(preds, dict) else preds
        if isinstance(student_depth, (tuple, list)):
            student_depth = student_depth[0]
        if student_depth.ndim == 3:
            student_depth = student_depth.unsqueeze(1)

        gt_depth = batch.get("depth")
        if gt_depth is not None and gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)

        terms: dict[str, torch.Tensor] = {}

        # --- mandatory ground-truth term ---
        gt_loss, _ = self.gt_criterion(preds, batch)
        # DepthLoss26 returns a per-component vector scaled by batch size; reduce
        # to the mean per-sample scalar so KD lambdas are batch-size independent.
        terms["gt"] = gt_loss.sum() / max(images.shape[0], 1)

        needs_teacher = any(
            x is not None for x in (self.depth_kd, self.feature_kd, self.boundary_kd, self.relative_kd, self.roi_kd)
        )
        if not needs_teacher:
            total, report = self.combiner(terms)
            return total * images.shape[0], {k: torch.tensor(v) for k, v in self._named(report).items()}

        # --- one teacher forward serves every KD term ---
        # NOTE: the student's activations were captured by the forward pass that
        # produced `preds`; they are read below and only cleared afterwards.
        want_features = self.feature_kd is not None and bool(self.student_layers)
        with torch.no_grad():
            if want_features:
                teacher_depth, teacher_feats = self.teacher.forward_with_features(images)
            else:
                teacher_depth, teacher_feats = self.teacher.forward(images), []

        teacher_depth = teacher_depth.to(student_depth.dtype)

        if self.depth_kd is not None:
            terms["depth"] = self.depth_kd(student_depth, teacher_depth, gt_depth)

        if self.feature_kd is not None:
            student_feats = self._captured_student_features()
            if student_feats is not None and teacher_feats:
                if self.projections is not None:
                    aligned_t, aligned_s = self.projections(teacher_feats, student_feats)
                else:
                    aligned_t, aligned_s = teacher_feats, student_feats
                terms["feature"] = self.feature_kd(aligned_t, aligned_s)

        if self.boundary_kd is not None:
            terms["boundary"] = self.boundary_kd(student_depth, teacher_depth, gt_depth)

        if self.relative_kd is not None:
            terms["relative"] = self.relative_kd(student_depth, teacher_depth, gt_depth)

        if self.roi_kd is not None:
            keys = batch.get("im_file")
            keys = list(keys) if isinstance(keys, (list, tuple)) else None
            boxes = self._detect_obstacles(images, keys)
            mask = boxes_to_mask(
                boxes,
                shape=tuple(student_depth.shape[-2:]),
                image_size=tuple(images.shape[-2:]),
                inner_ratio=self.roi_inner_ratio,
                device=student_depth.device,
            )
            terms["roi"] = self.roi_kd(student_depth, teacher_depth, mask, gt_depth)

        total, report = self.combiner(terms)

        # Release the captured activations now that every term has consumed them,
        # so they are not held alive until the next forward pass.
        _KD_FEATURES.pop(id(self.model), None)

        return total * images.shape[0], {
            k: torch.tensor(v, device=student_depth.device) for k, v in self._named(report).items()
        }

    def _named(self, report: dict[str, float]) -> dict[str, float]:
        """Map the combiner's report to the per-term columns the trainer logs."""
        return {f"{t}_loss": report.get(f"{t}_raw", 0.0) for t in self.combiner.active_terms()}

    def close(self) -> None:
        """Drop any captured student activations."""
        _KD_FEATURES.pop(id(self.model), None)


# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------
# The KD context lives at module scope, and KDDepthModel is a module-level class.
# Both are required for checkpointing to work:
#   * A class defined inside a factory function is a "local object" and cannot be
#     pickled, so torch.save of the model fails at the first checkpoint.
#   * A criterion stored as an instance attribute drags forward-hook handles, the
#     teacher and the detector into the pickle, which fails on _thread.lock.
# Keeping the class module-level and the criterion in a module-level registry
# leaves the model itself a plain, picklable DepthModel-shaped object.
_KD_CONTEXT: dict[str, Any] = {}
_KD_CRITERIA: dict[int, KDCriterion] = {}
# Student activations captured by the most recent forward, keyed by id(model).
_KD_FEATURES: dict[int, dict[int, torch.Tensor]] = {}


def set_kd_context(kd_config: dict[str, Any], teacher: TeacherDepthModel, projections, detector) -> None:
    """Register the KD context that ``KDDepthModel`` instances will use."""
    _KD_CONTEXT.clear()
    _KD_CRITERIA.clear()
    feat = (kd_config.get("kd", {}) or {}).get("feature", {}) or {}
    layers = list(feat.get("student_layers", []) or []) if feat.get("enabled") else []
    _KD_CONTEXT.update(
        {
            "config": kd_config,
            "teacher": teacher,
            "projections": projections,
            "detector": detector,
            "student_layers": layers,
        }
    )


def clear_kd_context() -> None:
    """Drop the KD context and detach every hook it created."""
    for crit in _KD_CRITERIA.values():
        crit.close()
    _KD_CRITERIA.clear()
    _KD_FEATURES.clear()
    _KD_CONTEXT.clear()


class KDDepthModel(DepthModel):
    """DepthModel whose loss is the full KD objective (spec section O).

    Behaves exactly like a stock ``DepthModel`` for inference and serialization;
    only ``loss()`` differs. The KD machinery is reached through the module-level
    registry rather than through instance state, so ``deepcopy`` and
    ``torch.save`` see nothing but ordinary tensors.
    """

    def _predict_once(self, x, profile=False, embed=None):
        """Forward pass that also records the KD tap layers' activations.

        WHY NOT FORWARD HOOKS FOR THE STUDENT
        -------------------------------------
        ``register_forward_hook`` stores the callback in the submodule's
        ``_forward_hooks`` OrderedDict, which is part of the module's picklable
        state. Saving a checkpoint then tries to pickle the hook closure and
        fails with ``Can't pickle local object ... hook``. The teacher keeps
        using hooks (it is never serialized); the student, which IS saved every
        epoch, captures features here instead.

        This mirrors ``BaseModel._predict_once`` and additionally copies the
        already-saved ``y[i]`` entries for the requested layers. Those layers
        feed the Depth head, so they are in ``self.save`` and are retained by the
        base implementation regardless -- no extra memory is held.
        """
        want = _KD_CONTEXT.get("student_layers") or []
        if not want:
            return super()._predict_once(x, profile=profile, embed=embed)

        captured: dict[int, torch.Tensor] = {}
        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed else {-1}
        max_idx = max(embed_set)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in want:
                captured[m.i] = x
            if m.i in embed_set:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        _KD_FEATURES[id(self)] = captured
        return x

    def _kd_criterion(self) -> KDCriterion:
        """Fetch (or lazily build) this instance's criterion from the registry."""
        crit = _KD_CRITERIA.get(id(self))
        if crit is None:
            if not _KD_CONTEXT:
                raise RuntimeError(
                    "KD context is not set. Call training.kd_trainer.set_kd_context(...) "
                    "before constructing KDDepthModel."
                )
            crit = KDCriterion(
                self,
                _KD_CONTEXT["config"],
                _KD_CONTEXT["teacher"],
                _KD_CONTEXT["projections"],
                _KD_CONTEXT["detector"],
            )
            _KD_CRITERIA[id(self)] = crit
        return crit

    # A class-level property: the trainer reads `unwrap_model(model).criterion`
    # directly, and a property never enters the instance __dict__.
    @property
    def criterion(self) -> KDCriterion:
        """The KD objective for this model instance."""
        return self._kd_criterion()

    @criterion.setter
    def criterion(self, value) -> None:
        """Ignore external assignment; the registry owns the criterion."""
        if value is not None:
            _KD_CRITERIA[id(self)] = value

    def loss(self, batch, preds=None):
        """Compute the KD loss for one batch."""
        crit = self._kd_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        return crit(preds, batch)

    def init_criterion(self):
        """Return the KD criterion (kept for API compatibility)."""
        return self._kd_criterion()


def strip_kd_wrapper(checkpoint_path: str | Path) -> bool:
    """Rewrite a saved checkpoint's model class from KDDepthModel to DepthModel.

    The KD wrapper is training scaffolding. Rewriting the class on the saved
    object makes the deployed student a stock YOLO26n-Depth model that loads with
    plain Ultralytics, with no dependency on this repository.

    Returns:
        (bool): True when a checkpoint was rewritten.
    """
    import torch
    from ultralytics.utils.patches import torch_load

    path = Path(checkpoint_path)
    if not path.exists():
        return False

    ckpt = torch_load(path, map_location="cpu")
    changed = False
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if m is not None and isinstance(m, KDDepthModel):
            m.__class__ = DepthModel
            changed = True
    if changed:
        torch.save(ckpt, path)
        LOG.info("Stripped the KD wrapper from %s; it is now a stock DepthModel.", path.name)
    return changed
