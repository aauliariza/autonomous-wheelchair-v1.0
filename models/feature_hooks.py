"""Intermediate feature extraction for feature-level KD (spec section K).

Layer indices are RESOLVED BY INTROSPECTION, never guessed. The YOLO26-Depth
head stores the indices it consumes in its ``.f`` attribute; ``resolve_depth_feature_layers``
reads that attribute so the KD taps always match the architecture actually loaded.

Verified in the session audit (ultralytics 8.4.138):
    yolo26n-depth  head.f = [16, 19, 22] -> channels ( 64, 128, 256)
    yolo26x-depth  head.f = [16, 19, 22] -> channels (384, 768, 768)
Spatial resolutions are identical between the two models at every level, so
feature KD requires channel projection only — no spatial resampling.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .model_utils import ModelLoadError, get_depth_head


def resolve_depth_feature_layers(model: nn.Module) -> list[int]:
    """Return the backbone/neck layer indices feeding the Depth head.

    Read from ``head.f`` on the loaded network, so a change of architecture or a
    custom YAML is followed automatically.
    """
    head = get_depth_head(model)
    f = getattr(head, "f", None)
    if f is None:
        raise ModelLoadError(
            "Depth head has no 'f' attribute listing its input layers. "
            "Recovery: pass explicit indices via kd.feature.teacher_layers / student_layers."
        )
    return [f] if isinstance(f, int) else list(f)


def describe_feature_layers(
    model: nn.Module, layers: list[int] | None = None, imgsz: int = 640, device: str | torch.device = "cpu"
) -> dict[int, tuple[int, ...]]:
    """Probe the real output shape of each requested layer with a forward pass.

    Used by the KD trainer to build correctly sized projections and by
    ``scripts`` to report architecture facts without hard-coding them.

    Returns:
        (dict): ``{layer_index: (C, H, W)}`` measured, not assumed.
    """
    layers = layers or resolve_depth_feature_layers(model)
    model = model.to(device).eval()
    with FeatureExtractor(model, layers) as fx, torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz, device=device))
        return {k: tuple(v.shape[1:]) for k, v in fx.features.items()}


class FeatureExtractor:
    """Context manager capturing intermediate activations via forward hooks.

    Hooks are the least invasive option available: the Ultralytics source is not
    modified and the network graph is untouched, satisfying the spec's
    "extension over modification" rule (spec section B, STEP 5).

    Hooks are ALWAYS removed on exit, including on exception, so repeated
    train/val cycles cannot leak handles and silently slow training down.

    Examples:
        >>> with FeatureExtractor(net, [16, 19, 22]) as fx:  # doctest: +SKIP
        ...     out = net(x)
        ...     feats = fx.features        # {16: Tensor, 19: Tensor, 22: Tensor}
    """

    def __init__(self, model: nn.Module, layers: list[int], detach: bool = False):
        """
        Args:
            model (nn.Module): Network to instrument.
            layers (list[int]): Indices into ``model.model`` (the nn.Sequential).
            detach (bool): Detach captured tensors. Use True for the FROZEN
                TEACHER so no graph is retained; keep False for the student, whose
                features must stay attached for backpropagation.
        """
        self.model = model.module if hasattr(model, "module") else model
        self.layers = list(layers)
        self.detach = detach
        self.features: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._validate()

    def _sequential(self) -> nn.Sequential:
        seq = getattr(self.model, "model", None)
        if not isinstance(seq, nn.Sequential):
            raise ModelLoadError(
                f"Expected {type(self.model).__name__}.model to be an nn.Sequential of layers; "
                f"got {type(seq).__name__}. Recovery: pass the underlying torch module, not the YOLO wrapper."
            )
        return seq

    def _validate(self) -> None:
        seq = self._sequential()
        n = len(seq)
        bad = [i for i in self.layers if not (-n <= i < n)]
        if bad:
            raise ModelLoadError(
                f"Feature layer index/indices {bad} are out of range for a model with {n} layers "
                f"(valid: 0..{n - 1}).\n  Recovery: run "
                f"`python scripts/inspect_model.py --model <ckpt>` to list the real layer indices."
            )

    def _make_hook(self, index: int):
        def hook(_module: nn.Module, _inp: Any, out: Any) -> None:
            # Some blocks return tuples/lists; the primary activation is element 0.
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            self.features[index] = tensor.detach() if self.detach else tensor

        return hook

    def attach(self) -> FeatureExtractor:
        """Register hooks. Idempotent — re-attaching does not duplicate handles."""
        if self._handles:
            return self
        seq = self._sequential()
        for i in self.layers:
            self._handles.append(seq[i].register_forward_hook(self._make_hook(i)))
        return self

    def remove(self) -> None:
        """Remove every registered hook and drop captured tensors."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self.features = {}

    def clear(self) -> None:
        """Drop captured tensors but keep the hooks registered (per-batch reset)."""
        self.features = {}

    def ordered(self) -> list[torch.Tensor]:
        """Captured features in the order the layers were requested."""
        missing = [i for i in self.layers if i not in self.features]
        if missing:
            raise RuntimeError(
                f"No activations captured for layer(s) {missing}. "
                f"Recovery: run a forward pass INSIDE the FeatureExtractor context before reading .features."
            )
        return [self.features[i] for i in self.layers]

    @property
    def is_attached(self) -> bool:
        """True while hooks are registered."""
        return bool(self._handles)

    def __enter__(self) -> FeatureExtractor:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.remove()
