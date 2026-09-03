"""Composite Optuna objective (spec section P).

WHY A COMPOSITE OBJECTIVE
-------------------------
A wheelchair is an embedded, safety-critical system, so a single-metric search
optimizes the wrong thing in both directions:

* Maximizing delta1 alone selects a student that may miss the real-time budget.
  A perception loop slower than the stale-frame timeout produces a permanent
  STOP, which is a non-functional system regardless of its accuracy.
* Minimizing latency alone selects a fast student whose distances are wrong,
  which is actively dangerous.

The score therefore carries both:

.. code-block:: text

    score = w_absrel * n(AbsRel) + w_rmse * n(RMSE) + w_delta1 * n(delta1)
            - w_latency * n(latency)

WEIGHT RATIONALE
----------------
delta1 carries the largest accuracy weight (0.40) because it is the standard
headline metric and is naturally bounded in [0,1], which makes its normalization
stable. AbsRel (0.25) and RMSE (0.15) are error terms and are sign-flipped by the
normalizer so that lower is better in all cases. Latency (0.20) is SUBTRACTED
rather than imposed as a hard constraint, so the search can still explore
accurate-but-slow regions and reveal the trade-off curve; a separate
``max_latency_ms`` enforces the hard real-time envelope by failing trials that
exceed it.

METRIC MODE
-----------
Metrics come from ``align="none"`` (MODE 1). Optimizing an aligned delta1 would
select for relative structure while ignoring the absolute scale the obstacle
distance depends on -- the exact failure this project exists to avoid.
"""

from __future__ import annotations

from typing import Any


def normalize(value: float, low: float, high: float, higher_is_better: bool) -> float:
    """Min-max normalize a metric into ``[0, 1]``, clamping outliers.

    Clamping matters: without it, one diverged trial with an AbsRel of 40 would
    compress every reasonable trial into an indistinguishable band and destroy
    the sampler's ability to rank them.
    """
    if high <= low:
        return 0.0
    x = (float(value) - low) / (high - low)
    x = max(0.0, min(1.0, x))
    return x if higher_is_better else 1.0 - x


class CompositeObjective:
    """Turns a trial's measurements into a single score to maximize.

    Args:
        config (dict): The ``objective`` block of ``configs/optuna.yaml``.

    Examples:
        >>> obj = CompositeObjective({
        ...     "weights": {"absrel": 0.25, "rmse": 0.15, "delta1": 0.40, "latency": 0.20},
        ...     "normalization": {"absrel": [0.0, 0.5], "rmse": [0.0, 2.0],
        ...                       "delta1": [0.0, 1.0], "latency_ms": [0.0, 200.0]},
        ... })
        >>> perfect = obj.score({"abs_rel": 0.0, "rmse": 0.0, "delta1": 1.0}, latency_ms=0.0)
        >>> round(perfect, 4)   # accuracy weights sum to 0.80; the latency term subtracts
        0.8
        >>> slower = obj.score({"abs_rel": 0.0, "rmse": 0.0, "delta1": 1.0}, latency_ms=100.0)
        >>> round(slower, 4)    # same accuracy, half the latency budget spent
        0.7
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        w = cfg.get("weights", {}) or {}
        self.w_absrel = float(w.get("absrel", 0.25))
        self.w_rmse = float(w.get("rmse", 0.15))
        self.w_delta1 = float(w.get("delta1", 0.40))
        self.w_latency = float(w.get("latency", 0.20))

        n = cfg.get("normalization", {}) or {}
        self.n_absrel = n.get("absrel", [0.0, 0.5])
        self.n_rmse = n.get("rmse", [0.0, 2.0])
        self.n_delta1 = n.get("delta1", [0.0, 1.0])
        self.n_latency = n.get("latency_ms", [0.0, 200.0])

        self.max_latency_ms = cfg.get("max_latency_ms")
        self.align = cfg.get("align", "none")

    def score(self, metrics: dict[str, float], latency_ms: float) -> float:
        """Compute the composite score for one trial.

        Args:
            metrics (dict): Must contain ``abs_rel``, ``rmse`` and ``delta1``
                from MODE 1 (metric) evaluation.
            latency_ms (float): Measured mean inference latency.

        Returns:
            (float): Higher is better.

        Raises:
            ValueError: If the trial exceeds the hard latency budget, so Optuna
                records it as failed rather than ranking an unusable model.
        """
        if self.max_latency_ms is not None and latency_ms > float(self.max_latency_ms):
            raise ValueError(
                f"Trial latency {latency_ms:.1f} ms exceeds the real-time budget "
                f"{float(self.max_latency_ms):.1f} ms; the model cannot meet the control loop."
            )

        return (
            self.w_absrel * normalize(metrics["abs_rel"], *self.n_absrel, higher_is_better=False)
            + self.w_rmse * normalize(metrics["rmse"], *self.n_rmse, higher_is_better=False)
            + self.w_delta1 * normalize(metrics["delta1"], *self.n_delta1, higher_is_better=True)
            - self.w_latency * normalize(latency_ms, *self.n_latency, higher_is_better=True)
        )

    def describe(self) -> dict[str, Any]:
        """Objective configuration, recorded with every study."""
        return {
            "weights": {
                "absrel": self.w_absrel,
                "rmse": self.w_rmse,
                "delta1": self.w_delta1,
                "latency": self.w_latency,
            },
            "normalization": {
                "absrel": self.n_absrel,
                "rmse": self.n_rmse,
                "delta1": self.n_delta1,
                "latency_ms": self.n_latency,
            },
            "max_latency_ms": self.max_latency_ms,
            "align": self.align,
        }


def suggest(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    """Suggest one hyperparameter from its YAML specification."""
    kind = spec.get("type", "float")
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)))
    return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))


def build_trial_config(
    base_config: dict[str, Any], params: dict[str, Any], trial_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Apply a trial's sampled parameters onto the base distillation config.

    A deep copy is taken so trials cannot contaminate each other through shared
    nested dictionaries -- a mutation bug here would silently invalidate an
    entire study.
    """
    import copy

    cfg = copy.deepcopy(base_config)

    train = cfg.setdefault("train", {})
    for key, target in (
        ("learning_rate", "lr0"),
        ("weight_decay", "weight_decay"),
        ("batch_size", "batch"),
        ("optimizer", "optimizer"),
    ):
        if key in params:
            train[target] = params[key]

    for k in ("epochs", "imgsz", "device", "workers"):
        if k in trial_cfg:
            train[k] = trial_cfg[k]
    if "data_fraction" in trial_cfg:
        train["fraction"] = trial_cfg["data_fraction"]

    kd = cfg.setdefault("kd", {})
    for term in ("gt", "depth", "feature", "boundary", "relative", "roi"):
        key = f"lambda_{term}"
        if key in params:
            node = kd.setdefault(term, {})
            node["lambda"] = params[key]
            # A sampled weight of exactly 0 means the term is off for this trial;
            # leaving it enabled would pay its compute cost for no contribution.
            if term != "gt":
                node["enabled"] = params[key] > 0.0

    if "depth_loss_type" in params:
        kd.setdefault("depth", {})["loss_type"] = params["depth_loss_type"]
    if "feature_loss_type" in params:
        kd.setdefault("feature", {})["loss_type"] = params["feature_loss_type"]
    if "roi_alpha" in params:
        kd.setdefault("roi", {})["alpha"] = params["roi_alpha"]
    if "temperature" in params:
        kd.setdefault("relative", {})["temperature"] = params["temperature"]

    return cfg
