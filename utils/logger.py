"""Logging and per-frame CSV telemetry (spec sections AI, AV)."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED: set[str] = set()

# Per-frame navigation telemetry schema (spec section AI). One row per obstacle;
# frames with no obstacles emit a single row with obstacle_id = -1 so that the
# command trace stays continuous and gaps are distinguishable from dropped frames.
FRAME_LOG_FIELDS = [
    "timestamp",
    "frame_id",
    "obstacle_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "depth_m",
    "euclidean_distance_m",
    "valid_depth_ratio",
    "sector",
    "sector_state",
    "command",
    "inference_latency_ms",
]


def setup_logger(
    name: str = "wheelchair", level: int = logging.INFO, log_file: str | Path | None = None
) -> logging.Logger:
    """Create a configured logger that writes to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED.add(name)
    return logger


def get_logger(name: str = "wheelchair") -> logging.Logger:
    """Return the shared logger, configuring it on first use."""
    if name not in _CONFIGURED:
        return setup_logger(name)
    return logging.getLogger(name)


class FrameLogger:
    """Append-only CSV writer for per-frame navigation telemetry.

    The file handle stays open for the session and is flushed after every frame,
    so a crash or an emergency stop still leaves a complete trace on disk up to
    the last processed frame.

    Examples:
        >>> with FrameLogger("outputs/run.csv") as fl:  # doctest: +SKIP
        ...     fl.log_frame(frame_id=0, obstacles=[], command="STOP", latency_ms=12.3)
    """

    def __init__(self, path: str | Path, fields: list[str] | None = None):
        self.path = Path(path)
        self.fields = fields or FRAME_LOG_FIELDS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately not a `with` block: the handle must stay open for the whole
        # session so each frame is appended and flushed as it is processed.
        # FrameLogger is itself a context manager, and close() is idempotent.
        self._fh = open(self.path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields, extrasaction="ignore")
        self._writer.writeheader()
        self._rows = 0

    def write_row(self, row: dict[str, Any]) -> None:
        """Write one raw row, filling missing columns with an empty string."""
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._rows += 1
        self._fh.flush()

    def log_frame(
        self,
        frame_id: int,
        obstacles: list[Any],
        command: str,
        latency_ms: float,
        timestamp: float | None = None,
    ) -> None:
        """Write one row per obstacle, or a single placeholder row when none were detected.

        Args:
            frame_id (int): Monotonic frame counter.
            obstacles (list): ``Obstacle`` dataclasses (see navigation.obstacle_fusion).
            command (str): Final navigation command emitted for this frame.
            latency_ms (float): Total pipeline latency for this frame.
            timestamp (float, optional): Frame timestamp; defaults to wall clock.
        """
        import time

        ts = time.time() if timestamp is None else timestamp
        if not obstacles:
            self.write_row(
                {
                    "timestamp": f"{ts:.6f}",
                    "frame_id": frame_id,
                    "obstacle_id": -1,
                    "command": command,
                    "inference_latency_ms": f"{latency_ms:.3f}",
                }
            )
            return

        for ob in obstacles:
            x1, y1, x2, y2 = ob.bbox
            self.write_row(
                {
                    "timestamp": f"{ts:.6f}",
                    "frame_id": frame_id,
                    "obstacle_id": ob.id,
                    "bbox_x1": f"{x1:.2f}",
                    "bbox_y1": f"{y1:.2f}",
                    "bbox_x2": f"{x2:.2f}",
                    "bbox_y2": f"{y2:.2f}",
                    "confidence": f"{ob.confidence:.4f}",
                    "depth_m": "" if ob.depth_m is None else f"{ob.depth_m:.4f}",
                    "euclidean_distance_m": (
                        "" if ob.euclidean_distance_m is None else f"{ob.euclidean_distance_m:.4f}"
                    ),
                    "valid_depth_ratio": f"{ob.valid_depth_ratio:.4f}",
                    "sector": "" if ob.sector is None else ob.sector,
                    "sector_state": "blocked" if ob.blocked else "free",
                    "command": command,
                    "inference_latency_ms": f"{latency_ms:.3f}",
                }
            )

    @property
    def rows_written(self) -> int:
        """Number of data rows written so far (excluding the header)."""
        return self._rows

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> FrameLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
