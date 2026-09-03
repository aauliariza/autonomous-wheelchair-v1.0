"""Pinhole camera intrinsics and depth-to-distance geometry (spec sections T, U).

THE DISTINCTION THIS MODULE ENFORCES
------------------------------------
A depth map stores **axial depth**: the distance along the camera's optical axis
(the Z coordinate), NOT the Euclidean distance from the camera centre to the
surface point. Calling a depth value "distance" is only correct at the principal
point; everywhere else it understates the true range, and the error grows toward
the image periphery.

Given intrinsics ``fx, fy, cx, cy`` and axial depth ``Z`` at pixel ``(u, v)``:

.. math::
    X = \\frac{(u - c_x)}{f_x} Z, \\qquad
    Y = \\frac{(v - c_y)}{f_y} Z, \\qquad Z = Z
.. math::
    d_{euclidean} = \\sqrt{X^2 + Y^2 + Z^2}
                  = Z \\sqrt{1 + \\left(\\tfrac{u-c_x}{f_x}\\right)^2
                              + \\left(\\tfrac{v-c_y}{f_y}\\right)^2}

The pipeline reports BOTH quantities and never silently substitutes one for the
other. For forward clearance the AXIAL value is the correct one — a wheelchair
advancing along its optical axis is limited by Z, not by slant range — so
``navigation.safety.distance_mode`` defaults to ``axial``, with the Euclidean
value logged alongside it.

Worked magnitude (computed, not estimated): at the corner of a 640x480 frame
with fx=fy=554.26 (60 deg HFOV), the Euclidean distance exceeds the axial depth
by 23.3% -- 1.0 m axial is 1.2332 m slant. At a 1.0 m safety threshold that is a
0.23 m discrepancy, which is easily enough to flip a STOP into a FORWARD. At the
mid-edge of the frame the gap is still ~15%. This is why the two quantities are
reported separately and never interchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class IntrinsicsError(Exception):
    """Raised when intrinsics are missing, malformed, or used uncalibrated."""


class CameraIntrinsics:
    """Pinhole intrinsics with resolution-aware scaling.

    Args:
        fx, fy (float): Focal lengths in pixels.
        cx, cy (float): Principal point in pixels.
        width, height (int): Resolution the intrinsics were measured at.
        distortion (list[float], optional): Brown-Conrady ``[k1,k2,p1,p2,k3]``.
        calibrated (bool): True only for values produced by a real calibration.
        meta (dict, optional): Calibration provenance.

    Examples:
        >>> K = CameraIntrinsics(554.26, 554.26, 320.0, 240.0, 640, 480)
        >>> round(K.euclidean_from_axial(1.0, 320.0, 240.0), 6)   # principal point
        1.0
        >>> round(K.euclidean_from_axial(1.0, 0.0, 0.0), 4)       # corner: slant > axial
        1.1417
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        distortion: list[float] | None = None,
        calibrated: bool = False,
        meta: dict[str, Any] | None = None,
    ):
        if fx <= 0 or fy <= 0:
            raise IntrinsicsError(f"Focal lengths must be positive, got fx={fx}, fy={fy}.")
        if width <= 0 or height <= 0:
            raise IntrinsicsError(f"Image size must be positive, got {width}x{height}.")

        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.width, self.height = int(width), int(height)
        self.distortion = list(distortion) if distortion is not None else [0.0] * 5
        self.calibrated = bool(calibrated)
        self.meta = meta or {}

    # ---------------- construction ----------------

    @classmethod
    def from_yaml(cls, path: str | Path, require_calibrated: bool = False) -> CameraIntrinsics:
        """Load intrinsics from ``configs/camera.yaml``.

        Args:
            require_calibrated (bool): Refuse placeholder values. Enable this
                anywhere a real metric distance is reported to a user.
        """
        from utils.io import load_yaml

        cfg = load_yaml(path)
        missing = [k for k in ("fx", "fy", "cx", "cy") if k not in cfg]
        if missing:
            raise IntrinsicsError(
                f"camera config {path} is missing {missing}.\n"
                f"  Recovery: run `python calibration/camera_calibration.py --output {path}`."
            )

        calibrated = bool(cfg.get("calibrated", False))
        if require_calibrated and not calibrated and not cfg.get("allow_uncalibrated", False):
            raise IntrinsicsError(
                f"{path} contains PLACEHOLDER intrinsics (calibrated: false).\n"
                f"  Euclidean distance computed from them is geometrically meaningless.\n"
                f"  Recovery: calibrate your camera with calibration/camera_calibration.py, "
                f"or set allow_uncalibrated: true to proceed for research purposes only."
            )

        return cls(
            fx=cfg["fx"],
            fy=cfg["fy"],
            cx=cfg["cx"],
            cy=cfg["cy"],
            width=cfg.get("image_width", 640),
            height=cfg.get("image_height", 480),
            distortion=cfg.get("distortion"),
            calibrated=calibrated,
            meta=cfg.get("calibration_meta", {}) or {},
        )

    @classmethod
    def from_fov(cls, hfov_deg: float, width: int, height: int) -> CameraIntrinsics:
        """Construct approximate intrinsics from a horizontal FOV.

        A FALLBACK ONLY — ``calibrated`` stays False. Assumes square pixels and a
        centred principal point, neither of which holds exactly for a real lens.
        """
        f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height, calibrated=False)

    def scaled(self, width: int, height: int) -> CameraIntrinsics:
        """Rescale intrinsics to a different resolution.

        Intrinsics are resolution-dependent: feeding 640x480-calibrated values to
        a 1280x720 frame doubles the effective error. Pipelines that resize MUST
        call this.
        """
        sx = width / float(self.width)
        sy = height / float(self.height)
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
            distortion=self.distortion,
            calibrated=self.calibrated,
            meta={**self.meta, "rescaled_from": f"{self.width}x{self.height}"},
        )

    # ---------------- geometry ----------------

    @property
    def matrix(self) -> np.ndarray:
        """3x3 intrinsic matrix ``K``."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def pixel_to_3d(self, u: float | np.ndarray, v: float | np.ndarray, z: float | np.ndarray):
        """Back-project pixel ``(u, v)`` with axial depth ``z`` to camera-frame ``(X, Y, Z)``."""
        x = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx * np.asarray(z, dtype=np.float64)
        y = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy * np.asarray(z, dtype=np.float64)
        return x, y, np.asarray(z, dtype=np.float64)

    def euclidean_from_axial(self, z: float | np.ndarray, u: float | np.ndarray, v: float | np.ndarray):
        """Convert axial depth ``Z`` at ``(u, v)`` to Euclidean range.

        Uses the closed form ``Z * sqrt(1 + a^2 + b^2)``, which avoids
        materializing X and Y and is exact.
        """
        a = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        b = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        return np.asarray(z, dtype=np.float64) * np.sqrt(1.0 + a * a + b * b)

    def horizontal_fov_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    def vertical_fov_deg(self) -> float:
        """Vertical field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy))))

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation for configs and experiment metadata."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_width": self.width,
            "image_height": self.height,
            "distortion": self.distortion,
            "calibrated": self.calibrated,
            "calibration_meta": self.meta,
        }

    def __repr__(self) -> str:
        state = "calibrated" if self.calibrated else "UNCALIBRATED(placeholder)"
        return (
            f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, "
            f"cy={self.cy:.2f}, {self.width}x{self.height}, {state})"
        )


def pixel_to_3d(u, v, z, intrinsics: CameraIntrinsics):
    """Module-level alias of :meth:`CameraIntrinsics.pixel_to_3d` (spec section U)."""
    return intrinsics.pixel_to_3d(u, v, z)


def depth_to_euclidean_distance(z, u, v, intrinsics: CameraIntrinsics):
    """Module-level alias of :meth:`CameraIntrinsics.euclidean_from_axial` (spec section U)."""
    return intrinsics.euclidean_from_axial(z, u, v)
