"""Depth map colorization (spec section AH).

Colour is used ONLY for display. Every number drawn on a frame comes from the
model or the navigation layer; nothing is hard-coded, and the colour map never
feeds back into a decision.
"""

from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(
    depth: np.ndarray,
    min_depth: float | None = None,
    max_depth: float | None = None,
    colormap: int = cv2.COLORMAP_INFERNO,
    invalid_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render a metric depth map as a BGR image.

    Invalid pixels are painted a flat colour rather than being normalized with
    the rest: mapping them through the colour ramp would make missing data look
    like a real near or far surface.

    Args:
        depth (np.ndarray): ``(H, W)`` depth in metres.
        min_depth, max_depth (float, optional): Normalization range. Defaults to
            the 2nd/98th percentiles of the valid pixels, which keeps a single
            outlier from flattening the whole image.
        colormap (int): An OpenCV colormap constant.
        invalid_color (tuple): BGR colour for invalid pixels.

    Returns:
        (np.ndarray): ``(H, W, 3)`` BGR image.
    """
    d = np.squeeze(np.asarray(depth, dtype=np.float32))
    valid = np.isfinite(d) & (d > 0)

    if not valid.any():
        return np.full((*d.shape, 3), invalid_color, dtype=np.uint8)

    lo = float(np.percentile(d[valid], 2)) if min_depth is None else float(min_depth)
    hi = float(np.percentile(d[valid], 98)) if max_depth is None else float(max_depth)
    if hi <= lo:
        hi = lo + 1e-3

    norm = np.zeros_like(d, dtype=np.float32)
    norm[valid] = np.clip((d[valid] - lo) / (hi - lo), 0.0, 1.0)

    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    colored[~valid] = invalid_color
    return colored


def depth_side_by_side(rgb: np.ndarray, depth: np.ndarray, **kwargs) -> np.ndarray:
    """Concatenate an RGB frame with its colorized depth map."""
    vis = colorize_depth(depth, **kwargs)
    if vis.shape[:2] != rgb.shape[:2]:
        vis = cv2.resize(vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.hstack([rgb, vis])


def depth_legend(image: np.ndarray, min_depth: float, max_depth: float, width: int = 24) -> np.ndarray:
    """Draw a vertical colour bar with metre labels on the right edge."""
    h = image.shape[0]
    bar = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(-1, 1)
    bar = cv2.applyColorMap((np.repeat(bar, width, axis=1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = np.hstack([image, bar])
    cv2.putText(out, f"{max_depth:.1f}m", (image.shape[1] - 44, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(out, f"{min_depth:.1f}m", (image.shape[1] - 44, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return out
