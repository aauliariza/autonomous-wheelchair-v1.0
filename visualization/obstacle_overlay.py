"""Obstacle bounding-box overlays (spec section AH).

Every box is labelled ``obstacle`` — never a recognised object class (spec A) —
followed by its measured distance in metres.

Colour convention (display only):
    red   = blocked (violates the safety distance, or distance is INVALID)
    green = free
    grey  = outside the navigation ROI, shown for information only
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

COLOR_BLOCKED = (0, 0, 255)      # BGR red
COLOR_FREE = (0, 200, 0)         # BGR green
COLOR_OUT_OF_ROI = (150, 150, 150)
COLOR_TEXT = (255, 255, 255)


def draw_obstacles(
    image: np.ndarray,
    obstacles: list[Any],
    show_confidence: bool = True,
    show_sector: bool = True,
    show_inner_roi: bool = False,
) -> np.ndarray:
    """Draw obstacle boxes with their labels and distances.

    Args:
        image (np.ndarray): BGR frame; drawn on a copy.
        obstacles (list): ``Obstacle`` records from ``ObstacleFusion``.
        show_confidence (bool): Append the detector confidence.
        show_sector (bool): Append the assigned sector.
        show_inner_roi (bool): Outline the inner-60% region depth was read from.

    Returns:
        (np.ndarray): Annotated copy of ``image``.
    """
    out = image.copy()

    for ob in obstacles:
        x1, y1, x2, y2 = (int(round(v)) for v in ob.bbox)

        if not ob.in_roi:
            color = COLOR_OUT_OF_ROI
        elif ob.blocked:
            color = COLOR_BLOCKED
        else:
            color = COLOR_FREE

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if show_inner_roi:
            stats = ob.depth_stats or {}
            roi = stats.get("roi")
            if roi:
                cv2.rectangle(out, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), color, 1, cv2.LINE_AA)

        # Distance comes from the model; "--" when it could not be measured.
        dist = f"{ob.distance_m:.2f}m" if ob.distance_m is not None else "-- m"
        parts = [f"{ob.label} {dist}"]
        if show_confidence:
            parts.append(f"c{ob.confidence:.2f}")
        if show_sector and ob.sector:
            parts.append(ob.sector)
        if not ob.in_roi:
            parts.append("out-of-ROI")
        label = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # Keep the caption on-screen for boxes that touch the top edge.
        ty = max(y1, th + 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    return out
