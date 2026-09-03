"""Navigation HUD: ROI, sectors, command and telemetry (spec section AH).

Reproduces the prototype display: the 60% navigation ROI, the five sectors
FL | L | CTR | R | FR with per-sector distance, and the current command.

EVERY NUMBER SHOWN IS MEASURED. Distances come from the depth model via
``ObstacleFusion``; latency and FPS come from the frame timer. Nothing on the HUD
is hard-coded (spec section AH).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

COLOR_BLOCKED = (0, 0, 255)
COLOR_FREE = (0, 200, 0)
COLOR_ROI = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (0, 0, 0)

COMMAND_COLORS = {
    "FORWARD": (0, 200, 0),
    "TURN_LEFT": (0, 200, 255),
    "TURN_RIGHT": (0, 200, 255),
    "STOP": (0, 0, 255),
    "EMERGENCY_STOP": (0, 0, 180),
}


def _panel(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.55) -> None:
    """Blend a translucent panel so text stays readable over any scene."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    image[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, COLOR_PANEL), alpha, 0)


def draw_navigation_overlay(
    image: np.ndarray,
    sector_map: Any,
    command: str,
    safety_distance_m: float,
    latency_ms: float | None = None,
    fps: float | None = None,
    num_obstacles: int = 0,
    safety_state: str | None = None,
    raw_command: str | None = None,
    hysteresis_state: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw the full navigation HUD.

    Args:
        image (np.ndarray): BGR frame; drawn on a copy.
        sector_map (SectorMap): Populated sector state for this frame.
        command (str): Final command after hysteresis and safety override.
        safety_distance_m (float): Threshold in metres, shown for context.
        latency_ms, fps (float, optional): Measured timing; omitted when unknown.
        num_obstacles (int): Obstacles detected this frame.
        safety_state (str, optional): OK | DEGRADED | EMERGENCY_STOP.
        raw_command (str, optional): Pre-hysteresis command, to show smoothing.
        hysteresis_state (dict, optional): Vote window for the HUD.

    Returns:
        (np.ndarray): Annotated copy of ``image``.
    """
    out = image.copy()
    h, w = out.shape[:2]
    roi = sector_map.roi

    # --- global navigation ROI ---
    cv2.rectangle(out, (roi.x1, roi.y1), (roi.x2, min(roi.y2, h) - 1), COLOR_ROI, 2)
    # Right-align the ROI caption: the telemetry panel occupies the top-left.
    roi_label = "60% NAV ROI"
    (rw, _), _ = cv2.getTextSize(roi_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(
        out,
        roi_label,
        (max(roi.x1 + 4, roi.x2 - rw - 6), max(16, roi.y1 + 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_ROI,
        1,
        cv2.LINE_AA,
    )

    # --- sectors ---
    # Vertical budget, bottom-up: command banner | distance labels | sector band.
    # Laid out explicitly so the banner never covers the per-sector distances.
    command_strip = 40  # reserved for the command banner
    distance_strip = 20  # reserved for per-sector distance labels
    band_bottom = min(roi.y2, h - command_strip - distance_strip)
    band_top = max(roi.y1, band_bottom - 46)

    for sector in sector_map:
        r = sector.region
        color = COLOR_BLOCKED if sector.blocked else COLOR_FREE

        # Tint the sector band so occupancy reads at a glance.
        sub = out[band_top:band_bottom, r.x1 : r.x2]
        if sub.size:
            out[band_top:band_bottom, r.x1 : r.x2] = cv2.addWeighted(sub, 0.65, np.full_like(sub, color), 0.35, 0)

        cv2.rectangle(out, (r.x1, band_top), (r.x2 - 1, band_bottom), color, 1)
        cv2.line(out, (r.x1, roi.y1), (r.x1, min(roi.y2, h) - 1), color, 1, cv2.LINE_AA)

        cx = (r.x1 + r.x2) // 2
        (tw, tht), _ = cv2.getTextSize(sector.name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        _panel(out, cx - tw // 2 - 4, band_top - tht - 10, cx + tw // 2 + 4, band_top - 2, alpha=0.5)
        cv2.putText(
            out, sector.name, (cx - tw // 2, band_top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )

        # Measured nearest distance, or "--" when nothing valid was seen.
        text = f"{sector.min_distance_m:.2f}m" if sector.min_distance_m is not None else "--"
        (tw2, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(
            out, text, (cx - tw2 // 2, band_bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1, cv2.LINE_AA
        )

    # --- telemetry panel ---
    lines = [f"obstacles: {num_obstacles}", f"safety dist: {safety_distance_m:.2f} m"]
    if latency_ms is not None:
        lines.append(f"latency: {latency_ms:.1f} ms")
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    if raw_command and raw_command != command:
        lines.append(f"raw: {raw_command}")
    if hysteresis_state:
        lines.append("vote: " + ",".join(c[:3] for c in hysteresis_state.get("history", [])))
    if safety_state and safety_state != "OK":
        lines.append(f"SAFETY: {safety_state}")

    _panel(out, 4, 4, 220, 14 + 18 * len(lines))
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 24 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    # --- command banner ---
    color = COMMAND_COLORS.get(str(command), COLOR_TEXT)
    (tw, th), _ = cv2.getTextSize(str(command), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    bx = (w - tw) // 2
    baseline = h - 10
    _panel(out, bx - 14, baseline - th - 8, bx + tw + 14, baseline + 6, alpha=0.7)
    cv2.putText(out, str(command), (bx, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    return out
