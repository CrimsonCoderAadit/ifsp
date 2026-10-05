"""Readable contact sheet for independent WideDepth stereo samples."""

from __future__ import annotations

import cv2
import numpy as np

from stereo.sgbm import disparity_visualization


PANEL_W, PANEL_H = 800, 400
HEADER_H, FOOTER_H, LABEL_H = 68, 68, 38
VIDEO_SIZE = (2 * PANEL_W, HEADER_H + 2 * (PANEL_H + LABEL_H) + FOOTER_H)


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    tile = np.zeros((PANEL_H + LABEL_H, PANEL_W, 3), dtype=np.uint8)
    tile[LABEL_H:] = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def _gt_overlay(left_bgr: np.ndarray, disparity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    view = cv2.resize(left_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    view = cv2.convertScaleAbs(view, alpha=0.45)
    ys, xs = np.nonzero(valid & np.isfinite(disparity) & (disparity > 0))
    if not len(xs):
        return view
    values = disparity[ys, xs]
    low, high = np.percentile(values, (2, 98))
    scaled = np.clip((values - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(scaled.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    # Only the color overlay is resized; numerical disparity stays untouched.
    scale_x, scale_y = PANEL_W / left_bgr.shape[1], PANEL_H / left_bgr.shape[0]
    for x, y, color in zip(xs, ys, colors):
        cv2.circle(view, (min(PANEL_W - 1, int(x * scale_x)),
                          min(PANEL_H - 1, int(y * scale_y))), 1,
                   tuple(int(channel) for channel in color), -1)
    return view


def compose_widedepth_demo(
    sample_id: str, sample_number: int, sample_count: int,
    left_bgr: np.ndarray, right_bgr: np.ndarray,
    gt_disparity: np.ndarray, gt_valid: np.ndarray,
    detections: list, measurements: list, comparison,
) -> np.ndarray:
    """Show one labeled stereo sample; samples are not consecutive video frames."""
    left_view = cv2.resize(left_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    sx, sy = PANEL_W / left_bgr.shape[1], PANEL_H / left_bgr.shape[0]
    for detection, measurement in zip(detections, measurements):
        x1, y1, x2, y2 = detection.bbox_xyxy
        pt1 = (round(x1 * sx), round(y1 * sy))
        pt2 = (round(x2 * sx), round(y2 * sy))
        color = (50, 210, 255) if measurement.disparity_px is not None else (80, 80, 255)
        cv2.rectangle(left_view, pt1, pt2, color, 2)
        depth_label = (f"{measurement.disparity_px:.1f}px" if measurement.disparity_px is not None
                       else "disp n/a")
        label = f"{detection.class_name} {detection.confidence:.2f} | {depth_label}"
        at = (max(3, pt1[0]), max(22, pt1[1] - 5))
        cv2.putText(left_view, label, at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.51, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(left_view, label, at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.51, color, 1, cv2.LINE_AA)

    if comparison is None:
        estimated_view = np.zeros_like(left_view)
        match_text = "SGBM disabled"
    else:
        estimated = cv2.rotate(comparison.estimated_rotated_px, cv2.ROTATE_90_CLOCKWISE)
        estimated_view = disparity_visualization(estimated)
        match_text = (f"SGBM MAE {comparison.mae_native_px:.2f} px"
                      if comparison.mae_native_px is not None else "SGBM MAE unavailable")
    panels = (
        _panel(left_view, "LEFT: YOLO detections + released disparity in boxes"),
        _panel(right_bgr, "RIGHT: matching stereo image"),
        _panel(_gt_overlay(left_bgr, gt_disparity, gt_valid),
               "RELEASED disparity: sparse colored points on left image"),
        _panel(estimated_view, "SGBM: estimated disparity (display colors only)"),
    )
    canvas = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
    canvas[HEADER_H:HEADER_H + PANEL_H + LABEL_H, :PANEL_W] = panels[0]
    canvas[HEADER_H:HEADER_H + PANEL_H + LABEL_H, PANEL_W:] = panels[1]
    canvas[HEADER_H + PANEL_H + LABEL_H:-FOOTER_H, :PANEL_W] = panels[2]
    canvas[HEADER_H + PANEL_H + LABEL_H:-FOOTER_H, PANEL_W:] = panels[3]
    cv2.putText(canvas, f"WideDepth sample {sample_number}/{sample_count}   ID {sample_id}",
                (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Independent dataset samples - playback is a contact sheet, not continuous motion",
                (20, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 210, 255), 2, cv2.LINE_AA)
    footer_y = VIDEO_SIZE[1] - FOOTER_H
    cv2.putText(canvas, f"GT valid: {100 * gt_valid.mean():.3f}%   YOLO objects: {len(detections)}   {match_text}",
                (20, footer_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Disparity is pixels, not metres. No ByteTrack IDs or risk scores across these samples.",
                (20, footer_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 210, 255), 2, cv2.LINE_AA)
    return canvas
