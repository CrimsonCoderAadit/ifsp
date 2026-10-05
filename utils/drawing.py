"""Small OpenCV overlay for relevant tracked objects and timing."""

import cv2
import numpy as np

from common.events import RiskEvent, RiskLevel, Track
from motion.ego_motion import EgoMotionResult
from risk.corridor import Corridor


def fit_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Shrink a complete annotated frame to fit the preview, preserving aspect ratio."""
    if max_width < 1 or max_height < 1:
        raise ValueError("display dimensions must be positive")
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def draw_bus_reads(frame: np.ndarray, reports: list[dict]) -> np.ndarray:
    """Add OCR and route labels after the existing risk overlay is drawn."""
    for report in reports:
        bbox = report.get("bus_box")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        matched = bool(report.get("route_number"))
        color = (30, 200, 60) if matched else (0, 180, 240)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        board = report.get("board_box")
        if board:
            a, b, c, d = map(int, board)
            cv2.rectangle(frame, (a, b), (c, d), color, 2)
        label = f"OCR {report.get('ocr_raw') or '?'} | {report.get('route_number') or 'no route'}"
        cv2.putText(frame, label, (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    return frame


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    source_fps: float,
    processing_fps: float,
    inference_ms: float,
    events: list[RiskEvent] | None = None,
    corridor: Corridor | None = None,
    show_corridor: bool = False,
    ego_motion: EgoMotionResult | None = None,
    motion_debug: bool = False,
) -> np.ndarray:
    event_by_id = {event.track_id: event for event in events or []}
    colors = {
        RiskLevel.LOW: (50, 220, 80),
        RiskLevel.MEDIUM: (0, 220, 255),
        RiskLevel.HIGH: (0, 120, 255),
        RiskLevel.CRITICAL: (0, 0, 255),
    }
    if show_corridor and corridor is not None:
        polygon = np.array(corridor.polygon(frame.shape[1], frame.shape[0]), dtype=np.int32)
        tint = frame.copy()
        cv2.fillPoly(tint, [polygon], (255, 180, 40))
        cv2.addWeighted(tint, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [polygon], True, (255, 180, 40), 2)
    if motion_debug and ego_motion and ego_motion.analysis_width > 0:
        scale = frame.shape[1] / ego_motion.analysis_width
        for x0, y0, x1, y1 in ego_motion.flow_vectors[:12]:
            cv2.arrowedLine(
                frame, (round(x0 * scale), round(y0 * scale)),
                (round(x1 * scale), round(y1 * scale)),
                (255, 255, 80), 2, tipLength=0.25,
            )

    stereo_dynamic_labels = any(event.risk_mode == "stereo_dynamic" for event in events or [])
    ui_scale = max(1.65 if stereo_dynamic_labels else 1.0, frame.shape[1] / 1200)
    font_scale = 0.55 * ui_scale
    label_scale = 0.42 * ui_scale
    box_thickness = round(2 * ui_scale)
    label_thickness = max(1, round(ui_scale))
    padding = round(6 * ui_scale)
    priority = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}
    for track in sorted(tracks, key=lambda item: priority[event_by_id[item.track_id].risk_level] if item.track_id in event_by_id else 0):
        event = event_by_id.get(track.track_id)
        color = colors[event.risk_level] if event else colors[RiskLevel.LOW]
        x1, y1, x2, y2 = (round(v) for v in track.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness + (1 if event and event.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 0))
        if event:
            lines = [f"{track.class_name.upper()} #{track.track_id}  {event.risk_level.value}"]
            if event.risk_mode == "stereo_dynamic":
                if event.risk_source == "metric_stereo":
                    distance = f"{event.distance_m:.1f}m" if event.distance_m is not None else "depth n/a"
                    velocity = f"{event.relative_velocity_ms:+.1f}m/s" if event.relative_velocity_ms is not None else "vel n/a"
                    ttc = f"TTC {event.ttc_s:.1f}s" if event.ttc_s is not None else "TTC n/a"
                    lines.append(f"{distance}  {velocity}  {ttc}")
                elif event.risk_source == "disparity_stereo":
                    disparity = f"{event.object_disparity_px:.1f}px" if event.object_disparity_px is not None else "disp n/a"
                    rate = f"{event.disparity_rate_px_s:+.1f}px/s" if event.disparity_rate_px_s is not None else "rate n/a"
                    lines.append(f"{disparity}  {rate}  {'closing' if event.disparity_closing else 'not closing'}")
                else:
                    lines.append(f"No stereo  2D {event.raw_approach_rate:+.2f}/s")
                lines.append(f"{event.risk_source}  Path {event.path_relevance:.2f}")
            elif event.risk_mode == "ego_motion":
                lines.append(
                    f"Raw {event.raw_approach_rate:+.2f}  Res {event.residual_approach_rate:+.2f}  Path {event.path_relevance:.2f}"
                )
            else:
                lines.append(
                    f"{'APPROACH' if event.approaching else 'NOT APPROACH'} {event.approach_rate:+.2f}/s  Path {event.path_relevance:.2f}"
                )
        else:
            lines = [f"{track.class_name} ID {track.track_id} {track.confidence:.2f}"]
        line_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)[0] for line in lines]
        text_width = max(size[0] for size in line_sizes)
        line_height = max(size[1] for size in line_sizes) + padding
        label_height = line_height * len(lines) + padding
        label_x = min(max(0, x1), max(0, frame.shape[1] - text_width - padding))
        label_top = max(0, y1 - label_height)
        cv2.rectangle(frame, (label_x, label_top), (label_x + text_width + padding, label_top + label_height), color, -1)
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (label_x + padding // 2, label_top + (index + 1) * line_height), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thickness, cv2.LINE_AA)

    status = f"Source {source_fps:.1f} FPS | Processing {processing_fps:.1f} FPS | Track {inference_ms:.1f} ms | Active {len(tracks)}"
    if events:
        high_count = sum(event.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) for event in events)
        status += f" | High+ {high_count}"
    (text_width, text_height), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_thickness)
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], text_width + 2 * padding), text_height + 2 * padding), (30, 30, 30), -1)
    cv2.putText(frame, status, (padding, text_height + padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), box_thickness, cv2.LINE_AA)
    if motion_debug and ego_motion is not None:
        motion_status = (
            f"Camera {ego_motion.state} | Expansion {ego_motion.background_expansion_rate:+.2f}/s"
            f" | Features {ego_motion.feature_count} | Inliers {ego_motion.inlier_ratio:.2f}"
            f" | Confidence {ego_motion.motion_confidence:.2f}"
        )
        (motion_width, motion_height), _ = cv2.getTextSize(motion_status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_thickness)
        top = text_height + 2 * padding
        cv2.rectangle(frame, (0, top), (min(frame.shape[1], motion_width + 2 * padding), top + motion_height + 2 * padding), (30, 30, 30), -1)
        cv2.putText(frame, motion_status, (padding, top + motion_height + padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), box_thickness, cv2.LINE_AA)
    return frame
