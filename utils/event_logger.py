"""Optional observation-level CSV for later baseline comparisons."""

import csv
from pathlib import Path

from common.events import RiskEvent, Track
from motion.ego_motion import EgoMotionResult, EgoMotionTiming
from perception.track_history import HistoryObservation
from stereo.object_depth import DepthObservation


FIELDNAMES = [
    "timestamp", "frame_index", "track_id", "class", "confidence",
    "bbox_area", "smoothed_bbox_area", "approach_rate", "approach_confidence",
    "path_relevance", "direction", "risk_score", "risk_level",
    "risk_mode", "raw_approach_rate", "raw_approach_confidence",
    "background_expansion_rate", "ego_motion_state", "ego_motion_confidence",
    "residual_approach_rate", "residual_approach_confidence",
    "motion_feature_count", "motion_inlier_ratio",
    "feature_detection_latency_ms", "optical_flow_latency_ms",
    "global_estimation_latency_ms", "global_motion_latency_ms",
    "detector_latency_ms", "risk_logic_latency_ms", "total_frame_latency_ms",
    "bus_ocr_raw", "bus_route_number", "bus_route_name", "bus_confidence", "bus_ocr_latency_ms",
]


class EventLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()

    def write(
        self,
        frame_index: int,
        track: Track,
        observation: HistoryObservation,
        event: RiskEvent,
        detector_latency_ms: float,
        risk_logic_latency_ms: float,
        total_frame_latency_ms: float,
        ego_motion: EgoMotionResult | None = None,
        ego_timing: EgoMotionTiming | None = None,
        bus_read: dict | None = None,
    ) -> None:
        self._writer.writerow({
            "timestamp": event.timestamp,
            "frame_index": frame_index,
            "track_id": track.track_id,
            "class": track.class_name,
            "confidence": track.confidence,
            "bbox_area": observation.bbox_area,
            "smoothed_bbox_area": observation.smoothed_bbox_area,
            "approach_rate": event.approach_rate,
            "approach_confidence": event.approach_confidence,
            "path_relevance": event.path_relevance,
            "direction": event.direction,
            "risk_score": event.risk_score,
            "risk_level": event.risk_level.value,
            "risk_mode": event.risk_mode,
            "raw_approach_rate": event.raw_approach_rate,
            "raw_approach_confidence": event.raw_approach_confidence,
            "background_expansion_rate": event.background_expansion_rate,
            "ego_motion_state": event.ego_motion_state,
            "ego_motion_confidence": event.ego_motion_confidence,
            "residual_approach_rate": event.residual_approach_rate,
            "residual_approach_confidence": event.residual_approach_confidence,
            "motion_feature_count": ego_motion.feature_count if ego_motion else 0,
            "motion_inlier_ratio": ego_motion.inlier_ratio if ego_motion else 0.0,
            "feature_detection_latency_ms": ego_timing.feature_detection_s * 1000 if ego_timing else 0.0,
            "optical_flow_latency_ms": ego_timing.optical_flow_s * 1000 if ego_timing else 0.0,
            "global_estimation_latency_ms": ego_timing.global_estimation_s * 1000 if ego_timing else 0.0,
            "global_motion_latency_ms": ego_timing.total_s * 1000 if ego_timing else 0.0,
            "detector_latency_ms": detector_latency_ms,
            "risk_logic_latency_ms": risk_logic_latency_ms,
            "total_frame_latency_ms": total_frame_latency_ms,
            "bus_ocr_raw": (bus_read or {}).get("ocr_raw", ""),
            "bus_route_number": (bus_read or {}).get("route_number", ""),
            "bus_route_name": (bus_read or {}).get("route_name", ""),
            "bus_confidence": (bus_read or {}).get("confidence", ""),
            "bus_ocr_latency_ms": (bus_read or {}).get("ocr_ms", ""),
        })

    def close(self) -> None:
        self._file.close()


STEREO_FIELDS = [
    "timestamp", "frame_index", "frame_id", "track_id", "class", "confidence",
    "raw_depth_m", "smoothed_depth_m", "depth_confidence", "valid_depth_ratio",
    "depth_mad_m", "temporal_depth_variance_m2", "distance_m",
    "raw_approach_rate", "residual_approach_rate", "path_relevance",
    "risk_score", "risk_level", "stereo_latency_ms", "stereo_input_latency_ms",
    "rectification_latency_ms", "sgbm_latency_ms", "depth_conversion_latency_ms",
    "object_depth_latency_ms", "depth_smoothing_latency_ms", "yolo_latency_ms",
    "total_latency_ms",
    "risk_source", "risk_reason", "closing", "relative_velocity_ms",
    "velocity_confidence", "velocity_fit_error", "ttc_s", "ttc_confidence",
    "object_disparity_px", "disparity_rate_px_s", "disparity_rate_confidence",
    "disparity_closing", "disparity_confidence", "bbox_approach_rate",
    "history_update_latency_ms", "velocity_regression_latency_ms",
    "disparity_rate_latency_ms", "ttc_latency_ms", "risk_fusion_latency_ms",
    "stage5_logic_latency_ms",
]


class StereoEventLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=STEREO_FIELDS)
        self._writer.writeheader()

    def write(
        self, frame_index: int, frame_id: str, track: Track, event: RiskEvent,
        depth: DepthObservation, timings_ms: dict[str, float],
    ) -> None:
        self._writer.writerow({
            "timestamp": event.timestamp,
            "frame_index": frame_index,
            "frame_id": frame_id,
            "track_id": track.track_id,
            "class": track.class_name,
            "confidence": track.confidence,
            "raw_depth_m": depth.raw_depth_m,
            "smoothed_depth_m": depth.smoothed_depth_m,
            "depth_confidence": depth.depth_confidence,
            "valid_depth_ratio": depth.valid_depth_ratio,
            "depth_mad_m": depth.depth_mad_m,
            "temporal_depth_variance_m2": depth.temporal_variance_m2,
            "distance_m": event.distance_m,
            "raw_approach_rate": event.raw_approach_rate,
            "residual_approach_rate": event.residual_approach_rate,
            "path_relevance": event.path_relevance,
            "risk_score": event.risk_score,
            "risk_level": event.risk_level.value,
            "stereo_latency_ms": sum(timings_ms.get(key, 0.0) for key in (
                "rectification", "sgbm", "depth_conversion", "object_depth", "depth_smoothing"
            )),
            "stereo_input_latency_ms": timings_ms.get("stereo_input", 0.0),
            "rectification_latency_ms": timings_ms.get("rectification", 0.0),
            "sgbm_latency_ms": timings_ms.get("sgbm", 0.0),
            "depth_conversion_latency_ms": timings_ms.get("depth_conversion", 0.0),
            "object_depth_latency_ms": timings_ms.get("object_depth", 0.0),
            "depth_smoothing_latency_ms": timings_ms.get("depth_smoothing", 0.0),
            "yolo_latency_ms": timings_ms.get("yolo", 0.0),
            "total_latency_ms": timings_ms.get("total", 0.0),
            "risk_source": event.risk_source,
            "risk_reason": event.risk_reason,
            "closing": event.closing,
            "relative_velocity_ms": event.relative_velocity_ms,
            "velocity_confidence": event.velocity_confidence,
            "velocity_fit_error": event.velocity_fit_error,
            "ttc_s": event.ttc_s,
            "ttc_confidence": event.ttc_confidence,
            "object_disparity_px": event.object_disparity_px,
            "disparity_rate_px_s": event.disparity_rate_px_s,
            "disparity_rate_confidence": event.disparity_rate_confidence,
            "disparity_closing": event.disparity_closing,
            "disparity_confidence": event.disparity_confidence,
            "bbox_approach_rate": event.raw_approach_rate,
            "history_update_latency_ms": timings_ms.get("history_update", 0.0),
            "velocity_regression_latency_ms": timings_ms.get("velocity_regression", 0.0),
            "disparity_rate_latency_ms": timings_ms.get("disparity_rate", 0.0),
            "ttc_latency_ms": timings_ms.get("ttc", 0.0),
            "risk_fusion_latency_ms": timings_ms.get("risk_fusion", 0.0),
            "stage5_logic_latency_ms": timings_ms.get("stage5_logic", 0.0),
        })

    def close(self) -> None:
        self._file.close()
