"""Structured observations emitted by the perception stage."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_center: tuple[float, float]
    bbox_bottom_center: tuple[float, float]
    bbox_width: float
    bbox_height: float
    bbox_area: float


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class RiskEvent:
    track_id: int
    object_class: str
    confidence: float
    direction: str
    approaching: bool
    approach_rate: float
    approach_confidence: float
    path_relevance: float
    risk_score: float
    risk_level: RiskLevel
    timestamp: float
    risk_mode: str = "stage2"
    raw_approach_rate: float = 0.0
    raw_approach_confidence: float = 0.0
    background_expansion_rate: float = 0.0
    ego_motion_state: str = "UNKNOWN"
    ego_motion_confidence: float = 0.0
    residual_approach_rate: float = 0.0
    residual_approach_confidence: float = 0.0
    distance_m: float | None = None
    relative_velocity_ms: float | None = None
    ttc_s: float | None = None
    risk_source: str = "monocular_ego_motion"
    risk_reason: str = "EGO_MOTION_FALLBACK"
    closing: bool | None = None
    velocity_confidence: float = 0.0
    velocity_fit_error: float | None = None
    ttc_confidence: float = 0.0
    object_disparity_px: float | None = None
    disparity_rate_px_s: float | None = None
    disparity_rate_confidence: float = 0.0
    disparity_closing: bool | None = None
    disparity_confidence: float = 0.0
