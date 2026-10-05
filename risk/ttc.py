"""Development-only radial time to contact, never collision probability."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TTCResult:
    ttc_s: float | None = None
    ttc_confidence: float = 0.0


class TTCEstimator:
    def __init__(self, minimum_closing_speed_ms: float = 0.5,
                 minimum_depth_confidence: float = 0.5,
                 minimum_velocity_confidence: float = 0.4,
                 maximum_ttc_s: float = 15.0) -> None:
        self.minimum_closing_speed_ms = minimum_closing_speed_ms
        self.minimum_depth_confidence = minimum_depth_confidence
        self.minimum_velocity_confidence = minimum_velocity_confidence
        self.maximum_ttc_s = maximum_ttc_s

    def estimate(self, distance_m: float | None, relative_velocity_ms: float | None,
                 closing: bool | None, depth_confidence: float,
                 velocity_confidence: float) -> TTCResult:
        if (distance_m is None or relative_velocity_ms is None or not closing
                or not math.isfinite(distance_m) or distance_m <= 0
                or not math.isfinite(relative_velocity_ms)
                or -relative_velocity_ms < self.minimum_closing_speed_ms
                or depth_confidence < self.minimum_depth_confidence
                or velocity_confidence < self.minimum_velocity_confidence):
            return TTCResult()
        ttc = distance_m / -relative_velocity_ms
        if not math.isfinite(ttc) or ttc <= 0 or ttc > self.maximum_ttc_s:
            return TTCResult()
        return TTCResult(ttc, min(depth_confidence, velocity_confidence))
