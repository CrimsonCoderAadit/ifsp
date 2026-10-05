"""Robust radial relative velocity from repeated metric object depths."""

from dataclasses import dataclass
from typing import Iterable

from motion.stereo_track_history import StereoTrackObservation
from motion.temporal_regression import fit_temporal_samples


@dataclass(frozen=True, slots=True)
class RadialVelocityResult:
    relative_velocity_ms: float | None = None
    closing: bool | None = None
    velocity_confidence: float = 0.0
    velocity_fit_error: float | None = None


class RadialVelocityEstimator:
    def __init__(self, minimum_valid_samples: int = 4, minimum_time_span_s: float = 0.18,
                 minimum_mean_depth_confidence: float = 0.4,
                 maximum_regression_error: float = 1.5,
                 minimum_closing_speed_ms: float = 0.3,
                 maximum_absolute_speed_ms: float = 35.0) -> None:
        self.minimum_valid_samples = minimum_valid_samples
        self.minimum_time_span_s = minimum_time_span_s
        self.minimum_mean_depth_confidence = minimum_mean_depth_confidence
        self.maximum_regression_error = maximum_regression_error
        self.minimum_closing_speed_ms = minimum_closing_speed_ms
        self.maximum_absolute_speed_ms = maximum_absolute_speed_ms

    def estimate(self, history: Iterable[StereoTrackObservation]) -> RadialVelocityResult:
        samples = [(o.timestamp, o.raw_depth_m, o.depth_confidence) for o in history
                   if o.raw_depth_m is not None and o.smoothed_depth_m is not None]
        fit = fit_temporal_samples(
            samples, min_samples=self.minimum_valid_samples,
            min_time_span_s=self.minimum_time_span_s,
            min_mean_confidence=self.minimum_mean_depth_confidence,
            max_regression_error=self.maximum_regression_error,
        )
        slope = fit.slope_per_s
        if slope is None or abs(slope) > self.maximum_absolute_speed_ms:
            return RadialVelocityResult(velocity_fit_error=fit.residual_rmse)
        return RadialVelocityResult(slope, slope < -self.minimum_closing_speed_ms,
                                    fit.confidence, fit.residual_rmse)
