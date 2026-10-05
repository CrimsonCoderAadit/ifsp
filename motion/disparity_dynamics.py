"""Pixel-disparity trend; deliberately no metric velocity or TTC conversion."""

from dataclasses import dataclass
from typing import Iterable

from motion.stereo_track_history import StereoTrackObservation
from motion.temporal_regression import fit_temporal_samples


@dataclass(frozen=True, slots=True)
class DisparityDynamicsResult:
    disparity_rate_px_s: float | None = None
    disparity_closing: bool | None = None
    disparity_rate_confidence: float = 0.0
    disparity_fit_error_px: float | None = None


class DisparityDynamicsEstimator:
    def __init__(self, minimum_valid_samples: int = 4, minimum_time_span_s: float = 0.18,
                 minimum_mean_disparity_confidence: float = 0.4,
                 maximum_regression_error_px: float = 3.0,
                 minimum_closing_rate_px_s: float = 0.5) -> None:
        self.minimum_valid_samples = minimum_valid_samples
        self.minimum_time_span_s = minimum_time_span_s
        self.minimum_mean_disparity_confidence = minimum_mean_disparity_confidence
        self.maximum_regression_error_px = maximum_regression_error_px
        self.minimum_closing_rate_px_s = minimum_closing_rate_px_s

    def estimate(self, history: Iterable[StereoTrackObservation],
                 disparity_increases_when_closer: bool | None) -> DisparityDynamicsResult:
        samples = [(o.timestamp, o.raw_disparity_px, o.disparity_confidence) for o in history
                   if o.raw_disparity_px is not None]
        fit = fit_temporal_samples(
            samples, min_samples=self.minimum_valid_samples,
            min_time_span_s=self.minimum_time_span_s,
            min_mean_confidence=self.minimum_mean_disparity_confidence,
            max_regression_error=self.maximum_regression_error_px,
        )
        slope = fit.slope_per_s
        closing = None
        if slope is not None and disparity_increases_when_closer is not None:
            signed = slope if disparity_increases_when_closer else -slope
            closing = signed > self.minimum_closing_rate_px_s
        return DisparityDynamicsResult(slope, closing, fit.confidence, fit.residual_rmse)
