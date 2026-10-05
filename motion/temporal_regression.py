"""Small robust linear fit shared by metric depth and pixel disparity."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalFit:
    slope_per_s: float | None = None
    confidence: float = 0.0
    residual_rmse: float | None = None
    samples_used: int = 0


def fit_temporal_samples(
    samples: list[tuple[float, float, float]], *, min_samples: int,
    min_time_span_s: float, min_mean_confidence: float,
    max_regression_error: float, outlier_mad_scale: float = 3.5,
    outlier_floor: float = 1.0,
) -> TemporalFit:
    valid = [(t, value, confidence) for t, value, confidence in samples
             if all(math.isfinite(v) for v in (t, value, confidence)) and value > 0 and confidence > 0]
    if len(valid) < min_samples:
        return TemporalFit()
    times, values, confidences = np.asarray(valid, dtype=np.float64).T
    if times[-1] - times[0] < min_time_span_s or np.any(np.diff(times) <= 0):
        return TemporalFit()
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    keep = np.abs(values - median) <= max(outlier_floor, outlier_mad_scale * 1.4826 * mad)
    times, values, confidences = times[keep], values[keep], confidences[keep]
    if (len(times) < min_samples or times[-1] - times[0] < min_time_span_s
            or float(np.mean(confidences)) < min_mean_confidence):
        return TemporalFit()
    centered = times - times[0]
    slope, intercept = np.polyfit(centered, values, 1)
    rmse = float(np.sqrt(np.mean((values - (slope * centered + intercept)) ** 2)))
    if not math.isfinite(slope) or rmse > max_regression_error:
        return TemporalFit(residual_rmse=rmse, samples_used=len(times))
    confidence = (float(np.mean(confidences)) * min(1.0, len(times) / (min_samples + 1))
                  * min(1.0, (times[-1] - times[0]) / max(1.5 * min_time_span_s, 1e-6))
                  / (1.0 + rmse / max_regression_error))
    return TemporalFit(float(slope), min(1.0, confidence), rmse, len(times))
