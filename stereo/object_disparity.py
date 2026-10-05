"""Robust released/estimated disparity extraction inside detection boxes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ObjectDisparity:
    disparity_px: float | None = None
    valid_ratio: float = 0.0
    mad_px: float | None = None
    confidence: float = 0.0


class ObjectDisparityExtractor:
    def __init__(
        self, roi_x_min: float = 0.2, roi_x_max: float = 0.8,
        roi_y_min: float = 0.4, roi_y_max: float = 0.9,
        trim_percent: float = 5.0, min_valid_pixels: int = 5,
        min_valid_ratio: float = 0.0005,
    ) -> None:
        if not (0 <= roi_x_min < roi_x_max <= 1 and 0 <= roi_y_min < roi_y_max <= 1):
            raise ValueError("invalid object disparity ROI")
        if not (0 <= trim_percent < 50 and min_valid_pixels > 0 and 0 <= min_valid_ratio <= 1):
            raise ValueError("invalid object disparity thresholds")
        self.x0, self.x1, self.y0, self.y1 = roi_x_min, roi_x_max, roi_y_min, roi_y_max
        self.trim_percent = trim_percent
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_ratio = min_valid_ratio

    def extract(
        self, bbox_xyxy: tuple[float, float, float, float],
        disparity_px: np.ndarray, valid_mask: np.ndarray,
    ) -> ObjectDisparity:
        if disparity_px.ndim != 2 or valid_mask.shape != disparity_px.shape:
            raise ValueError("disparity and mask dimensions differ")
        height, width = disparity_px.shape
        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            return ObjectDisparity()
        bw, bh = x2 - x1, y2 - y1
        xa = max(0, int(np.floor(x1 + self.x0 * bw)))
        xb = min(width, int(np.ceil(x1 + self.x1 * bw)))
        ya = max(0, int(np.floor(y1 + self.y0 * bh)))
        yb = min(height, int(np.ceil(y1 + self.y1 * bh)))
        if xa >= xb or ya >= yb:
            return ObjectDisparity()
        patch = disparity_px[ya:yb, xa:xb]
        mask = valid_mask[ya:yb, xa:xb] & np.isfinite(patch) & (patch > 0)
        values = patch[mask]
        ratio = float(values.size / patch.size)
        if values.size < self.min_valid_pixels or ratio < self.min_valid_ratio:
            return ObjectDisparity(valid_ratio=ratio)
        if self.trim_percent and values.size > 10:
            lo, hi = np.percentile(values, [self.trim_percent, 100 - self.trim_percent])
            values = values[(values >= lo) & (values <= hi)]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        # Sparse released LiDAR projection means a small valid ratio is normal.
        density = min(1.0, ratio / 0.01)
        dispersion = 1.0 / (1.0 + mad / max(median * 0.2, 1e-3))
        return ObjectDisparity(median, ratio, mad, float(density * dispersion))
