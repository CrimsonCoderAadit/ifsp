"""Keep pinhole metric depth separate from WideDepth's equirect disparity."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from stereo.calibration import StereoCalibration
from stereo.sgbm import depth_from_disparity


@dataclass(frozen=True, slots=True)
class StereoCapabilities:
    supports_metric_depth: bool
    supports_temporal_sequence: bool
    disparity_increases_when_closer: bool | None
    projection_model: str


class StereoDepthBackend(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> StereoCapabilities:
        """Declare geometry and temporal support; never infer it from arrays."""

    @abstractmethod
    def metric_depth_from_disparity(self, disparity_px: np.ndarray) -> np.ndarray | None:
        """Return metres only when the selected geometry supports conversion."""


class RobotCarPinholeBackend(StereoDepthBackend):
    """Calibrated, rectified horizontal pinhole stereo, including RobotCar LUT output."""

    def __init__(self, calibration: StereoCalibration, min_depth_m: float = 0.5,
                 max_depth_m: float = 80.0) -> None:
        self.calibration = calibration
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def metric_depth_from_disparity(self, disparity_px: np.ndarray) -> np.ndarray:
        return depth_from_disparity(
            disparity_px, self.calibration,
            min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m,
        )

    @property
    def capabilities(self) -> StereoCapabilities:
        return StereoCapabilities(True, True, True, "rectified_horizontal_pinhole")


class WideDepthBackend(StereoDepthBackend):
    """Released equirect disparity has no documented pinhole depth conversion."""

    def metric_depth_from_disparity(self, disparity_px: np.ndarray) -> None:
        return None

    @property
    def capabilities(self) -> StereoCapabilities:
        # Current selected IDs have no verified temporal cadence. Sign semantics
        # require a documented, temporally aligned equirectangular sequence.
        return StereoCapabilities(False, False, None, "vertical_equirectangular")

    @staticmethod
    def validate_released(
        left_rgb: np.ndarray, right_rgb: np.ndarray,
        disparity_px: np.ndarray, valid_mask: np.ndarray,
    ) -> None:
        if left_rgb.shape != right_rgb.shape or left_rgb.shape[:2] != disparity_px.shape:
            raise ValueError("WideDepth stereo arrays are not pixel aligned")
        if disparity_px.dtype != np.float32 or valid_mask.dtype != np.bool_ or valid_mask.shape != disparity_px.shape:
            raise ValueError("invalid WideDepth disparity or mask contract")
        if not np.isfinite(disparity_px).all() or not np.array_equal(valid_mask, disparity_px > 0):
            raise ValueError("WideDepth released mask and disparity disagree")
