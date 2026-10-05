"""StereoSGBM disparity in floating-point pixel units."""

import cv2
import numpy as np


class StereoMatcher:
    def __init__(
        self, minDisparity: int = 0, numDisparities: int = 128,
        blockSize: int = 5, P1: int = 0, P2: int = 0,
        disp12MaxDiff: int = 1, uniquenessRatio: int = 10,
        speckleWindowSize: int = 100, speckleRange: int = 2,
        preFilterCap: int = 31,
    ) -> None:
        if numDisparities <= 0 or numDisparities % 16:
            raise ValueError("numDisparities must be a positive multiple of 16")
        if blockSize < 3 or blockSize % 2 == 0:
            raise ValueError("blockSize must be odd and at least 3")
        if P1 < 0 or P2 < 0:
            raise ValueError("P1/P2 cannot be negative")
        p1 = P1 or 8 * blockSize * blockSize
        p2 = P2 or 32 * blockSize * blockSize
        if p2 <= p1:
            raise ValueError("P2 must exceed P1")
        self.min_disparity = minDisparity
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=minDisparity, numDisparities=numDisparities,
            blockSize=blockSize, P1=p1, P2=p2,
            disp12MaxDiff=disp12MaxDiff, uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize, speckleRange=speckleRange,
            preFilterCap=preFilterCap, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("stereo images must share dimensions")
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
        raw = self.matcher.compute(left_gray, right_gray)
        disparity = raw.astype(np.float32) / 16.0  # OpenCV's signed fixed-point scale.
        disparity[~np.isfinite(disparity) | (disparity <= max(0, self.min_disparity))] = np.nan
        return disparity


def depth_from_disparity(
    disparity: np.ndarray, calibration, *, min_depth_m: float = 0.3,
    max_depth_m: float = 80.0,
) -> np.ndarray:
    if not calibration.rectified:
        raise ValueError("depth requires rectified stereo calibration")
    if min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError("invalid metric depth bounds")
    denominator = disparity.astype(np.float64) - calibration.disparity_offset_px
    valid = np.isfinite(disparity) & (disparity > 0) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    computed = calibration.focal_px * calibration.baseline_m / denominator[valid]
    keep = np.isfinite(computed) & (computed >= min_depth_m) & (computed <= max_depth_m)
    candidate = np.full(computed.shape, np.nan, dtype=np.float32)
    candidate[keep] = computed[keep]
    depth[valid] = candidate
    return depth


def disparity_visualization(disparity: np.ndarray) -> np.ndarray:
    valid = disparity[np.isfinite(disparity) & (disparity > 0)]
    if valid.size == 0:
        return np.zeros((*disparity.shape, 3), np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    gray = np.clip((np.nan_to_num(disparity, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~np.isfinite(disparity)] = 0
    return colored


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    near = np.clip((hi - np.nan_to_num(depth, nan=hi)) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(near, cv2.COLORMAP_TURBO)
    colored[~np.isfinite(depth)] = 0
    return colored
