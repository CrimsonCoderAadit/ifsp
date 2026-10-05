"""Classical SGBM experiment on WideDepth's vertical equirect stereo pair."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from stereo.sgbm import StereoMatcher


@dataclass(frozen=True, slots=True)
class DisparityComparison:
    estimated_rotated_px: np.ndarray
    gt_rotated_px: np.ndarray
    gt_valid_rotated: np.ndarray
    estimated_valid_ratio: float
    evaluated_pixels: int
    gt_pixels_after_sampling: int
    mae_native_px: float | None
    median_absolute_error_native_px: float | None
    bad_3px_percent: float | None
    sgbm_ms: float
    comparison_ms: float
    absolute_errors_native_px: np.ndarray


def compare_sgbm(
    left_bgr: np.ndarray, right_bgr: np.ndarray,
    gt_disparity_px: np.ndarray, gt_valid: np.ndarray,
    matcher: StereoMatcher, processing_scale: float = 0.25,
) -> DisparityComparison:
    if not 0 < processing_scale <= 1:
        raise ValueError("processing_scale must be in (0, 1]")
    if left_bgr.shape != right_bgr.shape or left_bgr.shape[:2] != gt_disparity_px.shape:
        raise ValueError("WideDepth stereo shapes do not match")
    native_h, native_w = gt_disparity_px.shape
    size = (max(1, round(native_w * processing_scale)), max(1, round(native_h * processing_scale)))
    if abs(size[1] / native_h - size[0] / native_w) > 1e-4:
        raise ValueError("SGBM downsampling must preserve aspect ratio")
    actual_scale = size[1] / native_h  # Vertical native disparity becomes horizontal after CCW rotation.
    left_small = cv2.resize(left_bgr, size, interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right_bgr, size, interpolation=cv2.INTER_AREA)
    left_rot = cv2.rotate(left_small, cv2.ROTATE_90_COUNTERCLOCKWISE)
    right_rot = cv2.rotate(right_small, cv2.ROTATE_90_COUNTERCLOCKWISE)
    started = perf_counter()
    estimated = matcher.compute(left_rot, right_rot)
    sgbm_ms = (perf_counter() - started) * 1000

    started = perf_counter()
    # The released disparity is sparse. Nearest-neighbour sampling preserves
    # its encoded values without inventing targets between projected points.
    gt_small = cv2.resize(gt_disparity_px, size, interpolation=cv2.INTER_NEAREST) * actual_scale
    valid_small = cv2.resize(gt_valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    gt_rot = cv2.rotate(gt_small, cv2.ROTATE_90_COUNTERCLOCKWISE)
    valid_rot = cv2.rotate(valid_small.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE) > 0
    gt_supported = valid_rot & np.isfinite(gt_rot) & (gt_rot > 0)
    evaluated = gt_supported & np.isfinite(estimated) & (estimated > 0)
    errors = np.abs(estimated[evaluated] - gt_rot[evaluated]) / actual_scale
    comparison_ms = (perf_counter() - started) * 1000
    return DisparityComparison(
        estimated, gt_rot, valid_rot,
        float(np.isfinite(estimated).mean()), int(evaluated.sum()), int(gt_supported.sum()),
        float(np.mean(errors)) if errors.size else None,
        float(np.median(errors)) if errors.size else None,
        float(100 * np.mean(errors > 3)) if errors.size else None,
        sgbm_ms, comparison_ms, errors,
    )
