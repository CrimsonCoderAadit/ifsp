"""Robust background flow and non-metric global expansion."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, degrees, hypot
from time import perf_counter

import cv2
import numpy as np

from motion.background_features import (
    background_mask, detect_background_corners, grayscale_for_motion,
    points_outside_boxes,
)


@dataclass(frozen=True, slots=True)
class EgoMotionResult:
    translation_x: float = 0.0  # pixels/frame at analysis resolution
    translation_y: float = 0.0
    scale_change: float = 0.0  # affine image scale minus one, per frame
    rotation_deg: float = 0.0  # image-plane degrees/frame
    inlier_ratio: float = 0.0
    motion_confidence: float = 0.0
    background_expansion_rate: float = 0.0  # log radial change per second
    state: str = "UNKNOWN"
    feature_count: int = 0
    inlier_count: int = 0
    analysis_width: int = 0
    flow_vectors: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def reliable(self) -> bool:
        return self.state not in ("UNKNOWN", "UNSTABLE")


@dataclass(frozen=True, slots=True)
class EgoMotionTiming:
    feature_detection_s: float = 0.0
    optical_flow_s: float = 0.0
    global_estimation_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.feature_detection_s + self.optical_flow_s + self.global_estimation_s


def estimate_global_motion(
    previous: np.ndarray, current: np.ndarray, dt: float,
    shape: tuple[int, int], min_features: int = 25,
    ransac_threshold: float = 2.5, min_inlier_ratio: float = 0.5,
    min_confidence: float = 0.3, expansion_threshold: float = 0.08,
    lateral_threshold: float = 0.03, rotation_threshold_deg_s: float = 7.0,
) -> EgoMotionResult:
    """Fit an image-plane affine model and median background radial expansion."""
    count = len(previous)
    if dt <= 0 or count < min_features:
        return EgoMotionResult(feature_count=count, analysis_width=shape[1])
    matrix, inliers = cv2.estimateAffinePartial2D(
        previous, current, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold, maxIters=1000,
        confidence=0.99, refineIters=5,
    )
    if matrix is None or inliers is None:
        return EgoMotionResult(feature_count=count, analysis_width=shape[1], state="UNSTABLE")
    keep = inliers.ravel().astype(bool)
    inlier_count = int(keep.sum())
    ratio = inlier_count / count
    if inlier_count < min_features or ratio < min_inlier_ratio:
        return EgoMotionResult(
            inlier_ratio=ratio, feature_count=count, inlier_count=inlier_count,
            analysis_width=shape[1], state="UNSTABLE",
        )

    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = hypot(a, b)
    rotation = degrees(atan2(b, a))
    center = np.array([shape[1] / 2, shape[0] / 2], dtype=np.float64)
    moved_center = matrix[:, :2] @ center + matrix[:, 2]
    center_shift = moved_center - center

    # Subtract the affine translation/rotation of the image center. Pure pans
    # and rotations then have near-zero radial expansion; parallax may remain.
    p0, p1 = previous[keep].astype(np.float64), current[keep].astype(np.float64)
    radii_before = np.linalg.norm(p0 - center, axis=1)
    radii_after = np.linalg.norm(p1 - moved_center, axis=1)
    usable = radii_before >= 8.0
    if int(usable.sum()) < min_features:
        return EgoMotionResult(
            translation_x=float(center_shift[0]), translation_y=float(center_shift[1]),
            scale_change=scale - 1, rotation_deg=rotation, inlier_ratio=ratio,
            feature_count=count, inlier_count=inlier_count,
            analysis_width=shape[1], state="UNSTABLE",
        )
    rates = np.log((radii_after[usable] + 1) / (radii_before[usable] + 1)) / dt
    median = float(np.median(rates))
    mad = float(np.median(np.abs(rates - median)))
    retained = rates[np.abs(rates - median) <= 3 * max(mad, 0.02)]
    expansion = float(np.median(retained)) if len(retained) else median

    motion_confidence = ratio * min(1.0, inlier_count / 80)
    if abs(rotation) / dt >= rotation_threshold_deg_s:
        motion_confidence *= 0.6
        state = "ROTATING"
    elif motion_confidence < min_confidence:
        state = "UNSTABLE"
    elif expansion >= expansion_threshold:
        state = "FORWARD_EXPANSION"
    elif expansion <= -expansion_threshold:
        state = "BACKWARD_CONTRACTION"
    elif abs(center_shift[0]) / (shape[1] * dt) >= lateral_threshold:
        state = "LATERAL"
    else:
        state = "STABLE"
    if motion_confidence < min_confidence:
        state = "UNSTABLE"

    sample_indices = np.linspace(0, inlier_count - 1, min(12, inlier_count), dtype=int)
    flow_vectors = tuple(
        (float(p0[i, 0]), float(p0[i, 1]), float(p1[i, 0]), float(p1[i, 1]))
        for i in sample_indices
    )
    return EgoMotionResult(
        translation_x=float(center_shift[0]), translation_y=float(center_shift[1]),
        scale_change=scale - 1, rotation_deg=rotation, inlier_ratio=ratio,
        motion_confidence=motion_confidence, background_expansion_rate=expansion,
        state=state, feature_count=count, inlier_count=inlier_count,
        analysis_width=shape[1], flow_vectors=flow_vectors,
    )


class EgoMotionEstimator:
    """Keep background corners between frames; refresh only periodically."""

    def __init__(
        self, max_features: int = 180, min_features: int = 25,
        refresh_interval: int = 8, quality_level: float = 0.01,
        min_distance: float = 6.0, processing_width: int = 640,
        box_padding: int = 8, ransac_threshold: float = 2.5,
        min_inlier_ratio: float = 0.5, min_confidence: float = 0.3,
        expansion_threshold: float = 0.08, lateral_threshold: float = 0.03,
        rotation_threshold_deg_s: float = 7.0,
    ) -> None:
        if max_features < min_features or min_features < 3 or refresh_interval < 1 or processing_width < 32:
            raise ValueError("invalid ego-motion feature limits")
        self.max_features = max_features
        self.min_features = min_features
        self.refresh_interval = refresh_interval
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.processing_width = processing_width
        self.box_padding = box_padding
        self.ransac_threshold = ransac_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_confidence = min_confidence
        self.expansion_threshold = expansion_threshold
        self.lateral_threshold = lateral_threshold
        self.rotation_threshold_deg_s = rotation_threshold_deg_s
        self.previous_gray: np.ndarray | None = None
        self.previous_mask: np.ndarray | None = None
        self.previous_points = np.empty((0, 2), np.float32)
        self.previous_timestamp: float | None = None
        self.frame_count = 0

    def _detect(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return detect_background_corners(
            gray, mask, self.max_features, self.quality_level, self.min_distance,
        )

    def process(
        self, frame: np.ndarray, boxes: Sequence[tuple[float, float, float, float]],
        timestamp: float,
    ) -> tuple[EgoMotionResult, EgoMotionTiming]:
        feature_start = perf_counter()
        gray, scale = grayscale_for_motion(frame, self.processing_width)
        mask = background_mask(gray.shape, boxes, scale, self.box_padding)
        feature_seconds = perf_counter() - feature_start
        if self.previous_gray is None:
            feature_start = perf_counter()
            self.previous_points = self._detect(gray, mask)
            feature_seconds += perf_counter() - feature_start
            self.previous_gray, self.previous_mask = gray, mask
            self.previous_timestamp = timestamp
            self.frame_count = 1
            return EgoMotionResult(feature_count=len(self.previous_points), analysis_width=gray.shape[1]), EgoMotionTiming(feature_detection_s=feature_seconds)

        flow_start = perf_counter()
        previous = self.previous_points
        if len(previous):
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray, gray, previous.reshape(-1, 1, 2), None,
                winSize=(21, 21), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_points is not None and status is not None:
                current = next_points.reshape(-1, 2)
                valid = status.ravel().astype(bool)
                valid &= points_outside_boxes(previous, self.previous_mask)
                valid &= points_outside_boxes(current, mask)
                previous, current = previous[valid], current[valid]
            else:
                previous = current = np.empty((0, 2), np.float32)
        else:
            previous = current = np.empty((0, 2), np.float32)
        flow_seconds = perf_counter() - flow_start

        estimation_start = perf_counter()
        result = estimate_global_motion(
            previous, current, timestamp - self.previous_timestamp, gray.shape,
            min_features=self.min_features, ransac_threshold=self.ransac_threshold,
            min_inlier_ratio=self.min_inlier_ratio, min_confidence=self.min_confidence,
            expansion_threshold=self.expansion_threshold,
            lateral_threshold=self.lateral_threshold,
            rotation_threshold_deg_s=self.rotation_threshold_deg_s,
        )
        estimation_seconds = perf_counter() - estimation_start

        self.frame_count += 1
        feature_start = perf_counter()
        if self.frame_count % self.refresh_interval == 0 or len(current) < self.min_features:
            self.previous_points = self._detect(gray, mask)
        else:
            self.previous_points = current.astype(np.float32)
        feature_seconds += perf_counter() - feature_start
        self.previous_gray, self.previous_mask = gray, mask
        self.previous_timestamp = timestamp
        return result, EgoMotionTiming(feature_seconds, flow_seconds, estimation_seconds)
