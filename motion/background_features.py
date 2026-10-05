"""Small OpenCV helpers for corners outside detected objects."""

from collections.abc import Sequence

import cv2
import numpy as np


def grayscale_for_motion(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width)
    if scale < 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale


def background_mask(
    shape: tuple[int, int], boxes: Sequence[tuple[float, float, float, float]],
    scale: float, padding: int,
) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    height, width = shape
    for x1, y1, x2, y2 in boxes:
        left = max(0, round(x1 * scale) - padding)
        top = max(0, round(y1 * scale) - padding)
        right = min(width - 1, round(x2 * scale) + padding)
        bottom = min(height - 1, round(y2 * scale) + padding)
        cv2.rectangle(mask, (left, top), (right, bottom), 0, -1)
    return mask


def points_outside_boxes(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    height, width = mask.shape
    x = np.rint(points[:, 0]).astype(int)
    y = np.rint(points[:, 1]).astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] = mask[y[valid_indices], x[valid_indices]] > 0
    return valid


def detect_background_corners(gray: np.ndarray, mask: np.ndarray, max_features: int, quality: float, min_distance: float) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_features, qualityLevel=quality,
        minDistance=min_distance, mask=mask, blockSize=7,
    )
    return corners.reshape(-1, 2).astype(np.float32) if corners is not None else np.empty((0, 2), np.float32)
