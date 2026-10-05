"""Likely route-board regions in a detected bus, in priority order."""
from __future__ import annotations

import cv2
import numpy as np


def candidates(frame, bbox, *, include_frame=False, panel_boxes=()):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    for index, box in enumerate(panel_boxes):
        a, b, c, d = map(int, box)
        if c > a and d > b:
            yield f"panel_{index}", frame[b:d, a:c], (a, b, c, d)
    w, h = x2 - x1, y2 - y1
    regions = (
        # Small front destination signs are easily lost when OCR sees the
        # whole windshield. Keep this focused window ahead of broad crops.
        ("route_board", (x1 + round(.17 * w), y1 + round(.10 * h),
                         x1 + round(.65 * w), y1 + round(.24 * h))),
        ("top_front", (x1, y1, x2, y1 + round(.30 * h))),
        ("windshield", (x1 + round(.10 * w), y1 + round(.15 * h), x2 - round(.10 * w), y1 + round(.45 * h))),
        ("side_front", (x1, y1 + round(.24 * h), x1 + round(.58 * w), y1 + round(.62 * h))),
        ("bus", (x1, y1, x2, y2)),
    )
    for name, (a, b, c, d) in regions:
        if c > a and d > b:
            yield name, frame[b:d, a:c], (a, b, c, d)
    if include_frame:
        yield "image", frame, (0, 0, width, height)


def variants(image):
    if image.size == 0:
        return
    scale = max(1, min(6, round(32 / max(1, image.shape[0] / 8))))
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if scaled.ndim == 3 else scaled
    yield "original", scaled, scale
    enhanced = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    yield "clahe", enhanced, scale
    lines = cv2.HoughLinesP(cv2.Canny(enhanced, 60, 150), 1, np.pi / 180, 55,
                            minLineLength=max(20, gray.shape[1] // 4), maxLineGap=12)
    if lines is not None:
        angles = [np.degrees(np.arctan2(y2-y1, x2-x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
        angles = [angle for angle in angles if 5 <= abs(angle) <= 30]
        if angles:
            angle = float(np.median(angles))
            if 5 <= abs(angle) <= 30:
                matrix = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1)
                yield "deskew", cv2.warpAffine(enhanced, matrix, (gray.shape[1], gray.shape[0])), scale
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "inverted", cv2.bitwise_not(binary), scale
