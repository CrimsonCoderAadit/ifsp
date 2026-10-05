"""Image-space stereo overlap; outside regions keep monocular risk only."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoOverlapRegion:
    x_min: float = 0.0
    x_max: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.x_min < self.x_max <= 1:
            raise ValueError("overlap x bounds must satisfy 0 <= min < max <= 1")

    def mask(self, height: int, width: int) -> np.ndarray:
        if height < 1 or width < 1:
            raise ValueError("overlap mask dimensions must be positive")
        left = int(np.floor(self.x_min * width))
        right = int(np.ceil(self.x_max * width))
        result = np.zeros((height, width), dtype=bool)
        result[:, left:right] = True
        return result

    def zone(self, x_px: float, width: int) -> str:
        position = x_px / width
        if position < self.x_min:
            return "LEFT_ONLY"
        if position >= self.x_max:
            return "RIGHT_ONLY"
        return "STEREO_OVERLAP"
