"""Normalized forward walking corridor and image-relative direction."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Corridor:
    top_y: float = 0.38
    top_half_width: float = 0.09
    bottom_half_width: float = 0.38
    center_x: float = 0.5
    outside_falloff: float = 1.5

    def __post_init__(self) -> None:
        if not (0 <= self.top_y < 1 and 0 < self.top_half_width <= self.bottom_half_width <= 0.5):
            raise ValueError("invalid corridor geometry")
        if not (0 < self.center_x < 1 and self.outside_falloff > 1):
            raise ValueError("invalid corridor center or falloff")

    def half_width_at(self, y: float) -> float:
        fraction = min(1.0, max(0.0, (y - self.top_y) / (1 - self.top_y)))
        return self.top_half_width + fraction * (self.bottom_half_width - self.top_half_width)

    def point_inside_corridor(self, x: float, y: float) -> bool:
        return self.top_y <= y <= 1.0 and abs(x - self.center_x) <= self.half_width_at(y)

    def object_path_relevance(self, bottom_center: tuple[float, float], frame_width: int, frame_height: int) -> float:
        if frame_width <= 0 or frame_height <= 0:
            return 0.0
        x, y = bottom_center[0] / frame_width, bottom_center[1] / frame_height
        if y < self.top_y or y > 1.0:
            return 0.0
        lateral = abs(x - self.center_x) / (self.half_width_at(y) * self.outside_falloff)
        proximity = max(0.0, 1.0 - lateral)
        vertical = 0.6 + 0.4 * (y - self.top_y) / (1 - self.top_y)
        return min(1.0, proximity * vertical)

    def polygon(self, frame_width: int, frame_height: int) -> list[tuple[int, int]]:
        return [
            (round((self.center_x - self.top_half_width) * frame_width), round(self.top_y * frame_height)),
            (round((self.center_x + self.top_half_width) * frame_width), round(self.top_y * frame_height)),
            (round((self.center_x + self.bottom_half_width) * frame_width), frame_height - 1),
            (round((self.center_x - self.bottom_half_width) * frame_width), frame_height - 1),
        ]


def image_direction(center_x: float, frame_width: int, left_max: float = 0.4, right_min: float = 0.6) -> str:
    if not (0 <= left_max < right_min <= 1):
        raise ValueError("invalid direction thresholds")
    x = center_x / frame_width if frame_width > 0 else 0.5
    return "LEFT" if x < left_max else "RIGHT" if x > right_min else "CENTER"
