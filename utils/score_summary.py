"""Streaming score and event counts without unbounded frame storage."""

from collections import Counter

from common.events import RiskEvent, RiskLevel


class ScoreSummary:
    def __init__(self) -> None:
        self.observations = 0
        self.score_sum = 0.0
        self.score_squared_sum = 0.0
        self.level_counts: Counter[RiskLevel] = Counter()
        self.approach_events = 0
        self.raw_high_residual_low = 0
        self.example_frames: list[int] = []

    def add(
        self, event: RiskEvent, frame_index: int,
        raw_high_threshold: float = 0.5, residual_low_threshold: float = 0.12,
        min_ego_confidence: float = 0.45,
    ) -> None:
        self.observations += 1
        self.score_sum += event.risk_score
        self.score_squared_sum += event.risk_score * event.risk_score
        self.level_counts[event.risk_level] += 1
        self.approach_events += int(event.approaching)
        if (
            event.risk_mode == "ego_motion" and event.approaching
            and event.raw_approach_rate >= raw_high_threshold
            and event.residual_approach_rate <= residual_low_threshold
            and event.ego_motion_confidence >= min_ego_confidence
        ):
            self.raw_high_residual_low += 1
            if len(self.example_frames) < 20 and frame_index not in self.example_frames:
                self.example_frames.append(frame_index)

    @property
    def mean(self) -> float:
        return self.score_sum / self.observations if self.observations else 0.0

    @property
    def population_variance(self) -> float:
        return max(0.0, self.score_squared_sum / self.observations - self.mean ** 2) if self.observations else 0.0

    def counts_text(self) -> str:
        return ", ".join(f"{level.value} {self.level_counts[level]}" for level in RiskLevel)
