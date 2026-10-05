"""Rolling latency and end-to-end throughput measurements."""

from collections import deque
from dataclasses import dataclass
from statistics import mean

import numpy as np


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    acquisition_ms: float
    inference_ms: float
    total_ms: float
    mean_acquisition_ms: float
    mean_total_ms: float
    instantaneous_fps: float
    average_fps: float
    mean_inference_ms: float
    p95_inference_ms: float
    p95_total_ms: float


class Profiler:
    def __init__(self, window: int = 120) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self._acquisition = deque(maxlen=window)
        self._inference = deque(maxlen=window)
        self._total = deque(maxlen=window)
        self.frames_processed = 0
        self._all_total_seconds = 0.0

    def add(self, acquisition_s: float, inference_s: float, total_s: float) -> None:
        if min(acquisition_s, inference_s, total_s) < 0:
            raise ValueError("latencies cannot be negative")
        self._acquisition.append(acquisition_s)
        self._inference.append(inference_s)
        self._total.append(total_s)
        self.frames_processed += 1
        self._all_total_seconds += total_s

    def snapshot(self) -> ProfileSnapshot:
        if not self._total:
            return ProfileSnapshot(*(0.0,) * 10)
        last_total = self._total[-1]
        return ProfileSnapshot(
            acquisition_ms=self._acquisition[-1] * 1000,
            inference_ms=self._inference[-1] * 1000,
            total_ms=last_total * 1000,
            mean_acquisition_ms=mean(self._acquisition) * 1000,
            mean_total_ms=mean(self._total) * 1000,
            instantaneous_fps=1 / last_total if last_total else 0.0,
            average_fps=len(self._total) / sum(self._total) if sum(self._total) else 0.0,
            mean_inference_ms=mean(self._inference) * 1000,
            p95_inference_ms=float(np.percentile(self._inference, 95)) * 1000,
            p95_total_ms=float(np.percentile(self._total, 95)) * 1000,
        )

    @property
    def overall_fps(self) -> float:
        return self.frames_processed / self._all_total_seconds if self._all_total_seconds else 0.0


class RiskProfiler:
    """Per-frame cost of the three small Stage 2 computation steps."""

    def __init__(self) -> None:
        self.frames = 0
        self.history_seconds = 0.0
        self.approach_seconds = 0.0
        self.corridor_risk_seconds = 0.0

    def add(self, history_s: float, approach_s: float, corridor_risk_s: float) -> None:
        self.frames += 1
        self.history_seconds += history_s
        self.approach_seconds += approach_s
        self.corridor_risk_seconds += corridor_risk_s

    @property
    def mean_history_ms(self) -> float:
        return 1000 * self.history_seconds / self.frames if self.frames else 0.0

    @property
    def mean_approach_ms(self) -> float:
        return 1000 * self.approach_seconds / self.frames if self.frames else 0.0

    @property
    def mean_corridor_risk_ms(self) -> float:
        return 1000 * self.corridor_risk_seconds / self.frames if self.frames else 0.0

    @property
    def mean_total_ms(self) -> float:
        return self.mean_history_ms + self.mean_approach_ms + self.mean_corridor_risk_ms


class EgoMotionProfiler:
    """Bounded latency window with full-run means for feature motion stages."""

    def __init__(self, window: int = 120) -> None:
        self.frames = 0
        self.reliable_frames = 0
        self.feature_seconds = 0.0
        self.flow_seconds = 0.0
        self.estimation_seconds = 0.0
        self._totals = deque(maxlen=window)

    def add(self, feature_s: float, flow_s: float, estimation_s: float, reliable: bool) -> None:
        self.frames += 1
        self.reliable_frames += int(reliable)
        self.feature_seconds += feature_s
        self.flow_seconds += flow_s
        self.estimation_seconds += estimation_s
        self._totals.append(feature_s + flow_s + estimation_s)

    @property
    def mean_feature_ms(self) -> float:
        return 1000 * self.feature_seconds / self.frames if self.frames else 0.0

    @property
    def mean_flow_ms(self) -> float:
        return 1000 * self.flow_seconds / self.frames if self.frames else 0.0

    @property
    def mean_estimation_ms(self) -> float:
        return 1000 * self.estimation_seconds / self.frames if self.frames else 0.0

    @property
    def mean_total_ms(self) -> float:
        return self.mean_feature_ms + self.mean_flow_ms + self.mean_estimation_ms

    @property
    def p95_total_ms(self) -> float:
        return float(np.percentile(self._totals, 95)) * 1000 if self._totals else 0.0

    @property
    def reliable_percent(self) -> float:
        return 100 * self.reliable_frames / self.frames if self.frames else 0.0
