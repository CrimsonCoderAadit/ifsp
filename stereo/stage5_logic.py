"""Temporal stereo evidence, kept separate from image processing and IO."""

from dataclasses import dataclass
from time import perf_counter

from common.events import RiskEvent, Track
from motion.disparity_dynamics import DisparityDynamicsEstimator, DisparityDynamicsResult
from motion.radial_velocity import RadialVelocityEstimator, RadialVelocityResult
from motion.stereo_track_history import StereoTrackHistoryStore, StereoTrackObservation
from risk.risk_engine import RiskEngine
from risk.stability import RiskStabilizer
from risk.ttc import TTCEstimator, TTCResult
from stereo.depth_backend import StereoCapabilities
from stereo.object_depth import DepthObservation
from stereo.object_disparity import ObjectDisparity
from stereo.overlap import StereoOverlapRegion


@dataclass(frozen=True, slots=True)
class Stage5Timing:
    history_update_s: float = 0.0
    velocity_regression_s: float = 0.0
    disparity_rate_s: float = 0.0
    ttc_s: float = 0.0
    risk_fusion_s: float = 0.0

    @property
    def total_s(self) -> float:
        return sum((self.history_update_s, self.velocity_regression_s,
                    self.disparity_rate_s, self.ttc_s, self.risk_fusion_s))


class Stage5Processor:
    def __init__(self, risk_engine: RiskEngine, capabilities: StereoCapabilities,
                 overlap: StereoOverlapRegion, config: dict) -> None:
        self.risk_engine = risk_engine
        self.capabilities = capabilities
        self.overlap = overlap
        self.config = config
        self.histories = StereoTrackHistoryStore(**config["history"])
        self.velocity = RadialVelocityEstimator(**config["radial_velocity"])
        self.disparity = DisparityDynamicsEstimator(**config["disparity_dynamics"])
        self.ttc = TTCEstimator(**config["ttc"])
        self.stability = RiskStabilizer(**config["stability"])

    def process(self, track: Track, base: RiskEvent, depth: DepthObservation,
                disparity: ObjectDisparity, timestamp: float, frame_index: int,
                frame_width: int) -> tuple[RiskEvent, Stage5Timing]:
        in_overlap = self.overlap.zone(track.bbox_bottom_center[0], frame_width) == "STEREO_OVERLAP"
        start = perf_counter()
        smoothed_disparity = self.histories.smooth_disparity(track.track_id, disparity.disparity_px)
        observation = StereoTrackObservation(
            timestamp, frame_index, track.track_id, track.class_name,
            depth.raw_depth_m if in_overlap else None,
            depth.smoothed_depth_m if in_overlap else None,
            depth.depth_confidence if in_overlap else 0.0,
            disparity.disparity_px if in_overlap else None,
            smoothed_disparity if in_overlap else None,
            disparity.confidence if in_overlap else 0.0,
            depth.valid_depth_ratio if self.capabilities.supports_metric_depth else disparity.valid_ratio,
            track.bbox_xyxy, base.path_relevance,
        )
        history = self.histories.update(observation)
        history_s = perf_counter() - start

        start = perf_counter()
        velocity = (self.velocity.estimate(history)
                    if in_overlap and depth.smoothed_depth_m is not None
                    and self.capabilities.supports_temporal_sequence
                    and self.capabilities.supports_metric_depth else RadialVelocityResult())
        velocity_s = perf_counter() - start
        start = perf_counter()
        dynamics = (self.disparity.estimate(history, self.capabilities.disparity_increases_when_closer)
                    if in_overlap and disparity.disparity_px is not None
                    and self.capabilities.supports_temporal_sequence else DisparityDynamicsResult())
        disparity_s = perf_counter() - start
        start = perf_counter()
        ttc = (self.ttc.estimate(depth.smoothed_depth_m, velocity.relative_velocity_ms,
                                 velocity.closing, depth.depth_confidence, velocity.velocity_confidence)
               if in_overlap and self.capabilities.supports_metric_depth else TTCResult())
        ttc_s = perf_counter() - start
        start = perf_counter()
        event = self.risk_engine.evaluate_stereo_dynamic(
            base, track, depth, disparity, velocity, dynamics, ttc, in_overlap,
            self.config["fusion"],
        )
        event = self.stability.apply(event, self.risk_engine.config)
        fusion_s = perf_counter() - start
        return event, Stage5Timing(history_s, velocity_s, disparity_s, ttc_s, fusion_s)
