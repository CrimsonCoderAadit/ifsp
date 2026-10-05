"""Gated heuristic score; intentionally not a collision predictor."""

from dataclasses import replace

from common.events import RiskEvent, RiskLevel, Track
from motion.disparity_dynamics import DisparityDynamicsResult
from motion.ego_motion import EgoMotionResult
from motion.radial_velocity import RadialVelocityResult
from risk.approach import ApproachResult
from risk.corridor import Corridor, image_direction
from risk.ttc import TTCResult
from stereo.object_depth import DepthObservation
from stereo.object_disparity import ObjectDisparity


class RiskEngine:
    def __init__(
        self, corridor: Corridor, config: dict, class_priors: dict[str, float],
        direction: dict, ego_config: dict | None = None,
    ) -> None:
        self.corridor = corridor
        self.config = config
        self.class_priors = class_priors
        self.direction = direction
        self.ego_config = ego_config or {}

    def evaluate_stereo_dynamic(
        self, base: RiskEvent, track: Track, depth: DepthObservation,
        disparity: ObjectDisparity, velocity: RadialVelocityResult,
        disparity_dynamics: DisparityDynamicsResult, ttc: TTCResult,
        in_overlap: bool, config: dict,
    ) -> RiskEvent:
        """Fuse trustworthy stereo evidence; retain the Stage 3 fallback."""
        score = base.risk_score
        source = "monocular_ego_motion"
        reason = "STAGE2_FALLBACK" if base.risk_mode == "stage2" else "EGO_MOTION_FALLBACK"
        path = base.path_relevance
        prior = max(0.0, min(1.0, self.class_priors.get(track.class_name, 0.0)))
        path_factor = config["outside_path_floor"] + (1 - config["outside_path_floor"]) * path
        class_factor = 0.9 + 0.1 * prior
        detection_factor = min(1.0, track.confidence / 0.8)
        reliable_velocity = (velocity.relative_velocity_ms is not None
                             and velocity.velocity_confidence >= config["minimum_velocity_confidence"])
        reliable_disparity = (disparity.disparity_px is not None
                              and disparity_dynamics.disparity_closing is not None
                              and disparity_dynamics.disparity_rate_confidence >= config["minimum_disparity_rate_confidence"])
        if in_overlap and ttc.ttc_s is not None:
            urgency = (1.0 if ttc.ttc_s < config["critical_ttc_s"] else
                       0.82 if ttc.ttc_s < config["high_ttc_s"] else
                       0.58 if ttc.ttc_s < config["medium_ttc_s"] else 0.3)
            score = max(score * 0.45, urgency * path_factor * class_factor * detection_factor)
            source = "metric_stereo"
            reason = ("TTC_CRITICAL_PATH_HIGH" if urgency == 1.0 and path >= config["high_path_relevance"]
                      else "TTC_HIGH" if urgency >= 0.82 else "TTC_MODERATE")
        elif in_overlap and depth.smoothed_depth_m is not None and reliable_velocity:
            source = "metric_stereo"
            if velocity.closing:
                intensity = min(1.0, -velocity.relative_velocity_ms / config["fast_closing_speed_ms"])
                score = max(score * 0.6, (0.3 + 0.35 * intensity) * path_factor * class_factor * detection_factor)
                reason = "METRIC_CLOSING_FAST" if intensity >= 0.8 else "METRIC_CLOSING"
            else:
                score = min(score, config["stationary_score_cap"] * (0.5 + 0.5 * path))
                reason = "METRIC_NOT_CLOSING"
        elif in_overlap and reliable_disparity:
            source = "disparity_stereo"
            if disparity_dynamics.disparity_closing:
                intensity = min(1.0, abs(disparity_dynamics.disparity_rate_px_s)
                                / config["strong_disparity_rate_px_s"])
                score = max(score * 0.7, (0.22 + 0.4 * intensity) * path_factor * class_factor * detection_factor)
                reason = "DISPARITY_CLOSING_STRONG" if intensity >= 0.8 else "DISPARITY_CLOSING"
            else:
                score = min(score, config["stationary_score_cap"] * (0.5 + 0.5 * path))
                reason = "DISPARITY_NOT_CLOSING"
        elif not in_overlap:
            reason = "OUTSIDE_STEREO_OVERLAP"

        stereo_closing = (velocity.closing if source == "metric_stereo" else
                          disparity_dynamics.disparity_closing if source == "disparity_stereo" else None)
        disagreement = False
        if stereo_closing is not None:
            if base.approaching and not stereo_closing and base.approach_rate >= config["agreement_approach_rate"]:
                disagreement = True
            elif (not base.approaching and stereo_closing
                  and base.approach_rate <= -config["agreement_approach_rate"]):
                disagreement = True
            elif base.approaching and stereo_closing:
                reason += "+MULTI_SIGNAL_AGREEMENT"
        if disagreement:
            score *= config["disagreement_score_scale"]
            reason = "SIGNAL_DISAGREEMENT+" + reason
        if track.confidence < self.config["confidence_floor"]:
            score = 0.0
        score = max(0.0, min(1.0, score))
        level = (RiskLevel.CRITICAL if score >= self.config["critical_threshold"] else
                 RiskLevel.HIGH if score >= self.config["high_threshold"] else
                 RiskLevel.MEDIUM if score >= self.config["medium_threshold"] else RiskLevel.LOW)
        return replace(
            base, risk_mode="stereo_dynamic", risk_source=source, risk_reason=reason,
            risk_score=score, risk_level=level,
            distance_m=depth.smoothed_depth_m if in_overlap else None,
            relative_velocity_ms=velocity.relative_velocity_ms if in_overlap and depth.smoothed_depth_m is not None and reliable_velocity else None,
            closing=velocity.closing if in_overlap and depth.smoothed_depth_m is not None and reliable_velocity else None,
            velocity_confidence=velocity.velocity_confidence if in_overlap and depth.smoothed_depth_m is not None else 0.0,
            velocity_fit_error=velocity.velocity_fit_error if in_overlap and depth.smoothed_depth_m is not None else None,
            ttc_s=ttc.ttc_s if in_overlap else None,
            ttc_confidence=ttc.ttc_confidence if in_overlap else 0.0,
            object_disparity_px=disparity.disparity_px if in_overlap else None,
            disparity_rate_px_s=disparity_dynamics.disparity_rate_px_s if in_overlap else None,
            disparity_rate_confidence=disparity_dynamics.disparity_rate_confidence if in_overlap else 0.0,
            disparity_closing=disparity_dynamics.disparity_closing if in_overlap else None,
            disparity_confidence=disparity.confidence if in_overlap else 0.0,
        )

    def evaluate(
        self, track: Track, approach: ApproachResult, frame_width: int,
        frame_height: int, timestamp: float, smoothed_center_x: float | None = None,
        mode: str = "stage2", ego_motion: EgoMotionResult | None = None,
    ) -> RiskEvent:
        if mode not in ("bbox", "stage2", "ego_motion"):
            raise ValueError(f"Unknown risk mode: {mode}")
        path = self.corridor.object_path_relevance(track.bbox_bottom_center, frame_width, frame_height)
        direction = image_direction(
            smoothed_center_x if smoothed_center_x is not None else track.bbox_center[0],
            frame_width, **self.direction,
        )
        prior = max(0.0, min(1.0, self.class_priors.get(track.class_name, 0.0)))
        confidence = max(0.0, min(1.0, track.confidence))
        intensity = min(1.0, max(0.0, approach.approach_rate) / self.config["full_intensity_rate"])
        raw_motion = intensity * approach.approach_confidence if approach.approaching else 0.0
        base = (
            self.config["path_weight"] * path
            + self.config["class_weight"] * prior
            + self.config["confidence_weight"] * confidence
        )
        motion_weight = (
            self.config["approach_weight"]
            + self.config["approach_path_weight"] * path
            + self.config["approach_class_weight"] * prior
        )
        if mode == "ego_motion" and track.class_name == "person":
            # A footpoint in the corridor raises relevance, but ordinary
            # pedestrian box growth deserves less weight than vehicle growth.
            base -= self.config["path_weight"] * path * (
                1 - self.ego_config["person_path_weight_scale"]
            )
            path_gate = self.ego_config["person_path_floor"] + (
                1 - self.ego_config["person_path_floor"]
            ) * path
            motion_weight *= self.ego_config["person_motion_scale"] * path_gate
        residual_rate = approach.approach_rate
        residual_confidence = 0.0
        ego_confidence = 0.0
        if ego_motion is not None and ego_motion.reliable:
            ego_confidence = ego_motion.motion_confidence
            if ego_confidence >= self.ego_config.get("min_reliable_confidence", 0.45):
                residual_rate -= self.ego_config.get("background_alpha", 1.0) * ego_motion.background_expansion_rate
                residual_confidence = approach.approach_confidence * ego_confidence
            else:
                ego_confidence = 0.0

        if mode == "bbox":
            unscaled_score = raw_motion
        elif mode == "ego_motion" and ego_confidence > 0:
            retention = min(1.0, self.ego_config["raw_retention_floor"] + self.ego_config["raw_retention_path_weight"] * path)
            residual_intensity = min(1.0, max(0.0, residual_rate) / self.config["full_intensity_rate"])
            blended_motion = (
                raw_motion * (1 - ego_confidence * (1 - retention))
                + self.ego_config["residual_weight"] * residual_intensity * residual_confidence
            )
            unscaled_score = base + blended_motion * motion_weight
        else:
            unscaled_score = base + raw_motion * motion_weight
        score = min(1.0, unscaled_score * min(1.0, confidence / 0.8))
        if confidence < self.config["confidence_floor"]:
            score = 0.0
        if score >= self.config["critical_threshold"]:
            level = RiskLevel.CRITICAL
        elif score >= self.config["high_threshold"]:
            level = RiskLevel.HIGH
        elif score >= self.config["medium_threshold"]:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and (
            not approach.approaching
            or approach.approach_confidence < self.config["min_high_approach_confidence"]
            or confidence < self.config["min_high_detection_confidence"]
        ):
            level = RiskLevel.MEDIUM
            score = min(score, self.config["high_threshold"] - 0.001)
        return RiskEvent(
            track_id=track.track_id,
            object_class=track.class_name,
            confidence=track.confidence,
            direction=direction,
            approaching=approach.approaching,
            approach_rate=approach.approach_rate,
            approach_confidence=approach.approach_confidence,
            path_relevance=path,
            risk_score=score,
            risk_level=level,
            timestamp=timestamp,
            risk_mode=mode,
            raw_approach_rate=approach.approach_rate,
            raw_approach_confidence=approach.approach_confidence,
            background_expansion_rate=ego_motion.background_expansion_rate if ego_motion else 0.0,
            ego_motion_state=ego_motion.state if ego_motion else "UNKNOWN",
            ego_motion_confidence=ego_motion.motion_confidence if ego_motion else 0.0,
            residual_approach_rate=residual_rate,
            residual_approach_confidence=residual_confidence,
        )
