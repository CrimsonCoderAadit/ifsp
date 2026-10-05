"""Deterministic checks for temporal metric and pixel-only evidence."""

from dataclasses import replace
from pathlib import Path
import unittest

import yaml

from common.events import RiskEvent, RiskLevel, Track
from motion.disparity_dynamics import DisparityDynamicsEstimator, DisparityDynamicsResult
from motion.radial_velocity import RadialVelocityEstimator, RadialVelocityResult
from motion.stereo_track_history import StereoTrackHistoryStore, StereoTrackObservation
from risk.corridor import Corridor
from risk.risk_engine import RiskEngine
from risk.stability import RiskStabilizer
from risk.ttc import TTCEstimator, TTCResult
from stereo.depth_backend import StereoCapabilities
from stereo.object_depth import DepthObservation
from stereo.object_disparity import ObjectDisparity
from stereo.overlap import StereoOverlapRegion
from stereo.stage5_logic import Stage5Processor
from stereo.runner import _is_robotcar_hood


CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/config.yaml").read_text())


def observation(index: int, depth: float | None = None, disparity: float | None = None,
                track_id: int = 1) -> StereoTrackObservation:
    return StereoTrackObservation(
        float(index), index, track_id, "car", depth, depth,
        0.9 if depth is not None else 0.0,
        disparity, disparity, 0.9 if disparity is not None else 0.0,
        0.8, (40.0, 30.0, 60.0, 80.0), 0.9,
    )


def car() -> Track:
    return Track(1, 2, "car", 0.9, (40, 30, 60, 80), (50, 55), (50, 80), 20, 50, 1000)


def base_event(approaching: bool = True, rate: float = 0.8, score: float = 0.5,
               path: float = 0.95) -> RiskEvent:
    return RiskEvent(1, "car", 0.9, "CENTER", approaching, rate, 0.9,
                     path, score, RiskLevel.MEDIUM, 3.0, risk_mode="ego_motion",
                     raw_approach_rate=rate)


class TestTemporalStereo(unittest.TestCase):
    def setUp(self) -> None:
        self.velocity = RadialVelocityEstimator()
        self.disparity = DisparityDynamicsEstimator()
        self.ttc = TTCEstimator()

    def test_metric_static_closing_fast_receding(self) -> None:
        for depths, expected, closing in (
            ([10, 10, 10, 10], 0.0, False),
            ([10, 9, 8, 7], -1.0, True),
            ([10, 8, 6, 4], -2.0, True),
            ([4, 5, 6, 7], 1.0, False),
        ):
            with self.subTest(depths=depths):
                result = self.velocity.estimate([observation(i, depth=z) for i, z in enumerate(depths)])
                self.assertAlmostEqual(result.relative_velocity_ms, expected, places=5)
                self.assertEqual(result.closing, closing)
                result_ttc = self.ttc.estimate(depths[-1], result.relative_velocity_ms,
                                               result.closing, 0.9, result.velocity_confidence)
                self.assertAlmostEqual(result_ttc.ttc_s, depths[-1] / -expected, places=4) if closing else self.assertIsNone(result_ttc.ttc_s)

    def test_metric_outlier_and_insufficient_history(self) -> None:
        result = self.velocity.estimate([observation(i, depth=z) for i, z in enumerate([10, 9, 25, 8, 7])])
        self.assertIsNotNone(result.relative_velocity_ms)
        self.assertLess(abs(result.relative_velocity_ms), 2)
        self.assertIsNone(self.velocity.estimate([observation(i, depth=z) for i, z in enumerate([10, 9, 8])]).relative_velocity_ms)

    def test_disparity_trends_and_unknown_sign(self) -> None:
        for values, expected, closing in (
            ([10, 12, 14, 16], 2.0, True),
            ([16, 14, 12, 10], -2.0, False),
            ([10, 10.2, 9.9, 10.1], 0.0, False),
        ):
            result = self.disparity.estimate([observation(i, disparity=d) for i, d in enumerate(values)], True)
            self.assertAlmostEqual(result.disparity_rate_px_s, expected, delta=0.2)
            self.assertEqual(result.disparity_closing, closing)
        unknown = self.disparity.estimate([observation(i, disparity=d) for i, d in enumerate([10, 12, 14, 16])], None)
        self.assertGreater(unknown.disparity_rate_px_s, 0)
        self.assertIsNone(unknown.disparity_closing)

    def test_history_bounds_pruning_and_class_reset(self) -> None:
        store = StereoTrackHistoryStore(max_observations=3, stale_timeout_s=2, max_tracks=2)
        for i in range(4):
            store.update(observation(i, depth=10 - i))
        self.assertEqual(len(store.histories[1]), 3)
        store.update(replace(observation(5, depth=5), class_name="person"))
        self.assertEqual(len(store.histories[1]), 1)
        store.prune(8)
        self.assertNotIn(1, store.histories)


class TestFusion(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine(Corridor(**CONFIG["corridor"]), CONFIG["risk"],
                                 CONFIG["class_priors"], CONFIG["direction"], CONFIG["ego_risk"])
        self.fusion = CONFIG["stereo_dynamic"]["fusion"]
        self.depth = DepthObservation(4, 4, 0.9, 0.8)
        self.disp = ObjectDisparity(20, 0.8, 0.5, 0.9)

    def fuse(self, base: RiskEvent, velocity: RadialVelocityResult = RadialVelocityResult(),
             ttc: TTCResult = TTCResult(), in_overlap: bool = True,
             depth: DepthObservation | None = None) -> RiskEvent:
        return self.engine.evaluate_stereo_dynamic(base, car(), self.depth if depth is None else depth,
                                                    self.disp, velocity, DisparityDynamicsResult(), ttc,
                                                    in_overlap, self.fusion)

    def test_agreement_disagreement_and_path(self) -> None:
        closing = RadialVelocityResult(-4, True, 0.9, 0.1)
        high = self.fuse(base_event(), closing, TTCResult(1.0, 0.9))
        self.assertEqual(high.risk_source, "metric_stereo")
        self.assertIn("MULTI_SIGNAL_AGREEMENT", high.risk_reason)
        self.assertIn(high.risk_level, (RiskLevel.HIGH, RiskLevel.CRITICAL))
        lateral = self.fuse(base_event(path=0.05), closing, TTCResult(1.0, 0.9))
        self.assertLess(lateral.risk_score, high.risk_score)
        stable = self.fuse(base_event(), RadialVelocityResult(0, False, 0.9, 0.1))
        self.assertIn("SIGNAL_DISAGREEMENT", stable.risk_reason)
        self.assertLess(stable.risk_score, high.risk_score)

    def test_fallback_and_outside_overlap(self) -> None:
        fallback = self.fuse(base_event(), depth=DepthObservation())
        self.assertEqual(fallback.risk_source, "monocular_ego_motion")
        self.assertIsNone(fallback.ttc_s)
        outside = self.fuse(base_event(), RadialVelocityResult(-4, True, 0.9, 0.1),
                            TTCResult(1, 0.9), in_overlap=False)
        self.assertEqual(outside.risk_source, "monocular_ego_motion")
        self.assertIsNone(outside.distance_m)

    def test_widedepth_capability_blocks_temporal_rates(self) -> None:
        caps = StereoCapabilities(False, False, None, "vertical_equirectangular")
        processor = Stage5Processor(self.engine, caps, StereoOverlapRegion(), CONFIG["stereo_dynamic"])
        result = None
        for index in range(5):
            result, _ = processor.process(car(), replace(base_event(), timestamp=float(index)),
                                          DepthObservation(), ObjectDisparity(10 + index, 0.8, 0.5, 0.9),
                                          float(index), index, 100)
        self.assertIsNone(result.disparity_rate_px_s)
        self.assertIsNone(result.ttc_s)
        self.assertEqual(result.risk_source, "monocular_ego_motion")

    def test_temporal_disparity_only_has_no_metric_claim(self) -> None:
        caps = StereoCapabilities(False, True, True, "test_projection")
        processor = Stage5Processor(self.engine, caps, StereoOverlapRegion(), CONFIG["stereo_dynamic"])
        result = None
        for index in range(5):
            result, _ = processor.process(car(), replace(base_event(), timestamp=float(index)),
                                          DepthObservation(), ObjectDisparity(10 + 2 * index, 0.8, 0.5, 0.9),
                                          float(index), index, 100)
        self.assertEqual(result.risk_source, "disparity_stereo")
        self.assertGreater(result.disparity_rate_px_s, 0)
        self.assertTrue(result.disparity_closing)
        self.assertIsNone(result.distance_m)
        self.assertIsNone(result.relative_velocity_ms)
        self.assertIsNone(result.ttc_s)

    def test_missing_current_stereo_does_not_reuse_old_velocity(self) -> None:
        caps = StereoCapabilities(True, True, True, "rectified_horizontal_pinhole")
        processor = Stage5Processor(self.engine, caps, StereoOverlapRegion(), CONFIG["stereo_dynamic"])
        for index in range(4):
            processor.process(car(), replace(base_event(), timestamp=float(index)),
                              DepthObservation(10 - index, 10 - index, 0.9, 0.8),
                              ObjectDisparity(10 + index, 0.8, 0.5, 0.9), float(index), index, 100)
        event, _ = processor.process(car(), replace(base_event(), timestamp=4.0),
                                     DepthObservation(), ObjectDisparity(), 4.0, 4, 100)
        self.assertIsNone(event.distance_m)
        self.assertIsNone(event.relative_velocity_ms)
        self.assertIsNone(event.ttc_s)
        self.assertEqual(event.risk_source, "monocular_ego_motion")

    def test_hysteresis_holds_one_noisy_drop(self) -> None:
        stabilizer = RiskStabilizer(deescalation_persistence=3, cooldown_s=0)
        high = replace(base_event(), risk_level=RiskLevel.HIGH, risk_score=0.7)
        low = replace(high, timestamp=3.1, risk_level=RiskLevel.LOW, risk_score=0.1)
        self.assertEqual(stabilizer.apply(high, CONFIG["risk"]).risk_level, RiskLevel.HIGH)
        held = stabilizer.apply(low, CONFIG["risk"])
        self.assertEqual(held.risk_level, RiskLevel.HIGH)
        self.assertIn("HYSTERESIS_HOLD", held.risk_reason)
        faster = RiskStabilizer(escalation_persistence=1, cooldown_s=1.0)
        low_first = replace(low, timestamp=0.0)
        high_next = replace(high, timestamp=0.1)
        faster.apply(low_first, CONFIG["risk"])
        self.assertEqual(faster.apply(high_next, CONFIG["risk"]).risk_level, RiskLevel.HIGH)

    def test_hood_mask_only_for_configured_robotcar(self) -> None:
        class Stub:
            pass
        source, calibration = Stub(), Stub()
        source.raw_bayer = True
        calibration.robotcar_models_dir = Path("models")
        hood = replace(car(), bbox_xyxy=(0, 800, 900, 950))
        settings = CONFIG["stereo"]
        self.assertTrue(_is_robotcar_hood(hood, 1000, 1000, source, calibration, settings))
        source.raw_bayer = False
        self.assertFalse(_is_robotcar_hood(hood, 1000, 1000, source, calibration, settings))


if __name__ == "__main__":
    unittest.main()
