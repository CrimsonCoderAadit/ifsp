import unittest
from pathlib import Path

import yaml

from common.events import RiskLevel, Track
from perception.track_history import TrackHistoryStore
from risk.approach import ApproachEstimator, ApproachResult
from risk.corridor import Corridor
from risk.risk_engine import RiskEngine


CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "config.yaml").read_text())


def make_track(size: float, x: float = 500, bottom_y: float = 900, class_name: str = "car", confidence: float = 0.9) -> Track:
    x1, y1 = x - size / 2, bottom_y - size
    x2, y2 = x + size / 2, bottom_y
    return Track(
        track_id=7, class_id=2, class_name=class_name, confidence=confidence,
        bbox_xyxy=(x1, y1, x2, y2), bbox_center=(x, (y1 + y2) / 2),
        bbox_bottom_center=(x, bottom_y), bbox_width=size, bbox_height=size, bbox_area=size * size,
    )


class Stage2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.corridor = Corridor(**CONFIG["corridor"])
        self.engine = RiskEngine(self.corridor, CONFIG["risk"], CONFIG["class_priors"], CONFIG["direction"])
        self.estimator = ApproachEstimator(**CONFIG["approach"])

    def estimate_sequence(self, sizes: list[float]) -> list[ApproachResult]:
        store = TrackHistoryStore(**CONFIG["history"])
        results = []
        for index, size in enumerate(sizes):
            history, _ = store.update(make_track(size), index * 0.1)
            results.append(self.estimator.estimate(history, 1000))
        return results

    def test_constant_box_is_not_approaching(self):
        results = self.estimate_sequence([100] * 12)
        self.assertFalse(any(result.approaching for result in results))
        self.assertAlmostEqual(results[-1].approach_rate, 0.0, places=6)

    def test_slow_growth_is_positive(self):
        result = self.estimate_sequence([100 * 1.01 ** index for index in range(12)])[-1]
        self.assertTrue(result.approaching)
        self.assertGreater(result.approach_rate, 0.08)
        self.assertLess(result.approach_rate, 0.4)

    def test_fast_growth_is_stronger(self):
        slow = self.estimate_sequence([100 * 1.01 ** index for index in range(12)])[-1]
        fast = self.estimate_sequence([100 * 1.12 ** index for index in range(12)])[-1]
        self.assertTrue(fast.approaching)
        self.assertGreater(fast.approach_rate, slow.approach_rate * 3)
        self.assertGreater(fast.approach_confidence, slow.approach_confidence)

    def test_decreasing_box_is_receding(self):
        result = self.estimate_sequence([100 * 0.96 ** index for index in range(12)])[-1]
        self.assertFalse(result.approaching)
        self.assertLess(result.approach_rate, 0)

    def test_noisy_constant_box_does_not_false_trigger(self):
        sizes = [100, 102, 98, 101, 99, 100, 103, 97, 101, 99, 100, 102]
        results = self.estimate_sequence(sizes)
        self.assertFalse(any(result.approaching and result.approach_confidence >= 0.5 for result in results))
        self.assertLess(abs(results[-1].approach_rate), 0.2)

    def test_one_large_frame_does_not_count_as_approach(self):
        result = self.estimate_sequence([100] * 5 + [140])[-1]
        self.assertFalse(result.approaching)

    def test_corridor_center_relevance_exceeds_outside(self):
        inside = self.corridor.object_path_relevance((500, 900), 1000, 1000)
        outside = self.corridor.object_path_relevance((950, 900), 1000, 1000)
        self.assertTrue(self.corridor.point_inside_corridor(0.5, 0.9))
        self.assertFalse(self.corridor.point_inside_corridor(0.95, 0.9))
        self.assertGreater(inside, outside)

    def test_approaching_vehicle_in_path_scores_higher(self):
        approach = ApproachResult(True, 1.2, 0.95, 0)
        inside = self.engine.evaluate(make_track(100, x=500), approach, 1000, 1000, 1.0)
        outside = self.engine.evaluate(make_track(100, x=950), approach, 1000, 1000, 1.0)
        self.assertGreater(inside.risk_score, outside.risk_score)
        self.assertIn(inside.risk_level, (RiskLevel.HIGH, RiskLevel.CRITICAL))
        self.assertIn(outside.risk_level, (RiskLevel.MEDIUM, RiskLevel.HIGH))
        self.assertIsNone(inside.distance_m)
        self.assertIsNone(inside.ttc_s)

    def test_rapid_motorcycle_inside_can_be_critical(self):
        event = self.engine.evaluate(
            make_track(100, x=500, bottom_y=980, class_name="motorcycle"),
            ApproachResult(True, 1.5, 1.0, 0), 1000, 1000, 1.0,
        )
        self.assertEqual(event.risk_level, RiskLevel.CRITICAL)

    def test_stationary_vehicle_stays_below_approaching(self):
        stationary = self.engine.evaluate(make_track(100), ApproachResult(False, 0, 0, 0), 1000, 1000, 1.0)
        approaching = self.engine.evaluate(make_track(100), ApproachResult(True, 1.2, 0.95, 0), 1000, 1000, 1.0)
        self.assertLess(stationary.risk_score, approaching.risk_score)
        self.assertNotIn(stationary.risk_level, (RiskLevel.HIGH, RiskLevel.CRITICAL))

    def test_parked_vehicle_outside_is_low(self):
        event = self.engine.evaluate(make_track(100, x=950, class_name="bus"), ApproachResult(False, 0, 0, 0), 1000, 1000, 1.0)
        self.assertEqual(event.risk_level, RiskLevel.LOW)

    def test_low_confidence_suppresses_fast_growth(self):
        event = self.engine.evaluate(make_track(100, confidence=0.4), ApproachResult(True, 1.2, 1.0, 0), 1000, 1000, 1.0)
        self.assertEqual(event.risk_level, RiskLevel.LOW)

    def test_history_is_bounded_and_stale_ids_expire(self):
        store = TrackHistoryStore(max_observations=3, stale_timeout_s=0.5, max_tracks=2)
        for index in range(5):
            history, _ = store.update(make_track(100), index * 0.1)
        self.assertEqual(len(history.observations), 3)
        store.prune(1.0)
        self.assertEqual(len(store.histories), 0)


if __name__ == "__main__":
    unittest.main()
