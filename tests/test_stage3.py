import math
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from motion.background_features import background_mask, points_outside_boxes
from motion.ego_motion import EgoMotionEstimator, EgoMotionResult, estimate_global_motion
from risk.approach import ApproachResult
from risk.corridor import Corridor
from risk.risk_engine import RiskEngine
from tests.test_stage2 import make_track


CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "config.yaml").read_text())
SHAPE = (360, 640)
CENTER = np.array([320.0, 180.0], np.float32)
GRID = np.array([(x, y) for y in range(40, 320, 35) for x in range(60, 600, 45)], np.float32)


def motion_for(scale=1.0, shift=(0.0, 0.0), rotation_deg=0.0, count=None):
    previous = GRID if count is None else GRID[:count]
    theta = math.radians(rotation_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], np.float32)
    current = ((previous - CENTER) @ rotation.T) * scale + CENTER + np.array(shift, np.float32)
    return estimate_global_motion(previous, current.astype(np.float32), 0.1, SHAPE)


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine(
            Corridor(**CONFIG["corridor"]), CONFIG["risk"], CONFIG["class_priors"],
            CONFIG["direction"], CONFIG["ego_risk"],
        )
        self.car = make_track(100, x=500, bottom_y=900)

    def score(self, approach, ego, x=500, mode="ego_motion", class_name="car"):
        return self.engine.evaluate(
            make_track(100, x=x, bottom_y=900, class_name=class_name), approach,
            1000, 1000, 1.0, mode=mode, ego_motion=ego,
        )

    def test_static_scene_has_zero_expansion(self):
        result = motion_for()
        self.assertTrue(result.reliable)
        self.assertEqual(result.state, "STABLE")
        self.assertAlmostEqual(result.background_expansion_rate, 0.0, delta=0.01)
        event = self.score(ApproachResult(False, 0, 0, 0), result)
        self.assertAlmostEqual(event.residual_approach_rate, 0.0)

    def test_static_camera_approaching_object_keeps_positive_residual(self):
        event = self.score(ApproachResult(True, 0.6, 0.9, 0), motion_for())
        self.assertGreater(event.residual_approach_rate, 0.5)
        self.assertGreater(event.risk_score, 0.6)

    def test_forward_camera_static_object_retains_raw_hazard(self):
        ego = motion_for(scale=1.04)
        self.assertEqual(ego.state, "FORWARD_EXPANSION")
        self.assertGreater(ego.background_expansion_rate, 0.2)
        raw = ego.background_expansion_rate
        event = self.score(ApproachResult(True, raw, 0.9, 0), ego)
        self.assertLess(abs(event.residual_approach_rate), 0.03)
        self.assertGreater(event.risk_score, 0.3)
        outside = self.score(ApproachResult(True, raw, 0.9, 0), ego, x=950)
        self.assertLess(outside.risk_score, event.risk_score)

    def test_forward_camera_and_approaching_object_has_residual(self):
        ego = motion_for(scale=1.02)
        approach = ApproachResult(True, 0.7, 1.0, 0)
        event = self.score(approach, ego)
        shared_motion = self.score(
            ApproachResult(True, ego.background_expansion_rate, 1.0, 0), ego,
        )
        self.assertGreater(event.residual_approach_rate, 0.3)
        self.assertGreater(event.risk_score, shared_motion.risk_score)

    def test_stable_background_matches_stage2_score(self):
        ego = motion_for()
        approach = ApproachResult(True, 0.5, 0.9, 0)
        stage2 = self.score(approach, ego, mode="stage2")
        stage3 = self.score(approach, ego)
        self.assertAlmostEqual(stage3.risk_score, stage2.risk_score, delta=0.015)

    def test_pedestrian_in_path_is_lower_than_same_stage2_observation(self):
        approach = ApproachResult(True, 0.6, 0.8, 0)
        ego = motion_for()
        stage2 = self.score(approach, ego, mode="stage2", class_name="person")
        stage3 = self.score(approach, ego, class_name="person")
        self.assertLess(stage3.risk_score, stage2.risk_score)
        self.assertEqual(stage3.risk_level.value, "MEDIUM")

    def test_pedestrian_path_gate_and_fast_exception(self):
        ego = motion_for()
        approach = ApproachResult(True, 0.8, 1.0, 0)
        center = self.score(approach, ego, class_name="person")
        outside = self.score(approach, ego, x=950, class_name="person")
        self.assertGreater(center.path_relevance, outside.path_relevance)
        self.assertGreater(center.risk_score, outside.risk_score)
        self.assertEqual(outside.risk_level.value, "LOW")
        self.assertEqual(center.risk_level.value, "HIGH")

    def test_lateral_pan_has_little_expansion(self):
        result = motion_for(shift=(12, 0))
        self.assertEqual(result.state, "LATERAL")
        self.assertAlmostEqual(result.background_expansion_rate, 0.0, delta=0.02)

    def test_rotation_is_identified_and_less_confident(self):
        stable = motion_for()
        rotating = motion_for(rotation_deg=4)
        self.assertEqual(rotating.state, "ROTATING")
        self.assertLess(rotating.motion_confidence, stable.motion_confidence)
        self.assertAlmostEqual(rotating.background_expansion_rate, 0.0, delta=0.02)

    def test_too_few_features_falls_back_to_stage2(self):
        ego = motion_for(count=8)
        self.assertFalse(ego.reliable)
        approach = ApproachResult(True, 0.7, 0.9, 0)
        stage2 = self.score(approach, ego, mode="stage2")
        stage3 = self.score(approach, ego)
        self.assertEqual(stage3.risk_score, stage2.risk_score)
        self.assertEqual(stage3.risk_level, stage2.risk_level)
        self.assertEqual(stage3.residual_approach_confidence, 0.0)

    def test_background_mask_removes_object_points(self):
        mask = background_mask((1000, 1000), [self.car.bbox_xyxy], 1.0, 8)
        accepted = points_outside_boxes(np.array([[500, 850], [50, 50]], np.float32), mask)
        self.assertEqual(accepted.tolist(), [False, True])

    def test_estimator_tracks_features_and_handles_blank_frames(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        for x, y in GRID.astype(int):
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        moved = cv2.warpAffine(frame, cv2.getRotationMatrix2D((320, 180), 0, 1.02), (640, 360))
        estimator = EgoMotionEstimator(**CONFIG["ego_motion"])
        first, _ = estimator.process(frame, [], 0.0)
        second, timing = estimator.process(moved, [], 0.1)
        self.assertEqual(first.state, "UNKNOWN")
        self.assertTrue(second.reliable)
        self.assertGreater(second.background_expansion_rate, 0)
        self.assertGreater(timing.optical_flow_s, 0)
        blank = EgoMotionEstimator(**CONFIG["ego_motion"])
        blank.process(np.zeros_like(frame), [], 0.0)
        result, _ = blank.process(np.zeros_like(frame), [], 0.1)
        self.assertFalse(result.reliable)


if __name__ == "__main__":
    unittest.main()
