import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from input.stereo_source import StereoSource
from stereo.calibration import load_calibration
from stereo.object_depth import DepthHistoryStore, DepthObservation, ObjectDepthExtractor
from stereo.rectification import StereoRectifier
from stereo.sgbm import StereoMatcher, depth_from_disparity
from tests.test_stage2 import make_track


class Stage4Tests(unittest.TestCase):
    def test_stereo_source_rejects_missing_frame_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "left").mkdir()
            (root / "right").mkdir()
            image = np.zeros((12, 16, 3), np.uint8)
            cv2.imwrite(str(root / "left" / "0001.png"), image)
            cv2.imwrite(str(root / "right" / "0002.png"), image)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                StereoSource(root / "left", root / "right")

    def test_robotcar_archive_pairs_exact_timestamps(self):
        archive = Path(__file__).resolve().parents[1] / "sample_small.tar"
        if not archive.exists():
            self.skipTest("RobotCar sample archive not installed")
        source = StereoSource.from_robotcar_tar(archive)
        try:
            self.assertEqual(len(source), 49)
            first = source.read()
            second = source.read()
            self.assertEqual(first.frame_index, 0)
            self.assertEqual(first.left.shape, first.right.shape)
            self.assertEqual(first.left.ndim, 2)
            self.assertGreater(second.timestamp, first.timestamp)
            self.assertEqual(first.frame_id, "1418381798076682")
        finally:
            source.release()

    def test_kitti_projection_baseline_and_disparity_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calib.txt"
            P1 = np.array([[700, 0, 620, 0], [0, 700, 180, 0], [0, 0, 1, 0]])
            P2 = np.array([[700, 0, 625, -378], [0, 700, 180, 0], [0, 0, 1, 0]])
            path.write_text("P_rect_02: " + " ".join(map(str, P1.ravel())) + "\n" +
                            "P_rect_03: " + " ".join(map(str, P2.ravel())) + "\n")
            calibration = load_calibration(path)
            self.assertAlmostEqual(calibration.baseline_m, 0.54)
            self.assertAlmostEqual(calibration.focal_px, 700)
            depth = depth_from_disparity(np.array([[32.8]], np.float32), calibration)
            self.assertAlmostEqual(float(depth[0, 0]), 10.0, delta=0.1)

    def test_generic_calibration_requires_units_and_rectifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calib.npz"
            K = np.array([[120.0, 0, 80], [0, 120, 60], [0, 0, 1]])
            values = dict(K1=K, K2=K, D1=np.zeros(5), D2=np.zeros(5),
                          R=np.eye(3), T=np.array([-100.0, 0, 0]), rectified=False)
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "translation_unit"):
                load_calibration(path)
            np.savez(path, **values, translation_unit="mm")
            calibration = load_calibration(path)
            self.assertAlmostEqual(calibration.baseline_m, 0.1)
            image = np.zeros((120, 160, 3), np.uint8)
            left, right, rectified = StereoRectifier(calibration).rectify(image, image)
            self.assertEqual(left.shape, right.shape)
            self.assertTrue(rectified.rectified)
            self.assertAlmostEqual(rectified.baseline_m, 0.1)

    def test_sgbm_fixed_point_and_metric_depth(self):
        rng = np.random.default_rng(123)
        left = rng.integers(0, 256, (120, 192), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-12] = left[:, 12:]
        disparity = StereoMatcher(numDisparities=64, blockSize=5).compute(left, right)
        center = disparity[25:95, 80:170]
        self.assertAlmostEqual(float(np.nanmedian(center)), 12, delta=1)
        self.assertTrue(np.isnan(disparity).any())

    def test_object_roi_rejects_invalid_and_outside(self):
        extractor = ObjectDepthExtractor(min_valid_pixels=10)
        track = make_track(60, x=50, bottom_y=90)
        depth = np.full((100, 100), np.nan, np.float32)
        depth[54:84, 32:68] = 7.0
        depth[60, 50] = 70.0
        observation = extractor.extract(track, depth)
        self.assertAlmostEqual(observation.raw_depth_m, 7.0)
        self.assertGreater(observation.valid_depth_ratio, 0.5)
        self.assertGreater(observation.depth_confidence, 0)
        outside = extractor.extract(make_track(50, x=200, bottom_y=200), depth)
        self.assertIsNone(outside.raw_depth_m)
        self.assertIsNone(extractor.extract(track, np.full_like(depth, np.nan)).raw_depth_m)

    def test_temporal_depth_rejects_one_jump_and_recovers(self):
        store = DepthHistoryStore()
        base = DepthObservation(10.0, depth_confidence=0.8, valid_depth_ratio=0.8)
        self.assertAlmostEqual(store.update(7, base, 0.0).smoothed_depth_m, 10.0)
        single = store.update(7, replace(base, raw_depth_m=30.0), 0.1)
        self.assertIsNone(single.smoothed_depth_m)
        accepted = store.update(7, replace(base, raw_depth_m=29.0), 0.2)
        self.assertIsNotNone(accepted.smoothed_depth_m)
        invalid = store.update(7, DepthObservation(), 0.3)
        self.assertIsNone(invalid.smoothed_depth_m)
        store.prune(3.0)
        self.assertNotIn(7, store.history)


if __name__ == "__main__":
    unittest.main()
