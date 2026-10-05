import unittest
from pathlib import Path

import cv2
import numpy as np

from input.widedepth_source import WideDepthSource
from stereo.depth_backend import WideDepthBackend
from stereo.object_disparity import ObjectDisparityExtractor
from stereo.overlap import StereoOverlapRegion
from stereo.sgbm import StereoMatcher
from stereo.widedepth_match import compare_sgbm


ROOT = Path(__file__).resolve().parents[1]


class WideDepthTests(unittest.TestCase):
    def test_official_loader_sample_contract(self):
        data = ROOT / "data" / "widedepth_sample"
        code = ROOT / "third_party" / "WideDepth"
        if not data.exists() or not code.exists():
            self.skipTest("small official WideDepth sample not installed")
        source = WideDepthSource(data, code)
        self.assertEqual(len(source), 20)
        self.assertEqual(source.unmatched_files, {})
        frame = source.read()
        self.assertEqual(frame.sample_id, "00000")
        self.assertEqual(frame.left_rgb.shape, (2048, 4096, 3))
        self.assertEqual(frame.left_rgb.dtype, np.uint8)
        self.assertEqual(frame.gt_disparity_px.dtype, np.float32)
        self.assertEqual(frame.valid_mask.dtype, np.bool_)
        WideDepthBackend.validate_released(frame.left_rgb, frame.right_rgb,
                                           frame.gt_disparity_px, frame.valid_mask)
        self.assertIsNone(frame.timestamp)
        self.assertIsNone(WideDepthBackend().metric_depth_from_disparity(frame.gt_disparity_px))

    def test_overlap_labels_and_mask(self):
        region = StereoOverlapRegion(0.25, 0.75)
        mask = region.mask(10, 100)
        self.assertEqual(int(mask.sum()), 500)
        self.assertEqual(region.zone(5, 100), "LEFT_ONLY")
        self.assertEqual(region.zone(50, 100), "STEREO_OVERLAP")
        self.assertEqual(region.zone(95, 100), "RIGHT_ONLY")

    def test_sparse_object_disparity_uses_mask_and_robust_median(self):
        disparity = np.zeros((100, 100), np.float32)
        mask = np.zeros((100, 100), bool)
        for y, x in ((45, 40), (50, 45), (55, 50), (60, 55), (65, 60), (70, 50)):
            disparity[y, x] = 12
            mask[y, x] = True
        disparity[48, 48] = 200
        mask[48, 48] = True
        extractor = ObjectDisparityExtractor(min_valid_pixels=5)
        result = extractor.extract((20, 20, 80, 80), disparity, mask)
        self.assertAlmostEqual(result.disparity_px, 12)
        self.assertGreater(result.valid_ratio, 0)
        self.assertIsNone(extractor.extract((120, 20, 140, 80), disparity, mask).disparity_px)

    def test_vertical_pair_is_rotated_before_sgbm(self):
        rng = np.random.default_rng(42)
        left = rng.integers(0, 255, (128, 64, 3), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:-8] = left[8:]
        gt = np.full((128, 64), 8, np.float32)
        valid = np.ones((128, 64), bool)
        result = compare_sgbm(left, right, gt, valid,
                              StereoMatcher(numDisparities=16, blockSize=5), 1.0)
        self.assertGreater(result.evaluated_pixels, 1000)
        self.assertLess(result.median_absolute_error_native_px, 1.5)


if __name__ == "__main__":
    unittest.main()
