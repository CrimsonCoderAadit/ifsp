"""Optional stereo rectification and epipolar inspection."""

from dataclasses import replace

import cv2
import numpy as np

from stereo.calibration import StereoCalibration, _q_from_projections


class StereoRectifier:
    def __init__(self, calibration: StereoCalibration) -> None:
        self.calibration = calibration
        self._size: tuple[int, int] | None = None
        self._maps: tuple[np.ndarray, ...] | None = None
        self._rectified_calibration: StereoCalibration | None = None

    def _robotcar_maps(self, size: tuple[int, int]) -> None:
        if self._size == size:
            return
        maps = []
        for side in ("left", "right"):
            path = self.calibration.robotcar_models_dir / f"stereo_wide_{side}_distortion_lut.bin"
            values = np.fromfile(path, dtype=np.float64)
            if values.size != 2 * size[0] * size[1]:
                raise ValueError(f"RobotCar LUT does not match image size {size}")
            x, y = values.reshape(2, size[1], size[0])
            maps.extend((x.astype(np.float32), y.astype(np.float32)))
        self._maps = tuple(maps)
        self._rectified_calibration = replace(
            self.calibration, rectified=True,
            source=self.calibration.source + " (LUT rectified)",
        )
        self._size = size

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, StereoCalibration]:
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("left/right dimensions must match before rectification")
        size = (left.shape[1], left.shape[0])
        if self.calibration.image_size is not None and self.calibration.image_size != size:
            raise ValueError(f"image size {size} conflicts with calibration {self.calibration.image_size}")
        if self.calibration.robotcar_models_dir is not None:
            if left.ndim != 2 or right.ndim != 2:
                raise ValueError("RobotCar input must be raw single-channel Bayer images")
            self._robotcar_maps(size)
            left = cv2.cvtColor(left, cv2.COLOR_BayerGB2BGR)
            right = cv2.cvtColor(right, cv2.COLOR_BayerGB2BGR)
            m1l, m2l, m1r, m2r = self._maps
            return (
                cv2.remap(left, m1l, m2l, cv2.INTER_LINEAR),
                cv2.remap(right, m1r, m2r, cv2.INTER_LINEAR),
                self._rectified_calibration,
            )
        if self.calibration.rectified:
            return left, right, self.calibration
        if self._size != size:
            R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
                self.calibration.K_left, self.calibration.D_left,
                self.calibration.K_right, self.calibration.D_right,
                size, self.calibration.R, self.calibration.T_m.reshape(3, 1),
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            )
            f = float(P1[0, 0])
            baseline = -float(P2[0, 3]) / float(P2[0, 0]) + float(P1[0, 3]) / f
            if baseline <= 0 or abs(baseline - self.calibration.baseline_m) > max(0.01, 0.05 * baseline):
                raise ValueError("rectification produced invalid metric baseline or camera order")
            map_l = cv2.initUndistortRectifyMap(
                self.calibration.K_left, self.calibration.D_left, R1, P1, size, cv2.CV_32FC1,
            )
            map_r = cv2.initUndistortRectifyMap(
                self.calibration.K_right, self.calibration.D_right, R2, P2, size, cv2.CV_32FC1,
            )
            self._maps = (*map_l, *map_r)
            self._rectified_calibration = replace(
                self.calibration, K_left=P1[:, :3], K_right=P2[:, :3],
                D_left=np.zeros(5), D_right=np.zeros(5), R=np.eye(3),
                T_m=np.array([-baseline, 0, 0]), baseline_m=baseline,
                focal_px=f, cx_left=float(P1[0, 2]), cx_right=float(P2[0, 2]),
                Q=_q_from_projections(P1, P2, baseline), rectified=True,
                image_size=size, source=self.calibration.source + " (rectified)",
            )
            self._rectified_calibration.validate()
            self._size = size
        m1l, m2l, m1r, m2r = self._maps
        return (
            cv2.remap(left, m1l, m2l, cv2.INTER_LINEAR),
            cv2.remap(right, m1r, m2r, cv2.INTER_LINEAR),
            self._rectified_calibration,
        )


def epipolar_debug_image(left: np.ndarray, right: np.ndarray, interval: int = 40) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("rectified debug views must have matching image shape")
    joined = np.concatenate([left, right], axis=1)
    for y in range(0, joined.shape[0], interval):
        cv2.line(joined, (0, y), (joined.shape[1] - 1, y), (0, 255, 255), 1)
    return joined
