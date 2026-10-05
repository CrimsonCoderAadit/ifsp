"""Metric stereo calibration with explicit translation units."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    K_left: np.ndarray
    K_right: np.ndarray
    D_left: np.ndarray
    D_right: np.ndarray
    R: np.ndarray
    T_m: np.ndarray
    baseline_m: float
    focal_px: float
    cx_left: float
    cx_right: float
    Q: np.ndarray | None
    rectified: bool
    image_size: tuple[int, int] | None
    source: str
    robotcar_models_dir: Path | None = None

    @property
    def disparity_offset_px(self) -> float:
        return self.cx_left - self.cx_right

    def validate(self) -> None:
        for name in ("K_left", "K_right", "R"):
            value = getattr(self, name)
            if value.shape != (3, 3) or not np.isfinite(value).all():
                raise ValueError(f"invalid {name} matrix")
        if self.T_m.size != 3 or not np.isfinite(self.T_m).all():
            raise ValueError("invalid stereo translation")
        if not np.isfinite(self.baseline_m) or not 0.01 <= self.baseline_m <= 5:
            raise ValueError("implausible stereo baseline in meters")
        if not np.isfinite(self.focal_px) or self.focal_px <= 0:
            raise ValueError("invalid stereo focal length")
        if abs(np.linalg.norm(self.T_m) - self.baseline_m) > max(0.01, 0.05 * self.baseline_m):
            raise ValueError("baseline conflicts with translation")
        if self.Q is not None and (self.Q.shape != (4, 4) or not np.isfinite(self.Q).all()):
            raise ValueError("invalid reprojection matrix Q")


def _q_from_projections(P1: np.ndarray, P2: np.ndarray, baseline_m: float) -> np.ndarray:
    f = float(P1[0, 0])
    cx1, cx2, cy = float(P1[0, 2]), float(P2[0, 2]), float(P1[1, 2])
    return np.array([[1, 0, 0, -cx1], [0, 1, 0, -cy], [0, 0, 0, f],
                     [0, 0, 1 / baseline_m, -(cx1 - cx2) / baseline_m]], dtype=np.float64)


def load_kitti_calibration(path: Path | str) -> StereoCalibration:
    """Read KITTI color camera projection matrices (metric translations)."""
    values: dict[str, np.ndarray] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key.strip()] = np.fromstring(raw, sep=" ", dtype=np.float64)
        except ValueError:
            continue
    keys = ("P_rect_02", "P_rect_03") if "P_rect_02" in values else ("P2", "P3")
    if any(key not in values or values[key].size != 12 for key in keys):
        raise ValueError("KITTI calibration requires P_rect_02/P_rect_03 or P2/P3")
    P1, P2 = (values[key].reshape(3, 4) for key in keys)
    f1, f2 = float(P1[0, 0]), float(P2[0, 0])
    if f1 <= 0 or f2 <= 0 or abs(f1 - f2) > 0.02 * f1:
        raise ValueError("KITTI left/right focal lengths are inconsistent")
    left_center_x = -float(P1[0, 3]) / f1
    right_center_x = -float(P2[0, 3]) / f2
    baseline = right_center_x - left_center_x
    if baseline <= 0:
        raise ValueError("KITTI right camera must have positive baseline relative to left")
    size_key = "S_rect_02"
    image_size = tuple(int(round(v)) for v in values[size_key]) if size_key in values and values[size_key].size == 2 else None
    K1, K2 = P1[:, :3].copy(), P2[:, :3].copy()
    calibration = StereoCalibration(
        K1, K2, np.zeros(5), np.zeros(5), np.eye(3),
        np.array([-baseline, 0, 0]), baseline, f1, float(P1[0, 2]), float(P2[0, 2]),
        _q_from_projections(P1, P2, baseline), True, image_size, "KITTI rectified projection matrices",
    )
    calibration.validate()
    return calibration


def load_npz_calibration(path: Path | str) -> StereoCalibration:
    """Generic npz: K1,K2,D1,D2,R,T,translation_unit,rectified; optional P1,P2,Q,image_size."""
    with np.load(path, allow_pickle=False) as data:
        required = {"K1", "K2", "D1", "D2", "R", "T", "translation_unit", "rectified"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"generic calibration missing fields: {sorted(missing)}")
        unit = str(data["translation_unit"].item())
        if unit not in ("m", "mm"):
            raise ValueError("translation_unit must be explicitly 'm' or 'mm'")
        T = np.asarray(data["T"], dtype=np.float64).reshape(-1) * (0.001 if unit == "mm" else 1)
        baseline = float(np.linalg.norm(T))
        K1, K2 = np.asarray(data["K1"], np.float64), np.asarray(data["K2"], np.float64)
        rectified = bool(data["rectified"].item())
        P1 = np.asarray(data["P1"], np.float64) if "P1" in data else None
        P2 = np.asarray(data["P2"], np.float64) if "P2" in data else None
        if (P1 is None) != (P2 is None):
            raise ValueError("provide both P1 and P2, or neither")
        if P1 is not None and (P1.shape != (3, 4) or P2.shape != (3, 4)):
            raise ValueError("P1/P2 must be 3x4 matrices")
        if rectified and P1 is not None:
            f, cx1, cx2 = float(P1[0, 0]), float(P1[0, 2]), float(P2[0, 2])
            projected_baseline = -float(P2[0, 3]) / float(P2[0, 0]) + float(P1[0, 3]) / f
            if abs(projected_baseline - baseline) > max(0.01, 0.05 * baseline):
                raise ValueError("projection baseline conflicts with metric T")
        else:
            f, cx1, cx2 = float(K1[0, 0]), float(K1[0, 2]), float(K2[0, 2])
        Q = np.asarray(data["Q"], np.float64) if "Q" in data else None
        if Q is None and rectified and P1 is not None:
            Q = _q_from_projections(P1, P2, baseline)
        image_size = tuple(int(v) for v in data["image_size"].reshape(-1)) if "image_size" in data else None
        calibration = StereoCalibration(
            K1, K2, np.asarray(data["D1"], np.float64).reshape(-1),
            np.asarray(data["D2"], np.float64).reshape(-1), np.asarray(data["R"], np.float64),
            T, baseline, f, cx1, cx2, Q, rectified, image_size, "generic npz",
        )
    calibration.validate()
    return calibration


def load_robotcar_calibration(models_dir: Path | str) -> StereoCalibration:
    """Use Oxford RobotCar SDK wide stereo LUT models; translation is metres."""
    models_dir = Path(models_dir)
    models = []
    for side in ("left", "right"):
        path = models_dir / f"stereo_wide_{side}.txt"
        lut_path = models_dir / f"stereo_wide_{side}_distortion_lut.bin"
        if not path.is_file() or not lut_path.is_file():
            raise FileNotFoundError(f"RobotCar model or LUT missing for {side}: {models_dir}")
        rows = [np.fromstring(line, sep=" ") for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 5 or any(row.size != 4 for row in rows):
            raise ValueError(f"invalid RobotCar camera model: {path}")
        models.append((rows[0], np.stack(rows[1:])))
        if lut_path.stat().st_size != 2 * 1280 * 960 * 8:
            raise ValueError(f"RobotCar LUT size is not 1280x960: {lut_path}")
    intr_l, G_l = models[0]
    intr_r, G_r = models[1]
    if not np.allclose(intr_l, intr_r, atol=1e-3):
        raise ValueError("RobotCar wide camera rectified intrinsics differ unexpectedly")
    if not np.allclose(G_l[:3, :3], G_r[:3, :3], atol=1e-3):
        raise ValueError("RobotCar wide camera rectified axes differ unexpectedly")
    # SDK transforms image coordinates into the vehicle camera frame, whose
    # horizontal image axis is not the first vehicle-frame axis.
    displacement = G_l[:3, :3].T @ (G_r[:3, 3] - G_l[:3, 3])
    baseline = float(np.linalg.norm(displacement))
    if displacement[0] <= 0 or abs(displacement[0] - baseline) > 0.01:
        raise ValueError("RobotCar camera order or baseline is not horizontal")
    fx, fy, cx, cy = (float(v) for v in intr_l)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    P1 = np.column_stack((K, np.zeros(3)))
    P2 = P1.copy()
    P2[0, 3] = -fx * baseline
    calibration = StereoCalibration(
        K, K.copy(), np.zeros(5), np.zeros(5), np.eye(3),
        np.array([-baseline, 0, 0]), baseline, fx, cx, cx,
        _q_from_projections(P1, P2, baseline), False, (1280, 960),
        "Oxford RobotCar SDK wide stereo (raw Bayer + LUT)", models_dir,
    )
    calibration.validate()
    return calibration


def load_calibration(path: Path | str) -> StereoCalibration:
    path = Path(path)
    if path.is_dir():
        return load_robotcar_calibration(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing stereo calibration: {path}")
    if path.suffix.lower() == ".npz":
        return load_npz_calibration(path)
    if path.suffix.lower() == ".txt":
        return load_kitti_calibration(path)
    raise ValueError("stereo calibration must be KITTI .txt or generic .npz")
