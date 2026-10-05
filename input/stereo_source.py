"""Strictly matched stereo image pairs."""

from dataclasses import dataclass
from pathlib import Path
import tarfile

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class StereoFrame:
    left: np.ndarray
    right: np.ndarray
    timestamp: float
    frame_index: int
    frame_id: str


class StereoSource:
    def __init__(
        self, left_dir: Path | str | None = None, right_dir: Path | str | None = None,
        *, left_file: Path | str | None = None, right_file: Path | str | None = None,
        fps: float = 10.0, timestamps_file: Path | str | None = None,
        raw_bayer: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError("stereo fps must be positive")
        self.fps = fps
        self.index = 0
        self.raw_bayer = raw_bayer
        self._archive: tarfile.TarFile | None = None
        self._archive_names: list[tuple[str, str]] | None = None
        if left_file is not None or right_file is not None:
            if left_file is None or right_file is None or left_dir is not None or right_dir is not None:
                raise ValueError("provide exactly one left/right pair or two directories")
            left, right = Path(left_file), Path(right_file)
            if not left.is_file() or not right.is_file():
                raise FileNotFoundError("stereo pair is incomplete")
            self.pairs = [(left, right)]
            self.frame_ids = [left.stem]
        else:
            if left_dir is None or right_dir is None:
                raise ValueError("left and right image directories are required")
            left_paths = self._images(Path(left_dir))
            right_paths = self._images(Path(right_dir))
            left_ids = [path.stem for path in left_paths]
            right_ids = [path.stem for path in right_paths]
            if left_ids != right_ids:
                missing_right = sorted(set(left_ids) - set(right_ids))[:5]
                missing_left = sorted(set(right_ids) - set(left_ids))[:5]
                raise ValueError(f"stereo frame IDs mismatch: missing right {missing_right}; missing left {missing_left}")
            self.pairs = list(zip(left_paths, right_paths))
            self.frame_ids = left_ids
        if not self.pairs:
            raise ValueError("stereo source contains no image pairs")
        if timestamps_file is None:
            self.timestamps = [i / fps for i in range(len(self.pairs))]
        else:
            lines = [line.strip() for line in Path(timestamps_file).read_text().splitlines() if line.strip()]
            if len(lines) != len(self.pairs):
                raise ValueError("timestamp count does not match stereo pair count")
            try:
                self.timestamps = [float(line) for line in lines]
            except ValueError:
                from datetime import datetime
                parsed = [datetime.fromisoformat(line) for line in lines]
                self.timestamps = [(item - parsed[0]).total_seconds() for item in parsed]
            if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:])):
                raise ValueError("timestamps must increase strictly")

    @staticmethod
    def _images(directory: Path) -> list[Path]:
        if not directory.is_dir():
            raise FileNotFoundError(f"missing stereo image directory: {directory}")
        paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and not path.name.startswith("._"))
        stems = [path.stem for path in paths]
        if len(stems) != len(set(stems)):
            raise ValueError(f"duplicate stereo frame ID in {directory}")
        return paths

    @classmethod
    def from_dataset(
        cls, root: Path | str, *, left_subdir: str = "image_02/data",
        right_subdir: str = "image_03/data", fps: float = 10.0,
    ) -> "StereoSource":
        root = Path(root)
        if root.suffix.lower() == ".tar":
            return cls.from_robotcar_tar(root)
        timestamp_path = root / "timestamps.txt"
        return cls(root / left_subdir, root / right_subdir, fps=fps,
                   timestamps_file=timestamp_path if timestamp_path.exists() else None)

    @classmethod
    def from_robotcar_tar(cls, archive_path: Path | str) -> "StereoSource":
        archive = tarfile.open(archive_path, mode="r")
        try:
            names = [member.name for member in archive.getmembers() if member.isfile()]
            left = sorted(name for name in names if name.startswith("stereo/left/") and name.endswith(".png") and not Path(name).name.startswith("._"))
            right = sorted(name for name in names if name.startswith("stereo/right/") and name.endswith(".png") and not Path(name).name.startswith("._"))
            left_ids = [Path(name).stem for name in left]
            right_ids = [Path(name).stem for name in right]
            if not left or left_ids != right_ids:
                raise ValueError("RobotCar tar has missing or mismatched stereo frame IDs")
            if len(left_ids) != len(set(left_ids)):
                raise ValueError("RobotCar tar has duplicate stereo frame IDs")
            source = cls.__new__(cls)
            source.fps = 16.0
            source.index = 0
            source.raw_bayer = True
            source._archive = archive
            source._archive_names = list(zip(left, right))
            source.pairs = []
            source.frame_ids = left_ids
            first = int(left_ids[0])
            source.timestamps = [(int(frame_id) - first) / 1_000_000 for frame_id in left_ids]
            if any(b <= a for a, b in zip(source.timestamps, source.timestamps[1:])):
                raise ValueError("RobotCar frame timestamps must increase")
            return source
        except Exception:
            archive.close()
            raise

    def read(self) -> StereoFrame | None:
        if self.index >= len(self):
            return None
        if self._archive is not None:
            left_name, right_name = self._archive_names[self.index]
            left = self._decode_member(left_name)
            right = self._decode_member(right_name)
            names = (left_name, right_name)
        else:
            left_path, right_path = self.pairs[self.index]
            flag = cv2.IMREAD_UNCHANGED if self.raw_bayer else cv2.IMREAD_COLOR
            left = cv2.imread(str(left_path), flag)
            right = cv2.imread(str(right_path), flag)
            names = (str(left_path), str(right_path))
        if left is None or right is None:
            raise OSError(f"cannot read stereo pair {names}")
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"stereo image dimensions differ for ID {self.frame_ids[self.index]}")
        item = StereoFrame(left, right, self.timestamps[self.index], self.index, self.frame_ids[self.index])
        self.index += 1
        return item

    def _decode_member(self, name: str) -> np.ndarray | None:
        member = self._archive.extractfile(name)
        if member is None:
            return None
        encoded = np.frombuffer(member.read(), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    def release(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __len__(self) -> int:
        return len(self._archive_names) if self._archive_names is not None else len(self.pairs)
