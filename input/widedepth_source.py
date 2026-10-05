"""Adapter around the official, NumPy-first WideDepth training loader."""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True, slots=True)
class WideDepthFrame:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    gt_disparity_px: np.ndarray
    valid_mask: np.ndarray
    sample_id: str
    frame_index: int
    timestamp: float | None = None  # The release has IDs, not capture timestamps.

    @property
    def left_frame(self) -> np.ndarray:
        """Official RGB array; convert to BGR only at an OpenCV consumer."""
        return self.left_rgb

    @property
    def right_frame(self) -> np.ndarray:
        return self.right_rgb

    @property
    def frame_id(self) -> str:
        return self.sample_id


class WideDepthSource:
    def __init__(
        self, root_dir: str | Path, loader_repo: str | Path, *,
        task: str = "stereo", crop_equirect: bool = False,
        rotate_ccw: bool = False,
    ) -> None:
        if task != "stereo":
            raise ValueError("WideDepthSource requires the official stereo task")
        repository = Path(loader_repo).resolve()
        if not (repository / "widedepth" / "__init__.py").is_file():
            raise FileNotFoundError(f"official WideDepth loader not found: {repository}")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        from widedepth import WideDepthTrainDataset

        self.dataset = WideDepthTrainDataset(
            root_dir=root_dir, task="stereo",
            crop_equirect=crop_equirect, rotate_ccw=rotate_ccw,
        )
        self.unmatched_files = self.dataset.unmatched_files
        self.sample_ids = self.dataset.sample_ids
        self.index = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def read(self) -> WideDepthFrame | None:
        if self.index >= len(self.dataset):
            return None
        sample = self.dataset[self.index]
        left, right = sample["left_image"], sample["right_image"]
        disparity, valid = sample["disparity"], sample["valid_mask"]
        if (
            left.dtype != np.uint8 or right.dtype != np.uint8
            or left.ndim != 3 or right.ndim != 3
            or left.shape != right.shape or left.shape[2] != 3
            or disparity.shape != left.shape[:2] or disparity.dtype != np.float32
            or valid.shape != disparity.shape or valid.dtype != np.bool_
        ):
            raise ValueError(f"official loader returned incompatible arrays for {sample['sample_id']}")
        frame = WideDepthFrame(
            left, right, disparity, valid, str(sample["sample_id"]), self.index,
        )
        self.index += 1
        return frame

    def release(self) -> None:
        pass  # The official image-sequence loader keeps no open video handle.
