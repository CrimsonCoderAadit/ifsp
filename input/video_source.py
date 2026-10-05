"""OpenCV-backed video source with a small, stable interface."""

from pathlib import Path

import cv2


class VideoSource:
    def __init__(self, source: str | int):
        if isinstance(source, str) and source.isdecimal():
            source = int(source)
        if isinstance(source, str) and not Path(source).is_file():
            raise FileNotFoundError(f"Video file does not exist: {source}")

        self.source = source
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open video source: {source}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return (success, BGR frame); success is false at EOF or on read failure."""
        return self._capture.read()

    def release(self) -> None:
        self._capture.release()
