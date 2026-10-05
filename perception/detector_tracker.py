"""One YOLO instance and its persistent ByteTrack state."""

from collections.abc import Sequence

import numpy as np
from ultralytics import YOLO

from common.events import Track


class DetectorTracker:
    def __init__(
        self,
        model_name: str,
        allowed_classes: Sequence[str],
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        image_size: int = 640,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model = YOLO(model_name)  # Ultralytics downloads known weights if needed.
        self.names = self.model.names
        self.allowed_ids = [
            int(class_id)
            for class_id, name in self.names.items()
            if name in allowed_classes
        ]
        missing = set(allowed_classes) - set(self.names.values())
        if missing or not self.allowed_ids:
            raise ValueError(f"Unsupported model classes: {sorted(missing)}")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.tracker = tracker
        self.last_detection_boxes: list[tuple[float, float, float, float]] = []

    @property
    def device(self) -> str:
        if self.model.predictor is not None:
            return str(self.model.predictor.device)
        return str(self.model.device)

    def track(self, frame: np.ndarray) -> list[Track]:
        """Process the next frame in sequence; retain tracker state across calls."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            classes=self.allowed_ids,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
        )
        self.last_detection_boxes = []
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        self.last_detection_boxes = [
            tuple(float(value) for value in xyxy)
            for xyxy in boxes.xyxy.cpu().numpy()
        ]
        if boxes.id is None:  # Detections can be present before IDs are assigned.
            return []

        tracks: list[Track] = []
        for xyxy, track_id, class_id, confidence in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ):
            class_id = int(class_id)
            if class_id not in self.allowed_ids:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            tracks.append(
                Track(
                    track_id=int(track_id),
                    class_id=class_id,
                    class_name=self.names[class_id],
                    confidence=float(confidence),
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_center=((x1 + x2) / 2, (y1 + y2) / 2),
                    bbox_bottom_center=((x1 + x2) / 2, y2),
                    bbox_width=width,
                    bbox_height=height,
                    bbox_area=width * height,
                )
            )
        return tracks

    def detect(self, frame: np.ndarray) -> list[Track]:
        """Single-image detection using this instance; no persistent tracker state."""
        results = self.model.predict(
            source=frame, classes=self.allowed_ids, conf=self.confidence_threshold,
            iou=self.iou_threshold, imgsz=self.image_size, verbose=False,
        )
        tracks = []
        for result in results:
            if result.boxes is None:
                continue
            for xyxy, class_id, confidence in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                class_id = int(class_id)
                if class_id not in self.allowed_ids:
                    continue
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                width, height = max(0., x2-x1), max(0., y2-y1)
                tracks.append(Track(-1, class_id, self.names[class_id], float(confidence),
                                    (x1, y1, x2, y2), ((x1+x2)/2, (y1+y2)/2),
                                    ((x1+x2)/2, y2), width, height, width*height))
        return tracks
