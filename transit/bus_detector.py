"""Swappable bus detection with small greedy IoU tracking (not CV imports)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from core.config import ROOT


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    conf: float
    cls: int
    track_id: int | None = None


class DetectorBackend(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.0


def letterbox(frame, size=(640, 640)):
    height, width = frame.shape[:2]
    ratio = min(size[0]/height, size[1]/width)
    resized_w, resized_h = round(width*ratio), round(height*ratio)
    left, top = (size[1]-resized_w)//2, (size[0]-resized_h)//2
    padded = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    padded[top:top+resized_h, left:left+resized_w] = cv2.resize(frame, (resized_w, resized_h))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255
    return tensor, ratio, (left, top)


def decode_yolo(output, ratio, pad, original_shape, *, classes=(5,), confidence=.25, nms_iou=.45):
    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"Expected YOLO11 raw [1,84,N] output, got {predictions.shape}")
    if predictions.shape[0] == 84:
        predictions = predictions.T
    if predictions.shape[1] != 84:
        raise ValueError("Expected 4 box values + 80 COCO scores; export YOLO11 with nms=False")
    labels = predictions[:, 4:].argmax(axis=1)
    scores = predictions[np.arange(len(predictions)), labels+4]
    keep = (scores >= confidence) & np.isin(labels, classes)
    rows, labels, scores = predictions[keep], labels[keep], scores[keep]
    height, width = original_shape[:2]
    detections = []
    for row, label, score in zip(rows, labels, scores):
        x, y, w, h = row[:4]
        box = ((x-w/2-pad[0])/ratio, (y-h/2-pad[1])/ratio,
               (x+w/2-pad[0])/ratio, (y+h/2-pad[1])/ratio)
        box = (float(np.clip(box[0], 0, width)), float(np.clip(box[1], 0, height)),
               float(np.clip(box[2], 0, width)), float(np.clip(box[3], 0, height)))
        if box[2] > box[0] and box[3] > box[1] and np.isfinite(score):
            detections.append(Detection(box, float(score), int(label)))
    # Class-agnostic NMS across bus/truck avoids duplicate tracks of one vehicle.
    selected = []
    for detection in sorted(detections, key=lambda item: -item.conf):
        if all(iou(detection.bbox, old.bbox) < nms_iou for old in selected):
            selected.append(detection)
            if len(selected) >= 20:
                break
    return selected


class OnnxYoloBackend:
    def __init__(self, model=ROOT / "models/yolo11n.onnx", *, include_trucks=False,
                 confidence=.25, nms_iou=.45):
        if not Path(model).is_file():
            raise FileNotFoundError(f"Missing {model}; run scripts/download_models.py --only yolo")
        import onnxruntime as ort
        start = time.perf_counter()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.model_path = Path(model)
        self.load_time_ms = (time.perf_counter()-start)*1000
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.size = tuple(int(v) if isinstance(v, int) else 640 for v in shape[-2:])
        self.classes = (5, 7) if include_trucks else (5,)
        self.confidence, self.nms_iou = confidence, nms_iou

    def detect(self, frame_bgr):
        tensor, ratio, pad = letterbox(frame_bgr, self.size)
        output = self.session.run(None, {self.input.name: tensor})[0]
        return decode_yolo(output, ratio, pad, frame_bgr.shape, classes=self.classes,
                           confidence=self.confidence, nms_iou=self.nms_iou)


class UltralyticsBackend:
    def __init__(self, model=ROOT / "models/yolo11n.pt", *, include_trucks=False, confidence=.25):
        if not Path(model).is_file():
            raise FileNotFoundError(f"Local YOLO weights missing: {model}")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "models/ultralytics"))
        os.environ["YOLO_OFFLINE"] = "true"
        config_dir = Path(os.environ["YOLO_CONFIG_DIR"])
        (config_dir / "Ultralytics").mkdir(parents=True,exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR",str(ROOT / "models/matplotlib"))
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"sync": False})
        start = time.perf_counter()
        self.model = YOLO(str(model))
        self.model_path = Path(model)
        self.load_time_ms = (time.perf_counter()-start)*1000
        self.classes = [5, 7] if include_trucks else [5]
        self.confidence = confidence

    def detect(self, frame_bgr):
        results = self.model.predict(frame_bgr, classes=self.classes, conf=self.confidence,
                                     device="cpu", verbose=False)
        return [Detection(tuple(map(float, box.xyxy[0].tolist())), float(box.conf[0]), int(box.cls[0]))
                for result in results for box in result.boxes]


class PassthroughBackend:
    """Explicit close-up board mode; not evidence of real bus detection."""
    def detect(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        return [Detection((0, 0, width, height), 1.0, 5)]


class IoUTracker:
    def __init__(self, threshold=.3, max_missed=15):
        self.threshold, self.max_missed = threshold, max_missed
        self.tracks = {}
        self._next_id = 1

    def update(self, detections):
        matches = sorted([(iou(track["bbox"], det.bbox), track_id, index)
                          for track_id, track in self.tracks.items()
                          for index, det in enumerate(detections)], reverse=True)
        assigned, used = {}, set()
        for overlap, track_id, index in matches:
            if overlap >= self.threshold and index not in assigned and track_id not in used:
                assigned[index] = track_id
                used.add(track_id)
        for track_id in list(self.tracks):
            self.tracks[track_id]["missed"] += 1
            if self.tracks[track_id]["missed"] > self.max_missed:
                del self.tracks[track_id]
        result = []
        for index, detection in enumerate(detections):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            self.tracks[track_id] = {"bbox": detection.bbox, "missed": 0}
            result.append(replace(detection, track_id=track_id))
        return result


class BusDetector:
    def __init__(self, backend="auto", *, include_trucks=False, confidence=.25,
                 tracker=None, model=None):
        self.fallback_reason = None
        if not isinstance(backend, str):
            self.backend = backend
        elif backend == "passthrough":
            self.backend = PassthroughBackend()
        elif backend == "ultralytics":
            self.backend = UltralyticsBackend(model or ROOT / "models/yolo11n.pt", include_trucks=include_trucks, confidence=confidence)
        elif backend in {"auto", "onnx"}:
            try:
                self.backend = OnnxYoloBackend(model or ROOT / "models/yolo11n.onnx", include_trucks=include_trucks, confidence=confidence)
            except Exception as error:
                if backend == "onnx":
                    raise
                self.fallback_reason = str(error)
                try:
                    self.backend = UltralyticsBackend(include_trucks=include_trucks, confidence=confidence)
                except Exception as fallback:
                    raise RuntimeError(f"YOLO unavailable: {error}; {fallback}. Use --detector passthrough for board photos only.") from fallback
        else:
            raise ValueError(f"Unknown detector backend {backend}")
        self.tracker = tracker or IoUTracker()
        self.backend_name = type(self.backend).__name__
        self.load_time_ms = getattr(self.backend, "load_time_ms", 0.0)

    @property
    def board_mode(self):
        return isinstance(self.backend, PassthroughBackend)

    def detect(self, frame_bgr):
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.size:
            raise ValueError("Expected nonempty BGR frame")
        return self.tracker.update(self.backend.detect(frame_bgr))
