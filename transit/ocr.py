"""Local OCR backends and ordered bus-board crop/preprocessing attempts."""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from core.config import ROOT
from transit.route_matcher import MatchResult, extract_route_candidates, match_route


@dataclass(frozen=True)
class OCRLine:
    text: str
    conf: float
    box: tuple[tuple[float, float], ...]


class OCRBackend(Protocol):
    def read(self, image: np.ndarray) -> list[OCRLine]: ...


class RapidOCRBackend:
    def __init__(self, config_path=ROOT / "models/rapidocr/runtime_config.yaml", *, version="v5", size="mobile"):
        import yaml
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError("RapidOCR config missing; run scripts/download_models.py --only rapidocr")
        config = yaml.safe_load(config_path.read_text())
        params = {"EngineConfig.onnxruntime.intra_op_num_threads": 2,
                  "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                  "Det.limit_type": "max", "Det.limit_side_len": 960}
        # Validate local files BEFORE constructing RapidOCR, preventing any
        # implicit runtime download. Rebase paths when copied to a Pi.
        names = {}
        if version == "v5":
            from rapidocr import OCRVersion, ModelType, LangRec
            names = {"Det": f"ch_PP-OCRv5_det_{size}.onnx",
                     "Rec": f"ch_PP-OCRv5_rec_{size}.onnx"}
            model_type = ModelType.MOBILE if size == "mobile" else ModelType.SERVER
            params.update({"Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
                           "Det.model_type": model_type, "Rec.model_type": model_type,
                           "Rec.lang_type": LangRec.CH})
        for stage in ("Det", "Cls", "Rec"):
            path = config_path.parent / names.get(stage, Path(config[stage]["model_path"]).name)
            if not path.is_file() or path.stat().st_size < 100_000:
                raise FileNotFoundError(f"Local OCR model missing/incomplete: {path}")
            params[f"{stage}.model_path"] = str(path.resolve())
        self.model_names = [Path(params[f"{stage}.model_path"]).name for stage in ("Det", "Cls", "Rec")]
        from rapidocr import RapidOCR
        started = time.perf_counter()
        self.engine = RapidOCR(config_path=str(config_path), params=params)
        self.load_time_ms = (time.perf_counter()-started)*1000

    def read(self, image):
        result = self.engine(image)
        if result.txts is None:
            return []
        return [OCRLine(str(text), float(conf), tuple(tuple(map(float, p)) for p in box))
                for text, conf, box in zip(result.txts, result.scores, result.boxes)]


class PaddleOCRBackend:
    """Optional local PaddleOCR Tamil/Latin destination-text adapter."""
    def __init__(self, model_dir=ROOT / "models/paddleocr"):
        directory = Path(model_dir)
        paths = {"text_detection_model_dir": directory/"det",
                 "text_recognition_model_dir": directory/"rec"}
        for path in paths.values():
            if not (path / "inference.yml").is_file() or not any(path.glob("*.pdiparams")):
                raise FileNotFoundError(f"Provision local PaddleOCR Tamil model directory first: {path}")
        from paddleocr import PaddleOCR
        started = time.perf_counter()
        self.engine = PaddleOCR(**{key: str(value) for key, value in paths.items()},
                                lang="ta", ocr_version="PP-OCRv5",
                                use_doc_orientation_classify=False, use_doc_unwarping=False,
                                use_textline_orientation=False, device="cpu")
        self.load_time_ms = (time.perf_counter()-started)*1000

    def read(self, image):
        lines = []
        for result in self.engine.predict(image):
            data = result.json if hasattr(result, "json") else result
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            for text, conf, box in zip(data.get("rec_texts") or [],
                                       data.get("rec_scores") or [],
                                       data.get("rec_polys") or []):
                lines.append(OCRLine(str(text), float(conf), tuple(tuple(map(float, p)) for p in box)))
        return lines


class TesseractBackend:
    def __init__(self, psm=7):
        if psm not in {7, 11}:
            raise ValueError("Tesseract psm must be 7 or 11")
        if not shutil.which("tesseract"):
            raise RuntimeError("Tesseract binary missing; install it as an OS package")
        import pytesseract
        self.engine = pytesseract
        self.psm = psm
        self.load_time_ms = 0.0

    def read(self, image):
        # Try single-line mode first, then sparse text for full bus crops.
        for psm in dict.fromkeys((self.psm, 11)):
            data = self.engine.image_to_data(image, output_type=self.engine.Output.DICT, timeout=3,
                                            config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            lines = []
            for i, text in enumerate(data["text"]):
                if not text.strip() or float(data["conf"][i]) < 0:
                    continue
                x, y, w, h = [int(data[key][i]) for key in ("left", "top", "width", "height")]
                lines.append(OCRLine(text.strip(), float(data["conf"][i])/100,
                                     ((x,y), (x+w,y), (x+w,y+h), (x,y+h))))
            if lines:
                return lines
        return []


def create_ocr_backend(name="rapidocr"):
    if name == "tesseract":
        return TesseractBackend()
    if name == "paddleocr":
        return PaddleOCRBackend()
    if name not in {"rapidocr", "rapidocr-v4", "rapidocr-v5-mobile", "rapidocr-v5-server"}:
        raise ValueError(f"Unknown OCR backend {name}")
    try:
        version = "v4" if name in {"rapidocr", "rapidocr-v4"} else "v5"
        size = "server" if name == "rapidocr-v5-server" else "mobile"
        return RapidOCRBackend(version=version, size=size)
    except Exception as error:
        if name == "rapidocr":
            try:
                backend = RapidOCRBackend(version="v4")
                backend.fallback_reason = f"PP-OCRv5 unavailable: {error}"
                return backend
            except Exception:
                pass
        try:
            backend = TesseractBackend()
            backend.fallback_reason = str(error)
            return backend
        except Exception as fallback:
            raise RuntimeError(f"No offline OCR backend available: {error}; fallback: {fallback}") from fallback


def board_crops(frame, bbox, top_fraction=.35, *, include_top=True):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    if include_top:
        end = min(y2, y1+max(1, round((y2-y1)*top_fraction)))
        yield "top_front", frame[y1:end, x1:x2], (x1, y1, x2, end)
    yield "full", frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def perspective_crop(image):
    found = _perspective_transform(image)
    return None if found is None else found[0]


def _perspective_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    contours, _ = cv2.findContours(cv2.Canny(gray, 60, 180), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        polygon = cv2.approxPolyDP(contour, .03*cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or cv2.contourArea(polygon) < image.shape[0]*image.shape[1]*.15:
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered = np.array([points[sums.argmin()], points[differences.argmin()],
                            points[sums.argmax()], points[differences.argmax()]], dtype=np.float32)
        if len(np.unique(ordered, axis=0)) != 4:
            continue
        width = int(max(np.linalg.norm(ordered[1]-ordered[0]), np.linalg.norm(ordered[2]-ordered[3])))
        height = int(max(np.linalg.norm(ordered[3]-ordered[0]), np.linalg.norm(ordered[2]-ordered[1])))
        if width < 20 or height < 10:
            continue
        target = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, target)
        return cv2.warpPerspective(image, matrix, (width,height)), matrix
    return None


def box_to_frame(box, crop, crop_box, variant):
    """Map a box from a preprocessed variant back into original frame pixels."""
    points = np.asarray(box, dtype=np.float64).reshape(-1, 2)/2
    if variant == "perspective_x2":
        found = _perspective_transform(crop)
        if found is None:
            return None
        points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), np.linalg.inv(found[1])).reshape(-1, 2)
    points += crop_box[:2]
    return tuple((float(x), float(y)) for x, y in points)


def preprocessing_variants(image, *, perspective=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(scaled)
    yield "clahe_x2", enhanced
    sharp = cv2.addWeighted(enhanced, 1.8, cv2.GaussianBlur(enhanced, (0,0), 1), -.8, 0)
    yield "unsharp", sharp
    _, threshold = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    yield "otsu", threshold
    yield "inverted", cv2.bitwise_not(threshold)
    if perspective:
        corrected = perspective_crop(image)
        if corrected is not None:
            corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if corrected.ndim == 3 else corrected
            yield "perspective_x2", cv2.resize(corrected_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


@dataclass
class BoardRead:
    match: MatchResult = field(default_factory=lambda: MatchResult(None, 0, "none"))
    confidence: float = 0.0
    lines: list[OCRLine] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    crop: str = ""
    variant: str = ""
    frame_boxes: list = field(default_factory=list)


class BoardReader:
    def __init__(self, backend: OCRBackend, routes, *, top_fraction=.35, perspective=True,
                 max_attempts=10, max_crop_seconds=1.5, match_threshold=85):
        self.backend, self.routes = backend, set(routes)
        self.top_fraction, self.perspective = top_fraction, perspective
        self.max_attempts, self.max_crop_seconds = max_attempts, max_crop_seconds
        self.match_threshold = match_threshold

    def read(self, frame, bbox, *, board=False):
        """board=True means bbox is the whole sign, so the top-front crop is skipped."""
        result = BoardRead()
        started = time.perf_counter()
        for region, crop, crop_box in board_crops(frame, bbox, self.top_fraction, include_top=not board):
            for variant, image in preprocessing_variants(crop, perspective=self.perspective):
                if len(result.attempts) >= self.max_attempts or time.perf_counter()-started >= self.max_crop_seconds:
                    return result
                attempt_start = time.perf_counter()
                lines = self.backend.read(image)
                records = [{"text": line.text, "box": line.box} for line in lines]
                candidates = extract_route_candidates(records)
                # Rare GTFS aliases are preserved as exact matches, even when
                # the standard shape extractor deliberately rejects them.
                candidates += [line.text for line in lines if line.text.upper().strip() in self.routes]
                matched = MatchResult(None, 0, "none")
                for candidate in dict.fromkeys(candidates):
                    matched = match_route(candidate, self.routes, threshold=self.match_threshold)
                    if matched.route:
                        compact = lambda text: re.sub(r"[\s.\-]", "", text).upper()
                        token = compact(candidate)
                        supporting = [line.conf for line in lines
                                      if compact(line.text) and (compact(line.text) in token or token in compact(line.text))]
                        confidence = min(supporting) if supporting else min((line.conf for line in lines), default=0)
                        break
                elapsed = (time.perf_counter()-attempt_start)*1000
                result.attempts.append({"crop": region, "variant": variant, "crop_bbox": crop_box,
                                        "latency_ms": elapsed, "text": [line.text for line in lines],
                                        "conf": [line.conf for line in lines],
                                        "candidates": list(dict.fromkeys(candidates)),
                                        "route": matched.route, "method": matched.method})
                result.lines, result.crop, result.variant = lines, region, variant
                result.frame_boxes = [box_to_frame(line.box, crop, crop_box, variant) for line in lines]
                if matched.route:
                    result.match, result.confidence = matched, confidence
                    return result
        return result
