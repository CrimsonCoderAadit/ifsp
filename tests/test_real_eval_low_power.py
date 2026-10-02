import json

import numpy as np
import pytest

from bench.benchmark import failure_types, labelled_bbox, print_confusion
from core.config import (LOW_POWER_DETECTOR_MODEL, low_power_pipeline_config,
                         low_power_speech_config)
from transit.bus_detector import BusDetector


def test_real_label_failure_tags_and_box(capsys):
    frame = np.zeros((80, 120, 3), np.uint8)
    label = {"bbox": "[10, 5, 100, 70]", "failure_type": "glare|motion_blur"}
    assert labelled_bbox(label, frame) == (10, 5, 100, 70)
    assert failure_types(label) == ["glare", "motion_blur"]
    with pytest.raises(ValueError):
        labelled_bbox({"bbox": "[0,0,999,999]"}, frame)
    rows = [
        {"component":"ocr", "test_case":"real/bus1.jpg/pipeline", "success":"wrong",
         "notes":json.dumps({"expected":"M70", "route":"6D",
                              "failure_types":["glare", "motion_blur"]})},
        {"component":"ocr", "test_case":"real/bus2.jpg/pipeline", "success":"none",
         "notes":json.dumps({"expected":"21G", "route":None,
                              "failure_types":["glare"]})},
    ]
    print_confusion(rows, "real")
    output = capsys.readouterr().out
    assert "glare: n=2" in output and "wrong=1" in output and "none=1" in output
    assert "('M70', '6D')" in output and "('21G', '∅')" in output


def test_low_power_profile_and_exported_detector():
    config = low_power_pipeline_config()
    assert config.max_fps == 5 and config.ocr_every_n == 5
    assert low_power_speech_config().piper_model.name == "en_US-lessac-low.onnx"
    if not LOW_POWER_DETECTOR_MODEL.exists():
        pytest.skip("Run scripts/download_models.py --only yolo_low")
    detector = BusDetector("onnx", model=LOW_POWER_DETECTOR_MODEL)
    assert detector.backend.size == (320, 320)
    assert isinstance(detector.detect(np.zeros((240, 320, 3), np.uint8)), list)
