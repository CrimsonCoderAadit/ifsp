import csv
import json
import socket
import sys
import threading
import time
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config import ROOT, PipelineConfig
from core.events import BusEvent, RiskEvent
from core.pipeline import Assistant, bbox_direction, is_approaching
from core.state import AssistantState
from speech.scheduler import AudioScheduler, NullSink, SchedulerConfig
from transit.bus_detector import BusDetector, Detection, IoUTracker, decode_yolo, iou, letterbox
from transit.ocr import BoardReader, OCRLine, PaddleOCRBackend, board_crops, preprocessing_variants


class RecordingScheduler:
    def __init__(self):
        self.items = []
        self.repeats = self.stops = 0
        self.is_relevant = lambda item: True
    def submit(self, event, **kwargs):
        self.items.append((event, kwargs))
        return str(len(self.items))
    def stop(self):
        self.stops += 1
    def repeat(self):
        self.repeats += 1


class FakeOCR:
    def __init__(self, text="21 G", confidence=.95, delay=0):
        self.text, self.confidence, self.delay = text, confidence, delay
        self.calls = 0
    def read(self, image):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return [OCRLine(self.text,self.confidence,((0,0),(100,0),(100,50),(0,50)))]


def feed(assistant, frame):
    time.sleep(1/assistant.config.max_fps+.001)
    payload = assistant.handle_frame(frame)
    assert assistant.wait_for_frame(payload["submitted_frame_id"],2)
    assert assistant.overlay()["error"] is None
    return assistant.overlay()


def test_letterbox_and_coordinate_recovery():
    frame = np.zeros((100,200,3),np.uint8)
    tensor,ratio,pad = letterbox(frame)
    assert tensor.shape == (1,3,640,640) and tensor.dtype == np.float32
    assert ratio == 3.2 and pad == (0,160)
    rows = np.zeros((1,84,4),np.float32)
    # xywh in original pixels -> padded 640px space.
    for index, (box, label, confidence) in enumerate([
        ((50,50,80,60),5,.9), ((51,50,80,60),5,.8),
        ((150,50,60,60),7,.95), ((50,50,80,60),0,.99)]):
        x,y,w,h = box
        rows[0,:4,index] = [x*ratio+pad[0],y*ratio+pad[1],w*ratio,h*ratio]
        rows[0,4+label,index] = confidence
    buses = decode_yolo(rows,ratio,pad,frame.shape)
    assert len(buses) == 1 and buses[0].cls == 5
    assert buses[0].bbox == pytest.approx((10,20,90,80))
    both = decode_yolo(rows,ratio,pad,frame.shape,classes=(5,7))
    assert {item.cls for item in both} == {5,7}
    assert decode_yolo(rows.transpose(0,2,1),ratio,pad,frame.shape) == buses


def test_tracker_ids_are_unique_and_stable():
    tracker = IoUTracker(max_missed=1)
    first = tracker.update([Detection((0,0,100,100),.9,5),Detection((200,0,300,100),.8,5)])
    second = tracker.update([Detection((205,0,305,100),.9,5),Detection((3,0,103,100),.8,7)])
    assert [d.track_id for d in second] == [first[1].track_id,first[0].track_id]
    tracker.update([])
    tracker.update([])
    assert not tracker.tracks
    new = tracker.update([Detection((0,0,100,100),.9,5)])
    assert new[0].track_id not in {d.track_id for d in first}
    assert iou((0,0,0,0),(0,0,0,0)) == 0


def test_top_front_crop_then_full_fallback():
    class FullCropOCR(FakeOCR):
        def read(self,image):
            return [] if image.shape[0] < 200 else super().read(image)
    frame = np.zeros((200,400,3),np.uint8)
    crops = list(board_crops(frame,(-5,-5,405,205)))
    assert crops[0][1].shape == (70,400,3) and crops[1][1].shape == frame.shape
    read = BoardReader(FullCropOCR(),{"21G"},perspective=False).read(frame,(0,0,400,200))
    assert read.match.route == "21G" and read.crop == "full"
    assert [attempt["variant"] for attempt in read.attempts[:4]] == ["clahe_x2","unsharp","otsu","inverted"]
    assert len(read.attempts) == 5 and all(attempt["latency_ms"] >= 0 for attempt in read.attempts)


def test_ocr_stops_first_valid_variant():
    backend = FakeOCR()
    frame = np.zeros((200,400,3),np.uint8)
    result = BoardReader(backend,{"21G"}).read(frame,(0,0,400,200))
    assert backend.calls == 1 and result.match.method == "exact" and result.confidence == .95
    variants = list(preprocessing_variants(frame,perspective=False))
    assert np.array_equal(variants[3][1],cv2.bitwise_not(variants[2][1]))


@pytest.mark.parametrize("confidence,method,immediate",[(.91,"exact",True),(.9,"exact",False),(.99,"confusion_fix",False),(.99,"fuzzy",False)])
def test_temporal_vote_and_exact_shortcut(confidence,method,immediate):
    state = AssistantState()
    first = state.vote(1,"21G",confidence,method,100)
    assert (first == "21G") == immediate
    assert state.vote(1,"21G",confidence,method,101) == "21G"
    assert state.vote(2,None,0,"none",101) is None
    assert state.vote(2,"21G",.8,"exact",102) is None
    assert state.vote(2,"23C",.8,"exact",103) is None
    assert state.vote(2,"21G",.8,"exact",104) == "21G"
    assert state.recent_routes(110) == []
    state.prune({2})
    assert 1 not in state.votes


@pytest.mark.parametrize("box,direction",[((0,0,20,100),"on your left"),((40,0,60,100),"ahead"),((80,0,100,100),"on your right")])
def test_direction(box,direction):
    assert bbox_direction(box,100) == direction


def test_approaching_cue_requires_sustained_area_growth():
    assert not is_approaching([.10, .11])
    assert not is_approaching([.10, .13, .12])
    assert not is_approaching([.10, .105, .11])
    assert is_approaching([.10, .11, .13])

    class MovingBus:
        def __init__(self, sizes):
            self.sizes = iter(sizes)
        def detect(self, frame):
            side = next(self.sizes)
            return [Detection((50-side/2, 50-side/2, 50+side/2, 50+side/2), .95, 5)]

    frame = np.zeros((100,100,3),np.uint8)
    for sizes, expected in (([40,40,40], False), ([40,44,49], True)):
        scheduler = RecordingScheduler()
        detector = BusDetector(MovingBus(sizes))
        with Assistant(detector, FakeOCR(), scheduler, {"21G"},
                       PipelineConfig(max_fps=100, ocr_every_n=1,
                                      approaching_only=True)) as assistant:
            assistant.handle_transcript("find bus 21G")
            for _ in range(3):
                result = feed(assistant, frame)
            assert result["boxes"][0]["approaching"] is expected
            spoken = [event for event, _ in scheduler.items if isinstance(event, BusEvent)]
            assert bool(spoken) is expected


def test_local_paddle_tamil_adapter_uses_same_ocr_interface(tmp_path, monkeypatch):
    for name in ("det", "rec"):
        directory = tmp_path/name
        directory.mkdir()
        (directory/"inference.yml").write_text("model: local")
        (directory/"inference.pdiparams").write_bytes(b"local")
    calls = []
    class FakePaddle:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def predict(self, image):
            return [SimpleNamespace(json={"res": {
                "rec_texts": ["சென்னை", "21G"], "rec_scores": [.93, .97],
                "rec_polys": [[[0,0],[20,0],[20,8],[0,8]],
                              [[30,0],[50,0],[50,8],[30,8]]]}})]
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddle))
    backend = PaddleOCRBackend(tmp_path)
    lines = backend.read(np.zeros((30,60,3),np.uint8))
    assert [line.text for line in lines] == ["சென்னை", "21G"]
    assert calls[0]["lang"] == "ta" and calls[0]["ocr_version"] == "PP-OCRv5"
    assert not calls[0]["use_textline_orientation"]


def test_end_to_end_target_and_contract_risk():
    scheduler = RecordingScheduler()
    with Assistant(BusDetector("passthrough"),FakeOCR(),scheduler,{"21G","23C"},
                   PipelineConfig(max_fps=100,ocr_every_n=1)) as assistant:
        result = assistant.handle_transcript("find bus twenty one gee")
        assert result["target"] == "21G" and result["acknowledgement"] == "Looking for bus 21G."
        payload = feed(assistant,np.zeros((200,400,3),np.uint8))
        assert payload["matched_route"] == "21G" and payload["boxes"][0]["track_id"] == 1
        buses = [(event,opts) for event,opts in scheduler.items if isinstance(event,BusEvent)]
        assert len(buses) == 1 and buses[0][1]["text_override"] == "Bus 21G detected ahead."
        assistant.handle_risk({"type":"risk","object":"motorcycle","direction":"right","risk":"HIGH","timestamp":time.time(),"extra":True})
        assert isinstance(scheduler.items[-1][0],RiskEvent)
        assistant.handle_transcript("repeat")
        assistant.handle_transcript("stop")
        assert scheduler.repeats == 1 and scheduler.stops == 1
        assistant.handle_transcript("cancel bus")
        assert assistant.state.target_bus is None
        assert assistant.handle_transcript("what is ahead")["acknowledgement"] == "No confirmed buses yet."
        feed(assistant,np.zeros((200,400,3),np.uint8))
        assert assistant.handle_transcript("what is ahead")["acknowledgement"] == "Bus 21G ahead."
        json.dumps(assistant.overlay())


@pytest.mark.parametrize("verbose",[False,True])
def test_other_bus_is_quiet_by_default(verbose):
    scheduler = RecordingScheduler()
    with Assistant(BusDetector("passthrough"),FakeOCR("23 C"),scheduler,{"21G","23C"},
                   PipelineConfig(max_fps=100,verbose=verbose)) as assistant:
        assistant.handle_transcript("find bus 21G")
        feed(assistant,np.zeros((200,400,3),np.uint8))
        assert not any(isinstance(event,BusEvent) for event,_ in scheduler.items)
        assert any(event == "Bus 23C, not yours." for event,_ in scheduler.items) == verbose


def test_no_target_tracks_routes_without_auto_speech_and_reduces_ocr():
    backend,scheduler = FakeOCR(),RecordingScheduler()
    with Assistant(BusDetector("passthrough"),backend,scheduler,{"21G"},
                   PipelineConfig(max_fps=100,ocr_every_n=3)) as assistant:
        for _ in range(7):
            feed(assistant,np.zeros((200,400,3),np.uint8))
        assert backend.calls == 3
        assert not scheduler.items
        assert assistant.handle_transcript("what is ahead")["acknowledgement"] == "Bus 21G ahead."


def test_slow_ocr_does_not_stall_frame_submission():
    backend,scheduler = FakeOCR(delay=.15),RecordingScheduler()
    with Assistant(BusDetector("passthrough"),backend,scheduler,{"21G"},
                   PipelineConfig(max_fps=1000,ocr_every_n=1)) as assistant:
        frame = np.zeros((200,400,3),np.uint8)
        delays = []
        for _ in range(30):
            time.sleep(.002)
            before = time.monotonic()
            assistant.handle_frame(frame)
            delays.append(time.monotonic()-before)
        assert max(delays) < .03
        assert assistant.wait_for_frame(timeout=2)
        assert assistant.metrics()["dropped_busy"] > 10
        assert assistant.metrics()["processed"] < assistant.metrics()["submitted"]


def test_rate_limit_and_stale_frame():
    scheduler = RecordingScheduler()
    with Assistant(BusDetector("passthrough"),FakeOCR(),scheduler,{"21G"},
                   PipelineConfig(max_fps=1)) as assistant:
        frame = np.zeros((200,400,3),np.uint8)
        first = assistant.handle_frame(frame,timestamp=time.time()-5)
        assert assistant.handle_frame(frame)["status"] == "skipped_fps"
        assert assistant.wait_for_frame(first["submitted_frame_id"])
        assert assistant.overlay()["error"] == "dropped_stale_frame"


def test_synthetic_generator_and_video(tmp_path):
    from bench.make_synthetic_boards import generate
    video = tmp_path / "pan.mp4"
    rows = generate(tmp_path,count=6,seed=21,routes={"21G","570"},video=video,seconds=.25)
    assert len(rows) == 6 and rows[0]["route"] == "21G"
    with (tmp_path / "labels.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 6
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened() and capture.read()[0]
    capture.release()


@pytest.mark.slow
def test_real_onnx_backend_is_offline(monkeypatch):
    if not (ROOT / "models/yolo11n.onnx").exists():
        pytest.skip("Local YOLO ONNX model absent")
    monkeypatch.setattr(socket.socket,"connect",lambda *a,**k: (_ for _ in ()).throw(AssertionError("Network forbidden")))
    detector = BusDetector("onnx")
    result = detector.detect(np.zeros((200,400,3),np.uint8))
    assert result == []


@pytest.mark.slow
def test_real_rapidocr_is_offline(monkeypatch):
    pytest.importorskip("rapidocr")
    if not (ROOT / "models/rapidocr/runtime_config.yaml").exists():
        pytest.skip("Local RapidOCR models absent")
    from bench.make_synthetic_boards import render_board
    from transit.ocr import RapidOCRBackend
    monkeypatch.setattr(socket.socket,"connect",lambda *a,**k: (_ for _ in ()).throw(AssertionError("Network forbidden")))
    frame,_ = render_board("21G")
    result = BoardReader(RapidOCRBackend(),{"21G"}).read(frame,(0,0,1000,380))
    assert result.match.route == "21G" and result.match.method == "exact"


@pytest.mark.slow
def test_real_ultralytics_is_offline(monkeypatch):
    import importlib.util
    if importlib.util.find_spec("ultralytics") is None or not (ROOT / "models/yolo11n.pt").exists():
        pytest.skip("Optional Ultralytics or local weights absent")
    # Block connections before importing the backend/package too.
    monkeypatch.setattr(socket.socket,"connect",lambda *a,**k: (_ for _ in ()).throw(AssertionError("Network forbidden")))
    detector = BusDetector("ultralytics")
    assert detector.detect(np.zeros((200,400,3),np.uint8)) == []


class FixedBackend:
    def __init__(self, bbox):
        self.bbox = bbox
    def detect(self, frame):
        return [Detection(self.bbox,.9,5)]


class SilentTTS:
    def synthesize(self, text):
        return np.zeros(160,np.int16),16000


def played(scheduler, text):
    return sum(row["decision"] == "played" and row["text"] == text for row in list(scheduler.decisions))


def test_new_search_is_not_blocked_by_previous_bus_cooldown():
    frame = np.zeros((200,400,3),np.uint8)
    with AudioScheduler(SilentTTS(),NullSink(duration_s=.01),SchedulerConfig(log_path=None)) as scheduler:
        with Assistant(BusDetector("passthrough"),FakeOCR(),scheduler,{"21G","23C"},
                       PipelineConfig(max_fps=100,ocr_every_n=1)) as assistant:
            for expected in (1, 2):
                if expected == 2:
                    assistant.handle_transcript("cancel bus")
                    assert not assistant.state.confirmed and not assistant.state.votes
                assistant.handle_transcript("find bus 21G")
                deadline = time.monotonic()+2
                while played(scheduler,"Bus 21G detected ahead.") < expected and time.monotonic() < deadline:
                    feed(assistant,frame)
                    scheduler.wait_idle(1)
                assert played(scheduler,"Bus 21G detected ahead.") == expected
            assert any(row["decision"] == "cleared_cooldowns" for row in list(scheduler.decisions))


def test_board_mode_reads_full_crop_first_and_bus_mode_reads_top_first():
    frame = np.zeros((200,400,3),np.uint8)
    for detector, first in ((BusDetector("passthrough"),"full"),
                            (BusDetector(FixedBackend((100,40,300,190))),"top_front")):
        with Assistant(detector,FakeOCR("HELLO"),RecordingScheduler(),{"21G"},
                       PipelineConfig(max_fps=100,ocr_every_n=1)) as assistant:
            box = feed(assistant,frame)["boxes"][0]
            assert box["ocr_attempts"][0]["crop"] == first
            assert all(attempt["crop"] == "full" for attempt in box["ocr_attempts"]) == (first == "full")


def test_ocr_lines_include_frame_space_boxes():
    with Assistant(BusDetector(FixedBackend((100,40,300,190))),FakeOCR(),RecordingScheduler(),{"21G"},
                   PipelineConfig(max_fps=100,ocr_every_n=1)) as assistant:
        box = feed(assistant,np.zeros((200,400,3),np.uint8))["boxes"][0]
    line = box["ocr_lines"][0]
    assert box["ocr_crop"] == "top_front" and box["ocr_box_space"] == "preprocessed_crop"
    assert line["box"][2] == (100,50)
    assert line["box_frame"] == pytest.approx([(100,40),(150,40),(150,65),(100,65)])


@pytest.mark.parametrize("text,detector,status,status_text",[
    ("21 G",BusDetector("passthrough"),"confirmed",None),
    ("HELLO",BusDetector("passthrough"),"route_unreadable","HELLO"),
    ("999Z",BusDetector("passthrough"),"route_not_in_gtfs","999Z"),
    ("21 G",BusDetector(SimpleNamespace(detect=lambda frame: [])),"no_bus_in_view",None)])
def test_overlay_status_explains_silence(text, detector, status, status_text):
    with Assistant(detector,FakeOCR(text),RecordingScheduler(),{"21G"},
                   PipelineConfig(max_fps=100,ocr_every_n=1)) as assistant:
        payload = feed(assistant,np.zeros((200,400,3),np.uint8))
        assert (payload["status"],payload["status_text"]) == (status,status_text)
        json.dumps(payload)


def test_overlay_status_reading_board_before_vote_confirms():
    with Assistant(BusDetector("passthrough"),FakeOCR(confidence=.5),RecordingScheduler(),{"21G"},
                   PipelineConfig(max_fps=100,ocr_every_n=1)) as assistant:
        assert feed(assistant,np.zeros((200,400,3),np.uint8))["status"] == "reading_board"
        assert feed(assistant,np.zeros((200,400,3),np.uint8))["status"] == "confirmed"


def test_target_carries_gtfs_long_names():
    info = {"21G": {"route": "21G", "variants": ["21G"], "long_names": ["Kilambakkam TO Besant Nagar"]}}
    with Assistant(BusDetector("passthrough"),FakeOCR(),RecordingScheduler(),{"21G"},
                   PipelineConfig(max_fps=100),route_info=info) as assistant:
        state = assistant.handle_transcript("find bus 21G")["state"]
    assert state["target_bus"]["long_names"] == ["Kilambakkam TO Besant Nagar"]
