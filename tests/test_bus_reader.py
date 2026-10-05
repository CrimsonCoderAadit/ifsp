import time

import numpy as np

from common.events import Track
from speech.adapter import bus_phrase, submit_risk
from transit.bus_reader import BusReader
from transit.gtfs_loader import format_route_name, load_route_metadata
from transit.ocr import OCRLine
from common.events import RiskEvent, RiskLevel


def track(track_id=7):
    return Track(track_id, 5, "bus", .92, (0., 0., 300., 180.), (150., 90.),
                 (150., 180.), 300., 180., 54000.)


class OCR:
    def __init__(self, text="21 G", conf=.91):
        self.text, self.conf, self.calls = text, conf, 0
    def read(self, image):
        self.calls += 1
        return [OCRLine(self.text, self.conf, ((0,0),(50,0),(50,20),(0,20)))]


def test_voting_expiry_and_replacement():
    ocr = OCR()
    reader = BusReader(metadata={"21G": {"route_name": "A → B"}, "570": {"route_name": "C → D"}},
                       ocr=ocr, every_n=1, ttl=30)
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    reader.update(frame, [track()], frame_index=0, now=100)
    assert reader.current(100) is None
    reader.update(frame, [track()], frame_index=1, now=101)
    assert reader.current(101)["route_number"] == "21G"
    assert reader.current(132) is None
    ocr.text = "570"
    reader.update(frame, [track(8)], frame_index=2, now=133)
    reader.update(frame, [track(8)], frame_index=3, now=134)
    assert reader.current(134)["route_number"] == "570"


def test_low_confidence_photo_clears_previous_photo_and_is_not_committed():
    ocr = OCR(conf=.91)
    reader = BusReader(metadata={"21G": {"route_name": "A → B"}, "570": {"route_name": "C → D"}}, ocr=ocr)
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    reader.read_image(frame, [track()], now=100)
    assert reader.current(100)["route_number"] == "21G"
    ocr.text, ocr.conf = "570", .5
    reader.read_image(frame, [track(8)], now=101)
    assert reader.current(101) is None


def test_unknown_numeric_board_does_not_gain_a_gtfs_suffix():
    ocr = OCR(text="576", conf=.98)
    reader = BusReader(metadata={"57G": {"route_name": "A → B"}}, ocr=ocr)
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    report = reader.read_image(frame, [track()], now=100)
    assert report["ocr_raw"] == "576"
    assert report["route_number"] is None
    assert reader.current(100) is None


class Lines:
    def __init__(self, *lines):
        self.lines = [OCRLine(text, conf, ((0, top), (80, top), (80, top + height), (0, top + height)))
                      for text, conf, top, height in lines]
    def read(self, image):
        return self.lines


def test_small_valid_route_text_does_not_beat_the_route_board(monkeypatch):
    monkeypatch.setattr("transit.bus_reader.load_route_stops",
                        lambda route: [{"route_name": f"{route} out", "stops": ["A", "B"], "trip_count": 1}])
    ocr = Lines(("102A", .88, 10, 60), ("515", .95, 90, 14))
    reader = BusReader(metadata={"102A": {"route_name": "Thiruvanmiyur → Pudupakkam Rd.Jn"},
                                 "515": {"route_name": "Tambaram West → Mamallapuram"}}, ocr=ocr)
    report = reader.read_image(np.zeros((200, 320, 3), dtype=np.uint8), [track()], now=100)
    assert report["route_number"] == "102A"
    assert report["route_stops"][0]["route_name"] == "102A out"
    assert reader.current(100)["route_number"] == "102A"


def test_uncertain_route_read_stays_unconfirmed(monkeypatch):
    monkeypatch.setattr("transit.bus_reader.load_route_stops",
                        lambda route: (_ for _ in ()).throw(AssertionError("stops loaded for unconfirmed route")))
    reader = BusReader(metadata={"515": {"route_name": "Tambaram West → Mamallapuram"}},
                       ocr=Lines(("515", .6, 10, 40)))
    report = reader.read_image(np.zeros((200, 320, 3), dtype=np.uint8), [track()], now=100)
    assert report["ocr_raw"] == "515"
    assert (report["route_number"], report["normalized"], report["route_name"], report["route_stops"]) == (None, None, "", [])
    assert reader.current(100) is None


def test_speech_and_gtfs_name():
    assert format_route_name("Kilambakkam Bus Terminus TO Besant Nagar Terminus") == "Kilambakkam → Besant Nagar"
    assert bus_phrase({"route_number": "21G", "route_name": "Kilambakkam → Besant Nagar"}) == "Bus 21 G, Kilambakkam to Besant Nagar."
    assert bus_phrase(None) == "No bus identified."


def test_gtfs_canonical_name_uses_trip_count(tmp_path):
    from zipfile import ZipFile
    archive = tmp_path / "gtfs.zip"
    with ZipFile(archive, "w") as zipfile:
        zipfile.writestr("routes.txt", "route_id,route_type,route_short_name,route_long_name\n"
                         "main,3,21G,Kilambakkam Bus Terminus TO Besant Nagar Terminus\n"
                         "cut,3,21G CT,Tambaram TO Broadway\n"
                         "single,3,M70,Guindy TO Thiruvanmiyur\n")
        zipfile.writestr("trips.txt", "trip_id,route_id\n1,main\n2,main\n3,main\n4,cut\n5,single\n")
    data = load_route_metadata(zip_path=archive)
    assert data["21G"]["route_name"] == "Kilambakkam → Besant Nagar"
    assert data["M70"]["route_name"] == "Guindy → Thiruvanmiyur"
    assert "570" not in data


def test_only_high_risk_is_submitted():
    class Scheduler:
        def __init__(self): self.calls = []
        def submit(self, text, **kwargs): self.calls.append((text, kwargs))
    from dataclasses import replace
    event = RiskEvent(1, "car", .9, "RIGHT", True, 1., 1., 1., .8, RiskLevel.MEDIUM, time.time())
    sink = Scheduler()
    assert submit_risk(sink, event) is None
    submit_risk(sink, replace(event, risk_level=RiskLevel.HIGH))
    assert sink.calls[0][0] == "Car right."
