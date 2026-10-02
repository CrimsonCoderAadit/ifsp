"""Offline measurement runner: python -m bench.benchmark --device-label laptop."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

from core.config import ROOT, LOW_POWER_DETECTOR_MODEL, low_power_speech_config
from core.events import BusEvent
from speech.intents import parse_intent
from speech.scheduler import AudioScheduler, LocalSink, NullSink, SchedulerConfig
from speech.stt import SpeechRecognizer, resample_pcm
from speech.tts import SpeechSynthesizer
from transit.bus_detector import BusDetector
from transit.gtfs_loader import load_routes
from transit.ocr import BoardReader, board_crops, create_ocr_backend, preprocessing_variants
from transit.route_matcher import extract_route_candidates, match_route

FIELDS = ("component", "backend", "test_case", "latency_ms", "success", "notes", "timestamp")


def record(rows, device, component, backend, case, ms=None, success="", **notes):
    rows.append({"component": component, "backend": backend, "test_case": case,
                 "latency_ms": "" if ms is None else f"{ms:.3f}", "success": success,
                 "notes": json.dumps({"device": device, "system": platform.system(),
                                      "machine": platform.machine(), **notes},
                                     ensure_ascii=False, sort_keys=True),
                 "timestamp": datetime.now(timezone.utc).isoformat()})


def labels(directory):
    path = directory / "labels.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["filename"]: row for row in csv.DictReader(stream) if row.get("filename")}


def wav_pcm16(path):
    with wave.open(str(path)) as stream:
        if stream.getsampwidth() != 2:
            raise ValueError(f"Expected PCM16 WAV: {path}")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        if stream.getnchannels() > 1:
            samples = samples.reshape(-1, stream.getnchannels())
        return resample_pcm(samples, stream.getframerate()).tobytes()


def stt(rows, device):
    recognizer = SpeechRecognizer()
    for directory in (ROOT/"data/test_audio", ROOT/"data/test_audio/real"):
        expected = labels(directory)
        for path in sorted(directory.glob("*.wav")):
            pcm = wav_pcm16(path)
            command = expected.get(path.name, {}).get("command")
            if not command and directory.name != "real":
                command = path.stem.replace("_", " ")
            recognizer.start()
            for offset in range(0, len(pcm), 3200):
                recognizer.feed(pcm[offset:offset+3200])
            endpoint = time.perf_counter()
            transcript = recognizer.finish(endpoint)
            got = parse_intent(transcript)
            wanted = parse_intent(command) if command else None
            good = (got.type, got.target) == (wanted.type, wanted.target) if wanted else None
            record(rows, device, "stt", type(recognizer.backend).__name__,
                   f"{directory.name}/{path.name}", transcript.latency_ms,
                   "correct" if good else "wrong" if good is False else "unlabelled",
                   transcript=transcript.text, expected=command, intent=got.type.value,
                   target=got.target, confidence=round(transcript.confidence, 3),
                   timing="buffered feed; end-of-speech to final transcript")
    if not list((ROOT/"data/test_audio/real").glob("*.wav")):
        record(rows, device, "stt", type(recognizer.backend).__name__, "real/summary",
               success="unavailable", reason="no real WAV recordings")


def tts_and_sink(rows, device, low_power=False):
    tts = SpeechSynthesizer(low_power_speech_config() if low_power else None)
    record(rows, device, "tts", tts.backend_name, "model_load", tts.load_time_ms, "ok")
    phrase = "Benchmark bus 21G approaching ahead."
    for case in ("first_synthesis", "cached_synthesis"):
        started = time.perf_counter()
        tts.synthesize(phrase)
        record(rows, device, "tts", tts.backend_name, case,
               (time.perf_counter()-started)*1000, "ok")
    try:
        sink = LocalSink()
        tts.synthesize("Vehicle approaching right.")
        with AudioScheduler(tts, sink, SchedulerConfig(log_path=None)) as scheduler:
            submitted = time.monotonic()
            item_id = scheduler.submit({"type":"risk", "object":"car", "direction":"right",
                                        "risk":"HIGH", "timestamp":time.time()})
            deadline = submitted+5
            while time.monotonic() < deadline:
                decisions = list(scheduler.decisions)
                played = next((row for row in decisions if row["item_id"] == item_id
                               and row["decision"] == "played"), None)
                if played:
                    record(rows, device, "tts", "LocalSink", "event_to_audio_start",
                           (played["monotonic"]-submitted)*1000, "ok",
                           output_latency_ms=round(sink.output_latency_ms,3),
                           block_ms=sink.block_duration_ms, acoustic_start="not measured")
                    break
                if scheduler.last_error or any(row["decision"] == "sink_error" for row in decisions):
                    raise RuntimeError(scheduler.last_error or "LocalSink playback failed")
                time.sleep(.005)
            else:
                raise TimeoutError("LocalSink playback did not start")
            sink.stop()
    except Exception as error:
        record(rows, device, "tts", "LocalSink", "event_to_audio_start",
               success="unavailable", error=str(error))
    return tts


def verdict(route, expected, method):
    if route is None:
        return "none"
    if route != expected:
        return "wrong"
    return "exact" if method == "exact" else "fuzzy-correct"


def failure_types(label):
    raw = label.get("failure_type") or label.get("distortions") or ""
    if not raw:
        return ["unclassified"]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(value).strip().lower() for value in parsed] or ["clean"]
    except (ValueError, TypeError):
        pass
    return [part.strip().lower() for part in raw.replace("|", ";").split(";")
            if part.strip()] or ["unclassified"]


def labelled_bbox(label, frame):
    value = label.get("bbox", "").strip()
    if not value:
        return None
    coords = json.loads(value)
    if len(coords) != 4:
        raise ValueError("bbox must have four coordinates")
    x1, y1, x2, y2 = map(float, coords)
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        raise ValueError("bbox outside image")
    return (x1, y1, x2, y2)


def print_confusion(rows, dataset):
    groups = defaultdict(list)
    for row in rows:
        if row["component"] != "ocr" or not row["test_case"].startswith(dataset+"/") or not row["test_case"].endswith("/pipeline"):
            continue
        note = json.loads(row["notes"])
        for failure in note.get("failure_types", ["unclassified"]):
            groups[failure].append(row)
    print(f"{dataset} confusion by failure type:")
    if not groups:
        print("  N/A (no labelled images)")
    for failure, entries in sorted(groups.items()):
        counts = Counter(row["success"] for row in entries)
        wrong = Counter((json.loads(row["notes"])["expected"],
                         json.loads(row["notes"]).get("route") or "∅")
                        for row in entries if row["success"] in {"wrong", "none"})
        print(f"  {failure}: n={len(entries)} exact={counts['exact']} "
              f"fuzzy-correct={counts['fuzzy-correct']} wrong={counts['wrong']} "
              f"none={counts['none']} confusions={dict(wrong)}")


def ocr(rows, device, routes):
    backend = create_ocr_backend()
    name = type(backend).__name__
    real_detector = None
    for dataset in ("synthetic", "real"):
        directory = ROOT/"data/test_images"/dataset
        catalogue = labels(directory)
        images = sorted(path for path in (directory.iterdir() if directory.exists() else ())
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        and path.name in catalogue)
        if not images:
            record(rows, device, "ocr", name, f"{dataset}/summary",
                   success="unavailable", images=0)
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None:
                record(rows, device, "ocr", name, f"{dataset}/{path.name}",
                       success="unavailable", error="image decode failed")
                continue
            expected = catalogue[path.name]["route"].strip().upper()
            tags = failure_types(catalogue[path.name])
            bbox = labelled_bbox(catalogue[path.name], frame)
            if bbox is None and dataset == "real":
                real_detector = real_detector or BusDetector("auto")
                candidates = real_detector.detect(frame)
                bbox = max(candidates, key=lambda item: item.conf).bbox if candidates else None
            if bbox is None and dataset == "real":
                record(rows, device, "ocr", name, f"{dataset}/{path.name}/pipeline",
                       success="none", expected=expected, route=None,
                       failure_types=tags, reason="detector found no bus")
                continue
            board = bbox is None
            bbox = bbox or (0, 0, frame.shape[1], frame.shape[0])
            if board:
                started = time.perf_counter()
                BoardReader(backend, routes).read(frame, bbox)
                record(rows, device, "ocr", name, f"{dataset}/{path.name}/pipeline_top_first",
                       (time.perf_counter()-started)*1000, "measured",
                       timing="same close-up read in bus mode (top_front crop tried first)")
            started = time.perf_counter()
            read = BoardReader(backend, routes).read(frame, bbox, board=board)
            record(rows, device, "ocr", name, f"{dataset}/{path.name}/pipeline",
                   (time.perf_counter()-started)*1000,
                   verdict(read.match.route, expected, read.match.method),
                   expected=expected, route=read.match.route, method=read.match.method,
                   style=catalogue[path.name].get("style"),
                   failure_types=tags, bbox=bbox,
                   attempts=len(read.attempts), texts=[line.text for line in read.lines],
                   board_mode=board, first_crop=read.attempts[0]["crop"] if read.attempts else None)
            for crop_name, crop, _ in board_crops(frame, bbox):
                for variant_name, prepared in preprocessing_variants(crop):
                    started = time.perf_counter()
                    lines = backend.read(prepared)
                    candidates = extract_route_candidates(
                        [{"text": line.text, "box": line.box} for line in lines])
                    candidates += [line.text for line in lines
                                   if line.text.upper().strip() in routes]
                    matched = None
                    for candidate in dict.fromkeys(candidates):
                        possible = match_route(candidate, routes)
                        if possible.route:
                            matched = possible
                            break
                    record(rows, device, "ocr_variant", name,
                           f"{dataset}/{path.name}/{crop_name}/{variant_name}",
                           (time.perf_counter()-started)*1000,
                           verdict(matched.route if matched else None, expected,
                                   matched.method if matched else "none"),
                           expected=expected, route=matched.route if matched else None,
                           method=matched.method if matched else "none",
                           texts=[line.text for line in lines],
                           timing="OCR backend read plus route match")


def detector(rows, device, low_power=False):
    model = (BusDetector("onnx", model=LOW_POWER_DETECTOR_MODEL)
             if low_power else BusDetector("auto"))
    capture = cv2.VideoCapture(str(ROOT/"data/test_video.mp4"))
    if not capture.isOpened():
        record(rows, device, "detector", model.backend_name, "video",
               success="unavailable", error="test video missing")
    else:
        try:
            for index in range(30):
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.perf_counter()
                found = model.detect(frame)
                record(rows, device, "detector", model.backend_name,
                       f"synthetic_video/frame_{index:03}",
                       (time.perf_counter()-started)*1000, "measured",
                       detections=len(found), synthetic_video=True)
        finally:
            capture.release()


def scheduler_spam(rows, device, tts):
    tts.synthesize("Vehicle approaching right.")
    for enabled in (True, False):
        config = SchedulerConfig(log_path=None)
        if not enabled:
            config.cooldowns.update({priority: 0 for priority in config.cooldowns})
        case = f"spam_30fps_5s/cooldown_{'on' if enabled else 'off'}"
        with AudioScheduler(tts, NullSink(duration_s=.01), config) as scheduler:
            started = time.monotonic()
            for index in range(150):
                scheduler.submit({"type":"risk", "object":"car", "direction":"right",
                                  "risk":"HIGH", "timestamp":time.time()})
                time.sleep(max(0, started+(index+1)/30-time.monotonic()))
            scheduler.wait_idle(5)
            counts = Counter(row["decision"] for row in list(scheduler.decisions))
            record(rows, device, "scheduler", "AudioScheduler+NullSink", case,
                   (time.monotonic()-started)*1000, "ok", submitted=150,
                   played=counts["played"], suppressed_cooldown=counts["suppressed_cooldown"],
                   deduped=counts["deduped"], null_sink=True)


def bus_spam(rows, device, tts):
    """The pipeline submits a BUS event on every processed frame while the target is confirmed."""
    text = "Bus 21G detected ahead."
    tts.synthesize(text)
    for enabled in (True, False):
        config = SchedulerConfig(log_path=None)
        if not enabled:
            config.cooldowns.update({priority: 0 for priority in config.cooldowns})
        with AudioScheduler(tts, NullSink(duration_s=.01), config) as scheduler:
            started = time.monotonic()
            for index in range(150):
                scheduler.submit(BusEvent("21G", .99, (0, 0, 10, 10), time.time()), text_override=text)
                time.sleep(max(0, started+(index+1)/30-time.monotonic()))
            scheduler.wait_idle(5)
            counts = Counter(row["decision"] for row in list(scheduler.decisions))
            record(rows, device, "scheduler", "AudioScheduler+NullSink",
                   f"bus_spam_30fps_5s/cooldown_{'on' if enabled else 'off'}",
                   (time.monotonic()-started)*1000, "ok", submitted=150,
                   played=counts["played"], repeated=max(0, counts["played"]-1),
                   suppressed_cooldown=counts["suppressed_cooldown"],
                   deduped=counts["deduped"], null_sink=True)


def print_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["latency_ms"]:
            groups[(row["component"], row["backend"])].append(float(row["latency_ms"]))
    print(f"{'Component':<16} {'Backend':<27} {'n':>4} {'p50 ms':>9} {'p95 ms':>9}")
    for (component, backend), values in sorted(groups.items()):
        print(f"{component:<16} {backend:<27} {len(values):>4} "
              f"{np.percentile(values,50):>9.1f} {np.percentile(values,95):>9.1f}")
    stt_rows = [row for row in rows if row["component"] == "stt"
                and row["success"] in {"correct", "wrong"}]
    print("STT command accuracy:", f"{sum(row['success']=='correct' for row in stt_rows)}/{len(stt_rows)}"
          if stt_rows else "N/A")
    for dataset in ("synthetic", "real"):
        subset = [row for row in rows if row["component"] == "ocr"
                  and row["test_case"].startswith(dataset+"/")
                  and row["test_case"].endswith("/pipeline")]
        print(f"OCR {dataset}:", dict(Counter(row["success"] for row in subset))
              if subset else "N/A (no labelled images)")
        if not subset and dataset == "real":
            print("!!! NO REAL IMAGES: data/test_images/real/ has no labelled photos. "
                  "Route accuracy on real Chennai buses is UNMEASURED. !!!")
        compare = {row["test_case"].rsplit("/", 1)[0]: float(row["latency_ms"]) for row in rows
                   if row["test_case"].startswith(dataset+"/") and row["test_case"].endswith("/pipeline_top_first")}
        board = [float(row["latency_ms"]) - compare[row["test_case"].rsplit("/", 1)[0]] for row in subset
                 if row["test_case"].rsplit("/", 1)[0] in compare and row["latency_ms"]]
        if board:
            print(f"OCR {dataset} board-mode full-first minus top-first: "
                  f"p50 {np.percentile(board, 50):.1f} ms over {len(board)} images")
        print_confusion(rows, dataset)
    for row in rows:
        if row["component"] == "scheduler":
            print(row["test_case"], row["notes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-label", choices=("laptop", "pi"), required=True)
    parser.add_argument("--low-power", action="store_true",
                        help="Measure 320px detector and low Piper voice on this device")
    parser.add_argument("--output", type=str, default=str(ROOT/"logs/latency.csv"))
    args = parser.parse_args()
    rows = []
    stt(rows, args.device_label)
    synthesizer = tts_and_sink(rows, args.device_label, args.low_power)
    ocr(rows, args.device_label, load_routes())
    detector(rows, args.device_label, args.low_power)
    if args.low_power:
        record(rows, args.device_label, "profile", "low_power", "ocr_cadence",
               success="configured", every_n_frames=5, max_fps=5)
    scheduler_spam(rows, args.device_label, synthesizer)
    bus_spam(rows, args.device_label, synthesizer)
    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
