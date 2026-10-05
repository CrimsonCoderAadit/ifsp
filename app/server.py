"""Single-page local Rider server with one shared YOLO detector instance."""
from __future__ import annotations

import base64
import asyncio
import io
import os
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import ROOT, SpeechConfig
from perception.detector_tracker import DetectorTracker
from speech.adapter import bus_phrase
from speech.stt_keyword import BusKeyword
from speech.tts import SpeechSynthesizer
from transit.bus_reader import BusReader

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Bus Finder")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Runtime:
    def __init__(self):
        self.detector = None
        self.reader = None
        self.stt = None
        self.tts = None
        self.detector_lock = threading.RLock()

    def vision(self):
        if self.detector is None:
            self.detector = DetectorTracker(os.getenv("BUS_FINDER_MODEL", "yolo26n.pt"), ["bus", "truck"])
        if self.reader is None:
            self.reader = BusReader()
        return self.detector, self.reader

    def speech(self):
        if self.stt is None:
            self.stt = BusKeyword(SpeechConfig().vosk_model)
        if self.tts is None:
            self.tts = SpeechSynthesizer()
        return self.stt, self.tts


runtime = Runtime()
_risk_subscribers = set()
_loop = None


def publish_risk(event):
    """Called by the integrated video loop; enqueue a banner for web clients."""
    if _loop is None:
        return
    payload = {"type": "risk", "direction": event.direction,
               "text": f"{event.object_class.title()} {event.direction.lower()}",
               "level": event.risk_level.value}
    def enqueue():
        for queue in tuple(_risk_subscribers):
            queue.put_nowait(payload)
    _loop.call_soon_threadsafe(enqueue)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "detector_loaded": runtime.detector is not None,
            "ocr_loaded": runtime.reader is not None, "speech_loaded": runtime.stt is not None}


@app.get("/api/current_bus")
def current_bus():
    return {"current_bus": runtime.reader.current() if runtime.reader else None}


def analyze_frame(frame, *, video=False, frame_index=0):
    detector, reader = runtime.vision()
    with runtime.detector_lock:
        if video:
            reports = reader.update(frame, detector.track(frame), frame_index=frame_index)
            result = max(reports, key=lambda item: item["confidence"], default=None)
        else:
            result = reader.read_image(frame, detector.detect(frame))
    return result or {"detected": False, "ocr_raw": "", "normalized": None,
                      "route_number": None, "route_name": "", "confidence": 0.,
                      "bus_box": None, "board_box": None, "ocr_ms": 0.}


@app.post("/api/image")
async def image(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 100_000_000:
        return JSONResponse({"error": "File is too large"}, status_code=413)
    if (file.content_type or "").startswith("video/"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "video.mp4").suffix) as temporary:
            temporary.write(content)
            temporary.flush()
            capture = cv2.VideoCapture(temporary.name)
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.
            step = max(1, round(fps / 2))
            best = None
            frame_index = 0
            sampled = 0
            try:
                while sampled < 120:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % step == 0:
                        report = analyze_frame(frame, video=True, frame_index=frame_index)
                        if best is None or report["confidence"] > best["confidence"]:
                            best = report
                        sampled += 1
                    frame_index += 1
            finally:
                capture.release()
        return {"result": best or {"detected": False}, "current_bus": runtime.reader.current()}
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)
    result = analyze_frame(frame)
    return {"result": result, "current_bus": runtime.reader.current()}


def wav_base64(pcm, sr):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return base64.b64encode(stream.getvalue()).decode("ascii")


@app.websocket("/ws")
async def websocket(socket: WebSocket):
    global _loop
    _loop = asyncio.get_running_loop()
    await socket.accept()
    queue = asyncio.Queue()
    _risk_subscribers.add(queue)
    async def send_risks():
        while True:
            await socket.send_json(await queue.get())
    risk_task = asyncio.create_task(send_risks())
    try:
        while True:
            try:
                message = await socket.receive()
            except Exception:
                break
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                started = time.perf_counter()
                stt, tts = runtime.speech()
                with wave.open(io.BytesIO(message["bytes"]), "rb") as wav:
                    pcm = wav.readframes(wav.getnframes())
                    sample_rate = wav.getframerate()
                stt.start(sample_rate)
                found_during_feed = stt.feed(pcm)
                heard = stt.finish() or found_during_feed
                if heard:
                    caption = bus_phrase(runtime.reader.current() if runtime.reader else None)
                    audio, sr = tts.synthesize(caption)
                    await socket.send_json({"type": "speech", "caption": caption,
                                            "audio": wav_base64(audio, sr),
                                            "latency_ms": (time.perf_counter()-started)*1000})
                else:
                    await socket.send_json({"type": "keyword", "heard": False})
    finally:
        risk_task.cancel()
        _risk_subscribers.discard(queue)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8001)
