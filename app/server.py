"""Local debug console. Setup downloads are separate; runtime never downloads."""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import os
import subprocess
import threading
import time
import wave
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cli.demo import draw_overlay
from core.config import (ROOT, LOW_POWER_DETECTOR_MODEL, PipelineConfig,
                         low_power_pipeline_config, low_power_speech_config)
from core.pipeline import Assistant
from speech.scheduler import AudioScheduler, NullSink, SchedulerConfig
from speech.stt import SpeechRecognizer
from speech.tts import SpeechSynthesizer
from transit.bus_detector import BusDetector, PassthroughBackend
from transit.gtfs_loader import load_route_metadata, load_routes
from transit.ocr import create_ocr_backend

MAX_IMAGE_BYTES = 4 * 1024 * 1024


class Metrics:
    def __init__(self):
        self.values = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    def record(self, name, value):
        with self.lock:
            self.values[name].append(float(value))
    def summary(self):
        with self.lock:
            return {name: {"last": values[-1] if values else None,
                           "p50": float(np.percentile(values,50)) if values else None,
                           "p95": float(np.percentile(values,95)) if values else None,
                           "count": len(values)}
                    for name in ("stt","ocr","tts","detector")
                    for values in [list(self.values[name])]}


class MeasuredBackend:
    """Shared model with serialized calls; trackers/decoders remain per client."""
    def __init__(self, backend, method, metrics, name, *, serialize=True):
        self.backend, self.method, self.metrics, self.name = backend, method, metrics, name
        self.lock = threading.Lock()
        self.serialize = serialize
    def __getattr__(self, name):
        if name != self.method:
            return getattr(self.backend,name)
        def call(*args, **kwargs):
            with self.lock if self.serialize else contextlib.nullcontext():
                began = time.perf_counter()
                try:
                    return getattr(self.backend,name)(*args,**kwargs)
                finally:
                    self.metrics.record(self.name,(time.perf_counter()-began)*1000)
        return call


class ModelBundle:
    def __init__(self, detector="auto", *, low_power=False, approaching_only=False):
        self.metrics = Metrics()
        self.load_times = {}
        self.pipeline_config = (PipelineConfig(max_fps=5,ocr_every_n=5,
                                               approaching_only=approaching_only)
                                if low_power else
                                PipelineConfig(max_fps=5,ocr_every_n=1,
                                               approaching_only=approaching_only))
        def load(name, factory):
            began = time.perf_counter()
            result = factory()
            self.load_times[name] = (time.perf_counter()-began)*1000
            print(f"LOAD {name}={self.load_times[name]:.1f}ms",flush=True)
            return result
        self.routes = load("routes",load_routes)
        self.route_info = load("route_metadata",load_route_metadata)
        self.stt = load("stt",SpeechRecognizer)
        self.tts = MeasuredBackend(load("tts",lambda: SpeechSynthesizer(
            low_power_speech_config() if low_power else None)),"synthesize",self.metrics,"tts",serialize=False)
        self.ocr = MeasuredBackend(load("ocr",create_ocr_backend),"read",self.metrics,"ocr")
        detector_choice = "onnx" if low_power and detector == "auto" else detector
        model_path = LOW_POWER_DETECTOR_MODEL if low_power and detector_choice == "onnx" else None
        loaded = load("detector",lambda: BusDetector(detector_choice,model=model_path))
        self.detector_fallback_reason = loaded.fallback_reason
        self.detector = MeasuredBackend(loaded.backend,"detect",self.metrics,"detector")
        for phrase in ("Vehicle approaching right.","Motorcycle left!","Person ahead."):
            self.tts.synthesize(phrase)
    def recognizer(self):
        return SpeechRecognizer(backend=self.stt.backend.new_session())
    def health(self):
        tts, ocr, detector = self.tts.backend, self.ocr.backend, self.detector.backend
        stt_backend = self.stt.backend
        return {"models": {
                    "stt": {"backend": type(stt_backend).__name__,
                            "model": Path(getattr(getattr(stt_backend,"config",None),"vosk_model","")).name or None},
                    "tts": {"backend": tts.backend_name,
                            "model": Path(tts.config.piper_model).name if tts.backend_name == "PiperTTS" else None,
                            "fallback_reason": tts.fallback_reason},
                    "ocr": {"backend": type(ocr).__name__, "model": getattr(ocr,"model_names",None),
                            "fallback_reason": getattr(ocr,"fallback_reason",None)},
                    "detector": {"backend": type(detector).__name__,
                                 "model": Path(detector.model_path).name if getattr(detector,"model_path",None) else None,
                                 "fallback_reason": self.detector_fallback_reason}},
                "load_time_ms": self.load_times, "routes": len(self.routes)}


def git_commit():
    if os.environ.get("GIT_COMMIT"):
        return {"commit": os.environ["GIT_COMMIT"], "dirty": None}
    try:
        commit = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,timeout=2,check=True).stdout.strip()
        dirty = bool(subprocess.run(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,
                                    capture_output=True,text=True,timeout=2,check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


class SessionDetector:
    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()
        self.detector = BusDetector(backend)
    def set_mode(self, board):
        with self.lock:
            self.detector = BusDetector(PassthroughBackend() if board else self.backend)
    def detect(self, frame):
        with self.lock:
            return self.detector.detect(frame)
    @property
    def tracker(self):
        return self.detector.tracker
    @property
    def board_mode(self):
        with self.lock:
            return self.detector.board_mode


class WebSink:
    """Thread-safe browser sink; Python owns priority/pre-emption decisions."""
    def __init__(self, emit, *, clock=time.monotonic, grace_seconds=.75):
        self.emit, self.clock, self.grace = emit, clock, grace_seconds
        self.lock = threading.Lock()
        self.current = None
        self.deadline = 0
        self.item = None
        self.interrupted = False
        self.closed = False
    def prepare_item(self, item):
        self.item = item
    def play(self, pcm, sr, item_id):
        buffer = io.BytesIO()
        with wave.open(buffer,"wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(np.asarray(pcm,dtype="<i2").tobytes())
        with self.lock:
            if self.closed:
                raise RuntimeError("Browser disconnected")
            self.current = item_id
            self.deadline = self.clock()+len(pcm)/sr+self.grace
            interrupt, self.interrupted = self.interrupted, False
        self.emit({"type":"speak","id":item_id,"text":self.item.text,
                   "priority":int(self.item.priority),"interrupt":interrupt,
                   "audio_b64_wav":base64.b64encode(buffer.getvalue()).decode()})
    def stop(self):
        with self.lock:
            self.interrupted = self.current is not None
            self.current = None
        self.emit({"type":"stop_audio"})
    def playback_done(self, item_id):
        with self.lock:
            if item_id == self.current:
                self.current = None
    @property
    def is_playing(self):
        with self.lock:
            if self.current and self.clock() >= self.deadline:
                self.current = None
            return self.current is not None
    def close(self):
        self.stop()
        self.closed = True


def decode_image(data):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be between 1 byte and 4 MiB")
    frame = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if frame is None or frame.shape[0]*frame.shape[1] > 8_000_000:
        raise ValueError("Invalid image or image exceeds 8 megapixels")
    return frame


def create_app(bundle_factory=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.models = await asyncio.to_thread(bundle_factory or ModelBundle)
        app.state.git = await asyncio.to_thread(git_commit)
        yield
    application = FastAPI(lifespan=lifespan)
    application.mount("/static",StaticFiles(directory=ROOT/"app/static"),name="static")

    @application.middleware("http")
    async def revalidate_ui(request, call_next):
        # ES modules are fetched individually; a stale cached module breaks the whole page.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @application.get("/")
    async def index():
        return FileResponse(ROOT/"app/static/index.html")

    @application.get("/api/routes")
    async def routes(q: str = ""):
        catalogue = application.state.models.routes
        return {"routes":sorted(route for route in catalogue if q.upper().strip() in route)[:100]}

    @application.get("/api/route/{route}")
    async def route(route: str):
        models = application.state.models
        key = route.upper().strip()
        if key not in models.routes:
            raise HTTPException(404,f"Unknown route {route}")
        info = getattr(models,"route_info",{}).get(key,{})
        return {"route":key,"variants":info.get("variants",[key]),"long_names":info.get("long_names",[])}

    @application.get("/api/health")
    async def health():
        models = application.state.models
        report = models.health() if hasattr(models,"health") else {"load_time_ms":models.load_times,"routes":len(models.routes)}
        return {"status":"ok",**report,"git":application.state.git}

    @application.get("/api/metrics")
    async def metrics():
        models = application.state.models
        return {"latency_ms":models.metrics.summary(),"load_time_ms":models.load_times}

    @application.post("/api/image")
    async def image(file: UploadFile = File(...), command: str = Form(""), board: bool = Form(False)):
        models = application.state.models
        try:
            frame = await asyncio.to_thread(decode_image,await file.read(MAX_IMAGE_BYTES+1))
        except ValueError as error:
            raise HTTPException(400,str(error)) from error
        def run():
            with AudioScheduler(models.tts,NullSink(),SchedulerConfig(log_path=None)) as scheduler:
                detector = BusDetector(PassthroughBackend() if board else models.detector)
                with Assistant(detector,models.ocr,scheduler,models.routes,
                               PipelineConfig(ocr_every_n=1,max_fps=100),
                               route_info=getattr(models,"route_info",{})) as assistant:
                    if command:
                        assistant.handle_transcript(command)
                    for _ in range(3):
                        payload = assistant.handle_frame(frame)
                        if not assistant.wait_for_frame(payload["submitted_frame_id"],15):
                            raise TimeoutError("Still-image processing timed out")
                        payload = assistant.overlay()
                        if payload["matched_route"] or payload["error"]:
                            break
                        time.sleep(.011)
                    output = draw_overlay(frame,payload)
                    ok, jpeg = cv2.imencode(".jpg",output)
                    if not ok:
                        raise RuntimeError("JPEG encoding failed")
                    return {"overlay":payload,"annotated_jpeg_b64":base64.b64encode(jpeg).decode()}
        return await asyncio.to_thread(run)

    @application.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        models = application.state.models
        loop = asyncio.get_running_loop()
        outbound = asyncio.Queue(maxsize=256)
        disconnected = False
        def emit(message):
            def enqueue():
                if disconnected:
                    return
                try:
                    outbound.put_nowait(message)
                except asyncio.QueueFull:
                    asyncio.create_task(ws.close(code=1013,reason="Client is too slow"))
            loop.call_soon_threadsafe(enqueue)
        sink = WebSink(emit)
        scheduler = AudioScheduler(models.tts,sink,SchedulerConfig(log_path=ROOT/"logs/web_latency.csv"))
        detector = SessionDetector(models.detector)
        assistant = Assistant(detector,models.ocr,scheduler,models.routes,
                              getattr(models,"pipeline_config",PipelineConfig(max_fps=5,ocr_every_n=1)),
                              route_info=getattr(models,"route_info",{}))
        recognizer = models.recognizer()
        frame_task = None
        audio_active = False
        audio_bytes = 0
        next_binary_frame = False
        async def sender():
            while True:
                await ws.send_json(await outbound.get())
        async def monitor():
            while True:
                emit({"type":"state",**assistant.overlay()["state"],"scheduler":scheduler.snapshot()})
                emit({"type":"metrics","latency_ms":models.metrics.summary(),"pipeline":assistant.metrics()})
                await asyncio.sleep(.5)
        async def process_frame(data, request_id):
            try:
                frame = await asyncio.to_thread(decode_image,data)
                submitted = assistant.handle_frame(frame)
                if submitted["submitted_frame_id"] is not None:
                    if not await asyncio.to_thread(assistant.wait_for_frame,submitted["submitted_frame_id"],15):
                        raise TimeoutError("Frame processing timed out")
                emit({"type":"overlay",**assistant.overlay(),"request_id":request_id})
            except Exception as error:
                emit({"type":"error","message":str(error)})
            finally:
                emit({"type":"frame_ack","id":request_id})
        async def command(text):
            result = await asyncio.to_thread(assistant.handle_transcript,text)
            emit({"type":"intent",**result})
            emit({"type":"state",**result["state"],"scheduler":scheduler.snapshot()})
        sending, monitoring = asyncio.create_task(sender()), asyncio.create_task(monitor())
        emit({"type":"ready","load_time_ms":models.load_times,"detector":type(models.detector.backend).__name__})
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                try:
                    binary = message.get("bytes")
                    if binary is not None:
                        if next_binary_frame is not False:
                            request_id, next_binary_frame = next_binary_frame, False
                            if frame_task and not frame_task.done():
                                emit({"type":"frame_ack","id":request_id,"dropped":True})
                            else:
                                frame_task = asyncio.create_task(process_frame(binary,request_id))
                        elif audio_active:
                            audio_bytes += len(binary)
                            if audio_bytes > 16000*2*30:
                                audio_active = False
                                recognizer.backend.recognizer = None
                                raise ValueError("Audio limit is 30 seconds")
                            partial = await asyncio.to_thread(recognizer.feed,binary)
                            emit({"type":"partial_transcript","text":partial})
                        else:
                            raise ValueError("Tag JPEG with frame, or start audio before PCM chunks")
                        continue
                    control = json.loads(message.get("text") or "{}")
                    kind = control.get("type")
                    if kind == "text_command":
                        await command(str(control.get("text",""))[:500])
                    elif kind == "audio_start":
                        if audio_active:
                            raise ValueError("Audio stream already active")
                        await asyncio.to_thread(recognizer.start,16000)
                        audio_active, audio_bytes = True, 0
                        emit({"type":"partial_transcript","text":""})
                    elif kind == "audio_end":
                        endpoint = time.perf_counter()
                        if not audio_active:
                            raise ValueError("No active audio stream")
                        audio_active = False
                        transcript = await asyncio.to_thread(recognizer.finish,endpoint)
                        models.metrics.record("stt",transcript.latency_ms)
                        emit({"type":"transcript","text":transcript.text,"confidence":transcript.confidence,"latency_ms":transcript.latency_ms})
                        await command(transcript)
                    elif kind == "frame":
                        request_id = control.get("id")
                        if control.get("jpeg_b64"):
                            data = base64.b64decode(control["jpeg_b64"],validate=True)
                            if frame_task and not frame_task.done():
                                emit({"type":"frame_ack","id":request_id,"dropped":True})
                            else:
                                frame_task = asyncio.create_task(process_frame(data,request_id))
                        else:
                            next_binary_frame = request_id
                    elif kind == "source_mode":
                        if control.get("mode") not in {"board","bus"}:
                            raise ValueError("Mode must be board or bus")
                        await asyncio.to_thread(detector.set_mode,control["mode"] == "board")
                    elif kind == "mock_risk":
                        event = dict(control)
                        event.update(type="risk",timestamp=time.time())
                        assistant.handle_risk(event)
                    elif kind == "playback_done":
                        sink.playback_done(control.get("id"))
                    else:
                        raise ValueError("Unknown control message")
                except (ValueError,KeyError,TypeError,RuntimeError) as error:
                    emit({"type":"error","message":str(error)})
        except WebSocketDisconnect:
            pass
        finally:
            disconnected = True
            for task in (sending,monitoring):
                task.cancel()
            await asyncio.gather(sending,monitoring,return_exceptions=True)
            if frame_task:
                with contextlib.suppress(Exception):
                    await frame_task
            await asyncio.to_thread(assistant.close)
            await asyncio.to_thread(scheduler.close)
    return application


app = create_app()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=int(os.environ.get("PORT","8000")))
    parser.add_argument("--https",action="store_true")
    parser.add_argument("--low-power",action="store_true")
    parser.add_argument("--approaching-only",action="store_true")
    parser.add_argument("--cert",type=Path,default=ROOT/"models/tls/cert.pem")
    parser.add_argument("--key",type=Path,default=ROOT/"models/tls/key.pem")
    parser.add_argument("--cert-host",action="append",default=[],help="LAN IP/DNS name to include in certificate; repeatable")
    args = parser.parse_args()
    if args.https and (not args.cert.exists() or not args.key.exists()):
        args.cert.parent.mkdir(parents=True,exist_ok=True)
        args.key.parent.mkdir(parents=True,exist_ok=True)
        names = ["DNS:localhost","IP:127.0.0.1"]
        for host in args.cert_host:
            import ipaddress
            try:
                ipaddress.ip_address(host)
                names.append("IP:"+host)
            except ValueError:
                if not host or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in host):
                    raise ValueError("Invalid certificate host")
                names.append("DNS:"+host)
        subprocess.run(["openssl","req","-x509","-newkey","rsa:2048","-nodes","-days","30",
                        "-keyout",str(args.key),"-out",str(args.cert),"-subj","/CN=localhost",
                        "-addext","subjectAltName="+",".join(names)],check=True)
        args.key.chmod(0o600)
    import uvicorn
    application = (create_app(lambda: ModelBundle(low_power=args.low_power,
                      approaching_only=args.approaching_only))
                   if args.low_power or args.approaching_only else app)
    uvicorn.run(application,host=args.host,port=args.port,ws_max_size=6*1024*1024,
                ssl_certfile=str(args.cert) if args.https else None,
                ssl_keyfile=str(args.key) if args.https else None)


if __name__ == "__main__":
    main()
