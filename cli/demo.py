"""Offline image/video/camera demo using the same Assistant as the web UI."""
import argparse
import json
import logging
import sys
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from core.config import (ROOT, LOW_POWER_DETECTOR_MODEL, PipelineConfig,
                         low_power_pipeline_config, low_power_speech_config)
from core.pipeline import Assistant
from speech.scheduler import AudioScheduler, LocalSink, NullSink, SchedulerConfig
from speech.stt import SpeechRecognizer, resample_pcm
from speech.tts import SpeechSynthesizer
from transit.bus_detector import BusDetector
from transit.ocr import create_ocr_backend


def draw_overlay(frame, payload):
    output = frame.copy()
    for box in payload.get("boxes",[]):
        x1,y1,x2,y2 = map(int,box["bbox"])
        route = box.get("confirmed_route") or box.get("matched_route") or "?"
        color = (0,220,0) if box.get("confirmed_route") else (0,180,255)
        cv2.rectangle(output,(x1,y1),(max(x1,x2-1),max(y1,y2-1)),color,2)
        label = f"track {box['track_id']} route {route}: {' | '.join(box.get('ocr_text',[]))}"
        cv2.putText(output,label,(x1+5,y1+25),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    target = payload.get("state",{}).get("target_bus")
    cv2.putText(output,f"target: {target['route'] if target else '-'} | processed frame {payload.get('frame_id',0)}",
                (10,output.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.6,(80,180,255),2)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video",type=Path)
    source.add_argument("--image",type=Path)
    source.add_argument("--camera",type=int,default=0)
    parser.add_argument("--command",default="")
    parser.add_argument("--wav",type=Path)
    parser.add_argument("--mic",action="store_true")
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--save-overlay",type=Path)
    parser.add_argument("--mock-risk-at",type=float)
    parser.add_argument("--detector",choices=["auto","onnx","ultralytics","passthrough"],default="auto")
    parser.add_argument("--include-trucks",action="store_true")
    parser.add_argument("--ocr",choices=["rapidocr","tesseract","paddleocr"],default="rapidocr")
    parser.add_argument("--null-audio",action="store_true")
    parser.add_argument("--max-fps",type=float,default=8)
    parser.add_argument("--ocr-every",type=int,default=3)
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--approaching-only",action="store_true",
                        help="Announce a target only after sustained bbox-area growth")
    parser.add_argument("--low-power",action="store_true",
                        help="Use 320px ONNX detector, OCR every fifth frame, low Piper voice")
    parser.add_argument("--log",type=Path,default=ROOT / "logs/demo_decisions.csv")
    parser.add_argument("--console-log",type=Path)
    args = parser.parse_args()
    if args.low_power:
        args.max_fps, args.ocr_every = 5.0, 5
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.console_log:
        args.console_log.parent.mkdir(parents=True,exist_ok=True)
        handlers.append(logging.FileHandler(args.console_log,encoding="utf-8",mode="w"))
    logging.basicConfig(level=logging.INFO,format="%(message)s",handlers=handlers,force=True)
    # Mirror CLI reports into the same console/file handlers as pipeline logs.
    def report(message):
        logging.getLogger("demo").info("%s",message)
    detector_choice = "onnx" if args.low_power and args.detector == "auto" else args.detector
    detector_model = LOW_POWER_DETECTOR_MODEL if args.low_power and detector_choice == "onnx" else None
    detector = BusDetector(detector_choice,include_trucks=args.include_trucks,model=detector_model)
    ocr = create_ocr_backend(args.ocr)
    tts = SpeechSynthesizer(low_power_speech_config() if args.low_power else None)
    # Warm the mock warning so safety onset does not pay cold synthesis cost.
    if args.mock_risk_at is not None:
        tts.synthesize("Vehicle approaching right.")
    sink = NullSink(duration_s=None) if args.null_audio else LocalSink()
    config = (replace(low_power_pipeline_config(), verbose=args.verbose,
                      approaching_only=args.approaching_only) if args.low_power
              else PipelineConfig(max_fps=args.max_fps,ocr_every_n=args.ocr_every,
                                  verbose=args.verbose,approaching_only=args.approaching_only))
    report(f"BACKENDS detector={detector.backend_name} OCR={type(ocr).__name__} TTS={tts.backend_name} sink={type(sink).__name__}")
    report(f"LOAD detector={detector.load_time_ms:.1f}ms OCR={getattr(ocr,'load_time_ms',0):.1f}ms TTS={tts.load_time_ms:.1f}ms")
    writer = None
    capture = None
    with AudioScheduler(tts,sink,SchedulerConfig(log_path=args.log)) as scheduler:
        report(f"AUDIO STARTUP sink_prepare={scheduler.sink_prepare_time_ms:.1f}ms")
        with Assistant(detector,ocr,scheduler,config=config) as assistant:
            if args.command:
                assistant.handle_transcript(args.command)
            if args.wav:
                recognizer = SpeechRecognizer()
                with wave.open(str(args.wav),"rb") as stream:
                    if stream.getsampwidth() != 2:
                        raise ValueError("--wav expects PCM16 WAV")
                    pcm = np.frombuffer(stream.readframes(stream.getnframes()),dtype="<i2")
                    pcm = pcm.reshape(-1,stream.getnchannels())
                    mono = resample_pcm(pcm,stream.getframerate())
                transcript = recognizer.transcribe(mono.tobytes())
                report(f"STT {transcript.text!r} confidence={transcript.confidence:.3f} latency={transcript.latency_ms:.3f}ms")
                assistant.handle_transcript(transcript)
            if args.mic:
                recognizer = SpeechRecognizer()
                threading.Thread(target=lambda: recognizer.mic_loop(on_transcript=assistant.handle_transcript),daemon=True).start()
            try:
                if args.image:
                    frame = cv2.imread(str(args.image))
                    if frame is None:
                        raise FileNotFoundError(args.image)
                    payload = assistant.handle_frame(frame)
                    if not assistant.wait_for_frame(payload["submitted_frame_id"],10):
                        raise TimeoutError("Image OCR timed out")
                    payload = assistant.overlay()
                    # A second independent OCR read exercises temporal voting
                    # when the exact/high-confidence shortcut is unavailable.
                    if payload["matched_route"] is None:
                        for _ in range(args.ocr_every+1):
                            time.sleep(1/args.max_fps)
                            submitted = assistant.handle_frame(frame)
                            assistant.wait_for_frame(submitted["submitted_frame_id"],10)
                        payload = assistant.overlay()
                    output = draw_overlay(frame,payload)
                    if args.save_overlay:
                        args.save_overlay.parent.mkdir(parents=True,exist_ok=True)
                        if not cv2.imwrite(str(args.save_overlay),output):
                            raise RuntimeError("Could not write overlay image")
                    if args.show:
                        cv2.imshow("Voice + Transit",output)
                        cv2.waitKey(0)
                    report("OVERLAY "+json.dumps(payload,ensure_ascii=False))
                else:
                    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
                    if not capture.isOpened():
                        raise RuntimeError("Video/camera could not open")
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    fps = fps if 1 <= fps <= 120 else 25
                    started = time.monotonic()
                    frame_number = 0
                    mocked = False
                    while True:
                        ok,frame = capture.read()
                        if not ok:
                            break
                        if args.video:
                            # Pace recorded video independently of OCR.
                            delay = started+frame_number/fps-time.monotonic()
                            if delay > 0:
                                time.sleep(delay)
                        elapsed = frame_number/fps if args.video else time.monotonic()-started
                        if args.mock_risk_at is not None and elapsed >= args.mock_risk_at and not mocked:
                            assistant.handle_risk({"type":"risk","object":"car","direction":"right",
                                                   "risk":"HIGH","timestamp":time.time()})
                            mocked = True
                        payload = assistant.handle_frame(frame)
                        if args.show or args.save_overlay:
                            output = draw_overlay(frame,payload)
                            if args.save_overlay:
                                if writer is None:
                                    args.save_overlay.parent.mkdir(parents=True,exist_ok=True)
                                    writer = cv2.VideoWriter(str(args.save_overlay),cv2.VideoWriter_fourcc(*"mp4v"),fps,
                                                             (frame.shape[1],frame.shape[0]))
                                    if not writer.isOpened():
                                        raise RuntimeError("Overlay video writer failed")
                                writer.write(output)
                            if args.show:
                                cv2.imshow("Voice + Transit",output)
                                if cv2.waitKey(1)&0xFF in (27,ord("q")):
                                    break
                        frame_number += 1
                    assistant.wait_for_frame(timeout=10)
                    report("VIDEO frames="+str(frame_number)+" METRICS "+json.dumps(assistant.metrics()))
                    report("FINAL "+json.dumps(assistant.overlay(),ensure_ascii=False))
                if not scheduler.wait_idle(10):
                    raise TimeoutError("Speech did not finish")
                for row in list(scheduler.decisions):
                    if row["decision"] in {"played","pre-empted","dropped_ttl","sink_error","synthesis_error"}:
                        report(f"AUDIO {row['decision']} p={row['priority']} {row['text']} age={row['age_ms']:.1f}ms")
            finally:
                if capture:
                    capture.release()
                if writer:
                    writer.release()
                if args.show:
                    cv2.destroyAllWindows()
    report(f"Decision log: {args.log}")


if __name__ == "__main__":
    main()
