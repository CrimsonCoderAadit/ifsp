"""Real offline-model smoke test. Run pytest -m slow -s to see timings."""
import csv
import importlib.util
import socket
import time
import wave
from pathlib import Path

import pytest

from core.config import ROOT, SpeechConfig
from speech.intents import IntentType, parse_intent
from speech.stt import SpeechRecognizer, resample_pcm
from speech.tts import SpeechSynthesizer


@pytest.fixture(scope="module")
def speech_models():
    config = SpeechConfig()
    if (not config.piper_model.is_file() or not Path(str(config.piper_model)+".json").is_file()
            or not (config.vosk_model / "am/final.mdl").is_file()):
        pytest.skip("Piper/Vosk local models missing; run scripts/download_models.py")
    if importlib.util.find_spec("piper") is None or importlib.util.find_spec("vosk") is None:
        pytest.skip("Piper/Vosk packages missing from venv")
    tts = SpeechSynthesizer(config)
    assert tts.backend_name == "PiperTTS", tts.fallback_reason
    stt = SpeechRecognizer(config)
    print(f"\nMODEL LOAD: TTS={tts.load_time_ms:.3f}ms STT={stt.load_time_ms:.3f}ms")
    return tts, stt


@pytest.mark.slow
@pytest.mark.parametrize("phrase,kind,target", [
    ("find bus twenty one g", IntentType.FIND_BUS, "21G"),
    ("repeat", IntentType.REPEAT, None),
    ("stop", IntentType.STOP, None),
    ("what is ahead", IntentType.WHATS_AHEAD, None),
])
def test_roundtrip(speech_models, phrase, kind, target, monkeypatch):
    # No network calls are permitted during model inference.
    def blocked(*args, **kwargs):
        raise AssertionError("Runtime attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    tts, stt = speech_models
    started = time.perf_counter()
    pcm, sr = tts.synthesize(phrase)
    first_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    cached, cached_sr = tts.synthesize(phrase)
    cached_ms = (time.perf_counter()-started)*1000
    assert cached is pcm and cached_sr == sr
    filename = ROOT / "data/test_audio" / (phrase.replace(" ", "_")+".wav")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), "wb") as stream:
        stream.setparams((1, 2, sr, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.astype("<i2", copy=False).tobytes())
    # Read the written WAV, then resample to mono 16kHz for the batch API.
    with wave.open(str(filename), "rb") as stream:
        import numpy as np
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        audio16 = resample_pcm(samples, stream.getframerate())
    transcript = stt.transcribe(audio16.tobytes())
    intent = parse_intent(transcript)
    stt.start()
    raw = audio16.tobytes()
    chunk_bytes = 3200  # 100 ms at 16 kHz, streamed without realtime pacing
    endpoint = None
    for offset in range(0, len(raw), chunk_bytes):
        if offset+chunk_bytes >= len(raw):
            endpoint = time.perf_counter()
        stt.feed(raw[offset:offset+chunk_bytes])
    streamed = stt.finish(endpoint)
    assert (parse_intent(streamed).type, parse_intent(streamed).target) == (kind, target)
    print(f"\n{phrase!r} -> {transcript.text!r}: {intent.type.value}({intent.target}) | "
          f"first={first_ms:.3f}ms cached={cached_ms:.3f}ms STT={transcript.latency_ms:.3f}ms "
          f"stream_end={streamed.latency_ms:.3f}ms confidence={transcript.confidence:.3f}")
    streaming_log = ROOT / "logs/streaming_latency.csv"
    streaming_log.parent.mkdir(parents=True, exist_ok=True)
    with streaming_log.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if streaming_log.stat().st_size == 0:
            writer.writerow(["phrase", "last_chunk_to_transcript_ms", "transcript"])
        writer.writerow([phrase, streamed.latency_ms, streamed.text])
    log = ROOT / "logs/speech_latency.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(["phrase", "tts_load_ms", "stt_load_ms", "tts_first_ms", "tts_cached_ms", "stt_batch_latency_ms", "transcript", "intent", "target"])
        writer.writerow([phrase, tts.load_time_ms, stt.load_time_ms, first_ms, cached_ms,
                         transcript.latency_ms, transcript.text, intent.type.value, intent.target])
    assert (intent.type, intent.target) == (kind, target)
