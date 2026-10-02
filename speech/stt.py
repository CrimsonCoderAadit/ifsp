"""Grammar-constrained offline STT with a common batch/stream interface."""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.config import SpeechConfig
from speech.numbers import NUMBER_WORDS, LETTER_WORDS

COMMAND_WORDS = set("find bus number look looking for search where is the here cancel stop what whats ahead in front of me repeat say it that again silence quiet speaking be help can i please".split())
COMMAND_PHRASES = {"what is ahead", "what's ahead", "what is in front of me",
                   "find bus", "look for", "where is the", "stop looking",
                   "cancel bus", "say that again", "stop speaking", "what can i say"}
GRAMMAR = sorted(COMMAND_WORDS | NUMBER_WORDS | LETTER_WORDS | COMMAND_PHRASES | {"[unk]", "what's"})


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float
    latency_ms: float


class STTBackend(Protocol):
    def start(self, sr: int = 16000) -> None: ...
    def feed(self, chunk: bytes) -> str: ...
    def finish(self, end_of_speech_at: float | None = None) -> Transcript: ...
    def transcribe(self, pcm_int16_bytes: bytes, sr: int = 16000) -> Transcript: ...


def resample_pcm(pcm: np.ndarray, source_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Anti-aliased resampling of mono or frames-by-channels int16 audio."""
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")
    samples = np.asarray(pcm, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    elif samples.ndim != 1:
        raise ValueError("Expected mono or frames-by-channels PCM")
    if samples.size and source_sr != target_sr:
        from scipy.signal import resample_poly
        divisor = math.gcd(source_sr, target_sr)
        samples = resample_poly(samples, target_sr // divisor, source_sr // divisor)
    return np.clip(np.rint(samples), -32768, 32767).astype("<i2")


class VoskSTT:
    def __init__(self, config: SpeechConfig | None = None, *, grammar=None):
        self.config = config or SpeechConfig()
        if not self.config.vosk_model.is_dir():
            raise FileNotFoundError(f"Vosk model missing: {self.config.vosk_model}")
        import vosk
        start = time.perf_counter()
        self.model = vosk.Model(str(self.config.vosk_model))
        self.load_time_ms = (time.perf_counter() - start) * 1000
        self.grammar = list(GRAMMAR if grammar is None else grammar)
        if "[unk]" not in self.grammar:
            self.grammar.append("[unk]")
        self.recognizer = None
        self._segments = []

    def new_session(self):
        """Independent decoder state sharing the already-loaded Vosk model."""
        session = object.__new__(type(self))
        session.config, session.model = self.config, self.model
        session.grammar = list(self.grammar)
        session.load_time_ms = 0.0
        session.recognizer, session._segments = None, []
        return session

    def start(self, sr: int = 16000) -> None:
        if sr <= 0:
            raise ValueError("Sample rate must be positive")
        if self.recognizer is not None:
            raise RuntimeError("Finish the current stream before starting another")
        from vosk import KaldiRecognizer
        self.recognizer = KaldiRecognizer(self.model, sr, json.dumps(self.grammar))
        self.recognizer.SetWords(True)
        self._segments = []

    def feed(self, chunk: bytes) -> str:
        if self.recognizer is None:
            raise RuntimeError("Call start() before feed()")
        if len(chunk) % 2:
            raise ValueError("PCM int16 chunks must contain complete samples")
        if self.recognizer.AcceptWaveform(chunk):
            result = json.loads(self.recognizer.Result())
            self._segments.append(result)
            partial = ""
        else:
            partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
        return " ".join([segment.get("text", "") for segment in self._segments] + [partial]).strip()

    def finish(self, end_of_speech_at: float | None = None) -> Transcript:
        if self.recognizer is None:
            raise RuntimeError("Call start() before finish()")
        endpoint = time.perf_counter() if end_of_speech_at is None else end_of_speech_at
        try:
            self._segments.append(json.loads(self.recognizer.FinalResult()))
            text = " ".join(segment.get("text", "") for segment in self._segments).strip()
            words = [word for segment in self._segments for word in segment.get("result", [])]
            confidence = float(np.mean([word.get("conf", 0) for word in words])) if words else 0.0
            return Transcript(text, confidence, max(0, (time.perf_counter() - endpoint) * 1000))
        finally:
            self.recognizer = None

    def transcribe(self, pcm_int16_bytes: bytes, sr: int = 16000) -> Transcript:
        # Input is already recorded: start timing when the buffered utterance
        # is available, including stream initialization and all decoding.
        endpoint = time.perf_counter()
        self.start(sr)
        try:
            self.feed(pcm_int16_bytes)
            return self.finish(endpoint)
        except Exception:
            self.recognizer = None
            raise


class SpeechRecognizer:
    """One instance per stream/client. Backends share the small STT protocol."""
    def __init__(self, config: SpeechConfig | None = None, *, backend: STTBackend | None = None):
        self.config = config or SpeechConfig()
        self.backend = backend if backend is not None else VoskSTT(self.config)
        self.load_time_ms = getattr(self.backend, "load_time_ms", 0.0)

    def transcribe(self, pcm_int16_bytes: bytes, sr: int = 16000) -> Transcript:
        return self.backend.transcribe(pcm_int16_bytes, sr)

    def start(self, sr: int = 16000) -> None:
        self.backend.start(sr)

    def feed(self, chunk: bytes) -> str:
        return self.backend.feed(chunk)

    def finish(self, end_of_speech_at: float | None = None) -> Transcript:
        return self.backend.finish(end_of_speech_at)

    def mic_loop(self, *, mode: str = "ptt", on_transcript=None, energy_threshold: float = 500,
                 silence_seconds: float = .7, max_seconds: float = 15) -> None:
        """Enter to start/stop, or VAD with RMS threshold in int16 units.

        Callback receives Transcript; default prints it. Ctrl-C exits. VAD
        latency includes the silence timeout. No audio is sent over a network.
        """
        if mode not in {"ptt", "vad"}:
            raise ValueError("mode must be ptt or vad")
        if energy_threshold < 0 or silence_seconds <= 0 or max_seconds <= 0:
            raise ValueError("Invalid VAD thresholds")
        import sounddevice as sd
        report = on_transcript or print
        sr = self.config.sample_rate
        chunks: queue.Queue = queue.Queue(maxsize=200)
        capture_error = threading.Event()

        def callback(indata, frames, timing, status):
            try:
                chunks.put_nowait((bytes(indata), time.perf_counter(), bool(status)))
            except queue.Full:
                capture_error.set()

        try:
            while True:
                if mode == "ptt":
                    input("Press Enter to speak (Ctrl-C exits)... ")
                while not chunks.empty():
                    chunks.get_nowait()
                capture_error.clear()
                stop = threading.Event()
                stopped_at = []
                if mode == "ptt":
                    def wait_for_enter():
                        try:
                            input("Press Enter to finish... ")
                        except EOFError:
                            pass
                        stopped_at.append(time.perf_counter())
                        stop.set()
                    threading.Thread(target=wait_for_enter, daemon=True).start()
                self.start(sr)
                active = mode == "ptt"
                began = time.perf_counter()
                last_voice = began
                endpoint = None
                previous_capture = None
                preroll = deque(maxlen=4)
                with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                                       blocksize=int(sr * .05), callback=callback):
                    while not stop.is_set():
                        if capture_error.is_set():
                            raise RuntimeError("Microphone queue overflow; retry the command")
                        try:
                            chunk, captured_at, overflow = chunks.get(timeout=.1)
                        except queue.Empty:
                            continue
                        if overflow or (previous_capture and captured_at-previous_capture > .3):
                            raise RuntimeError("Microphone input overflow; retry the command")
                        previous_capture = captured_at
                        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float64)
                        voiced = bool(samples.size and np.sqrt(np.mean(samples*samples)) >= energy_threshold)
                        if voiced:
                            if not active:
                                began = captured_at
                                for previous in preroll:
                                    self.feed(previous)
                                preroll.clear()
                            active = True
                            last_voice = captured_at
                        if active:
                            self.feed(chunk)
                        else:
                            preroll.append(chunk)
                        if mode == "vad" and active and captured_at-last_voice >= silence_seconds:
                            endpoint = last_voice
                            break
                        if mode == "vad" and active and captured_at-began >= max_seconds:
                            endpoint = captured_at
                            break
                # Drain audio captured before the Enter endpoint.
                endpoint = endpoint or (stopped_at[0] if stopped_at else time.perf_counter())
                while not chunks.empty():
                    chunk, captured_at, overflow = chunks.get_nowait()
                    if captured_at <= endpoint:
                        if overflow:
                            raise RuntimeError("Microphone input overflow")
                        self.feed(chunk)
                report(self.finish(endpoint))
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            if getattr(self.backend, "recognizer", None) is not None:
                self.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ptt", "vad"], default="ptt")
    SpeechRecognizer().mic_loop(mode=parser.parse_args().mode)
