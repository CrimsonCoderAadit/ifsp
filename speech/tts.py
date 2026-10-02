"""Offline synthesis only. Playback and interruption belong to the scheduler."""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import numpy as np

from core.config import SpeechConfig


class TTSBackend(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]: ...


class PiperTTS:
    def __init__(self, config: SpeechConfig | None = None):
        config = config or SpeechConfig()
        model = Path(config.piper_model)
        if not model.is_file() or not Path(str(model) + ".json").is_file():
            raise FileNotFoundError(f"Piper model/config missing: {model}")
        from piper import PiperVoice
        start = time.perf_counter()
        self.voice = PiperVoice.load(str(model), use_cuda=False)
        self.load_time_ms = (time.perf_counter() - start) * 1000
        self.sample_rate = self.voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.empty(0, dtype=np.int16), self.sample_rate
        chunks = [chunk.audio_int16_array for chunk in self.voice.synthesize(text)]
        return (np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16), self.sample_rate)


class Pyttsx3TTS:
    """Local OS voice. espeak-ng/espeak also works if pyttsx3 cannot initialize.

    File synthesis avoids starting playback, so the scheduler can interrupt PCM.
    macOS pyttsx3 requires its driver on the main thread. Linux uses espeak.
    """
    def __init__(self, config: SpeechConfig | None = None):
        config = config or SpeechConfig()
        start = time.perf_counter()
        self.engine = None
        self.espeak = None
        self.rate = config.speech_rate
        try:
            import pyttsx3
            self.engine = pyttsx3.init(driverName=config.pyttsx3_driver)
            self.engine.setProperty("rate", self.rate)
            if config.pyttsx3_voice:
                self.engine.setProperty("voice", config.pyttsx3_voice)
        except Exception as error:
            self.espeak = shutil.which("espeak-ng") or shutil.which("espeak")
            if not self.espeak:
                raise RuntimeError("No local fallback TTS: install pyttsx3/OS driver or espeak-ng") from error
        self.load_time_ms = (time.perf_counter() - start) * 1000

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.empty(0, dtype=np.int16), 22050
        if self.espeak:
            output = subprocess.run([self.espeak, "--stdout", "-s", str(self.rate), "--", text],
                                    check=True, capture_output=True, timeout=30).stdout
            with wave.open(io.BytesIO(output)) as stream:
                pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
                return pcm, stream.getframerate()
        import soundfile as sf
        with tempfile.TemporaryDirectory(prefix="ifsp-tts-") as directory:
            filename = str(Path(directory) / "speech.wav")
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            if not Path(filename).exists() or Path(filename).stat().st_size < 44:
                raise RuntimeError("OS TTS driver did not produce audio")
            # libsndfile handles WAV and macOS AIFF regardless of extension.
            pcm, sr = sf.read(filename, dtype="int16", always_2d=True)
            mono = np.rint(pcm.astype(np.float64).mean(axis=1)).astype(np.int16)
            if not mono.size:
                raise RuntimeError("OS TTS driver produced an empty audio file")
            return mono, int(sr)


class SpeechSynthesizer:
    def __init__(self, config: SpeechConfig | None = None, *, backend: TTSBackend | None = None):
        self.config = config or SpeechConfig()
        if self.config.cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        start = time.perf_counter()
        self.fallback_reason: str | None = None
        if backend is not None:
            self.backend = backend
        elif self.config.tts_backend == "piper":
            try:
                self.backend = PiperTTS(self.config)
            except Exception as error:
                self.fallback_reason = f"{type(error).__name__}: {error}"
                logging.warning("Piper unavailable; using local fallback: %s", self.fallback_reason)
                self.backend = Pyttsx3TTS(self.config)
        elif self.config.tts_backend == "pyttsx3":
            self.backend = Pyttsx3TTS(self.config)
        else:
            raise ValueError(f"Unknown TTS backend: {self.config.tts_backend}")
        self.load_time_ms = (time.perf_counter() - start) * 1000
        self.backend_name = type(self.backend).__name__
        self.last_synthesis_ms = 0.0
        self.cache_hits = self.cache_misses = 0
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        start = time.perf_counter()
        text = text.strip()
        with self._lock:
            if text in self._cache:
                self.cache_hits += 1
                self._cache.move_to_end(text)
                result = self._cache[text]
                self.last_synthesis_ms = (time.perf_counter() - start) * 1000
                return result
        # Keep cache hits independent of slow synthesis. Backends remain
        # serialized, while a warmed warning can bypass a cold bus inference.
        with self._inference_lock:
            with self._lock:
                if text in self._cache:
                    self.cache_hits += 1
                    self._cache.move_to_end(text)
                    self.last_synthesis_ms = (time.perf_counter() - start) * 1000
                    return self._cache[text]
                self.cache_misses += 1
            pcm, sr = self.backend.synthesize(text)
            if pcm.dtype != np.int16 or pcm.ndim != 1 or sr <= 0:
                raise ValueError("TTS backend must return mono int16 PCM and positive sample rate")
            if text and not pcm.size:
                raise RuntimeError("TTS backend returned empty audio for nonempty text")
            pcm = pcm.copy()
            pcm.setflags(write=False)
            result = pcm, sr
            with self._lock:
                if self.config.cache_size:
                    self._cache[text] = result
                    if len(self._cache) > self.config.cache_size:
                        self._cache.popitem(last=False)
        self.last_synthesis_ms = (time.perf_counter() - start) * 1000
        return result

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
