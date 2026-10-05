"""Offline streaming recognition of the single command 'bus'."""
from __future__ import annotations

import json
import time
from pathlib import Path


class BusKeyword:
    def __init__(self, model_path: Path, *, threshold=.7, debounce_s=1.5):
        from vosk import Model
        self.model = Model(str(model_path))
        self.threshold, self.debounce_s = threshold, debounce_s
        self.last_fired = float("-inf")
        self.recognizer = None

    def start(self, sample_rate=16000):
        from vosk import KaldiRecognizer
        self.recognizer = KaldiRecognizer(self.model, sample_rate, json.dumps(["bus", "[unk]"]))
        self.recognizer.SetWords(True)

    def feed(self, pcm: bytes, *, now=None):
        if self.recognizer is None:
            self.start()
        if self.recognizer.AcceptWaveform(pcm):
            return self._accept(json.loads(self.recognizer.Result()), now)
        return False

    def finish(self, *, now=None):
        if self.recognizer is None:
            return False
        result = self._accept(json.loads(self.recognizer.FinalResult()), now)
        self.recognizer = None
        return result

    def _accept(self, result, now=None):
        now = time.monotonic() if now is None else now
        words = result.get("result", [])
        heard = any(word.get("word") == "bus" and float(word.get("conf", 0)) >= self.threshold
                    for word in words)
        if heard and now - self.last_fired >= self.debounce_s:
            self.last_fired = now
            return True
        return False
