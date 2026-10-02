import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core.config import SpeechConfig
from speech.stt import GRAMMAR, SpeechRecognizer, VoskSTT, resample_pcm
from speech.tts import SpeechSynthesizer


class DummyTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return np.array([1, 2, 3], dtype=np.int16), 22050


def test_tts_cache_lru_and_readonly():
    backend = DummyTTS()
    tts = SpeechSynthesizer(replace(SpeechConfig(), cache_size=2), backend=backend)
    first = tts.synthesize("Bus 21G detected.")
    assert tts.synthesize(" Bus 21G detected. ")[0] is first[0]
    assert not first[0].flags.writeable
    tts.synthesize("repeat")
    tts.synthesize("Bus 21G detected.")
    tts.synthesize("stop")
    tts.synthesize("repeat")
    assert backend.calls == ["Bus 21G detected.", "repeat", "stop", "repeat"]
    assert tts.cache_hits == 2 and tts.cache_misses == 4
    assert tts.load_time_ms >= 0


def test_tts_load_fallback():
    with patch("speech.tts.PiperTTS", side_effect=RuntimeError("bad voice")), patch("speech.tts.Pyttsx3TTS", return_value=DummyTTS()):
        tts = SpeechSynthesizer()
        assert "bad voice" in tts.fallback_reason
        assert tts.synthesize("hello")[0].dtype == np.int16


def test_explicit_fallback_and_invalid_config():
    with patch("speech.tts.Pyttsx3TTS", return_value=DummyTTS()):
        assert SpeechSynthesizer(replace(SpeechConfig(), tts_backend="pyttsx3")).fallback_reason is None
    with pytest.raises(ValueError):
        SpeechSynthesizer(replace(SpeechConfig(), tts_backend="cloud"))


class DummyKaldi:
    def __init__(self, model, sr, grammar):
        self.grammar = json.loads(grammar)
        self.calls = 0

    def SetWords(self, enabled):
        assert enabled

    def AcceptWaveform(self, chunk):
        self.calls += 1
        return self.calls == 1

    def Result(self):
        return json.dumps({"text": "find bus", "result": [{"conf": .8}, {"conf": 1}]})

    def PartialResult(self):
        return json.dumps({"partial": "twenty one g"})

    def FinalResult(self):
        return json.dumps({"text": "twenty one g", "result": [{"conf": .9}]})


def test_stream_preserves_finalized_segments_and_grammar(tmp_path):
    module = SimpleNamespace(Model=lambda path: object(), KaldiRecognizer=DummyKaldi)
    with patch.dict(sys.modules, vosk=module):
        backend = VoskSTT(replace(SpeechConfig(), vosk_model=tmp_path))
        recognizer = SpeechRecognizer(backend=backend)
        recognizer.start()
        assert "[unk]" in backend.recognizer.grammar
        assert "find" in backend.recognizer.grammar and "seventy" in backend.recognizer.grammar
        assert recognizer.feed(b"\x00\x00") == "find bus"
        assert recognizer.feed(b"\x00\x00") == "find bus twenty one g"
        result = recognizer.finish()
        assert result.text == "find bus twenty one g"
        assert result.confidence == pytest.approx(.9)
        assert result.latency_ms >= 0
        with pytest.raises(RuntimeError):
            recognizer.finish()
        recognizer.start()
        with pytest.raises(RuntimeError):
            recognizer.start()
        with pytest.raises(ValueError):
            recognizer.feed(b"\x00")
        recognizer.finish()


def test_resample_and_downmix():
    pcm = np.full((22050, 2), 200, dtype=np.int16)
    output = resample_pcm(pcm, 22050)
    assert output.dtype == np.int16 and output.shape == (16000,)
    assert output[8000] == pytest.approx(200, abs=1)
    assert resample_pcm(np.empty(0, dtype=np.int16), 22050).size == 0
    with pytest.raises(ValueError):
        resample_pcm(pcm, 0)


def test_grammar_vocabulary():
    assert {"[unk]", "find", "bus", "twenty", "gee", "em", "repeat", "stop", "ahead"} <= set(GRAMMAR)
    assert "what is ahead" in GRAMMAR


def test_reject_empty_synthesis():
    class BrokenTTS:
        def synthesize(self, text):
            return np.empty(0, dtype=np.int16), 16000
    with pytest.raises(RuntimeError, match="empty audio"):
        SpeechSynthesizer(backend=BrokenTTS()).synthesize("hello")


def test_warmed_warning_cache_bypasses_cold_inference():
    import threading
    import time
    entered, release = threading.Event(), threading.Event()
    class SlowBackend(DummyTTS):
        def synthesize(self, text):
            if text == "cold bus":
                entered.set()
                release.wait(2)
            return super().synthesize(text)
    tts = SpeechSynthesizer(backend=SlowBackend())
    expected = tts.synthesize("warning")
    worker = threading.Thread(target=lambda: tts.synthesize("cold bus"))
    worker.start()
    try:
        assert entered.wait(1)
        before = time.monotonic()
        assert tts.synthesize("warning")[0] is expected[0]
        assert time.monotonic()-before < .01
    finally:
        release.set()
        worker.join(1)
