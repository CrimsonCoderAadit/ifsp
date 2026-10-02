"""Local configuration; model loading never downloads resources."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Must precede any onnxruntime import: its telemetry thread uploads to
# mobile.events.data.microsoft.com and can abort the interpreter at exit.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class SpeechConfig:
    tts_backend: str = "piper"
    piper_model: Path = ROOT / "models/en_US-lessac-medium.onnx"
    vosk_model: Path = ROOT / "models/vosk-model-small-en-in-0.4"
    cache_size: int = 128
    sample_rate: int = 16000
    pyttsx3_driver: str | None = None
    pyttsx3_voice: str | None = None
    speech_rate: int = 175


@dataclass(frozen=True)
class PipelineConfig:
    max_fps: float = 8.0
    ocr_every_n: int = 3
    vote_k: int = 2
    vote_n: int = 3
    recent_seconds: float = 3.0
    verbose: bool = False
    max_tracks_per_frame: int = 4
    max_frame_age_seconds: float = 2.5
    approaching_only: bool = False
    approach_min_growth: float = 1.12
    approach_window: int = 3


def low_power_speech_config() -> SpeechConfig:
    return SpeechConfig(piper_model=ROOT / "models/en_US-lessac-low.onnx")


def low_power_pipeline_config() -> PipelineConfig:
    return PipelineConfig(max_fps=5.0, ocr_every_n=5)


LOW_POWER_DETECTOR_MODEL = ROOT / "models/yolo11n-320.onnx"
