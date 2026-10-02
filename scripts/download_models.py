"""One-time online setup. Runtime modules never call this downloader."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
VOICE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
VOICE_LOW = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/"
GTFS = "https://raw.githubusercontent.com/ungalsoththu/ChennaiGTFS/main/data/mtc-gtfs.zip"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def download(url: str, path: Path, minimum: int) -> None:
    """Atomic downloads; check Content-Length, minimum size, and cached hash.

    Cached hashes detect local corruption, not upstream authenticity. Delete a
    file and its receipt to refresh a weekly feed or changed upstream model.
    """
    receipt = path.with_name(path.name + ".download.json")
    if path.exists() and receipt.exists():
        info = json.loads(receipt.read_text())
        if (info.get("url") == url and path.stat().st_size == info.get("size")
                and path.stat().st_size >= minimum and digest(path) == info.get("sha256")):
            print(f"Cached: {path.name} ({path.stat().st_size:,} bytes)", flush=True)
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    context = ssl.create_default_context()
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "IFSP-setup/1.0"})
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                expected = int(response.headers.get("Content-Length", "0"))
                downloaded = 0
                last_report = 0.0
                with partial.open("wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                        downloaded += len(block)
                        if time.monotonic() - last_report > 1:
                            total = f" / {expected:,}" if expected else ""
                            print(f"{path.name}: {downloaded:,}{total} bytes", flush=True)
                            last_report = time.monotonic()
                if downloaded < minimum or (expected and downloaded != expected):
                    raise ValueError(f"Size check failed: {downloaded} bytes (expected {expected})")
            partial.replace(path)
            receipt.write_text(json.dumps({"url": url, "size": downloaded, "sha256": digest(path)}))
            print(f"Ready: {path.name} ({downloaded:,} bytes)", flush=True)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            print(f"Retrying {path.name} ({attempt + 1}/2)", flush=True)
            time.sleep(1)


def prepare_vosk() -> None:
    name = "vosk-model-small-en-in-0.4"
    archive_path = MODELS / f"{name}.zip"
    download(f"https://alphacephei.com/vosk/models/{name}.zip", archive_path, 10_000_000)
    marker = MODELS / name / ".complete"
    expected = digest(archive_path)
    if marker.exists() and marker.read_text() == expected:
        print("Cached: extracted Vosk model", flush=True)
        return
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt Vosk zip")
        for member in archive.infolist():
            destination = (MODELS / member.filename).resolve()
            if not destination.is_relative_to(MODELS.resolve()):
                raise ValueError("Unsafe zip path")
        archive.extractall(MODELS)
    marker.write_text(expected)


def prepare_yolo() -> None:
    weights = MODELS / "yolo11n.pt"
    download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
             weights, 1_000_000)
    output = MODELS / "yolo11n.onnx"
    if output.exists() and output.stat().st_size > 1_000_000:
        print("Cached: yolo11n.onnx", flush=True)
        return
    if importlib.util.find_spec("ultralytics") is None:
        print("YOLO export pending. Run .venv/bin/python -m pip install ultralytics onnx\n"
              "Then rerun this script with --only yolo.", flush=True)
        return
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS / "ultralytics"))
    (Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MODELS / "matplotlib"))
    from ultralytics import YOLO
    exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=640, opset=17,
                                            simplify=False, dynamic=False))
    if exported.resolve() != output.resolve():
        shutil.copyfile(exported, output)
    if output.stat().st_size < 1_000_000:
        raise ValueError("YOLO ONNX size check failed")


def prepare_yolo_low() -> None:
    prepare_yolo()
    output = MODELS / "yolo11n-320.onnx"
    if output.exists() and output.stat().st_size > 1_000_000:
        print("Cached: yolo11n-320.onnx", flush=True)
        return
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError("Install optional ultralytics and onnx to export 320px YOLO")
    source = MODELS / "yolo11n.pt"
    low_weights = MODELS / "yolo11n-320.pt"
    shutil.copyfile(source, low_weights)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS / "ultralytics"))
    (Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MODELS / "matplotlib"))
    from ultralytics import YOLO
    exported = Path(YOLO(str(low_weights)).export(format="onnx", imgsz=320, opset=17,
                                                  simplify=False, dynamic=False))
    if exported.resolve() != output.resolve():
        shutil.copyfile(exported, output)
    if output.stat().st_size < 1_000_000:
        raise ValueError("Low-power YOLO ONNX export too small")


def prepare_piper_low() -> None:
    for filename, minimum in (("en_US-lessac-low.onnx", 10_000_000),
                              ("en_US-lessac-low.onnx.json", 100)):
        download(VOICE_LOW + filename, MODELS / filename, minimum)


def prepare_rapidocr() -> None:
    import numpy as np
    from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
    directory = MODELS / "rapidocr"
    directory.mkdir(parents=True, exist_ok=True)
    params = {"Global.model_root_dir": str(directory)}
    for stage in ("Det", "Cls", "Rec"):
        params.update({f"{stage}.engine_type": EngineType.ONNXRUNTIME,
                       f"{stage}.model_type": ModelType.MOBILE,
                       f"{stage}.ocr_version": OCRVersion.PPOCRV4,
                       f"{stage}.model_dir": str(directory)})
    engine = RapidOCR(params=params)
    engine(np.full((100, 300, 3), 255, dtype=np.uint8))
    paths = sorted(directory.glob("*.onnx"))
    if len(paths) < 3 or any(path.stat().st_size < 100_000 for path in paths):
        raise ValueError("RapidOCR models missing or too small")
    # Save explicit paths for a later runtime adapter; no download is needed.
    for stage, attr in (("Det", "text_det"), ("Cls", "text_cls"), ("Rec", "text_rec")):
        model_path = str(getattr(engine, attr).session.session._model_path)
        params[f"{stage}.model_path"] = model_path
    serialized = {key: getattr(value, "value", value) for key, value in params.items()}
    # YAML config accepts enum strings; params= requires Enum instances.
    import yaml
    import rapidocr
    nested = yaml.safe_load((Path(rapidocr.__file__).parent / "config.yaml").read_text())
    for key, value in serialized.items():
        stage, option = key.split(".", 1)
        nested.setdefault(stage, {})[option] = value
    (directory / "runtime_config.yaml").write_text(yaml.safe_dump(nested))
    print(f"RapidOCR warmed: {len(paths)} ONNX models in {directory}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=["vosk", "piper", "yolo", "rapidocr", "gtfs",
                                                       "piper_low", "yolo_low"])
    args = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    tasks = {
        "vosk": prepare_vosk,
        "piper": lambda: [download(VOICE + filename, MODELS / filename, minimum)
                           for filename, minimum in [("en_US-lessac-medium.onnx", 10_000_000),
                                                     ("en_US-lessac-medium.onnx.json", 100)]],
        "yolo": prepare_yolo,
        "yolo_low": prepare_yolo_low,
        "piper_low": prepare_piper_low,
        "rapidocr": prepare_rapidocr,
        "gtfs": lambda: download(GTFS, ROOT / "data/mtc-gtfs.zip", 1_000),
    }
    failed = []
    for name in args.only or ("vosk", "piper", "yolo", "rapidocr", "gtfs"):
        try:
            tasks[name]()
        except Exception as error:
            failed.append(name)
            print(f"FAILED {name}: {error}. Rerun --only {name} after fixing setup.", file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
