"""One-time web asset build: OFL fonts, demo samples and the "Ready." clip.

Run once with network access, then commit app/static/{fonts,samples,audio}.
The web console itself never fetches anything from the internet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_models import download  # noqa: E402

STATIC = ROOT / "app/static"
FONTS = "https://github.com/google/fonts/raw/main/ofl/"
FONT_FILES = {
    "doto": ("Doto%5BROND,wght%5D.ttf", "Doto.ttf"),
    "atkinsonhyperlegiblenext": ("AtkinsonHyperlegibleNext%5Bwght%5D.ttf", "AtkinsonHyperlegibleNext.ttf"),
    "atkinsonhyperlegiblemono": ("AtkinsonHyperlegibleMono%5Bwght%5D.ttf", "AtkinsonHyperlegibleMono.ttf"),
}
# Latin, punctuation, arrows, check/cross marks and the en/em dashes used in the UI.
UNICODES = "U+0020-007E,U+00A0-00FF,U+2013-2014,U+2018-201D,U+2022,U+2026,U+2190-2193,U+2713,U+2715,U+00D7,U+00B7"
STILLS = [
    ("000_21G_printed.jpg", "21G printed board", True),
    ("005_21G_led.jpg", "21G amber LED board", True),
    ("001_23C_led.jpg", "23C LED board (another route)", True),
    ("019_M70_led.jpg", "M70 LED, JPEG artefacts (known misread)", True),
]


def fonts(cache: Path) -> None:
    from fontTools import subset
    out = STATIC / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    for family, (remote, local) in FONT_FILES.items():
        source = cache / local
        download(FONTS + family + "/" + remote, source, 20_000)
        download(FONTS + family + "/OFL.txt", cache / f"{family}-OFL.txt", 1_000)
        if family == "doto":
            from fontTools.ttLib import TTFont
            from fontTools.varLib.instancer import instantiateVariableFont
            pinned = cache / "Doto-round-bold.ttf"
            instantiateVariableFont(TTFont(source), {"ROND": 100, "wght": 900}).save(pinned)
            source = pinned
        subset.main([str(source), f"--unicodes={UNICODES}", "--layout-features=*",
                     f"--output-file={out / local}"])
        shutil.copyfile(cache / f"{family}-OFL.txt", out / f"{local.removesuffix('.ttf')}-OFL.txt")
        print(f"Font: {local} {(out / local).stat().st_size:,} bytes")


def samples() -> None:
    import cv2
    out = STATIC / "samples"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, label, board in STILLS:
        shutil.copyfile(ROOT / "data/test_images/synthetic" / name, out / name)
        manifest.append({"file": name, "label": label, "kind": "image", "board": board})
    capture = cv2.VideoCapture(str(ROOT / "data/test_video.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20
    width, height = int(capture.get(3)), int(capture.get(4))
    clip = out / "board_pan.webm"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    for _ in range(int(fps * 6)):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    capture.release()
    if not clip.exists() or clip.stat().st_size < 10_000:
        raise RuntimeError("VP8 WebM encoding failed; OpenCV needs FFmpeg with libvpx")
    manifest.append({"file": clip.name, "label": "Moving 21G board, 6 s", "kind": "video", "board": True})
    (out / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Samples: {len(manifest)} items")


def ready_clip() -> None:
    from speech.tts import SpeechSynthesizer
    pcm, sr = SpeechSynthesizer().synthesize("Ready.")
    out = STATIC / "audio/ready.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sr)
        stream.writeframes(pcm.astype("<i2").tobytes())
    print(f"Audio: {out.name} {out.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=("fonts", "samples", "audio"),
                        default=("fonts", "samples", "audio"))
    args = parser.parse_args()
    if "fonts" in args.only:
        fonts(ROOT / "models/fonts")
    if "samples" in args.only:
        samples()
    if "audio" in args.only:
        ready_clip()


if __name__ == "__main__":
    main()
