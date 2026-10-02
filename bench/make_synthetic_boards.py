"""Generate deterministic OCR evaluation images, labels and a panning clip."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.config import ROOT
from transit.gtfs_loader import load_routes

FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]
TAMIL_PATHS = ["/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc",
               "/System/Library/Fonts/Tamil Sangam MN.ttc",
               "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf"]


def font(size, tamil=False):
    for path in TAMIL_PATHS if tamil else FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render_board(route, style="printed", size=(1000, 380)):
    led = style == "led"
    image = Image.new("RGB", size, (8,8,8) if led else (240,235,212))
    draw = ImageDraw.Draw(image)
    color = (255,180,35) if led else (10,15,15)
    draw.rectangle((4,4,size[0]-5,size[1]-5), outline=(120,95,30), width=4)
    displayed = route[:-1]+" "+route[-1] if route[-1].isalpha() else route
    draw.text((45,12), displayed, font=font(110), fill=color, anchor="lt")
    tamil = any(Path(path).exists() for path in TAMIL_PATHS)
    destination = "சென்னை - தாம்பரம்" if tamil else "TAMBARAM - CHENNAI"
    draw.text((45,205), destination, font=font(35,tamil=tamil), fill=color, anchor="lt")
    if led:
        # Subtle dark horizontal scan lines leave route glyphs legible.
        for y in range(15, 155, 5):
            draw.line((12,y,size[0]-12,y), fill=(12,10,5), width=1)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), destination


def distort(image, rng, names):
    output = image.copy()
    if "blur" in names:
        output = cv2.GaussianBlur(output, (5,5), rng.uniform(.5,1.8))
    if "motion" in names:
        kernel = np.zeros((7,7), dtype=np.float32)
        kernel[3,:] = 1/7
        output = cv2.filter2D(output, -1, kernel)
    if "glare" in names:
        height, width = output.shape[:2]
        layer = output.astype(np.float32)
        x, y = rng.integers(width//3,width), rng.integers(height)
        yy, xx = np.mgrid[:height,:width]
        glow = np.exp(-((xx-x)**2+(yy-y)**2)/(2*50**2))[:,:,None]*rng.uniform(40,130)
        output = np.clip(layer+glow, 0,255).astype(np.uint8)
    if "perspective" in names:
        h,w = output.shape[:2]
        source = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        destination = source+rng.uniform(-15,15,(4,2)).astype(np.float32)
        output = cv2.warpPerspective(output, cv2.getPerspectiveTransform(source,destination),
                                     (w,h), borderMode=cv2.BORDER_REPLICATE)
    if "jpeg" in names:
        ok, compressed = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35,70))])
        if ok:
            output = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    return output


def make_video(image, output, seconds=8, fps=20):
    if seconds <= 0 or fps <= 0:
        raise ValueError("Video duration and FPS must be positive")
    height,width = image.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer unavailable")
    try:
        for index in range(round(seconds*fps)):
            # Mild translation keeps the entire route visible during the pan.
            shift = 20*np.sin(index/max(1,seconds*fps-1)*2*np.pi)
            matrix = np.float32([[1,0,shift],[0,1,3*np.cos(index/fps)]])
            writer.write(cv2.warpAffine(image,matrix,(width,height),borderMode=cv2.BORDER_REPLICATE))
    finally:
        writer.release()


def generate(output=ROOT / "data/test_images/synthetic", *, count=24, seed=21, routes=None,
             video=ROOT / "data/test_video.mp4", seconds=8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    real = ROOT / "data/test_images/real"
    real.mkdir(parents=True, exist_ok=True)
    fields = ["filename","route","style","distortions","destination","width","height"]
    if not (real / "labels.csv").exists():
        (real / "labels.csv").write_text(",".join(fields)+"\n")
    catalogue = sorted(load_routes() if routes is None else routes)
    catalogue = [route for route in catalogue if len(route) <= 7]
    ordered = (["21G"] if "21G" in catalogue else [])+[route for route in catalogue if route != "21G"]
    if count < 1 or not ordered:
        raise ValueError("A positive count and nonempty route catalogue are required")
    rng = np.random.default_rng(seed)
    rows = []
    baseline = None
    effects = ["blur","motion","glare","perspective","jpeg"]
    for index in range(count):
        route = ordered[index % len(ordered)]
        style = "printed" if index % 2 == 0 else "led"
        image, destination = render_board(route,style)
        if index == 0:
            names = []
            baseline = image
        else:
            names = [effects[(index-1) % len(effects)]]
            if index >= len(effects)+1:
                names = list(rng.choice(effects, size=int(rng.integers(1,4)), replace=False))
        image = distort(image,rng,names)
        filename = f"{index:03d}_{route}_{style}.jpg"
        if not cv2.imwrite(str(output / filename),image):
            raise RuntimeError(f"Cannot write {filename}")
        rows.append(dict(zip(fields,[filename,route,style,json.dumps(names),destination,image.shape[1],image.shape[0]])))
    with (output / "labels.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if video is not None and baseline is not None:
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        make_video(baseline,Path(video),seconds)
    print(f"Generated {len(rows)} labelled boards in {output}; video={video}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT / "data/test_images/synthetic")
    parser.add_argument("--count",type=int,default=24)
    parser.add_argument("--seed",type=int,default=21)
    parser.add_argument("--routes",nargs="+")
    parser.add_argument("--video",type=Path,default=ROOT / "data/test_video.mp4")
    parser.add_argument("--seconds",type=float,default=8)
    args = parser.parse_args()
    generate(args.output,count=args.count,seed=args.seed,routes=args.routes,video=args.video,seconds=args.seconds)
