"""Resume-friendly collection of licensed bus images and optional Roboflow data."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/real"
API = "https://commons.wikimedia.org/w/api.php"
CATEGORIES = ["Category:Metropolitan Transport Corporation (Chennai)",
              "Category:Buses in Chennai", "Category:Buses in Tamil Nadu"]
USER_AGENT = "IFSP Bus Finder academic evaluation (contact: https://github.com/aaditpraveennath)"
ALLOWED = ("cc0", "cc-by", "cc by", "public domain", "pd-")
last_request = 0.


def api(params):
    global last_request
    time.sleep(max(0, 1 - (time.monotonic() - last_request)))
    url = API + "?" + urllib.parse.urlencode({"format": "json", **params})
    response = subprocess.run(["curl", "--http1.1", "-fsSL", "--retry", "5", "--retry-all-errors", "--max-time", "30",
                               "-A", USER_AGENT, url], capture_output=True, check=True)
    payload = json.loads(response.stdout)
    last_request = time.monotonic()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def members(category):
    continuation = {}
    while True:
        result = api({"action": "query", "list": "categorymembers", "cmtitle": category,
                      "cmlimit": "max", "cmtype": "file|subcat", **continuation})
        yield from result.get("query", {}).get("categorymembers", [])
        continuation = result.get("continue")
        if not continuation:
            break


def commons():
    directory = ROOT / "commons"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.csv"
    known = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as stream:
            known = {row["title"]: row for row in csv.DictReader(stream)}
    queue = [(category, 0) for category in CATEGORIES]
    visited = set()
    titles = []
    seen_titles = set()
    while queue:
        category, depth = queue.pop(0)
        if category in visited:
            continue
        visited.add(category)
        try:
            for item in members(category):
                if item["ns"] == 14 and depth < 2:
                    queue.append((item["title"], depth + 1))
                elif item["ns"] == 6 and item["title"] not in seen_titles:
                    titles.append(item["title"])
                    seen_titles.add(item["title"])
        except Exception as error:
            print(f"Skipping category {category}: {error}")
    for title in titles:
        if title in known and (directory / known[title]["file"]).exists():
            continue
        try:
            result = api({"action": "query", "prop": "imageinfo", "titles": title,
                          "iiprop": "url|extmetadata", "iiurlwidth": 1600})
        except Exception as error:
            print(f"Skipping image {title}: {error}")
            continue
        page = next(iter(result["query"]["pages"].values()))
        info = (page.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata", {})
        license_name = re.sub(r"<[^>]+>", "", metadata.get("LicenseShortName", {}).get("value", ""))
        if not any(tag in license_name.lower() for tag in ALLOWED):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url or not re.search(r"\.(?:jpe?g|png|webp)(?:\?|$)", url, re.I):
            continue
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", title.removeprefix("File:"))[:150]
        target = directory / name
        if not target.exists():
            time.sleep(max(0, 1 - (time.monotonic() - last_request)))
            try:
                subprocess.run(["curl", "--http1.1", "-fsSL", "--retry", "2", "--retry-all-errors", "--max-time", "20",
                                "-A", USER_AGENT, "-o", str(target), url], check=True)
            except Exception as error:
                print(f"Skipping download {title}: {error}")
                continue
        known[title] = {"title": title, "file": name, "url": info.get("descriptionurl", url),
                        "license": license_name,
                        "author": re.sub(r"<[^>]+>", "", metadata.get("Artist", {}).get("value", ""))}
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["title", "file", "url", "license", "author"])
            writer.writeheader()
            writer.writerows(known.values())
    print(f"Commons: {len(known)} licensed images; review manually for readable MTC routes")


def roboflow():
    key = os.getenv("ROBOFLOW_API_KEY")
    env = Path(__file__).resolve().parents[1] / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        print("Roboflow skipped: ROBOFLOW_API_KEY is absent")
        return
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    for project_name in ("tsrtc", "bus-route-detection"):
        directory = ROOT / "roboflow" / project_name
        if any(directory.rglob("data.yaml")):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        project = rf.workspace("aaj").project(project_name)
        version_number = max(int(version.version) for version in project.versions())
        project.version(version_number).download("yolov8", location=str(directory))
        print(f"Roboflow {project_name}: version {version_number}")


def own_frames():
    import cv2
    directory = ROOT / "own"
    directory.mkdir(parents=True, exist_ok=True)
    for video in directory.iterdir():
        if video.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % max(1, round(fps)) == 0:
                target = directory / f"{video.stem}_{index // max(1, round(fps)):04d}.jpg"
                if not target.exists():
                    cv2.imwrite(str(target), frame)
            index += 1
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["all", "commons", "roboflow", "own"], default="all")
    selected = parser.parse_args().source
    if selected in {"all", "commons"}: commons()
    if selected in {"all", "roboflow"}: roboflow()
    if selected in {"all", "own"}: own_frames()
