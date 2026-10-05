"""Evaluate confirmed real route labels; hold out the test split until --test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.detector_tracker import DetectorTracker
from transit.bus_reader import BusReader
from transit.ocr import create_ocr_backend

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["source", "file", "board_type", "lighting", "expected", "detected", "predicted",
          "outcome", "detector_ms", "ocr_ms", "total_ms", "ocr_raw"]


def partitions(rows):
    """Assign exactly about 30% of each source to test, by stable image hash."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    held_out = set()
    for source, group in groups.items():
        ranked = sorted(group, key=lambda row: hashlib.sha256((source + ":" + row["file"]).encode()).digest())
        count = round(len(ranked) * .30)
        held_out.update((source, row["file"]) for row in ranked[:count])
    return {(row["source"], row["file"]):
            "test" if (row["source"], row["file"]) in held_out else "dev" for row in rows}


def summarize(rows):
    lines = []
    for key in ("source", "board_type", "lighting"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        lines += [f"## By {key}", "| Group | Images | Bus detected | Exact | Correct blank | Wrong route | No read | Mean ms/image |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for group, subset in sorted(groups.items()):
            n = len(subset)
            pct = lambda count: f"{100 * count / n:.1f}%" if n else "n/a"
            lines.append(f"| {group} | {n} | {pct(sum(r['detected']=='1' for r in subset))} | "
                         f"{pct(sum(r['outcome']=='exact' for r in subset))} | "
                         f"{pct(sum(r['outcome']=='correct_no_route' for r in subset))} | "
                         f"{pct(sum(r['outcome']=='wrong_route' for r in subset))} | "
                         f"{pct(sum(r['outcome']=='no_read' for r in subset))} | "
                         f"{sum(float(r['total_ms']) for r in subset)/n:.1f} |")
        lines.append("")
    failures = [row for row in rows if row["outcome"] not in {"exact", "correct_no_route"}][:10]
    lines += ["## First 10 failures", "| File | Expected | Read | Outcome |", "|---|---|---|---|"]
    lines += [f"| {r['file']} | {r['expected']} | {r['predicted']} | {r['outcome']} |" for r in failures]
    return "\n".join(lines) + "\n"


def main(test=False, ocr_backend="rapidocr"):
    labels = ROOT / "data/real/labels.csv"
    if not labels.exists():
        raise SystemExit("No confirmed real labels. Run scripts/fetch_datasets.py and scripts/label_tool.py first.")
    with labels.open(newline="", encoding="utf-8") as stream:
        source = [row for row in csv.DictReader(stream) if row.get("confirmed") == "1"]
    assignments = partitions(source)
    subset = [row for row in source if assignments[(row["source"], row["file"])] == ("test" if test else "dev")]
    if not subset:
        raise SystemExit("Selected split has no confirmed labels; no accuracy claim is possible.")
    detector = DetectorTracker("yolo26n.pt", ["bus", "truck"])
    reader = BusReader(ocr=create_ocr_backend(ocr_backend))
    output = ROOT / "experiments" / ("route_ocr_eval.csv" if test else f"route_ocr_dev_{ocr_backend}.csv")
    rows = []
    for label in subset:
        frame = cv2.imread(str(ROOT / "data/real" / label["file"]))
        if frame is None:
            continue
        started = time.perf_counter()
        tracks = detector.detect(frame)
        detect_ms = (time.perf_counter()-started)*1000
        report = reader.read_image(frame, tracks)
        predicted = report.get("route_number") or ""
        expected = label["route_number"].strip().upper()
        outcome = ("exact" if predicted == expected and expected else
                   "correct_no_route" if not predicted and not expected else
                   "wrong_route" if predicted else "no_read")
        rows.append({"source": label["source"], "file": label["file"],
                     "board_type": label.get("board_type", ""), "lighting": label.get("lighting", ""),
                     "expected": expected, "detected": str(int(report["detected"])),
                     "predicted": predicted, "outcome": outcome,
                     "detector_ms": f"{detect_ms:.1f}", "ocr_ms": f"{report.get('ocr_ms', 0):.1f}",
                     "total_ms": f"{(time.perf_counter()-started)*1000:.1f}",
                     "ocr_raw": report.get("ocr_raw", "")})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    summary = ROOT / "experiments" / ("route_ocr_summary.md" if test else f"route_ocr_dev_{ocr_backend}.md")
    summary.write_text(f"# {'Test' if test else 'Dev'} route OCR evaluation\n\nConfirmed real images: {len(rows)}\n\n" + summarize(rows))
    print(summary.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--test", action="store_true")
    parser.add_argument("--ocr-backend", choices=["rapidocr-v4", "rapidocr-v5-mobile", "rapidocr-v5-server"], default="rapidocr-v4")
    args = parser.parse_args()
    main(args.test, args.ocr_backend)
