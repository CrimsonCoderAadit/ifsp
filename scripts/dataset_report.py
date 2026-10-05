"""Count collected images and confirmed labels without mixing suggestions in accuracy."""
from collections import Counter
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data/real"
for source in ("commons", "roboflow", "own"):
    files = [p for p in (root / source).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    print(f"{source}: {len(files)} images")
labels = root / "labels.csv"
rows = list(csv.DictReader(labels.open(newline="", encoding="utf-8"))) if labels.exists() else []
confirmed = [row for row in rows if row.get("confirmed") == "1"]
print(f"labels: {len(rows)} rows; {len(confirmed)} confirmed; {len(rows)-len(confirmed)} suggestions")
print(f"confirmed readable routes: {sum(bool(row.get('route_number')) for row in confirmed)}")
print("confirmed board types:", dict(Counter(row.get("board_type", "") for row in confirmed)))
