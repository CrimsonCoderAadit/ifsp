"""Offline route catalogue. Deliberately never opens calendar.txt."""
from __future__ import annotations

import argparse
import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ROUTE_SHAPE = re.compile(r"^[A-Z]{0,2}\d{1,3}[A-Z]{0,2}$")
VARIANT = re.compile(r"(?:\s*(?:CUT\d*|CT\d*|R\d*|ET))$", re.I)


def base_route(short_name: str) -> str:
    value = short_name.strip().upper()
    while True:
        cleaned = VARIANT.sub("", value).strip()
        if cleaned == value:
            return value
        value = cleaned


@dataclass(frozen=True)
class LoadStats:
    raw_routes: int
    bus_routes: int
    base_routes: int
    rejected_shape_routes: int  # distinct bases retained as exact-only aliases
    rejected_shape_rows: int


def build_routes(zip_path: Path = ROOT / "data/mtc-gtfs.zip",
                 output: Path = ROOT / "data/routes.csv") -> LoadStats:
    grouped: dict[str, tuple[set[str], set[str]]] = {}
    raw = buses = rejected_rows = 0
    rejected: set[str] = set()
    with ZipFile(zip_path) as archive:
        with archive.open("routes.txt") as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")):
                raw += 1
                if row.get("route_type", "").strip() != "3":
                    continue
                buses += 1
                short = row.get("route_short_name", "").strip()
                base = base_route(short)
                if not base:
                    logging.warning("Empty route short name at row %s", raw)
                    continue
                if not ROUTE_SHAPE.fullmatch(base):
                    rejected_rows += 1
                    if base not in rejected:
                        logging.warning("Rejected shape; keeping exact-only alias: %r", base)
                    rejected.add(base)
                names, longs = grouped.setdefault(base, (set(), set()))
                names.add(short)
                if row.get("route_long_name"):
                    longs.add(row["route_long_name"].strip())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["base_route", "raw_short_names", "sample_long_names"])
        for base, (names, longs) in sorted(grouped.items()):
            writer.writerow([base, " | ".join(sorted(names)), " | ".join(sorted(longs)[:3])])
    temporary.replace(output)
    return LoadStats(raw, buses, len(grouped), len(rejected), rejected_rows)


def load_routes(full: Path = ROOT / "data/routes.csv",
                fallback: Path = ROOT / "data/routes_review1.csv") -> set[str]:
    path = full if full.exists() else fallback
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["base_route"] for row in csv.DictReader(stream) if row["base_route"]}



def load_route_metadata(zip_path: Path = ROOT / "data/mtc-gtfs.zip",
                        full: Path = ROOT / "data/routes.csv",
                        fallback: Path = ROOT / "data/routes_review1.csv") -> dict[str, dict]:
    """Map base route to its raw variants and every unique "A TO B" long name.

    Reads routes.txt from the GTFS ZIP when present; the route CSV keeps
    only three sample long names per base.
    """
    grouped: dict[str, tuple[set[str], dict[str, str]]] = {}

    def add(base, short, long_name):
        names, longs = grouped.setdefault(base, (set(), {}))
        if short:
            names.add(short)
        long_name = re.sub(r"\s+", " ", long_name or "").strip()
        if long_name:
            longs.setdefault(long_name.upper(), long_name)

    if zip_path.exists():
        with ZipFile(zip_path) as archive, archive.open("routes.txt") as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")):
                short = row.get("route_short_name", "").strip()
                if row.get("route_type", "").strip() == "3" and base_route(short):
                    add(base_route(short), short, row.get("route_long_name"))
    else:
        path = full if full.exists() else fallback
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for short in (row.get("raw_short_names") or row["base_route"]).split(" | "):
                    add(row["base_route"], short.strip(), None)
                for long_name in (row.get("sample_long_names") or "").split(" | "):
                    add(row["base_route"], None, long_name)
    return {base: {"route": base, "variants": sorted(names), "long_names": sorted(longs.values())}
            for base, (names, longs) in grouped.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT / "data/mtc-gtfs.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/routes.csv")
    args = parser.parse_args()
    print(build_routes(args.zip, args.output))
