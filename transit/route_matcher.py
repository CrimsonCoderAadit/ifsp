"""Small, offline matching interface; scores use RapidFuzz's 0–100 scale.

Ambiguous valid text is preserved by normalize_route. Catalogue matching can
resolve e.g. 2IG -> 21G or 216 -> 21G, but an existing exact route always wins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from rapidfuzz.fuzz import ratio
from transit.gtfs_loader import ROUTE_SHAPE

TO_DIGIT = dict(zip("OILSBZG", "0115826"))
TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
DEFAULT_THRESHOLD = 85.0


def _compact(text: str) -> str:
    return re.sub(r"[\s.\-]+", "", text.strip()).upper()


def _variants(value: str) -> list[tuple[str, int]]:
    if not re.fullmatch(r"[A-Z0-9]{1,7}", value) or not any(c.isdigit() for c in value):
        return []
    if re.search(r"\d{4,}", value):
        return []
    result: dict[str, int] = {}
    for prefix in range(3):
        for digits in range(1, 4):
            suffix = len(value) - prefix - digits
            if not 0 <= suffix <= 2:
                continue
            fixed = []
            changes = 0
            for i, char in enumerate(value):
                digit_position = prefix <= i < prefix + digits
                mapping = TO_DIGIT if digit_position else TO_LETTER
                desired = char.isdigit() if digit_position else char.isalpha()
                new = char if desired else mapping.get(char)
                if new is None:
                    break
                fixed.append(new)
                changes += new != char
            else:
                candidate = "".join(fixed)
                result[candidate] = min(changes, result.get(candidate, 99))
    return sorted(result.items(), key=lambda item: (item[1], item[0]))


def normalize_route(text: str) -> str | None:
    value = _compact(text)
    if ROUTE_SHAPE.fullmatch(value):
        return value
    options = _variants(value)
    if not options:
        return None
    best = [route for route, cost in options if cost == options[0][1]]
    return best[0] if len(best) == 1 else None


@dataclass(frozen=True)
class MatchResult:
    route: str | None
    score: float
    method: str


def match_route(candidate: str, routes: Iterable[str],
                threshold: float = DEFAULT_THRESHOLD) -> MatchResult:
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    catalogue = {route.strip().upper() for route in routes}
    value = _compact(candidate)
    literal = candidate.strip().upper()
    if literal in catalogue:
        return MatchResult(literal, 100.0, "exact")
    # Aliases with unusual shapes are exact-only. Do not strip punctuation
    # from catalogue aliases, as that could create unintended matches.
    if value in catalogue and ROUTE_SHAPE.fullmatch(value):
        return MatchResult(value, 100.0, "exact")
    variants = _variants(value)
    fixes = [(route, cost) for route, cost in variants if cost and route in catalogue]
    if fixes:
        best_cost = min(cost for _, cost in fixes)
        best = [route for route, cost in fixes if cost == best_cost]
        if len(best) == 1:
            return MatchResult(best[0], max(0.0, 100.0 - best_cost * 5), "confusion_fix")
        return MatchResult(None, 0.0, "none")
    normalized = normalize_route(candidate)
    if normalized is None:
        return MatchResult(None, 0.0, "none")
    digit_block = re.search(r"\d+", normalized).group()
    scored = [(route, ratio(normalized, route)) for route in catalogue
              if ROUTE_SHAPE.fullmatch(route)
              and re.search(r"\d+", route).group() == digit_block]
    scored.sort(key=lambda item: (-item[1], item[0]))
    if scored and scored[0][1] >= threshold:
        if len(scored) == 1 or scored[0][1] > scored[1][1]:
            return MatchResult(scored[0][0], scored[0][1], "fuzzy")
    return MatchResult(None, 0.0, "none")


def extract_route_candidates(ocr_lines: Iterable) -> list[str]:
    """Accept strings, dicts {text, box}, or RapidOCR [box, text, score].

    Boxes are four [x,y] points. Return unique normalized strings ranked by
    shortness, then height and top position. Confidence is not route validity.
    """
    ranked: dict[str, float] = {}
    fragments = []
    for index, line in enumerate(ocr_lines):
        box = None
        if isinstance(line, str):
            text = line
        elif isinstance(line, dict):
            text, box = line.get("text", ""), line.get("box")
        else:
            box, text = line[:2]
        fragments.append((text, box))
        height = top = 0.0
        if box is not None:
            ys = [float(point[1]) for point in box]
            height, top = max(ys) - min(ys), min(ys)
        words = re.findall(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*\.?", text)
        # Whole lines and adjacent words cover 'M 70', '21 G', and 'M 70 A'.
        if normalize_route(text):
            pieces = [text]
        else:
            spans = [(i, i+n, " ".join(words[i:i+n])) for n in (1, 2, 3)
                     for i in range(len(words)-n+1)
                     if normalize_route(" ".join(words[i:i+n]))]
            # Suppress '21' inside '21 G' even on a line with a destination.
            pieces = [piece for start, end, piece in spans
                      if not any(a <= start and end <= b and b-a > end-start
                                 for a, b, _ in spans)]
        for piece in pieces:
            route = normalize_route(piece)
            if route:
                score = 100 - len(route) * 2 + min(height, 100) * .2 - max(top, 0) * .01 - index * .001
                ranked[route] = max(score, ranked.get(route, float("-inf")))
    for (left, left_box), (right, right_box) in zip(fragments, fragments[1:]):
        a, b = _compact(left), _compact(right)
        # OCR can split a single route board into '21' and 'G', or 'M' and
        # '70'. Never concatenate arbitrary destination words or full routes.
        if not ((re.fullmatch(r"[A-Z]{0,2}\d{1,3}", a)
                 and re.fullmatch(r"[A-Z]{1,2}", b))
                or (re.fullmatch(r"[A-Z]{1,2}", a)
                    and re.fullmatch(r"\d{1,3}[A-Z]{0,2}", b))):
            continue
        if left_box is not None and right_box is not None:
            ly = [float(p[1]) for p in left_box]
            ry = [float(p[1]) for p in right_box]
            lx = [float(p[0]) for p in left_box]
            rx = [float(p[0]) for p in right_box]
            height = max(max(ly)-min(ly), max(ry)-min(ry), 1)
            if min(max(ly), max(ry)) <= max(min(ly), min(ry)):
                continue
            if not 0 <= min(rx)-max(lx) <= height * 3:
                continue
        joined = normalize_route(a+b)
        if joined:
            ranked[joined] = max(ranked.get(joined, 0),
                                 max(ranked.get(a, 90), ranked.get(b, 90)) + 1)
    return sorted(ranked, key=lambda route: (-ranked[route], route))
