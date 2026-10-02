"""Deterministic command parsing with offline route validation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from speech.numbers import spoken_route_candidates
from transit.gtfs_loader import load_routes
from transit.route_matcher import match_route


class IntentType(str, Enum):
    FIND_BUS = "FIND_BUS"
    CANCEL_BUS = "CANCEL_BUS"
    WHATS_AHEAD = "WHATS_AHEAD"
    REPEAT = "REPEAT"
    STOP = "STOP"
    HELP = "HELP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Intent:
    type: IntentType
    target: str | None
    raw: str

    @property
    def prompt(self) -> str | None:
        return "Which bus number?" if self.type == IntentType.FIND_BUS and self.target is None else None


BUS_PATTERNS = [
    r"(?:find|look for|search for)(?: the)?(?: bus)?(?: number)?(?:\s+(?P<route>.+))?",
    r"where is(?: the)?(?: bus)?(?: number)?(?:\s+(?P<route>.+))?",
    r"bus(?: number)?(?:\s+(?P<route>.+))?",
    r"is(?: the)?(?: bus)?\s+(?P<route>.+?)\s+here",
]


def parse_intent(transcript, routes: Iterable[str] | None = None) -> Intent:
    raw = transcript if isinstance(transcript, str) else transcript.text
    text = re.sub(r"\s+", " ", raw.lower().strip().rstrip(".!?"))
    text = re.sub(r"^please\s+|\s+please$", "", text)
    rules = [
        (r"(?:stop looking|stop looking for(?: the)? bus|cancel(?: the)? bus|cancel bus search)", IntentType.CANCEL_BUS),
        # Vosk mishears Piper's "what" as "for"/"where"; "ahead" is unique to this command.
        (r"(?:(?:[a-z']+ )?(?:is )?ahead(?: of me)?|what is in front(?: of me)?)", IntentType.WHATS_AHEAD),
        (r"(?:repeat|say (?:it|that) again|repeat that)", IntentType.REPEAT),
        (r"(?:stop|silence|quiet|stop speaking|be quiet)", IntentType.STOP),
        (r"(?:help|help me|what can i say)", IntentType.HELP),
    ]
    for pattern, intent_type in rules:
        if re.fullmatch(pattern, text):
            return Intent(intent_type, None, raw)
    for pattern in BUS_PATTERNS:
        found = re.fullmatch(pattern, text)
        if found:
            phrase = found.groupdict().get("route") or ""
            if not phrase:
                return Intent(IntentType.FIND_BUS, None, raw)
            catalogue = load_routes() if routes is None else set(routes)
            candidates = [phrase] + spoken_route_candidates(phrase)
            validated = []
            for candidate in candidates:
                result = match_route(candidate, catalogue)
                # Commands should never silently pick a fuzzy route suffix.
                if result.route and result.method in {"exact", "confusion_fix"} and result.route not in validated:
                    validated.append(result.route)
            if len(validated) == 1:
                return Intent(IntentType.FIND_BUS, validated[0], raw)
            if spoken_route_candidates(phrase):
                return Intent(IntentType.FIND_BUS, None, raw)
            return Intent(IntentType.UNKNOWN, None, raw)
    return Intent(IntentType.UNKNOWN, None, raw)
