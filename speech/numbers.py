"""Spoken route candidates, without assuming which routes exist.

Number groups are concatenated (five seventy -> 570); conventional hundreds
are also supported. Ambiguous 'oh' can be zero or the letter O.
"""
from __future__ import annotations

import itertools
import re

SMALL = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20)))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
LETTERS = {char: char.upper() for char in "abcdefghijklmnopqrstuvwxyz"}
LETTERS.update({"ay": "A", "bee": "B", "be": "B", "see": "C", "cee": "C",
                "dee": "D", "de": "D", "ee": "E", "ef": "F", "eff": "F",
                "gee": "G", "aitch": "H", "eye": "I", "jay": "J", "kay": "K",
                "el": "L", "ell": "L", "em": "M", "en": "N", "oh": "O",
                "pee": "P", "pe": "P", "cue": "Q", "queue": "Q", "ar": "R",
                "are": "R", "es": "S", "ess": "S", "tee": "T", "tea": "T",
                "you": "U", "vee": "V", "ex": "X", "why": "Y", "zed": "Z", "zee": "Z"})
NUMBER_WORDS = set(SMALL) | set(TENS) | {"oh", "hundred", "and"}
LETTER_WORDS = set(LETTERS) | {"double"}
SHAPE = re.compile(r"^[A-Z]{0,2}\d{1,3}[A-Z]{0,2}$")


def _number_groups(tokens: list[str]) -> str | None:
    groups = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word == "and":
            return None
        if word.isdigit():
            groups.append(word)
            i += 1
            continue
        value = SMALL.get(word, TENS.get(word))
        if value is None:
            return None
        i += 1
        if i < len(tokens) and tokens[i] == "hundred":
            if not 1 <= value <= 9:
                return None
            value *= 100
            i += 1
            if i < len(tokens) and tokens[i] == "and":
                i += 1
            if i < len(tokens) and tokens[i] in TENS:
                value += TENS[tokens[i]]
                i += 1
                if i < len(tokens) and tokens[i] in SMALL and 0 < SMALL[tokens[i]] < 10:
                    value += SMALL[tokens[i]]
                    i += 1
            elif i < len(tokens) and tokens[i] in SMALL:
                value += SMALL[tokens[i]]
                i += 1
        elif word in TENS and i < len(tokens) and tokens[i] in SMALL and 0 < SMALL[tokens[i]] < 10:
            value += SMALL[tokens[i]]
            i += 1
        groups.append(str(value))
    return "".join(groups)


def spoken_route_candidates(text: str) -> list[str]:
    """Return ordered, deduplicated candidates; reject unrelated words.

    Only a route phrase belongs here; intents remove command words first.
    Unknown GTFS aliases can still be matched literally by the intent parser.
    """
    tokens = re.findall(r"[a-z]+|\d+", text.lower().replace("-", " "))
    if not tokens or len(tokens) > 12 or re.search(r"[^a-zA-Z0-9\s.\-]", text):
        return []
    # 'double you' is the letter W, not a digit repetition operator.
    tokens = " ".join(tokens).replace("double you", "w").split()
    alternatives = []
    for token in tokens:
        if token == "oh":
            alternatives.append([("number", "zero"), ("letter", "O")])
        elif token in SMALL or token in TENS or token in {"hundred", "and"} or token.isdigit():
            alternatives.append([("number", token)])
        elif token in LETTERS:
            alternatives.append([("letter", LETTERS[token])])
        else:
            return []
    results = []
    for combination in itertools.product(*alternatives):
        kinds = [kind for kind, _ in combination]
        if "number" not in kinds:
            continue
        start = kinds.index("number")
        end = len(kinds) - kinds[::-1].index("number")
        if any(kind != "number" for kind in kinds[start:end]):
            continue
        digits = _number_groups([value for _, value in combination[start:end]])
        if digits is None:
            continue
        candidate = ("".join(value for _, value in combination[:start]) + digits
                     + "".join(value for _, value in combination[end:]))
        if SHAPE.fullmatch(candidate) and candidate not in results:
            results.append(candidate)
    return results


parse_spoken_route = spoken_route_candidates
