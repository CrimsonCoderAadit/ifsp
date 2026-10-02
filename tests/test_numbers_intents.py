import pytest

from speech.intents import IntentType, parse_intent
from speech.numbers import spoken_route_candidates
from speech.stt import Transcript


@pytest.mark.parametrize("text,expected", [
    ("twenty one g", "21G"), ("two one g", "21G"), ("twenty one gee", "21G"),
    ("five seventy", "570"), ("five seven zero", "570"), ("m seventy", "M70"),
    ("em seventy", "M70"), ("one oh two", "102"), ("a one", "A1"),
    ("nineteen b", "19B"), ("twenty three c", "23C"), ("twenty three see", "23C"),
    ("twenty one dee", "21D"), ("d seventy", "D70"), ("ex seventy", "X70"),
    ("x seventy", "X70"), ("en forty five bee", "N45B"),
    ("one hundred and two", "102"), ("five hundred seventy", "570"),
    ("five hundred and seventy", "570"), ("fifty four", "54"),
    ("one five nine ay", "159A"), ("21 G", "21G"), ("M-70", "M70"),
    ("5c", "5C"), ("twenty one ee", "21E"), ("double you seventy", "W70"),
    ("TWENTY-ONE G", "21G"),
])
def test_spoken_routes(text, expected):
    assert expected in spoken_route_candidates(text)


@pytest.mark.parametrize("text", ["", "hello world", "twenty cats", "bus twenty one g", "one thousand", "hundred", "and one", "1234", "21/G", "21தமிழ்"])
def test_invalid_routes(text):
    assert spoken_route_candidates(text) == []


def test_ambiguous_oh():
    assert spoken_route_candidates("oh one") == ["01", "O1"]
    assert spoken_route_candidates("one oh") == ["10", "1O"]


@pytest.mark.parametrize("text,target", [
    ("find bus 21G", "21G"), ("look for 21 G", "21G"), ("where is the 570", "570"),
    ("bus twenty one g", "21G"), ("is 21G here", "21G"),
    ("find bus twenty one gee", "21G"), ("look for bus two one g", "21G"),
    ("where is bus five seventy", "570"), ("bus em seventy", "M70"),
    ("find bus one oh two", "102"), ("find bus a one", "A1"),
    ("bus nineteen bee", "19B"), ("Please find bus twenty three see.", "23C"),
    ("search for the bus 21G please", "21G"),
])
def test_bus_intents(text, target):
    intent = parse_intent(text, {"21G", "570", "M70", "102", "A1", "19B", "23C"})
    assert (intent.type, intent.target, intent.raw) == (IntentType.FIND_BUS, target, text)


@pytest.mark.parametrize("text,kind", [
    ("stop looking", IntentType.CANCEL_BUS), ("cancel bus", IntentType.CANCEL_BUS),
    ("stop looking for the bus", IntentType.CANCEL_BUS),
    ("what is ahead", IntentType.WHATS_AHEAD), ("what's ahead", IntentType.WHATS_AHEAD),
    ("whats ahead", IntentType.WHATS_AHEAD), ("what is in front of me", IntentType.WHATS_AHEAD),
    ("for is ahead", IntentType.WHATS_AHEAD), ("where is ahead", IntentType.WHATS_AHEAD),
    ("ahead", IntentType.WHATS_AHEAD),
    ("repeat", IntentType.REPEAT), ("say that again", IntentType.REPEAT),
    ("stop", IntentType.STOP), ("silence", IntentType.STOP), ("stop speaking", IntentType.STOP),
    ("help", IntentType.HELP), ("what can i say", IntentType.HELP),
    ("hello", IntentType.UNKNOWN), ("", IntentType.UNKNOWN),
    ("don't stop", IntentType.UNKNOWN), ("find bus elephants", IntentType.UNKNOWN),
])
def test_command_intents(text, kind):
    assert parse_intent(text).type == kind


@pytest.mark.parametrize("text", ["find bus", "look for", "where is the bus", "bus", "find bus number"])
def test_clarify_route(text):
    intent = parse_intent(text)
    assert intent.type == IntentType.FIND_BUS and intent.target is None
    assert intent.prompt == "Which bus number?"


def test_transcript_input():
    assert parse_intent(Transcript("repeat", .9, 20)).type == IntentType.REPEAT


def test_validate_routes_and_reject_ambiguity():
    assert parse_intent("bus oh one", {"O1"}).target == "O1"
    assert parse_intent("bus oh one", {"01", "O1"}).target is None
    assert parse_intent("bus 21G", {"23G"}).target is None
    assert parse_intent("bus N45BB", {"N45B"}).target is None


def test_exact_gtfs_alias():
    assert parse_intent("find bus 29AEX", {"29AEX"}).target == "29AEX"
