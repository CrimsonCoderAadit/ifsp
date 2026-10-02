import pytest

from transit.route_matcher import extract_route_candidates, match_route, normalize_route


@pytest.mark.parametrize("text,expected", [
    ("21 G", "21G"), ("21-G", "21G"), ("21g", "21G"), ("21G.", "21G"),
    ("M 70", "M70"), ("M-70", "M70"), (" 570 ", "570"), ("N45B", "N45B"),
    ("S31", "S31"), ("A1", "A1"), ("D70", "D70"), ("159A", "159A"),
    ("1", "1"), ("5C", "5C"), ("21GG", "21GG"), ("M7O", "M7O"),
    ("M7l", "M7L"), ("M7I", "M7I"), ("M7S", "M7S"),
    ("M7O7", "M707"), ("M7I7", "M717"), ("M7S7", "M757"),
    ("M7B7", "M787"), ("M7Z7", "M727"), ("M7G7", "M767"),
    ("", None), ("BUS", None), ("சென்னை", None), ("1234", None),
    ("21/G", None), ("@21G", None), ("MAA1", None), ("29AEX", None),
])
def test_normalize(text, expected):
    assert normalize_route(text) == expected


@pytest.mark.parametrize("text,route,method", [
    ("21 G", "21G", "exact"), ("21-G", "21G", "exact"),
    ("21g", "21G", "exact"), ("21G.", "21G", "exact"),
    ("M 70", "M70", "exact"), ("M-70", "M70", "exact"),
    ("M7O", "M70", "confusion_fix"), ("M7I", "M71", "confusion_fix"),
    ("M7l", "M71", "confusion_fix"), ("M7S", "M75", "confusion_fix"),
    ("M7B", "M78", "confusion_fix"), ("M7Z", "M72", "confusion_fix"),
    ("M7G", "M76", "confusion_fix"), ("216", "21G", "confusion_fix"),
    ("218", "21B", "confusion_fix"), ("215", "21S", "confusion_fix"),
    ("212", "21Z", "confusion_fix"), ("210", "21O", "confusion_fix"),
    ("211", "21I", "confusion_fix"), ("2IG", "21G", "confusion_fix"),
    ("2lG", "21G", "confusion_fix"), ("2SG", "25G", "confusion_fix"),
    ("2BG", "28G", "confusion_fix"), ("2ZG", "22G", "confusion_fix"),
    ("2GG", "26G", "confusion_fix"), ("29AEX", "29AEX", "exact"),
    ("MAA1", "MAA1", "exact"),
    ("221 RE1", "221 RE1", "exact"),
])
def test_matching(text, route, method):
    result = match_route(text, {route})
    assert (result.route, result.method) == (route, method)


@pytest.mark.parametrize("candidate,routes", [
    ("21G", {"23G"}), ("23G", {"21G"}), ("21G", {"21", "23G"}),
    ("999X", {"21G", "570"}), ("INVALID", {"21G"}),
    ("29AEX", {"29AE"}), ("MAA1", {"MA1"}),
])
def test_reject(candidate, routes):
    assert match_route(candidate, routes).method == "none"


def test_digit_guard_even_at_zero_threshold():
    assert match_route("21G", {"23G"}, threshold=0).route is None


def test_fuzzy_same_digits():
    result = match_route("N45BB", {"N45B"})
    assert (result.route, result.method) == ("N45B", "fuzzy")


def test_exact_beats_confusion():
    assert match_route("216", {"216", "21G"}).method == "exact"
    assert match_route("M7S", {"M7S", "M75"}).route == "M7S"


def test_ambiguous_confusions_rejected():
    assert match_route("2IG", {"21G", "216"}).route == "21G"
    assert match_route("21B", {"218", "2IB"}).route is None


def test_candidate_lines():
    candidates = extract_route_candidates(["சென்னை", "TAMBARAM", "21 G", "M-70"])
    assert "21G" in candidates and "M70" in candidates
    assert "TAMBARAM" not in candidates
    assert extract_route_candidates(["21 G TAMBARAM"])[0] == "21G"


def test_box_ranking():
    lines = [
        {"text": "23C", "box": [[0, 200], [30, 200], [30, 210], [0, 210]]},
        {"text": "21G", "box": [[0, 0], [90, 0], [90, 60], [0, 60]]},
    ]
    assert extract_route_candidates(lines)[0] == "21G"
    assert extract_route_candidates([[lines[1]["box"], "21 G", .95]])[0] == "21G"


@pytest.mark.parametrize("lines,expected", [(["21", "G"], "21G"), (["M", "70"], "M70")])
def test_split_route_tokens(lines, expected):
    assert extract_route_candidates(lines)[0] == expected


def test_do_not_join_different_board_rows():
    lines = [
        {"text": "21", "box": [[0, 0], [30, 0], [30, 20], [0, 20]]},
        {"text": "G", "box": [[35, 100], [45, 100], [45, 120], [35, 120]]},
    ]
    assert "21G" not in extract_route_candidates(lines)
