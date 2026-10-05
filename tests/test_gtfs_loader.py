import csv
from zipfile import ZipFile

import pytest

from transit.gtfs_loader import ROOT, base_route, build_routes, load_route_stops, load_routes


@pytest.mark.parametrize("raw,base", [
    ("21G CT2", "21G"), ("21G R3", "21G"), ("21G Cut1", "21G"),
    ("21G ct7", "21G"), ("11 R", "11"), ("22 CT", "22"),
    ("593Ct", "593"), ("21EEt", "21E"), ("C56CEt R", "C56C"),
    ("21G", "21G"), ("29AEX", "29AEX"), ("MAA1", "MAA1"),
])
def test_variants(raw, base):
    assert base_route(raw) == base


def test_only_routes_and_bus_filter(tmp_path):
    archive = tmp_path / "feed.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("routes.txt", "\ufeffroute_short_name,route_long_name,route_type\n"
                        "21G CT2,A to B,3\n21G R3,C to D,3\n29AEX,Alias,3\nMetro,Train,1\n")
        stream.writestr("calendar.txt", "junk,do not parse")
    output = tmp_path / "routes.csv"
    stats = build_routes(archive, output)
    assert (stats.raw_routes, stats.bus_routes, stats.base_routes,
            stats.rejected_shape_routes, stats.rejected_shape_rows) == (4, 3, 2, 1, 1)
    assert load_routes(output) == {"21G", "29AEX"}
    with output.open() as stream:
        row = next(csv.DictReader(stream))
        assert row["raw_short_names"] == "21G CT2 | 21G R3"


def test_fallback():
    from pathlib import Path
    routes = load_routes(Path("does-not-exist.csv"))
    assert len(routes) == 20 and "21G" in routes


def test_route_stops_follow_stop_sequence_per_direction(tmp_path):
    archive = tmp_path / "feed.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("routes.txt", "route_id,route_short_name,route_long_name,route_type\n"
                        "out,102A,Thiruvanmiyur TO Pudupakkam Rd.Jn,3\n"
                        "back,102A,Pudupakkam Rd.Jn TO Thiruvanmiyur,3\n"
                        "cut,102A CT,Thiruvanmiyur TO Navalur,3\n"
                        "other,515,Tambaram West TO Mamallapuram,3\n")
        stream.writestr("trips.txt", "route_id,trip_id\nout,o1\nout,o2\nout,o3\n"
                        "back,b1\nback,b2\ncut,c1\nother,x1\n")
        stream.writestr("stops.txt", "stop_id,stop_name\nT,Thiruvanmiyur\nK,Kandanchavadi\n"
                        "N,Navalur\nP,Pudupakkam\nM,Mamallapuram\n")
        stream.writestr("stop_times.txt", "trip_id,stop_id,stop_sequence\n"
                        "o1,P,10\no1,T,1\no1,N,9\no1,K,2\n"
                        "o2,T,1\no2,K,2\no2,N,9\no2,P,10\n"
                        "o3,T,1\no3,N,2\n"
                        "b1,P,1\nb1,N,2\nb1,K,3\nb1,T,4\n"
                        "b2,P,1\nb2,N,2\nb2,K,3\nb2,T,4\n"
                        "c1,T,1\nc1,N,2\nx1,M,1\n")
    result = load_route_stops("102A", zip_path=archive)
    assert result == [
        {"route_name": "Pudupakkam Rd.Jn → Thiruvanmiyur",
         "stops": ["Pudupakkam", "Navalur", "Kandanchavadi", "Thiruvanmiyur"], "trip_count": 2},
        {"route_name": "Thiruvanmiyur → Pudupakkam Rd.Jn",
         "stops": ["Thiruvanmiyur", "Kandanchavadi", "Navalur", "Pudupakkam"], "trip_count": 2},
        {"route_name": "Thiruvanmiyur → Navalur", "stops": ["Thiruvanmiyur", "Navalur"], "trip_count": 1},
    ]
    assert load_route_stops("576", zip_path=archive) == []


@pytest.mark.skipif(not (ROOT / "data/mtc-gtfs.zip").exists(), reason="MTC GTFS feed not installed")
def test_real_102a_has_two_complete_opposite_directions():
    first, second = load_route_stops("102A")
    assert {first["route_name"], second["route_name"]} == {
        "Thiruvanmiyur → Pudupakkam Rd.Jn", "Pudupakkam Rd.Jn → Thiruvanmiyur"}
    assert len(first["stops"]) >= 30 and len(second["stops"]) >= 30
    assert first["stops"][0] == second["stops"][-1] and first["stops"][-1] == second["stops"][0]
    assert "Thiruvanmiyur Bus Stand" in first["stops"] and "Tidel Park" in second["stops"]
