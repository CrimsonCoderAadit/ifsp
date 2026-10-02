import csv
from zipfile import ZipFile

import pytest

from transit.gtfs_loader import base_route, build_routes, load_routes


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
