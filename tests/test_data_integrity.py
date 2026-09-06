"""Integrity checks on the data layer."""

from __future__ import annotations

import json

import pytest

from tokyoline.data import (
    DATA_DIR,
    load_congestion,
    load_gradients,
    load_ridership,
    load_segments,
    load_stations,
)
from tokyoline.fleet import Bounded, load_fleet
from tokyoline.network import load_coordinates

VALID_STATUSES = {
    "published", "derived", "modelled", "bounded", "unknown",
    "not_modelled", "published_ambiguous", "published_unverified_definition",
    "published_qualitative", "published_baseline_unverified",
    "published_section_only", "not_yet_sourced", "not_sourced",
}


def test_thirty_stations():
    assert len(load_stations()) == 30
    assert len(load_segments()) == 30


def test_segment_distances_sum_to_the_published_loop():
    """34.5 km is published. The loader asserts this too, but a test states."""
    assert sum(s.distance_m for s in load_segments()) == pytest.approx(34_500,
                                                                      abs=1)


def test_cumulative_kilometrage_is_consistent_with_the_gaps():
    df = load_stations()
    cum = df["cumulative_km"].tolist()
    gaps = df["dist_from_prev_km"].tolist()
    for i in range(1, len(cum)):
        assert cum[i] - cum[i - 1] == pytest.approx(gaps[i], abs=0.001)


def test_segments_form_a_closed_ring():
    segs = load_segments()
    for i, s in enumerate(segs):
        assert s.to_station == segs[(i + 1) % len(segs)].from_station


def test_every_segment_is_a_plausible_length():
    for s in load_segments():
        assert 400 <= s.distance_m <= 2100, s


def test_jy_numbering_is_complete_and_unique():
    """JR East's published station numbering. JY01 to JY30, each used once."""
    df = load_stations()
    numbers = df["jy_number"].tolist()
    assert len(set(numbers)) == 30
    assert set(numbers) == {f"JY{i:02d}" for i in range(1, 31)}


def test_jy_numbering_runs_opposite_to_the_file_order():
    """JY01 is Tokyo and the numbers increase in the inner-loop direction,."""
    df = load_stations()
    by_name = dict(zip(df["name_en"], df["jy_number"]))
    assert by_name["Tokyo"] == "JY01"
    assert by_name["Shinagawa"] == "JY25"
    assert by_name["Shinjuku"] == "JY17"
    assert by_name["Yurakucho"] == "JY30"
    # Shinagawa is first in file order but numbered 25.
    assert df.iloc[0]["name_en"] == "Shinagawa"


def test_coordinates_cover_every_station():
    coords = load_coordinates()
    names = {s.from_station for s in load_segments()}
    assert set(coords["name_en"]) == names


def test_coordinates_are_inside_tokyo():
    coords = load_coordinates()
    assert coords["lat"].between(35.5, 35.9).all()
    assert coords["lon"].between(139.6, 139.9).all()


def test_ridership_covers_every_station_exactly_once():
    r = load_ridership()
    names = {s.from_station for s in load_segments()}
    assert set(r["name_en"]) == names
    assert len(r) == 30


def test_ridership_ranks_are_one_to_thirty():
    r = load_ridership()
    assert sorted(r["rank_2024"]) == list(range(1, 31))


def test_ridership_is_ordered_by_rank():
    r = load_ridership().sort_values("rank_2024")
    boardings = r["boardings_2024"].tolist()
    assert all(b <= a for a, b in zip(boardings, boardings[1:]))
    assert boardings[0] > 600_000       # Shinjuku
    assert boardings[-1] > 0


def test_gradient_profile_covers_every_segment():
    assert len(load_gradients()) == 30


def test_elevation_returns_to_its_starting_point():
    """A closed loop cannot have a net climb. If this fails the profile was."""
    import csv

    with open(DATA_DIR / "gradients.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(
            line for line in fh if not line.startswith("#"))]
    net_rise = sum(float(r["rise_m"]) for r in rows)
    assert net_rise == pytest.approx(0.0, abs=0.1)


def test_gradients_are_consistent_with_rise_and_distance():
    import csv

    with open(DATA_DIR / "gradients.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(
            line for line in fh if not line.startswith("#"))]
    for r in rows:
        expected = 1000.0 * float(r["rise_m"]) / float(r["distance_m"])
        assert float(r["grade_permille"]) == pytest.approx(expected, abs=0.01)


def _tagged_nodes(node, path=""):
    """Yield (path, dict) for every mapping that carries a 'value' key."""
    if isinstance(node, dict):
        if "value" in node:
            yield path, node
        for k, v in node.items():
            yield from _tagged_nodes(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _tagged_nodes(v, f"{path}[{i}]")


#: Files using the value/status convention, and their minimum node count.
TAGGED_FILES = {
    "rolling_stock.json": 15,
    "fleet.json": 50,
    "timetable_anchors.json": 5,
}


@pytest.mark.parametrize("filename,minimum", TAGGED_FILES.items())
def test_every_valued_parameter_declares_its_provenance(filename, minimum):
    with open(DATA_DIR / filename, encoding="utf-8") as fh:
        blob = json.load(fh)

    nodes = list(_tagged_nodes(blob))
    assert len(nodes) >= minimum, (
        f"{filename}: only {len(nodes)} valued parameters found; the check "
        "would be close to vacuous"
    )

    untagged = [
        path for path, node in nodes
        if "status" not in node and not path.startswith("_")
    ]
    assert not untagged, f"{filename}: untagged parameters {untagged}"


@pytest.mark.parametrize("filename", TAGGED_FILES)
def test_declared_statuses_are_from_the_known_set(filename):
    with open(DATA_DIR / filename, encoding="utf-8") as fh:
        blob = json.load(fh)
    for path, node in _tagged_nodes(blob):
        status = node.get("status")
        if status is not None:
            assert status in VALID_STATUSES, f"{filename}:{path} -> {status}"


def test_the_provenance_walker_actually_detects_a_missing_tag():
    """Negative control. A checker that cannot fail is not a check."""
    untagged = [
        path for path, node in _tagged_nodes({"speed": {"value": 90.0}})
        if "status" not in node
    ]
    assert untagged == ["speed"]


def test_prose_documented_files_still_cite_their_sources():
    """congestion.json and regeneration.json record provenance in prose rather."""
    for filename in ("congestion.json", "regeneration.json"):
        with open(DATA_DIR / filename, encoding="utf-8") as fh:
            raw = fh.read()
        blob = json.loads(raw)
        assert "_source" in blob or "_comment" in blob, filename
        # Every figure in these two came from a named body.
        assert "MLIT" in raw or "JR East" in raw or "国土交通省" in raw, filename


def test_fleet_has_the_five_documented_classes():
    assert set(load_fleet()) == {"E235-0", "E233-0", "E5", "E7", "N700S"}


def test_n700s_mass_stays_a_bound():
    """Published only as 'under 700 t'. If this ever becomes a point value,."""
    n700s = load_fleet()["N700S"]
    assert n700s.tare_mass_t is None
    assert n700s.tare_mass_bound_t == pytest.approx(700.0)
    assert n700s.mass_is_bound
    assert isinstance(n700s.power_to_weight_kw_per_t(), Bounded)


def test_classes_with_known_mass_do_not_report_bounds():
    for key in ("E235-0", "E233-0", "E5", "E7"):
        m = load_fleet()[key]
        assert not m.mass_is_bound
        assert not isinstance(m.power_to_weight_kw_per_t(), Bounded)


def test_unpublished_deceleration_stays_unpublished():
    """The E5 and N700S publish no service rate. Filling one in with a peer."""
    for key in ("E5", "N700S"):
        assert load_fleet()[key].service_deceleration_kmh_s is None
        assert load_fleet()[key].stopping_distance_m() is None


def test_e7_keeps_its_speed_dependent_curve():
    e7 = load_fleet()["E7"]
    assert e7.deceleration_curve is not None
    speeds = [p[0] for p in e7.deceleration_curve]
    rates = [p[1] for p in e7.deceleration_curve]
    assert speeds == sorted(speeds)
    assert rates[0] > rates[-1], "rate must fall as speed rises"


def test_motor_counts_are_consistent_with_formation():
    for m in load_fleet().values():
        assert 0 < m.n_motor_cars <= m.n_cars
        assert 0.0 < m.motored_fraction <= 1.0


def test_congestion_records_only_the_measured_section():
    """MLIT publishes one critical section per direction. Claiming more would."""
    c = load_congestion()
    outer = c["critical_sections"]["outer_loop"]
    assert outer["from"] == "Ueno"
    assert outer["to"] == "Okachimachi"
    assert outer["congestion_rate_percent"] == 125
    # The inner loop rate is unknown and must stay null.
    assert c["critical_sections"]["inner_loop"]["congestion_rate_percent"] is None
