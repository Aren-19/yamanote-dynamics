"""Guard the numbers the README states publicly.

Unlike the property tests, these deliberately pin specific values. The README
makes quantitative claims to a reader who cannot run the code, so those claims
need to fail loudly when the model moves rather than drift quietly out of date.

A failure here is not necessarily a bug. It may mean the model improved and the
README needs updating. What it must never mean is that the published numbers
are wrong and nobody noticed.

Tolerances are wide enough to absorb integration noise and narrow enough to
catch a real change in behaviour.
"""

from __future__ import annotations

import pytest

from tokyoline import build_model, load_segments, simulate_segment
from tokyoline.energy import circuit_energy
from tokyoline.fleet import load_fleet
from tokyoline.validate import (
    decompose,
    dwell_implied_by_margin,
    dwell_implied_by_zero_margin,
    load_anchors,
)

PUBLISHED_CIRCUIT_S = 3948
DWELL_SCENARIO_S = 32.0


@pytest.fixture(scope="module")
def runs():
    traction, brake = build_model(load_factor=1.0)
    segments = load_segments()
    results = [
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
        for s in segments
    ]
    return traction, brake, segments, results


def test_line_is_30_segments_and_34_5_km(runs):
    _, _, segments, _ = runs
    assert len(segments) == 30
    assert sum(s.distance_m for s in segments) == pytest.approx(34_500, abs=1)


def test_published_circuit_time(runs):
    assert load_anchors()["loop_times"]["inner_loop_mean_s"]["value"] == \
        PUBLISHED_CIRCUIT_S


def test_minimum_circuit_time_is_2309_s(runs):
    """README: 'a minimum circuit time of 2,309 s'."""
    _, _, _, results = runs
    assert sum(r.run_time_s for r in results) == pytest.approx(2309, abs=3)


def test_standard_run_time_is_2390_s(runs):
    """README: 'a standard run time of 2,390 s'."""
    traction, brake, segments, _ = runs
    d = decompose(traction, brake, segments, dwell_s=DWELL_SCENARIO_S,
                  published_loop_s=PUBLISHED_CIRCUIT_S, line_speed_kmh=90.0)
    assert d.standard_run_time_s == pytest.approx(2390, abs=5)


def test_implied_dwell_is_39_to_50_seconds(runs):
    """README: 'Implied mean station dwell is 39-50 s'.

    The range spans two braking assumptions and margins from 3 to 10 percent.
    """
    traction, brake, segments, _ = runs
    d = decompose(traction, brake, segments, dwell_s=DWELL_SCENARIO_S,
                  published_loop_s=PUBLISHED_CIRCUIT_S, line_speed_kmh=90.0)
    at_10 = dwell_implied_by_margin(d.standard_run_time_s,
                                    PUBLISHED_CIRCUIT_S, 30, 0.10)
    at_3 = dwell_implied_by_margin(d.standard_run_time_s,
                                   PUBLISHED_CIRCUIT_S, 30, 0.03)
    assert 39.0 <= at_10 <= 50.0
    assert 39.0 <= at_3 <= 50.0
    assert at_3 > at_10


def test_zero_margin_dwell_bound_is_51_9_s(runs):
    """README: 'A mean dwell of 51.9 s or more would imply no recovery
    margin at all'."""
    traction, brake, segments, _ = runs
    d = decompose(traction, brake, segments, dwell_s=DWELL_SCENARIO_S,
                  published_loop_s=PUBLISHED_CIRCUIT_S, line_speed_kmh=90.0)
    bound = dwell_implied_by_zero_margin(d.standard_run_time_s,
                                         PUBLISHED_CIRCUIT_S, 30)
    assert bound == pytest.approx(51.9, abs=0.3)


def test_92_percent_of_traction_work_goes_to_the_brakes(runs):
    """README: '92% of traction work is dissipated in braking'."""
    traction, _, _, results = runs
    total, _ = circuit_energy(results, traction, dwell_s=DWELL_SCENARIO_S)
    share = 100 * total.brake_system_j / total.traction_j
    assert share == pytest.approx(92.2, abs=1.0)


def test_specific_consumption_is_1_82_kwh_per_car_km(runs):
    """README: '1.82 kWh per car-km ... below the published 2-4 band'."""
    traction, _, _, results = runs
    total, _ = circuit_energy(results, traction, dwell_s=DWELL_SCENARIO_S)
    per_car = total.kwh_per_car_km(traction.spec.n_cars)
    assert per_car == pytest.approx(1.82, abs=0.05)
    assert per_car < 2.0, "README states this sits BELOW the published band"


def test_model_regeneration_rate_is_54_7_percent(runs):
    """README: 'The model produces 54.7%' against a published 59.0%."""
    traction, _, _, results = runs
    total, _ = circuit_energy(results, traction, dwell_s=DWELL_SCENARIO_S)
    rate = 100 * total.regenerated_j / total.traction_j
    assert rate == pytest.approx(54.7, abs=1.0)
    assert rate < 59.0, "README states the model falls short of the published rate"


def test_platform_error_is_zero(runs):
    """README: 'Final position error 0.0 m on all 30 segments'."""
    _, _, _, results = runs
    assert max(abs(r.position_error_m) for r in results) < 1e-9


def test_e7_stops_in_5670_m_from_275_kmh():
    """README: 'Integrating the real curve gives a 5,670 m stop'."""
    e7 = load_fleet()["E7"]
    assert e7.stopping_distance_m(jerk=0.75) == pytest.approx(5670, abs=15)


def test_constant_rate_assumption_understates_the_e7_stop_by_30_percent():
    """README: 'assuming the low-speed rate throughout gives 3,943 m,
    understating by 30%'."""
    from tokyoline import BrakeProfile

    e7 = load_fleet()["E7"]
    real = e7.stopping_distance_m(jerk=0.75)
    naive = BrakeProfile(a_max=2.69 / 3.6, jerk=0.75).distance_to_stop(275 / 3.6)
    assert naive == pytest.approx(3943, abs=20)
    assert (real - naive) / real == pytest.approx(0.30, abs=0.03)


def test_power_per_passenger_separates_the_categories():
    """README: '13 kW/passenger for a Shinkansen against 2 for a commuter'."""
    fleet = load_fleet()
    commuter = fleet["E235-0"].total_power_kw / fleet["E235-0"].capacity
    highspeed = fleet["E5"].total_power_kw / fleet["E5"].capacity
    assert commuter == pytest.approx(1.95, abs=0.15)
    assert highspeed == pytest.approx(13.1, abs=0.5)
