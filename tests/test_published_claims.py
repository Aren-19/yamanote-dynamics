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

import numpy as np
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


def test_distance_rounding_is_worth_about_one_second_over_the_loop(runs):
    """README: published kilometrage is stated to 0.1 km, and the resulting
    uncertainty on the circuit minimum is under a second.

    It stays that small because the loop is closed. Each station position
    appears in two segments with opposite sign, so its rounding error largely
    cancels; only the residual survives. If a change ever broke that structure
    -- distances read independently per segment rather than differenced from
    cumulative kilometrage -- the spread would grow by a factor of about the
    square root of thirty, and this would catch it."""
    traction, brake, segments, _ = runs
    distances = np.array([s.distance_m for s in segments])
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])[:-1]
    loop_m = distances.sum()

    rng = np.random.default_rng(20260905)
    totals = []
    for _ in range(40):
        positions = cumulative + rng.uniform(-50.0, 50.0, size=cumulative.size)
        perturbed = np.diff(np.concatenate([positions, [positions[0] + loop_m]]))
        totals.append(sum(
            simulate_segment(traction, brake, float(d),
                             speed_limit_ms=traction.spec.max_speed_ms,
                             dt=0.2).run_time_s
            for d in perturbed
        ))

    assert np.std(totals) < 1.5
    assert abs(np.mean(totals) - 2309) < 3


def test_fifty_metres_is_worth_two_to_three_seconds_on_one_segment(runs):
    """The other half of the same finding, and the reason per-segment results
    carry no claim: a single segment has no cancelling partner, so the same
    50 m of rounding lands whole."""
    traction, brake, _, _ = runs
    for distance in (500.0, 1200.0, 2000.0):
        pair = [
            simulate_segment(traction, brake, d,
                             speed_limit_ms=traction.spec.max_speed_ms,
                             dt=0.05).run_time_s
            for d in (distance, distance + 50.0)
        ]
        assert 1.8 < pair[1] - pair[0] < 3.0


def test_input_rounding_dominates_integration_error(runs):
    """The claim that justifies stopping at dt = 0.1 s. Refining the step buys
    nothing while the input is known only to 50 m."""
    traction, brake, segments, _ = runs
    coarse = sum(
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.1).run_time_s
        for s in segments
    )
    fine = sum(
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.01).run_time_s
        for s in segments
    )
    # Two orders of magnitude below the 0.83 s that rounding contributes.
    assert abs(coarse - fine) < 0.01


def test_eleven_of_thirty_segments_never_reach_line_speed(runs):
    """README: a speed restriction costs nothing where the train would not have
    reached it, and on a third of the line it would not have.

    Those segments are short enough that the train is still accelerating when
    it must start braking, so the peak is set by the distance."""
    *_, results = runs
    below = [r for r in results if not r.reached_line_speed]
    assert len(below) == 11
    assert min(r.max_speed_kmh for r in below) == pytest.approx(67.3, abs=0.5)


def test_a_51_kmh_line_wide_cap_would_absorb_the_whole_residual(runs):
    """README: restrictions alone could explain the 598 s residual only under
    a line-wide cap 43 percent below the published 90 km/h line speed.

    The bound works because run time falls strictly with the speed limit, so a
    uniform cap is an upper bound on any set of per-segment restrictions no
    lower than it."""
    traction, brake, segments, _ = runs

    def circuit(cap_kmh):
        return sum(
            simulate_segment(traction, brake, s.distance_m,
                             speed_limit_ms=cap_kmh / 3.6, dt=0.1).run_time_s
            for s in segments
        )

    baseline = circuit(90.0)
    lo, hi = 20.0, 90.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if circuit(mid) - baseline > 598.0:
            lo = mid
        else:
            hi = mid
    assert 49.0 < 0.5 * (lo + hi) < 53.0


def test_a_severe_75_kmh_cap_accounts_for_a_seventh_of_the_residual(runs):
    """README: the realistic order of the restriction effect. Stated so the
    bound is not read as a claim that restrictions are large."""
    traction, brake, segments, _ = runs
    cost = sum(
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=75.0 / 3.6, dt=0.1).run_time_s
        for s in segments
    ) - 2309.0
    assert 0.10 < cost / 598.0 < 0.20


def test_load_moves_implied_dwell_by_under_four_seconds():
    """README: passenger load moves implied dwell by 3.7 s across the whole
    plausible range, and lighter trains imply MORE dwell, not less.

    The direction matters more than the size. Every headline figure assumes
    100 percent of published capacity while the published loop time it is
    compared against is the mean of all services, most of them far lighter. So
    the reported dwell is, if anything, understated -- the bias runs against
    the result rather than toward it."""
    segments = load_segments()
    implied = {}
    for load in (0.0, 1.0, 1.6):
        traction, brake = build_model(load_factor=load)
        d = decompose(traction, brake, segments, dwell_s=DWELL_SCENARIO_S,
                      published_loop_s=PUBLISHED_CIRCUIT_S, line_speed_kmh=90.0)
        implied[load] = dwell_implied_by_margin(
            d.standard_run_time_s, PUBLISHED_CIRCUIT_S, len(segments), 0.05)

    assert implied[0.0] > implied[1.0] > implied[1.6]
    assert implied[0.0] - implied[1.6] < 4.5
    for value in implied.values():
        assert 45.0 < value < 51.9


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
