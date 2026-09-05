"""The energy balance across every driving regime.

A segment starts and ends at rest, so the kinetic term drops out and

    E_traction = E_resistance + E_brake_system + E_grade

must hold exactly. Both sides are computed independently, which makes this a
test of the accounting rather than a restatement of it.

The identity was previously written without the gravitational term and tested
only on level track, where that term is zero. On a gradient it failed by up to
104 percent. Three separate omissions had to be fixed:

  * gravitational work was not counted at all;
  * the brake system was credited with the whole commanded deceleration, when
    uphill gravity supplies part of it and downhill opposes it;
  * holding a constant speed downhill requires braking, which the cruise phase
    recorded as no force at all, so the work gravity did was absorbed by
    nothing.

The sign of the failure flipping between climb and descent is what identified
the second and third. These tests exist so that cannot happen again quietly.
"""

from __future__ import annotations

import pytest

from tokyoline import build_model, load_segments, simulate_segment
from tokyoline.energy import segment_energy

TOLERANCE = 5e-3


@pytest.fixture(scope="module")
def setup():
    traction, brake = build_model(load_factor=1.0)
    segment = max(load_segments(), key=lambda s: s.distance_m)
    return traction, brake, segment


def balance(setup, **kwargs):
    traction, brake, segment = setup
    result = simulate_segment(traction, brake, segment.distance_m,
                              speed_limit_ms=traction.spec.max_speed_ms,
                              dt=0.02, **kwargs)
    breakdown, _ = segment_energy(result, traction, auxiliary_kw=0.0)
    return breakdown


def test_balance_holds_on_level_track(setup):
    assert balance(setup).balance_error_fraction < TOLERANCE


def test_balance_holds_while_coasting(setup):
    _, _, segment = setup
    b = balance(setup, coast_start_m=segment.distance_m * 0.4)
    assert b.balance_error_fraction < TOLERANCE


@pytest.mark.parametrize("grade", [2.0, 5.0, 10.0, 20.0, 34.0])
def test_balance_holds_climbing(setup, grade):
    b = balance(setup, grade_permille=grade)
    assert b.balance_error_fraction < TOLERANCE
    assert b.grade_j > 0.0


@pytest.mark.parametrize("grade", [-2.0, -5.0, -10.0, -20.0, -34.0])
def test_balance_holds_descending(setup, grade):
    """Descents were the worst case: braking to hold speed downhill was
    recorded as no force at all."""
    b = balance(setup, grade_permille=grade)
    assert b.balance_error_fraction < TOLERANCE
    assert b.grade_j < 0.0


def test_balance_holds_coasting_downhill(setup):
    _, _, segment = setup
    b = balance(setup, coast_start_m=segment.distance_m * 0.5,
                grade_permille=-10.0)
    assert b.balance_error_fraction < TOLERANCE


def test_gravitational_work_matches_the_closed_form(setup):
    """Path independent: m*g*sin(theta)*L, whatever the speed profile."""
    traction, _, segment = setup
    grade = 15.0
    b = balance(setup, grade_permille=grade)
    mass = traction.spec.mass_kg(traction.load_factor)
    expected = mass * 9.80665 * (grade / 1000.0) * segment.distance_m
    assert b.grade_j == pytest.approx(expected, rel=1e-9)


def test_grade_work_is_zero_on_level_track(setup):
    assert balance(setup).grade_j == 0.0


def test_climbing_costs_more_traction_than_descending(setup):
    up = balance(setup, grade_permille=15.0)
    down = balance(setup, grade_permille=-15.0)
    assert up.traction_j > down.traction_j


def test_grade_work_cancels_over_a_closed_loop(setup):
    """The reason circuit-level results are unaffected by gradient: whatever
    is climbed is descended."""
    traction, brake, _ = setup
    total = 0.0
    for i, seg in enumerate(load_segments()):
        # Alternating grades, so the loop returns to its starting height.
        grade = 12.0 if i % 2 == 0 else -12.0
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=traction.spec.max_speed_ms,
                             dt=0.1, grade_permille=grade)
        b, _ = segment_energy(r, traction, auxiliary_kw=0.0)
        total += b.grade_j
    # Segment lengths differ, so this does not cancel to zero exactly; it must
    # be small against the work of any single climb.
    single = abs(balance(setup, grade_permille=12.0).grade_j)
    assert abs(total) < single


def test_balance_error_falls_with_step_size(setup):
    """Confirms the remaining residual is discretisation, not a missing term."""
    errors = []
    for dt in (0.2, 0.05, 0.01):
        traction, brake, segment = setup
        r = simulate_segment(traction, brake, segment.distance_m,
                             speed_limit_ms=traction.spec.max_speed_ms,
                             dt=dt, grade_permille=10.0)
        b, _ = segment_energy(r, traction, auxiliary_kw=0.0)
        errors.append(b.balance_error_fraction)
    assert errors[-1] < errors[0]
