"""Tests for speed-dependent braking."""

from __future__ import annotations

import pytest

from tokyoline import (
    BrakeProfile,
    SpeedDependentBrakeProfile,
    build_model,
    load_segments,
    simulate_segment,
)
from tokyoline.fleet import load_fleet

JERK = 0.75


@pytest.fixture(scope="module")
def e7_curve():
    """The E7 publishes a real speed-dependent deceleration curve."""
    return list(load_fleet()["E7"].deceleration_curve)


@pytest.fixture(scope="module")
def flat():
    """A curve that is constant, so the closed form applies to it too."""
    return SpeedDependentBrakeProfile([(0, 4.2), (120, 4.2)], jerk=JERK)


def test_flat_curve_matches_closed_form_distance(flat):
    exact = BrakeProfile(a_max=4.2 / 3.6, jerk=JERK)
    for v0 in (10.0, 17.0, 25.0):
        assert flat.distance_to_stop(v0) == pytest.approx(
            exact.distance_to_stop(v0), abs=0.005)


def test_flat_curve_matches_closed_form_time(flat):
    exact = BrakeProfile(a_max=4.2 / 3.6, jerk=JERK)
    for v0 in (10.0, 17.0, 25.0):
        assert flat.time_to_stop(v0) == pytest.approx(
            exact.time_to_stop(v0), abs=0.005)


def test_rate_interpolates_between_published_points(e7_curve):
    p = SpeedDependentBrakeProfile(e7_curve, jerk=JERK)
    # 2.69 km/h/s below 70, 1.44 at 275.
    assert p.rate_ms2(0.0) == pytest.approx(2.69 / 3.6, rel=1e-6)
    assert p.rate_ms2(275 / 3.6) == pytest.approx(1.44 / 3.6, rel=1e-6)
    mid = p.rate_ms2(150 / 3.6)
    assert 1.44 / 3.6 < mid < 2.23 / 3.6


def test_rate_is_held_flat_outside_the_published_range(e7_curve):
    """Extrapolating an empirical curve eventually gives a negative rate."""
    p = SpeedDependentBrakeProfile(e7_curve, jerk=JERK)
    assert p.rate_ms2(400 / 3.6) == pytest.approx(1.44 / 3.6, rel=1e-6)
    assert p.rate_ms2(0.0) > 0.0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        SpeedDependentBrakeProfile([], jerk=JERK)
    with pytest.raises(ValueError):
        SpeedDependentBrakeProfile([(0, 2.0)], jerk=-1.0)
    with pytest.raises(ValueError):
        SpeedDependentBrakeProfile([(0, -2.0), (100, 2.0)], jerk=JERK)


def test_speed_dependence_lengthens_the_stop(e7_curve):
    variable = SpeedDependentBrakeProfile(e7_curve, jerk=JERK)
    naive = BrakeProfile(a_max=2.69 / 3.6, jerk=JERK)
    v0 = 275 / 3.6
    assert variable.distance_to_stop(v0) > naive.distance_to_stop(v0) * 1.2


def test_distance_and_time_increase_with_speed(e7_curve):
    p = SpeedDependentBrakeProfile(e7_curve, jerk=JERK)
    speeds = [30 / 3.6, 100 / 3.6, 200 / 3.6, 275 / 3.6]
    dists = [p.distance_to_stop(v) for v in speeds]
    times = [p.time_to_stop(v) for v in speeds]
    assert all(b > a for a, b in zip(dists, dists[1:]))
    assert all(b > a for a, b in zip(times, times[1:]))


def test_stopping_from_rest_is_free(e7_curve):
    p = SpeedDependentBrakeProfile(e7_curve, jerk=JERK)
    assert p.distance_to_stop(0.0) == pytest.approx(0.0, abs=1e-9)
    assert p.time_to_stop(0.0) == pytest.approx(0.0, abs=1e-9)


def test_trajectory_ends_at_rest(e7_curve):
    p = SpeedDependentBrakeProfile(e7_curve, jerk=JERK)
    traj = p.trajectory(275 / 3.6, dt=0.05)
    assert traj["v"][-1] == pytest.approx(0.0, abs=1e-9)
    assert traj["a"][-1] == pytest.approx(0.0, abs=1e-9)
    assert traj["v"].max() == pytest.approx(275 / 3.6, rel=1e-3)


def test_trajectory_never_exceeds_the_rate_limit(e7_curve):
    """Deceleration must respect the curve at the speed it is travelling."""
    p = SpeedDependentBrakeProfile(e7_curve, jerk=JERK)
    traj = p.trajectory(275 / 3.6, dt=0.02)
    for v, a in zip(traj["v"], traj["a"]):
        assert abs(a) <= p.rate_ms2(v) + 1e-6


def test_drops_into_the_segment_solver():
    """The interface matches BrakeProfile, so simulate_segment takes either."""
    traction, _ = build_model(load_factor=1.0)
    seg = load_segments()[0]
    flat = SpeedDependentBrakeProfile([(0, 4.2), (120, 4.2)], jerk=JERK)
    r = simulate_segment(traction, flat, seg.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
    assert r.v[-1] == pytest.approx(0.0, abs=1e-9)
    # Numerical rather than closed-form, so sub-millimetre.
    assert abs(r.position_error_m) < 0.005


def test_matches_the_constant_rate_solver_run():
    """Same physics, two implementations: run times must agree closely."""
    traction, closed = build_model(load_factor=1.0)
    seg = load_segments()[0]
    flat = SpeedDependentBrakeProfile(
        [(0, closed.a_max * 3.6), (120, closed.a_max * 3.6)], jerk=closed.jerk)
    a = simulate_segment(traction, closed, seg.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
    b = simulate_segment(traction, flat, seg.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
    assert b.run_time_s == pytest.approx(a.run_time_s, abs=0.02)


def test_fleet_delegates_to_the_shared_implementation(e7_curve):
    e7 = load_fleet()["E7"]
    direct = SpeedDependentBrakeProfile(e7_curve, jerk=JERK,
                                        v_max_ms=275 / 3.6)
    assert e7.stopping_distance_m(jerk=JERK) == pytest.approx(
        direct.distance_to_stop(275 / 3.6), rel=1e-9)


def test_constant_rate_classes_still_use_the_closed_form():
    e235 = load_fleet()["E235-0"]
    assert isinstance(e235.brake_profile(), BrakeProfile)
    e7 = load_fleet()["E7"]
    assert isinstance(e7.brake_profile(), SpeedDependentBrakeProfile)
