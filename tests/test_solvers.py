"""Tests for the coasting solver, loop geometry and rail conditions.

The coasting module previously shipped a bisection with an inverted
monotonicity flag. It did not crash: it pinned the search at its lower bound
and produced a plausible-looking table of savings that happened to be
nonsense. Several tests here exist specifically to catch that class of failure,
by checking that a solver actually hits the target it was asked for rather than
merely returning a number.
"""

from __future__ import annotations

import pytest

from tokyoline import (
    CONDITIONS,
    LoopGeometry,
    StalledRun,
    build_model,
    load_segments,
    simulate_segment,
)
from tokyoline.coasting import (
    evaluate,
    pareto_front,
    solve_coast_for_time,
    solve_speed_for_time,
)
from tokyoline.network import simulate_circuit


@pytest.fixture(scope="module")
def model():
    return build_model(load_factor=1.0)


@pytest.fixture(scope="module")
def long_segment():
    return max(load_segments(), key=lambda s: s.distance_m)


# ------------------------------------------------------------ coast solver

def test_coast_solver_hits_the_requested_time(model, long_segment):
    """The check that would have caught the inverted bisection flag: a solver
    must reach the target it was given, not just return something."""
    traction, brake = model
    v = traction.spec.max_speed_ms
    flat = evaluate(traction, brake, long_segment.distance_m, speed_limit_ms=v,
                    dt=0.1)
    for factor in (1.05, 1.15, 1.30):
        target = flat.run_time_s * factor
        s = solve_coast_for_time(traction, brake, long_segment.distance_m,
                                 target, speed_limit_ms=v, dt=0.1)
        assert s is not None
        assert s.run_time_s == pytest.approx(target, rel=0.01)


def test_slower_cruise_solver_hits_the_requested_time(model, long_segment):
    """Same check for the comparison strategy. This is the one that was
    broken: it pinned at 18 km/h for every target."""
    traction, brake = model
    v = traction.spec.max_speed_ms
    flat = evaluate(traction, brake, long_segment.distance_m, speed_limit_ms=v,
                    dt=0.1)
    for factor in (1.05, 1.15, 1.30):
        target = flat.run_time_s * factor
        s = solve_speed_for_time(traction, brake, long_segment.distance_m,
                                 target, speed_limit_ms=v, dt=0.1)
        assert s is not None
        assert s.run_time_s == pytest.approx(target, rel=0.01)
        # A sane answer sits below line speed but well above a crawl.
        assert 0.4 * v < s.cruise_speed_kmh / 3.6 <= v


def test_unreachable_targets_return_none(model, long_segment):
    """Faster than flat out is impossible, and must not silently return the
    nearest achievable strategy."""
    traction, brake = model
    v = traction.spec.max_speed_ms
    flat = evaluate(traction, brake, long_segment.distance_m, speed_limit_ms=v,
                    dt=0.1)
    assert solve_coast_for_time(traction, brake, long_segment.distance_m,
                                flat.run_time_s * 0.9, speed_limit_ms=v,
                                dt=0.1) is None
    assert solve_speed_for_time(traction, brake, long_segment.distance_m,
                                flat.run_time_s * 0.9, speed_limit_ms=v,
                                dt=0.1) is None


def test_more_time_costs_less_energy(model, long_segment):
    """The Pareto front must slope the right way."""
    traction, brake = model
    front = pareto_front(traction, brake, long_segment.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         time_factors=(1.0, 1.05, 1.15, 1.30), dt=0.1)
    energies = [p.coast.net_energy_j for p in front if p.coast]
    times = [p.coast.run_time_s for p in front if p.coast]
    assert len(energies) == 4
    assert all(b < a for a, b in zip(energies, energies[1:]))
    assert all(b > a for a, b in zip(times, times[1:]))


def test_coasting_beats_cruising_slower_at_equal_time(model, long_segment):
    """The result the whole module exists to establish."""
    traction, brake = model
    front = pareto_front(traction, brake, long_segment.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         time_factors=(1.10, 1.30), dt=0.1)
    for p in front:
        assert p.coast is not None and p.slower is not None
        # Same run time, so the comparison is fair.
        assert p.coast.run_time_s == pytest.approx(p.slower.run_time_s, rel=0.02)
        assert p.coast.net_energy_j < p.slower.net_energy_j
        assert 0.0 < p.saving_fraction < 0.15


def test_coasting_actually_coasts(model, long_segment):
    traction, brake = model
    flat = evaluate(traction, brake, long_segment.distance_m,
                    speed_limit_ms=traction.spec.max_speed_ms, dt=0.1)
    s = solve_coast_for_time(traction, brake, long_segment.distance_m,
                             flat.run_time_s * 1.2,
                             speed_limit_ms=traction.spec.max_speed_ms, dt=0.1)
    assert s.coast_distance_m > 100.0
    assert 0.0 < s.coast_start_m < long_segment.distance_m


# ------------------------------------------------------------ stalled runs

def test_coasting_from_too_early_raises_rather_than_grinding(model, long_segment):
    """Coast from the first metre and resistance stops the train long before
    the platform. The solver used to integrate two million steps -- twenty-two
    seconds -- and then report a convergence failure, which named the
    integrator rather than the cause."""
    traction, brake = model
    with pytest.raises(StalledRun, match="coast onset"):
        simulate_segment(traction, brake, long_segment.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.1, coast_start_m=1.0)


def test_an_unclimbable_gradient_raises(model, long_segment):
    """The other way to stall: not enough tractive effort at rest. 200 per-mille
    is far beyond anything adhesion rail does, which is the point."""
    traction, brake = model
    with pytest.raises(StalledRun, match="gradient"):
        simulate_segment(traction, brake, long_segment.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.1, grade_permille=200.0)


def test_the_stall_is_reported_quickly(model, long_segment):
    """A guard that takes as long as the step budget it replaces is no guard.
    The bound is loose; the failure it catches was three orders of magnitude
    slower."""
    import time

    traction, brake = model
    start = time.perf_counter()
    with pytest.raises(StalledRun):
        simulate_segment(traction, brake, long_segment.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.1, coast_start_m=1.0)
    assert time.perf_counter() - start < 2.0


def test_the_solver_lower_bound_never_stalls(model):
    """solve_coast_for_time probes 5 percent of the distance to bracket its
    search. If any real segment stalled there the solver would raise instead
    of returning None, so this is what makes that bound safe."""
    traction, brake = model
    for seg in load_segments():
        simulate_segment(traction, brake, seg.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.1, coast_start_m=0.05 * seg.distance_m)


def test_gradients_the_line_actually_has_do_not_stall(model, long_segment):
    """34 per-mille is the published maximum, between Tabata and
    Nishi-Nippori. Nothing near it should trip the guard."""
    traction, brake = model
    for grade in (-34.0, -20.0, 0.0, 20.0, 34.0, 60.0):
        r = simulate_segment(traction, brake, long_segment.distance_m,
                             speed_limit_ms=traction.spec.max_speed_ms,
                             dt=0.1, grade_permille=grade)
        assert r.run_time_s > 0.0


# ---------------------------------------------------------- loop geometry

@pytest.fixture(scope="module")
def geom():
    return LoopGeometry.build()


def test_geometry_covers_every_station(geom):
    assert len(geom.names) == 30
    assert geom.total_m == pytest.approx(34_500, abs=1)


def test_position_wraps_around_the_loop(geom):
    """It is a circle: distance zero and one full lap are the same place."""
    a = geom.position_at(0.0)
    b = geom.position_at(geom.total_m)
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert a[1] == pytest.approx(b[1], abs=1e-9)


def test_position_at_a_station_matches_that_station(geom):
    for i, cum in enumerate(geom.cumulative_m):
        lat, lon = geom.position_at(float(cum))
        assert lat == pytest.approx(geom.lat[i], abs=1e-9)
        assert lon == pytest.approx(geom.lon[i], abs=1e-9)


def test_negative_distance_wraps_backwards(geom):
    """Used when drawing a train whose tail is behind the origin."""
    behind = geom.position_at(-100.0)
    ahead = geom.position_at(geom.total_m - 100.0)
    assert behind[0] == pytest.approx(ahead[0])
    assert behind[1] == pytest.approx(ahead[1])


def test_closed_path_repeats_the_first_station(geom):
    lat, lon = geom.closed_path()
    assert len(lat) == len(geom.names) + 1
    assert lat[0] == lat[-1]
    assert lon[0] == lon[-1]


# --------------------------------------------------------------- circuit

def test_circuit_totals_are_consistent(model):
    traction, brake = model
    segments = load_segments()
    dwell = 30.0
    c = simulate_circuit(traction, brake, segments,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dwell_s=dwell, dt=0.1, sample_dt=1.0)
    assert len(c.run_times_s) == 30
    assert c.total_time_s == pytest.approx(sum(c.run_times_s) + 30 * dwell,
                                           rel=1e-3)
    assert c.distance_m[-1] <= 34_500 + 1
    assert c.speed_kmh.max() <= traction.spec.max_speed_ms * 3.6 + 1e-6


def test_circuit_carries_acceleration_and_force(model):
    """The live view needs these; they must span the published limits."""
    traction, brake = model
    c = simulate_circuit(traction, brake, load_segments(),
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dwell_s=30.0, dt=0.1, sample_dt=1.0)
    assert c.accel_kmh_s.max() == pytest.approx(3.0, abs=0.1)
    assert c.accel_kmh_s.min() == pytest.approx(-4.2, abs=0.1)
    assert c.force_n.max() > 0.0
    assert c.power_w.max() == pytest.approx(traction.power_w, rel=0.02)


def test_circuit_sample_wraps(model):
    traction, brake = model
    c = simulate_circuit(traction, brake, load_segments(),
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dwell_s=30.0, dt=0.1, sample_dt=1.0)
    a = c.sample(10.0)
    b = c.sample(10.0 + c.total_time_s)
    assert a["distance_m"] == pytest.approx(b["distance_m"])


# ------------------------------------------------------- rail conditions

def test_worse_rail_lowers_the_adhesion_ceiling():
    ceilings = [CONDITIONS[k].max_braking_ms2(0.0)
                for k in ("dry", "damp", "wet", "leaf_fall")]
    assert all(b < a for a, b in zip(ceilings, ceilings[1:]))


def test_only_the_worst_condition_limits_braking():
    """The asymmetry the module exists to show: rail condition barely touches
    a 4.2 km/h/s service rate until the surface is genuinely contaminated."""
    service = 4.2 / 3.6
    assert not CONDITIONS["dry"].is_adhesion_limited(service)
    assert not CONDITIONS["damp"].is_adhesion_limited(service)
    assert not CONDITIONS["wet"].is_adhesion_limited(service)
    assert CONDITIONS["leaf_fall"].is_adhesion_limited(service)


def test_effective_rate_never_exceeds_the_service_rate():
    service = 4.2 / 3.6
    for cond in CONDITIONS.values():
        assert cond.effective_brake_rate_ms2(service) <= service + 1e-12
