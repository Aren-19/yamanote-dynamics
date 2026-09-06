"""Structural invariants of a segment trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from tokyoline import build_model, load_segments, simulate_segment
from tokyoline.energy import segment_energy
from tokyoline.segment import Phase

# The control sequence is a total order; phases may be skipped.
PHASE_ORDER = {Phase.ACCEL: 0, Phase.CRUISE: 1, Phase.COAST: 2, Phase.BRAKE: 3}


@pytest.fixture(scope="module")
def model():
    return build_model(load_factor=1.0)


@pytest.fixture(scope="module")
def long_segment():
    return max(load_segments(), key=lambda s: s.distance_m)


def run(model, distance, **kwargs):
    traction, brake = model
    kwargs.setdefault("speed_limit_ms", traction.spec.max_speed_ms)
    kwargs.setdefault("dt", 0.05)
    return simulate_segment(traction, brake, distance, **kwargs)


def transitions(result):
    return [
        p for i, p in enumerate(result.phase)
        if i == 0 or p != result.phase[i - 1]
    ]



def test_phases_never_go_backwards(model, long_segment):
    seq = transitions(run(model, long_segment.distance_m))
    assert [p.value for p in seq] == ["accel", "cruise", "brake"]


def test_coasting_inserts_exactly_one_phase(model, long_segment):
    """Coast sits between cruise and brake and nowhere else."""
    seq = transitions(run(model, long_segment.distance_m,
                          coast_start_m=long_segment.distance_m * 0.6))
    assert [p.value for p in seq] == ["accel", "cruise", "coast", "brake"]


def test_a_short_segment_skips_cruise(model):
    """500 m is the shortest on the line, and the train is still accelerating."""
    seq = transitions(run(model, 500.0))
    assert [p.value for p in seq] == ["accel", "brake"]
    assert not run(model, 500.0).reached_line_speed


@pytest.mark.parametrize("distance", [500.0, 900.0, 1300.0, 2000.0])
def test_phase_order_holds_at_every_segment_length(model, distance):
    seq = transitions(run(model, distance))
    assert all(PHASE_ORDER[a] < PHASE_ORDER[b] for a, b in zip(seq, seq[1:])), (
        " -> ".join(p.value for p in seq)
    )



def test_phase_times_sum_to_the_run_time(model, long_segment):
    r = run(model, long_segment.distance_m)
    assert sum(r.phase_times_s.values()) == pytest.approx(r.run_time_s, abs=1e-6)


def test_phase_distances_sum_to_the_segment(model, long_segment):
    r = run(model, long_segment.distance_m)
    assert sum(r.phase_distances_m.values()) == pytest.approx(r.distance_m,
                                                              abs=1e-3)


def test_bookkeeping_holds_with_coasting_too(model, long_segment):
    r = run(model, long_segment.distance_m,
            coast_start_m=long_segment.distance_m * 0.5)
    assert sum(r.phase_times_s.values()) == pytest.approx(r.run_time_s, abs=1e-6)
    assert sum(r.phase_distances_m.values()) == pytest.approx(r.distance_m,
                                                              abs=1e-3)
    assert r.phase_times_s[Phase.COAST.value] > 0.0


def test_time_and_position_advance_monotonically(model, long_segment):
    r = run(model, long_segment.distance_m)
    assert (np.diff(r.t) > 0).all()
    assert (np.diff(r.x) >= -1e-12).all()


def test_every_array_has_the_same_length(model, long_segment):
    """A trajectory is read by index throughout the application; a ragged one."""
    r = run(model, long_segment.distance_m)
    n = len(r.t)
    assert len(r.x) == len(r.v) == len(r.a) == len(r.phase) == n
    assert len(r.tractive_force_n) == n



def test_no_tractive_effort_while_coasting_or_braking(model, long_segment):
    """The defining property of both phases. Recording effort here would show."""
    r = run(model, long_segment.distance_m,
            coast_start_m=long_segment.distance_m * 0.5)
    unpowered = [p in (Phase.COAST, Phase.BRAKE) for p in r.phase]
    assert np.abs(r.tractive_force_n[unpowered]).max() < 1e-9


def test_cruise_records_the_force_that_holds_speed(model, long_segment):
    r = run(model, long_segment.distance_m)
    cruising = [p is Phase.CRUISE for p in r.phase]
    assert r.tractive_force_n[cruising].min() > 0.0


def test_tractive_effort_never_exceeds_the_starting_value(model, long_segment):
    """Effort is greatest at rest and falls under the power and adhesion."""
    traction, _ = model
    r = run(model, long_segment.distance_m)
    assert r.tractive_force_n.max() <= traction.effort_n(0.0) + 1.0


def test_deceleration_never_exceeds_the_service_rate(model, long_segment):
    _, brake = model
    r = run(model, long_segment.distance_m)
    braking = [p is Phase.BRAKE for p in r.phase]
    assert -r.a[braking].min() <= brake.a_max + 1e-6


def test_acceleration_signs_match_the_phase(model, long_segment):
    r = run(model, long_segment.distance_m)
    accel = [p is Phase.ACCEL for p in r.phase]
    braking = [p is Phase.BRAKE for p in r.phase]
    # The last sample of each phase sits on the boundary.
    assert (r.a[accel][:-1] > 0.0).all()
    assert (r.a[braking][:-1] < 0.0).all()



def test_run_time_falls_as_the_speed_limit_rises(model, long_segment):
    times = [
        run(model, long_segment.distance_m, speed_limit_ms=v / 3.6).run_time_s
        for v in (40, 50, 60, 70, 80, 90)
    ]
    assert (np.diff(times) < 0).all(), times


def test_later_coast_onset_is_faster_and_costs_more(model, long_segment):
    traction, _ = model
    times, energies = [], []
    for fraction in (0.3, 0.4, 0.5, 0.6, 0.7):
        r = run(model, long_segment.distance_m,
                coast_start_m=long_segment.distance_m * fraction)
        breakdown, _ = segment_energy(r, traction, auxiliary_kw=0.0)
        times.append(r.run_time_s)
        energies.append(breakdown.traction_j)
    assert (np.diff(times) < 0).all(), times
    assert (np.diff(energies) > 0).all(), energies


def test_coasting_trades_time_for_energy_against_running_flat_out(model,
                                                                  long_segment):
    traction, _ = model
    flat_out = run(model, long_segment.distance_m)
    coasted = run(model, long_segment.distance_m,
                  coast_start_m=long_segment.distance_m * 0.5)
    e_flat, _ = segment_energy(flat_out, traction, auxiliary_kw=0.0)
    e_coast, _ = segment_energy(coasted, traction, auxiliary_kw=0.0)
    assert coasted.run_time_s > flat_out.run_time_s
    assert e_coast.traction_j < e_flat.traction_j


def test_coast_onset_beyond_the_brake_point_changes_nothing(model,
                                                            long_segment):
    """The train is already braking, so there is nothing left to coast."""
    plain = run(model, long_segment.distance_m)
    late = run(model, long_segment.distance_m,
               coast_start_m=long_segment.distance_m * 0.999)
    assert late.run_time_s == pytest.approx(plain.run_time_s, abs=1e-6)
