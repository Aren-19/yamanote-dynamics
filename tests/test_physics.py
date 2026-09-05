"""Property tests for the physics core.

These assert identities and bounds that must hold regardless of parameter
values, rather than pinning specific numbers. A test that hard-codes "circuit
time is 2309 s" fails whenever a coefficient is retuned and teaches nothing; a
test that asserts energy is conserved catches the class of bug that actually
occurred during development.
"""

from __future__ import annotations

import pytest

from tokyoline import (
    BrakeProfile,
    DavisCoefficients,
    TractionModel,
    adhesion_coefficient,
    build_model,
    load_segments,
    load_spec,
    simulate_segment,
)
from tokyoline.energy import segment_energy
from tokyoline.units import KMH_TO_MS, KMHS_TO_MS2, ms, ms2


@pytest.fixture(scope="module")
def model():
    return build_model(load_factor=1.0)


@pytest.fixture(scope="module")
def segments():
    return load_segments()


# ------------------------------------------------------------------- units

def test_kmh_to_ms_exact():
    assert ms(90.0) == pytest.approx(25.0)
    assert pytest.approx(1 / 3.6) == KMH_TO_MS


def test_kmhs_to_ms2_is_the_same_factor():
    """1 km/h/s = 1/3.6 m/s^2. Confusing these is a factor-3.6 error."""
    assert pytest.approx(1 / 3.6) == KMHS_TO_MS2
    assert ms2(3.0) == pytest.approx(0.8333, abs=1e-4)


# -------------------------------------------------------------- resistance

def test_davis_is_monotonic_in_speed():
    d = DavisCoefficients(A=4000.0, B=90.0, C=9.5)
    values = [d(v) for v in (0, 5, 10, 15, 20, 25)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_davis_never_negative():
    d = DavisCoefficients(A=4000.0, B=90.0, C=9.5)
    assert all(d(v) >= 0 for v in (0, 1, 10, 30, 100))


def test_davis_symmetric_in_direction():
    """Resistance opposes motion; its magnitude cannot depend on sign."""
    d = DavisCoefficients(A=4000.0, B=90.0, C=9.5)
    assert d(-12.0) == pytest.approx(d(12.0))


def test_estimated_coefficients_land_in_the_expected_band(model):
    """A commuter EMU sits near 10-15 N/t at rest, 25-35 N/t at 90 km/h."""
    traction, _ = model
    mass = traction.spec.mass_kg(traction.load_factor)
    assert 8.0 <= traction.davis.specific_n_per_tonne(0.0, mass) <= 15.0
    assert 22.0 <= traction.davis.specific_n_per_tonne(25.0, mass) <= 38.0


# ---------------------------------------------------------------- adhesion

def test_adhesion_falls_with_speed():
    values = [adhesion_coefficient(v) for v in (0, 10, 20, 25)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_adhesion_never_binds_for_this_class(model):
    """Six of eleven cars powered leaves a wide margin. If this fails, the
    model has silently become wheelslip-limited."""
    traction, _ = model
    assert not any(traction.adhesion_binds(i * 0.5) for i in range(51))


# ----------------------------------------------------------------- braking

def test_jerk_penalty_is_independent_of_initial_speed():
    """For a trapezoidal stop the penalty is exactly a_max/j, whatever v0."""
    b = BrakeProfile(a_max=1.1667, jerk=0.75)
    expected = b.a_max / b.jerk
    for v0 in (15.0, 20.0, 25.0):
        assert b.time_to_stop(v0) - v0 / b.a_max == pytest.approx(expected, abs=1e-6)


def test_jerk_limited_stop_is_longer_than_ideal():
    b = BrakeProfile(a_max=1.1667, jerk=0.75)
    for v0 in (10.0, 25.0):
        assert b.distance_to_stop(v0) > b.idealised_distance_to_stop(v0)


def test_triangular_profile_below_threshold():
    b = BrakeProfile(a_max=1.1667, jerk=0.75)
    threshold = b.speed_threshold_for_full_braking
    assert not b.is_trapezoidal(threshold * 0.5)
    assert b.peak_deceleration(threshold * 0.5) < b.a_max


def test_brake_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        BrakeProfile(a_max=0.0, jerk=0.75)
    with pytest.raises(ValueError):
        BrakeProfile(a_max=1.0, jerk=-1.0)


def test_brake_trajectory_ends_at_rest():
    b = BrakeProfile(a_max=1.1667, jerk=0.75)
    traj = b.trajectory(25.0, dt=0.01)
    assert traj["v"][-1] == pytest.approx(0.0, abs=1e-9)
    assert traj["a"][-1] == pytest.approx(0.0, abs=1e-9)
    assert traj["x"][-1] == pytest.approx(b.distance_to_stop(25.0), rel=1e-6)


# ---------------------------------------------------------------- traction

def test_effort_is_zero_at_maximum_speed(model):
    """The driver stops accelerating at the limit."""
    traction, _ = model
    assert traction.effort_n(traction.spec.max_speed_ms) == 0.0


def test_capability_is_positive_at_maximum_speed(model):
    """The machine can still produce force there, which is what holds cruise.
    Conflating this with effort_n made the energy balance fail at 7.3%."""
    traction, _ = model
    assert traction.capability_n(traction.spec.max_speed_ms) > 0.0


def test_holding_force_balances_resistance_at_cruise(model):
    traction, _ = model
    v = traction.spec.max_speed_ms
    assert traction.holding_force_n(v) == pytest.approx(traction.davis(v), rel=1e-9)


def test_effort_follows_constant_power_above_base_speed(model):
    traction, _ = model
    vb = traction.base_speed_ms
    for v in (vb * 1.5, vb * 2.0):
        if v < traction.spec.max_speed_ms:
            assert traction.effort_n(v) == pytest.approx(traction.power_w / v,
                                                         rel=1e-6)


# ----------------------------------------------------- segment integration

def test_train_stops_exactly_at_the_platform(model, segments):
    """The backward brake-curve intersection exists to make this exact."""
    traction, brake = model
    for seg in segments:
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
        assert abs(r.position_error_m) < 1e-6, seg


def test_segment_starts_and_ends_at_rest(model, segments):
    traction, brake = model
    r = simulate_segment(traction, brake, segments[0].distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
    assert r.v[0] == pytest.approx(0.0)
    assert r.v[-1] == pytest.approx(0.0, abs=1e-9)


def test_speed_never_exceeds_the_limit(model, segments):
    traction, brake = model
    limit = traction.spec.max_speed_ms
    for seg in segments[:6]:
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=limit, dt=0.05)
        assert r.v.max() <= limit + 1e-6


def test_longer_segment_takes_longer(model):
    traction, brake = model
    limit = traction.spec.max_speed_ms
    times = [
        simulate_segment(traction, brake, d, speed_limit_ms=limit,
                         dt=0.05).run_time_s
        for d in (600.0, 1000.0, 1500.0, 2000.0)
    ]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_run_time_converges_with_step_size(model):
    """Halving the step must not move the answer meaningfully."""
    traction, brake = model
    limit = traction.spec.max_speed_ms
    coarse = simulate_segment(traction, brake, 1500.0, speed_limit_ms=limit, dt=0.2)
    fine = simulate_segment(traction, brake, 1500.0, speed_limit_ms=limit, dt=0.01)
    assert coarse.run_time_s == pytest.approx(fine.run_time_s, abs=0.05)


def test_impossible_segment_is_rejected(model):
    traction, brake = model
    with pytest.raises(ValueError):
        simulate_segment(traction, brake, -5.0)


# ------------------------------------------------------------------ energy

def test_energy_balance_holds(model, segments):
    """E_traction = E_resistance + E_brake, because each segment starts and
    ends at rest so the kinetic term drops out. Both sides are computed
    independently, so this tests the integrator rather than restating an
    assumption. This is the check that caught the cruise-force bug."""
    traction, brake = model
    for seg in segments[:8]:
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=traction.spec.max_speed_ms, dt=0.02)
        b, _ = segment_energy(r, traction, auxiliary_kw=0.0)
        assert b.balance_error_fraction < 1e-3, seg


def test_energy_balance_improves_with_step_size(model, segments):
    traction, brake = model
    errors = []
    for dt in (0.2, 0.05, 0.01):
        r = simulate_segment(traction, brake, segments[0].distance_m,
                             speed_limit_ms=traction.spec.max_speed_ms, dt=dt)
        b, _ = segment_energy(r, traction, auxiliary_kw=0.0)
        errors.append(b.balance_error_fraction)
    assert errors[0] > errors[-1]


def test_regeneration_cannot_exceed_braking_energy(model, segments):
    traction, brake = model
    r = simulate_segment(traction, brake, segments[0].distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
    b, _ = segment_energy(r, traction, regen_receptivity=1.0,
                          regen_efficiency=1.0)
    assert b.regenerated_j <= b.brake_system_j + 1e-6


def test_net_energy_falls_as_receptivity_rises(model, segments):
    traction, brake = model
    r = simulate_segment(traction, brake, segments[0].distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
    nets = [segment_energy(r, traction, regen_receptivity=rec)[0].net_j
            for rec in (0.0, 0.5, 1.0)]
    assert all(b < a for a, b in zip(nets, nets[1:]))


# -------------------------------------------------------------------- mass

def test_effective_mass_exceeds_static_mass():
    spec = load_spec()
    assert spec.effective_mass_kg(1.0) > spec.mass_kg(1.0)


def test_loading_increases_mass():
    spec = load_spec()
    assert spec.mass_kg(1.0) > spec.mass_kg(0.0)


def test_heavier_train_accelerates_more_slowly():
    spec = load_spec()
    davis = DavisCoefficients.estimate_for_emu(
        spec.mass_kg(0.0), spec.n_cars, spec.frontal_area_m2)
    light = TractionModel(spec, davis, load_factor=0.0)
    heavy = TractionModel(spec, davis, load_factor=1.5)
    assert heavy.acceleration_ms2(10.0) < light.acceleration_ms2(10.0)
