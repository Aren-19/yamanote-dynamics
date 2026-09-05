"""Tests for journey time between stations.

Every number on the departure board comes from this function, so an error here
would be visible to a reader and invisible to the physics tests. It lived in
app.py until now, which put the arithmetic behind the one file the test suite
cannot import.

The awkward part is the loop. Travelling inbound from station i covers segment
i-1, not segment i, and both directions have to wrap. A sign error there gives
plausible-looking times that are simply wrong.
"""

from __future__ import annotations

import pytest

from tokyoline import build_model, journey_time_s, load_segments, simulate_segment

# Four stations, one minute apart, thirty second dwells: small enough to
# verify by hand.
UNIFORM = [60.0, 60.0, 60.0, 60.0]
DWELL = 30.0


# ---------------------------------------------------------------- degenerate

def test_journey_to_the_same_station_is_zero():
    """Not a full circuit. Someone asking for Shinjuku to Shinjuku wants
    zero, not sixty-five minutes."""
    assert journey_time_s(2, 2, UNIFORM, DWELL, True) == 0.0
    assert journey_time_s(2, 2, UNIFORM, DWELL, False) == 0.0


def test_empty_line_is_zero():
    assert journey_time_s(0, 1, [], DWELL, True) == 0.0


# ------------------------------------------------------------------- outbound

def test_one_hop_outbound_is_a_single_run():
    """No dwell: the journey ends on arrival, not on departure."""
    assert journey_time_s(0, 1, UNIFORM, DWELL, True) == pytest.approx(60.0)


def test_two_hops_outbound_includes_one_intermediate_dwell():
    assert journey_time_s(0, 2, UNIFORM, DWELL, True) == pytest.approx(150.0)


def test_three_hops_outbound_includes_two_dwells():
    assert journey_time_s(0, 3, UNIFORM, DWELL, True) == pytest.approx(240.0)


def test_outbound_wraps_past_the_origin():
    assert journey_time_s(3, 0, UNIFORM, DWELL, True) == pytest.approx(60.0)
    assert journey_time_s(3, 1, UNIFORM, DWELL, True) == pytest.approx(150.0)


# -------------------------------------------------------------------- inbound

def test_one_hop_inbound_traverses_the_preceding_segment():
    """Leaving station 1 inbound covers segment 0, which joins 0 and 1."""
    assert journey_time_s(1, 0, UNIFORM, DWELL, False) == pytest.approx(60.0)


def test_inbound_wraps_backwards_past_the_origin():
    assert journey_time_s(0, 3, UNIFORM, DWELL, False) == pytest.approx(60.0)
    assert journey_time_s(0, 2, UNIFORM, DWELL, False) == pytest.approx(150.0)


def test_the_two_directions_cover_the_whole_ring():
    """On a loop of four, one direction takes k hops and the other 4-k."""
    for target in (1, 2, 3):
        out = journey_time_s(0, target, UNIFORM, DWELL, True)
        inn = journey_time_s(0, target, UNIFORM, DWELL, False)
        out_hops = round((out + DWELL) / (60.0 + DWELL))
        inn_hops = round((inn + DWELL) / (60.0 + DWELL))
        assert out_hops + inn_hops == 4


# ------------------------------------------------------- direction asymmetry

def test_unequal_segments_make_the_directions_differ():
    """With uniform segments the two directions can coincide, which would hide
    a sign error. Uneven ones cannot."""
    uneven = [10.0, 20.0, 40.0, 80.0]
    out = journey_time_s(0, 2, uneven, 0.0, True)     # segments 0 and 1
    inn = journey_time_s(0, 2, uneven, 0.0, False)    # segments 3 and 2
    assert out == pytest.approx(30.0)
    assert inn == pytest.approx(120.0)
    assert out != inn


def test_a_full_circuit_either_way_is_the_same_total():
    """Going all the way round covers every segment exactly once, whichever
    way you travel."""
    uneven = [10.0, 20.0, 40.0, 80.0]
    n = len(uneven)
    out = sum(journey_time_s(i, (i + 1) % n, uneven, 0.0, True) for i in range(n))
    inn = sum(journey_time_s(i, (i - 1) % n, uneven, 0.0, False) for i in range(n))
    assert out == pytest.approx(sum(uneven))
    assert inn == pytest.approx(sum(uneven))


# ------------------------------------------------------------ against the line

def test_real_line_journey_is_bounded_by_the_circuit():
    """No journey between two stations can exceed a full lap."""
    traction, brake = build_model(load_factor=1.0)
    segments = load_segments()
    run_times = [
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.1).run_time_s
        for s in segments
    ]
    dwell = 32.0
    circuit = sum(run_times) + dwell * len(run_times)

    for target in range(0, 30, 7):
        for outer in (True, False):
            t = journey_time_s(0, target, run_times, dwell, outer)
            assert 0.0 <= t < circuit


def test_shinjuku_to_tokyo_is_roughly_half_the_loop():
    """Shinjuku is at 10.6 km and Tokyo at 27.7 km, so both directions are
    close to 17 km and should take a similar time."""
    traction, brake = build_model(load_factor=1.0)
    segments = load_segments()
    names = [s.from_station for s in segments]
    run_times = [
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=traction.spec.max_speed_ms,
                         dt=0.1).run_time_s
        for s in segments
    ]
    i, j = names.index("Shinjuku"), names.index("Tokyo")
    out = journey_time_s(i, j, run_times, 32.0, True)
    inn = journey_time_s(i, j, run_times, 32.0, False)
    # Within a few minutes of each other, and both well under an hour.
    assert abs(out - inn) < 300
    assert 20 * 60 < min(out, inn) < 40 * 60
