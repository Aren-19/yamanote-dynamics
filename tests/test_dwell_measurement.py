"""Tests for the dwell-measurement logic."""

from __future__ import annotations

import io
from unittest import mock

import pytest

from tokyoline.odpt import (
    Sample,
    aggregate,
    episodes_from_samples,
    short_station,
)


def stream(train: str, entries: list[tuple[float, str | None, str | None]]):
    """Build samples from (time, from_station, to_station) triples."""
    return [
        Sample(t=t, train=train, railway="test", from_station=f,
               to_station=to, delay_s=0)
        for t, f, to in entries
    ]



def test_short_station_takes_the_last_segment():
    assert short_station("odpt.Station:JR-East.Yamanote.Shibuya") == "Shibuya"
    assert short_station(None) is None


def test_standing_is_defined_by_a_missing_destination():
    running = Sample(0.0, "t", "r", "Ueno", "Okachimachi", 0)
    standing = Sample(0.0, "t", "r", "Ueno", None, 0)
    assert not running.standing
    assert standing.standing



def test_bracketed_episode_is_detected_with_both_bounds():
    s = stream("A", [
        (0.0, "Ueno", "Okachimachi"),    # running
        (10.0, "Okachimachi", None),     # arrived
        (20.0, "Okachimachi", None),     # still standing
        (30.0, "Okachimachi", "Akihabara"),  # departed
    ])
    episodes = episodes_from_samples(s)
    assert len(episodes) == 1
    e = episodes[0]
    assert e.station == "Okachimachi"
    assert e.bracketed
    # Observed standing span is the lower bound.
    assert e.lower_bound_s == pytest.approx(10.0)
    # The gap between the bracketing running observations is the upper bound.
    assert e.upper_bound_s == pytest.approx(30.0)
    assert e.midpoint_s == pytest.approx(20.0)


def test_episode_open_at_the_start_is_not_counted():
    """Sampling began with the train already standing, so there is no lower."""
    s = stream("A", [
        (0.0, "Ueno", None),
        (10.0, "Ueno", None),
        (20.0, "Ueno", "Okachimachi"),
    ])
    episodes = episodes_from_samples(s)
    assert len(episodes) == 1
    assert not episodes[0].bracketed
    assert episodes[0].upper_bound_s is None
    assert episodes[0].midpoint_s is None


def test_episode_open_at_the_end_is_not_counted():
    s = stream("A", [
        (0.0, "Ueno", "Okachimachi"),
        (10.0, "Okachimachi", None),
        (20.0, "Okachimachi", None),
    ])
    episodes = episodes_from_samples(s)
    assert len(episodes) == 1
    assert not episodes[0].bracketed


def test_two_stops_give_two_episodes():
    s = stream("A", [
        (0.0, "Ueno", "Okachimachi"),
        (10.0, "Okachimachi", None),
        (20.0, "Okachimachi", "Akihabara"),
        (30.0, "Akihabara", None),
        (40.0, "Akihabara", "Kanda"),
    ])
    episodes = episodes_from_samples(s)
    assert [e.station for e in episodes] == ["Okachimachi", "Akihabara"]
    assert all(e.bracketed for e in episodes)


def test_trains_are_tracked_separately():
    s = stream("A", [
        (0.0, "Ueno", "Okachimachi"),
        (10.0, "Okachimachi", None),
        (20.0, "Okachimachi", "Akihabara"),
    ]) + stream("B", [
        (0.0, "Shibuya", "Harajuku"),
        (10.0, "Harajuku", None),
        (20.0, "Harajuku", "Yoyogi"),
    ])
    episodes = episodes_from_samples(s)
    assert {e.train for e in episodes} == {"A", "B"}
    assert {e.station for e in episodes} == {"Okachimachi", "Harajuku"}


def test_samples_are_sorted_before_grouping():
    entries = [
        (30.0, "Okachimachi", "Akihabara"),
        (0.0, "Ueno", "Okachimachi"),
        (20.0, "Okachimachi", None),
        (10.0, "Okachimachi", None),
    ]
    episodes = episodes_from_samples(stream("A", entries))
    assert len(episodes) == 1
    assert episodes[0].lower_bound_s == pytest.approx(10.0)


def test_a_train_never_seen_standing_yields_nothing():
    s = stream("A", [
        (0.0, "Ueno", "Okachimachi"),
        (10.0, "Okachimachi", "Akihabara"),
    ])
    assert episodes_from_samples(s) == []



def test_aggregate_uses_only_bracketed_episodes():
    good = stream("A", [
        (0.0, "Ueno", "Okachimachi"),
        (10.0, "Okachimachi", None),
        (20.0, "Okachimachi", None),
        (30.0, "Okachimachi", "Akihabara"),
    ])
    # Train B is standing for the whole window: no bounds at all.
    open_ended = stream("B", [
        (0.0, "Shibuya", None),
        (10.0, "Shibuya", None),
        (20.0, "Shibuya", None),
    ])
    stats = aggregate(episodes_from_samples(good + open_ended))
    assert stats.n_episodes == 2
    assert stats.n_bracketed == 1
    assert stats.mean_lower_s == pytest.approx(10.0)
    assert stats.mean_upper_s == pytest.approx(30.0)
    assert stats.mean_estimate_s == pytest.approx(20.0)


def test_aggregate_returns_none_when_nothing_is_usable():
    """Parked trains at end of service produce episodes but no measurements."""
    parked = stream("A", [(float(t), "Shibuya", None) for t in range(0, 60, 10)])
    assert aggregate(episodes_from_samples(parked)) is None


def test_per_station_breakdown_groups_correctly():
    s = stream("A", [
        (0.0, "Ueno", "Okachimachi"),
        (10.0, "Okachimachi", None),
        (20.0, "Okachimachi", "Akihabara"),
    ]) + stream("B", [
        (0.0, "Ueno", "Okachimachi"),
        (10.0, "Okachimachi", None),
        (30.0, "Okachimachi", "Akihabara"),
    ])
    stats = aggregate(episodes_from_samples(s))
    assert set(stats.per_station) == {"Okachimachi"}
    count, lower, upper = stats.per_station["Okachimachi"]
    assert count == 2
    assert lower == pytest.approx(0.0)   # each seen standing once only
    assert upper == pytest.approx(25.0)  # mean of 20 and 30


def test_bounds_always_bracket_the_truth():
    """Whatever the sampling, the lower bound cannot exceed the upper."""
    s = stream("A", [
        (0.0, "Ueno", "Okachimachi"),
        (5.0, "Okachimachi", None),
        (11.0, "Okachimachi", None),
        (17.0, "Okachimachi", None),
        (23.0, "Okachimachi", "Akihabara"),
    ])
    for e in episodes_from_samples(s):
        assert e.lower_bound_s <= e.upper_bound_s


def test_http_errors_do_not_leak_the_consumer_key():
    """The query string carries the key, and this message reaches the console."""
    import urllib.error

    from tokyoline.odpt import OdptClient, OdptError

    client = OdptClient(consumer_key="SECRET-KEY-VALUE")

    def raise_500(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://example.test", 500, "boom", {}, io.BytesIO(b"server error"))

    with mock.patch("urllib.request.urlopen", raise_500),             pytest.raises(OdptError) as caught:
        client.trains()

    assert "SECRET-KEY-VALUE" not in str(caught.value)
    assert "consumerKey" not in str(caught.value)
