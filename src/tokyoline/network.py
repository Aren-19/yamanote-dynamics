"""Loop geometry and whole-circuit simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .brake import BrakeProfile
from .data import DATA_DIR, Segment, load_segments
from .segment import simulate_segment
from .traction import TractionModel


def load_coordinates(path: Path | str | None = None) -> pd.DataFrame:
    """Station coordinates, fetched from OpenStreetMap. Display only."""
    path = Path(path) if path else DATA_DIR / "station_coordinates.csv"
    return pd.read_csv(path, comment="#")


@dataclass
class LoopGeometry:
    """Station positions along the loop, in both distance and coordinates."""

    names: list[str]
    cumulative_m: np.ndarray
    """Distance from Shinagawa to each station, from PUBLISHED kilometrage."""
    lat: np.ndarray
    lon: np.ndarray
    total_m: float

    @classmethod
    def build(cls, segments: list[Segment] | None = None) -> LoopGeometry:
        segments = segments or load_segments()
        coords = load_coordinates()

        names = [s.from_station for s in segments]
        by_name = {r.name_en: (r.lat, r.lon) for r in coords.itertuples()}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"no coordinates for: {missing}")

        cum = np.concatenate([[0.0], np.cumsum([s.distance_m for s in segments])])
        total = float(cum[-1])

        return cls(
            names=names,
            cumulative_m=cum[:-1],
            lat=np.array([by_name[n][0] for n in names]),
            lon=np.array([by_name[n][1] for n in names]),
            total_m=total,
        )

    def position_at(self, distance_m: float) -> tuple[float, float]:
        """Interpolated (lat, lon) at a distance along the loop."""
        d = float(distance_m) % self.total_m
        edges = np.append(self.cumulative_m, self.total_m)
        i = int(np.searchsorted(edges, d, side="right") - 1)
        i = min(max(i, 0), len(self.names) - 1)

        seg_start = edges[i]
        seg_len = edges[i + 1] - seg_start
        frac = 0.0 if seg_len <= 0 else (d - seg_start) / seg_len

        j = (i + 1) % len(self.names)
        return (
            float(self.lat[i] + frac * (self.lat[j] - self.lat[i])),
            float(self.lon[i] + frac * (self.lon[j] - self.lon[i])),
        )

    def closed_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Station coordinates with the first repeated at the end, for drawing."""
        return (
            np.append(self.lat, self.lat[0]),
            np.append(self.lon, self.lon[0]),
        )

    def next_station_index(self, distance_m: float) -> int:
        d = float(distance_m) % self.total_m
        edges = np.append(self.cumulative_m, self.total_m)
        i = int(np.searchsorted(edges, d, side="right") - 1)
        return (min(max(i, 0), len(self.names) - 1) + 1) % len(self.names)


@dataclass
class Circuit:
    """A full simulated circuit, sampled on a uniform time grid."""

    t: np.ndarray
    distance_m: np.ndarray
    speed_ms: np.ndarray
    accel_ms2: np.ndarray
    """Instantaneous acceleration. Negative while braking."""
    force_n: np.ndarray
    """Applied tractive effort. Zero while coasting or braking."""
    dwelling: np.ndarray
    """Boolean: True while stopped at a platform."""
    segment_index: np.ndarray
    arrival_times_s: list[float]
    departure_times_s: list[float]
    run_times_s: list[float]
    dwell_s: float
    total_time_s: float

    @property
    def speed_kmh(self) -> np.ndarray:
        return self.speed_ms * 3.6

    @property
    def accel_kmh_s(self) -> np.ndarray:
        """Acceleration in km/h/s, the unit Japanese stock is specified in."""
        return self.accel_ms2 * 3.6

    @property
    def power_w(self) -> np.ndarray:
        """Instantaneous traction power at the wheel."""
        return self.force_n * self.speed_ms

    def sample(self, t_query: float) -> dict:
        """State at an arbitrary time, wrapping around the circuit."""
        tq = float(t_query) % self.total_time_s
        i = int(np.searchsorted(self.t, tq, side="right") - 1)
        i = min(max(i, 0), len(self.t) - 1)
        return {
            "t": tq,
            "distance_m": float(self.distance_m[i]),
            "speed_ms": float(self.speed_ms[i]),
            "speed_kmh": float(self.speed_ms[i] * 3.6),
            "dwelling": bool(self.dwelling[i]),
            "segment_index": int(self.segment_index[i]),
        }


def simulate_circuit(
    traction: TractionModel,
    brake: BrakeProfile,
    segments: list[Segment],
    *,
    speed_limit_ms: float,
    dwell_s: float,
    dt: float = 0.1,
    sample_dt: float = 1.0,
) -> Circuit:
    """Simulate a complete circuit including dwells, on a uniform time grid."""
    times: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    accels: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    dwell_flags: list[np.ndarray] = []
    seg_idx: list[np.ndarray] = []

    arrivals: list[float] = []
    departures: list[float] = []
    run_times: list[float] = []

    t_offset = 0.0
    x_offset = 0.0

    for i, seg in enumerate(segments):
        departures.append(t_offset)
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=speed_limit_ms, dt=dt)
        run_times.append(r.run_time_s)

        times.append(r.t + t_offset)
        dists.append(r.x + x_offset)
        speeds.append(r.v)
        accels.append(r.a)
        forces.append(r.tractive_force_n)
        dwell_flags.append(np.zeros_like(r.t, dtype=bool))
        seg_idx.append(np.full(r.t.shape, i, dtype=int))

        t_offset += r.run_time_s
        x_offset += seg.distance_m
        arrivals.append(t_offset)

        # Dwell at the destination platform.
        n_dwell = max(int(dwell_s / sample_dt) + 1, 2)
        t_dwell = np.linspace(0.0, dwell_s, n_dwell)
        times.append(t_dwell + t_offset)
        dists.append(np.full(n_dwell, x_offset))
        speeds.append(np.zeros(n_dwell))
        accels.append(np.zeros(n_dwell))
        forces.append(np.zeros(n_dwell))
        dwell_flags.append(np.ones(n_dwell, dtype=bool))
        seg_idx.append(np.full(n_dwell, i, dtype=int))
        t_offset += dwell_s

    t_raw = np.concatenate(times)
    x_raw = np.concatenate(dists)
    v_raw = np.concatenate(speeds)
    a_raw = np.concatenate(accels)
    f_raw = np.concatenate(forces)
    d_raw = np.concatenate(dwell_flags)
    s_raw = np.concatenate(seg_idx)

    # Interpolation needs strictly increasing time.
    keep = np.concatenate([[True], np.diff(t_raw) > 1e-12])
    t_raw, x_raw, v_raw, a_raw, f_raw, d_raw, s_raw = (
        arr[keep] for arr in (t_raw, x_raw, v_raw, a_raw, f_raw, d_raw, s_raw)
    )

    total = float(t_raw[-1])
    grid = np.arange(0.0, total, sample_dt)

    return Circuit(
        t=grid,
        distance_m=np.interp(grid, t_raw, x_raw),
        speed_ms=np.interp(grid, t_raw, v_raw),
        accel_ms2=np.interp(grid, t_raw, a_raw),
        force_n=np.interp(grid, t_raw, f_raw),
        dwelling=np.interp(grid, t_raw, d_raw.astype(float)) > 0.5,
        segment_index=np.rint(np.interp(grid, t_raw, s_raw)).astype(int),
        arrival_times_s=arrivals,
        departure_times_s=departures,
        run_times_s=run_times,
        dwell_s=dwell_s,
        total_time_s=total,
    )


def journey_time_s(from_idx: int, to_idx: int, run_times: list[float],
                   dwell_s: float, outer: bool) -> float:
    """Modelled time between two stations, including intermediate dwells."""
    n = len(run_times)
    if n == 0 or from_idx == to_idx:
        return 0.0

    total = 0.0
    i = from_idx
    for _ in range(n):
        if outer:
            total += run_times[i % n]
            i = (i + 1) % n
        else:
            total += run_times[(i - 1) % n]
            i = (i - 1) % n
        if i == to_idx:
            return total
        total += dwell_s
    return total
