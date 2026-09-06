"""Simulated minimum run time against the published timetable."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .brake import BrakeProfile
from .data import DATA_DIR, Segment
from .segment import simulate_segment
from .traction import TractionModel


@dataclass(frozen=True)
class SegmentRun:
    segment: Segment
    run_time_s: float
    max_speed_kmh: float
    cruise_distance_m: float


@dataclass
class LoopRun:
    """Result of simulating all segments once, under one set of assumptions."""

    label: str
    runs: list[SegmentRun]
    speed_limit_kmh: float
    round_up_to_s: float | None = None

    @property
    def raw_total_s(self) -> float:
        return sum(r.run_time_s for r in self.runs)

    @property
    def total_s(self) -> float:
        """Total after any round-up-to-standard-run-time step."""
        if self.round_up_to_s is None:
            return self.raw_total_s
        q = self.round_up_to_s
        return sum(math.ceil(r.run_time_s / q) * q for r in self.runs)

    @property
    def n_segments(self) -> int:
        return len(self.runs)


@dataclass
class Rung:
    """One step of the decomposition ladder."""

    name: str
    cumulative_s: float
    delta_s: float
    source: str
    status: str  # published | modelled | derived | residual


@dataclass
class Decomposition:
    published_loop_s: float
    rungs: list[Rung]
    dwell_s: float
    n_stations: int
    residual_margin_s: float
    standard_run_time_s: float

    @property
    def residual_margin_fraction(self) -> float:
        """Residual margin as a fraction of the standard run time."""
        return self.residual_margin_s / self.standard_run_time_s

    def table(self) -> str:
        w = max(len(r.name) for r in self.rungs) + 2
        lines = [
            f"  {'layer':<{w}} {'cumulative':>12} {'delta':>10}   source",
            "  " + "-" * (w + 60),
        ]
        for r in self.rungs:
            lines.append(
                f"  {r.name:<{w}} {r.cumulative_s:10.0f} s {r.delta_s:+9.0f} s"
                f"   [{r.status}] {r.source}"
            )
        return "\n".join(lines)


def simulate_loop(
    traction: TractionModel,
    brake: BrakeProfile,
    segments: list[Segment],
    *,
    label: str,
    speed_limit_kmh: float,
    dt: float = 0.05,
    round_up_to_s: float | None = None,
) -> LoopRun:
    """Simulate every segment once at a given speed limit."""
    v_limit = speed_limit_kmh / 3.6
    runs = []
    for seg in segments:
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=v_limit, dt=dt)
        runs.append(
            SegmentRun(
                segment=seg,
                run_time_s=r.run_time_s,
                max_speed_kmh=r.max_speed_kmh,
                cruise_distance_m=r.phase_distances_m.get("cruise", 0.0),
            )
        )
    return LoopRun(label=label, runs=runs, speed_limit_kmh=speed_limit_kmh,
                   round_up_to_s=round_up_to_s)


def load_practice(path: Path | str | None = None) -> dict:
    path = Path(path) if path else DATA_DIR / "operational_practice.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_anchors(path: Path | str | None = None) -> dict:
    path = Path(path) if path else DATA_DIR / "timetable_anchors.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def decompose(
    traction: TractionModel,
    brake: BrakeProfile,
    segments: list[Segment],
    *,
    dwell_s: float,
    published_loop_s: float,
    line_speed_kmh: float = 90.0,
    derate_kmh: float = 3.0,
    round_up_to_s: float = 5.0,
    dt: float = 0.05,
) -> Decomposition:
    """Build the layered decomposition from theoretical minimum to timetable."""
    minimum = simulate_loop(traction, brake, segments, label="theoretical minimum",
                            speed_limit_kmh=line_speed_kmh, dt=dt)
    derated = simulate_loop(traction, brake, segments, label="de-rated run curve",
                            speed_limit_kmh=line_speed_kmh - derate_kmh, dt=dt)
    rounded = simulate_loop(traction, brake, segments, label="standard run time",
                            speed_limit_kmh=line_speed_kmh - derate_kmh, dt=dt,
                            round_up_to_s=round_up_to_s)

    n = len(segments)
    total_dwell = dwell_s * n
    standard_run = rounded.total_s
    residual = published_loop_s - standard_run - total_dwell

    rungs = [
        Rung(
            "theoretical minimum",
            minimum.raw_total_s,
            minimum.raw_total_s,
            "this model: full tractive effort, max service brake, no restrictions",
            "derived",
        ),
        Rung(
            f"run curve at -{derate_kmh:.0f} km/h",
            derated.raw_total_s,
            derated.raw_total_s - minimum.raw_total_s,
            "MLIT survey: operators draw run curves 2-5 km/h below the limit",
            "modelled",
        ),
        Rung(
            f"round up to {round_up_to_s:.0f} s (基準運転時分)",
            standard_run,
            standard_run - derated.raw_total_s,
            "MLIT survey: JR East rounds the computed run time up",
            "modelled",
        ),
        Rung(
            f"dwell at {dwell_s:.0f} s x {n} stations",
            standard_run + total_dwell,
            total_dwell,
            "peer-operator dwell figures; NOT published by JR East",
            "modelled",
        ),
        Rung(
            "residual margin (余裕時分)",
            published_loop_s,
            residual,
            "whatever is left: added margin, driver conservatism, signal restrictions",
            "residual",
        ),
    ]

    return Decomposition(
        published_loop_s=published_loop_s,
        rungs=rungs,
        dwell_s=dwell_s,
        n_stations=n,
        residual_margin_s=residual,
        standard_run_time_s=standard_run,
    )


def identifiability_locus(
    standard_run_time_s: float,
    published_loop_s: float,
    n_stations: int,
    dwell_range_s: tuple[float, float] = (10.0, 55.0),
    n_points: int = 46,
) -> dict[str, np.ndarray]:
    """The set of (dwell, margin) pairs consistent with the published loop time."""
    dwell = np.linspace(dwell_range_s[0], dwell_range_s[1], n_points)
    margin = (published_loop_s - n_stations * dwell - standard_run_time_s) / standard_run_time_s
    return {"dwell_s": dwell, "margin_fraction": margin}


def dwell_implied_by_margin(standard_run_time_s: float, published_loop_s: float,
                            n_stations: int, margin_fraction: float) -> float:
    """Invert the problem: given a margin, what mean dwell does the loop imply?"""
    return (published_loop_s - standard_run_time_s * (1.0 + margin_fraction)) / n_stations


def dwell_implied_by_zero_margin(standard_run_time_s: float, published_loop_s: float,
                                 n_stations: int) -> float:
    """Mean dwell that would account for the entire gap with no margin at all."""
    return (published_loop_s - standard_run_time_s) / n_stations


@dataclass
class SegmentComparison:
    """Per-segment simulated vs scheduled, for the bar chart."""

    runs: list[SegmentRun]
    scheduled_s: list[float] | None = None
    quantization_s: float = 30.0
    warning: str = field(
        default=(
            "Per-segment scheduled times are published only to the minute. "
            "Quantization dominates per-segment error; use the loop total for "
            "any quantitative claim."
        )
    )
