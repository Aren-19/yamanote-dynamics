"""Phase 4: cross-class comparison.

The contrast the comparison exists to make legible: a commuter EMU and a
high-speed set are optimised for opposite things. The E235 accelerates at 3.0
km/h/s and tops out at 90; the E5 accelerates at 1.71 and tops out at 320. Neither
is "better" -- they are solving different problems, and a scatter of acceleration
against maximum speed shows the trade-off in one glance.

Three things this module is careful about:

1. **Bounded quantities stay bounded.** N700S formation mass is published only as
   "under 700 t". Power-to-weight computed from it is therefore a LOWER bound,
   and it is returned tagged as one rather than silently becoming a point value.

2. **Missing quantities stay missing.** Service deceleration is not published for
   the E5 or N700S in the sources consulted. They are excluded from braking
   comparisons rather than filled in with a peer value.

3. **Speed-dependent braking is modelled where it is published.** The E7 publishes
   a deceleration that falls from 2.69 km/h/s at low speed to 1.44 at 275 km/h,
   because regenerative brake effort is power-limited just as tractive effort is.
   Treating that as a constant would understate its stopping distance badly. For a
   90 km/h commuter EMU the constant assumption is harmless; for high-speed stock
   it is not, and that difference is itself a result worth showing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .brake import (
    DEFAULT_JERK_LIMIT,
    BrakeProfile,
    SpeedDependentBrakeProfile,
)
from .data import DATA_DIR
from .units import KMH_TO_MS, KMHS_TO_MS2


@dataclass(frozen=True)
class Bounded:
    """A quantity known only as a one-sided bound.

    Exists so that a bound cannot be mistaken for a measurement further
    downstream. Anything derived from it should carry `is_bound` through.
    """

    value: float
    kind: str  # "upper" | "lower"

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        sign = "<" if self.kind == "upper" else ">"
        return f"{sign} {self.value:g}"


@dataclass(frozen=True)
class FleetMember:
    key: str
    display_name: str
    category: str
    operator: str

    n_cars: int
    n_motor_cars: int
    formation: str

    tare_mass_t: float | None
    tare_mass_bound_t: float | None
    capacity: int

    total_power_kw: float
    max_operating_speed_kmh: float
    design_max_speed_kmh: float
    starting_acceleration_kmh_s: float
    service_deceleration_kmh_s: float | None
    deceleration_curve: list[tuple[float, float]] | None

    car_length_m: float
    body_width_m: float
    body_height_m: float

    decel_is_ambiguous: bool = False

    # ------------------------------------------------------------- derived

    @property
    def motored_fraction(self) -> float:
        return self.n_motor_cars / self.n_cars

    @property
    def frontal_area_m2(self) -> float:
        return self.body_width_m * self.body_height_m

    @property
    def train_length_m(self) -> float:
        return self.n_cars * self.car_length_m

    @property
    def mass_for_ratios_t(self) -> float:
        """Mass to use in ratios. Falls back to the bound when exact is unknown."""
        if self.tare_mass_t is not None:
            return self.tare_mass_t
        if self.tare_mass_bound_t is not None:
            return self.tare_mass_bound_t
        raise ValueError(f"{self.key}: no mass available")

    @property
    def mass_is_bound(self) -> bool:
        return self.tare_mass_t is None and self.tare_mass_bound_t is not None

    def power_to_weight_kw_per_t(self) -> float | Bounded:
        """Installed power per tonne of tare mass.

        Returns a Bounded when formation mass is only bounded: dividing by an
        upper-bound mass yields a LOWER bound on power-to-weight.
        """
        ratio = self.total_power_kw / self.mass_for_ratios_t
        if self.mass_is_bound:
            return Bounded(ratio, "lower")
        return ratio

    def specific_power_kw_per_passenger(self) -> float | Bounded:
        return self.total_power_kw / self.capacity

    # ------------------------------------------------------------- braking

    def deceleration_at_kmh(self, v_kmh: float) -> float | None:
        """Service deceleration at a given speed, in km/h/s.

        Uses the published speed-dependent curve when there is one, otherwise
        the single published figure, otherwise None.
        """
        if self.deceleration_curve:
            xs = [p[0] for p in self.deceleration_curve]
            ys = [p[1] for p in self.deceleration_curve]
            return float(np.interp(v_kmh, xs, ys))
        return self.service_deceleration_kmh_s

    def brake_profile(self, jerk: float = DEFAULT_JERK_LIMIT,
                      dt: float = 0.005):
        """The braking model for this class, or None if none is published.

        Constant-rate classes get the closed-form BrakeProfile, which is exact.
        Classes publishing a speed-dependent curve get the numerical profile.
        Both satisfy the same interface, so callers do not branch.
        """
        if self.deceleration_at_kmh(0.0) is None:
            return None
        if self.deceleration_curve:
            return SpeedDependentBrakeProfile(
                list(self.deceleration_curve), jerk=jerk, dt=dt,
                v_max_ms=self.max_operating_speed_kmh * KMH_TO_MS,
            )
        return BrakeProfile(
            a_max=self.service_deceleration_kmh_s * KMHS_TO_MS2, jerk=jerk)

    def stopping_distance_m(self, from_kmh: float | None = None,
                            jerk: float = DEFAULT_JERK_LIMIT,
                            dt: float = 0.005) -> float | None:
        """Distance to stop from a given speed under a shared jerk limit.

        Defaults to the class's own maximum operating speed, so each train
        brakes from its own cruise under identical comfort constraints.

        Returns None when the class publishes no deceleration figure. Callers
        must handle that rather than receiving a plausible-looking default.
        """
        profile = self.brake_profile(jerk, dt)
        if profile is None:
            return None
        return profile.distance_to_stop(self._reference_speed_ms(from_kmh))

    def stopping_time_s(self, from_kmh: float | None = None,
                        jerk: float = DEFAULT_JERK_LIMIT,
                        dt: float = 0.005) -> float | None:
        profile = self.brake_profile(jerk, dt)
        if profile is None:
            return None
        return profile.time_to_stop(self._reference_speed_ms(from_kmh))

    def _reference_speed_ms(self, from_kmh: float | None) -> float:
        speed = from_kmh if from_kmh is not None else self.max_operating_speed_kmh
        return speed * KMH_TO_MS


def to_train_spec(m: FleetMember, *, passenger_mass_kg: float = 60.0,
                  lambda_rot: float = 0.08):
    """Convert a fleet entry into a simulatable :class:`TrainSpec`.

    Raises for classes that cannot be run. That is deliberate: it is better for
    an unusable class to fail loudly than to be silently simulated with an
    invented deceleration figure.
    """
    from .stock import TrainSpec
    from .units import ms, ms2

    if m.service_deceleration_kmh_s is None:
        raise ValueError(
            f"{m.key}: no published service deceleration, cannot be simulated"
        )
    if m.tare_mass_t is None:
        raise ValueError(f"{m.key}: formation mass is only bounded, cannot be simulated")

    return TrainSpec(
        name=m.display_name,
        n_cars=m.n_cars,
        n_motor_cars=m.n_motor_cars,
        tare_mass_kg=m.tare_mass_t * 1000.0,
        capacity=m.capacity,
        passenger_mass_kg=passenger_mass_kg,
        total_power_w=m.total_power_kw * 1000.0,
        max_speed_ms=ms(m.max_operating_speed_kmh),
        start_accel_ms2=ms2(m.starting_acceleration_kmh_s),
        service_decel_ms2=ms2(m.service_deceleration_kmh_s),
        lambda_rot=lambda_rot,
        adhesive_mass_fraction=m.n_motor_cars / m.n_cars,
        body_width_m=m.body_width_m,
        body_height_m=m.body_height_m,
        car_length_m=m.car_length_m,
    )


def simulatable(m: FleetMember) -> tuple[bool, str]:
    """Whether a class can be run on this line's model, and why not if it cannot.

    High-speed stock is excluded on purpose.

    Speed-dependent braking is no longer the obstacle it once was:
    SpeedDependentBrakeProfile implements it and satisfies the same interface
    as the constant-rate profile, so simulate_segment accepts either. What
    remains:

    * The Davis estimator was built for a blunt-nosed commuter EMU and does not
      transfer to a streamlined 400 m trainset. fleet.json carries per-class
      aerodynamic coefficients, but marks the high-speed ones as weakly
      grounded, and tunnel aerodynamics are not modelled at all -- a
      first-order omission on the Tokaido.
    * Yamanote geometry is wrong for the question. Running a 400 m Shinkansen
      between stops 1.15 km apart produces a number, but not a meaningful one.

    The refusal is therefore about the resistance model and the choice of line,
    not about the solver.
    """
    if m.category != "commuter EMU":
        return False, ("high-speed stock: the commuter Davis estimator does not "
                       "transfer, and Yamanote geometry is wrong for the "
                       "question")
    if m.service_deceleration_kmh_s is None:
        return False, "no published service deceleration"
    if m.tare_mass_t is None:
        return False, "formation mass is only bounded"
    return True, ""


def load_fleet(path: Path | str | None = None) -> dict[str, FleetMember]:
    """Load the comparison fleet, preserving bounds and gaps."""
    path = Path(path) if path else DATA_DIR / "fleet.json"
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)

    def val(node, key, default=None):
        n = node.get(key)
        if n is None:
            return default
        return n.get("value", default) if isinstance(n, dict) else n

    out: dict[str, FleetMember] = {}
    for key, d in blob["classes"].items():
        mass_node = d["tare_mass_t"]
        decel_node = d["service_deceleration_kmh_s"]
        curve_node = d.get("deceleration_curve_kmh_s")

        out[key] = FleetMember(
            key=key,
            display_name=d["display_name"],
            category=d["category"],
            operator=d["operator"],
            n_cars=d["formation"]["n_cars"],
            n_motor_cars=d["formation"]["n_motor_cars"],
            formation=d["formation"]["value"],
            tare_mass_t=mass_node.get("value"),
            tare_mass_bound_t=mass_node.get("upper_bound"),
            capacity=int(val(d, "capacity")),
            total_power_kw=float(val(d, "total_power_kw")),
            max_operating_speed_kmh=float(val(d, "max_operating_speed_kmh")),
            design_max_speed_kmh=float(val(d, "design_max_speed_kmh")),
            starting_acceleration_kmh_s=float(val(d, "starting_acceleration_kmh_s")),
            service_deceleration_kmh_s=decel_node.get("value"),
            deceleration_curve=(
                [tuple(p) for p in curve_node["points"]] if curve_node else None
            ),
            car_length_m=float(val(d, "car_length_m")),
            body_width_m=float(val(d, "body_width_m")),
            body_height_m=float(val(d, "body_height_m")),
            decel_is_ambiguous=decel_node.get("status") == "published_ambiguous",
        )
    return out
