"""Rolling stock specification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .units import G, ms, ms2


@dataclass(frozen=True)
class TrainSpec:
    """Physical and performance parameters for one train class."""

    name: str
    n_cars: int
    n_motor_cars: int

    tare_mass_kg: float
    capacity: int
    passenger_mass_kg: float

    total_power_w: float
    """One-hour rating. See the note in :meth:`tractive_effort_limit`."""

    max_speed_ms: float
    start_accel_ms2: float
    service_decel_ms2: float

    lambda_rot: float
    """Rotational inertia allowance. m_eff = m * (1 + lambda)."""

    adhesive_mass_fraction: float
    body_width_m: float
    body_height_m: float
    car_length_m: float


    def mass_kg(self, load_factor: float = 0.0) -> float:
        """Static mass at a given load."""
        return self.tare_mass_kg + load_factor * self.capacity * self.passenger_mass_kg

    def effective_mass_kg(self, load_factor: float = 0.0) -> float:
        """Mass including the rotational inertia of wheels, axles and motors."""
        return self.mass_kg(load_factor) * (1.0 + self.lambda_rot)

    def adhesive_mass_kg(self, load_factor: float = 0.0) -> float:
        """Mass carried on powered axles, which is what generates adhesion."""
        return self.mass_kg(load_factor) * self.adhesive_mass_fraction


    @property
    def frontal_area_m2(self) -> float:
        """Body width x height above rail."""
        return self.body_width_m * self.body_height_m

    @property
    def train_length_m(self) -> float:
        return self.n_cars * self.car_length_m


    def tractive_effort_limit(self, resistance_at_rest_n: float,
                              load_factor: float = 0.0) -> float:
        """Maximum tractive effort in the constant-effort region, in newtons."""
        return self.effective_mass_kg(load_factor) * self.start_accel_ms2 + resistance_at_rest_n

    def base_speed_ms(self, resistance_at_rest_n: float,
                      load_factor: float = 0.0) -> float:
        """Speed where constant effort gives way to constant power."""
        return self.total_power_w / self.tractive_effort_limit(resistance_at_rest_n, load_factor)

    def adhesion_limit_n(self, v_ms: float, load_factor: float = 0.0) -> float:
        """Ceiling on tractive effort set by wheel-rail friction, in newtons."""
        return adhesion_coefficient(v_ms) * self.adhesive_mass_kg(load_factor) * G


    @classmethod
    def from_json(cls, path: Path | str, key: str = "E235-0") -> TrainSpec:
        """Build a spec from the provenance-tagged JSON data file."""
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        d = blob[key]

        def val(*path_parts):
            node = d
            for part in path_parts:
                node = node[part]
            return node["value"] if isinstance(node, dict) and "value" in node else node

        return cls(
            name=d["designation"],
            n_cars=d["formation"]["n_cars"],
            n_motor_cars=d["formation"]["n_motor_cars"],
            tare_mass_kg=val("tare_mass_t") * 1000.0,
            capacity=int(val("capacity_passengers")),
            passenger_mass_kg=val("passenger_mass_kg"),
            total_power_w=val("total_power_kw") * 1000.0,
            max_speed_ms=ms(val("max_operating_speed_kmh")),
            start_accel_ms2=ms2(val("starting_acceleration_kmh_s")),
            service_decel_ms2=ms2(val("service_deceleration_kmh_s")),
            lambda_rot=val("rotational_inertia_lambda"),
            adhesive_mass_fraction=val("adhesive_mass_fraction"),
            body_width_m=val("body_width_mm") / 1000.0,
            body_height_m=val("body_height_mm") / 1000.0,
            car_length_m=val("car_length_intermediate_mm") / 1000.0,
        )


def adhesion_coefficient(v_ms: float) -> float:
    """Wheel-rail adhesion coefficient as a function of speed."""
    v_kmh = max(v_ms, 0.0) * 3.6
    return 0.161 + 7.5 / (v_kmh + 44.0)
