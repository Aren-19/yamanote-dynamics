"""Running, grade and curve resistance."""

from __future__ import annotations

from dataclasses import dataclass

from .units import RHO_AIR, G


@dataclass(frozen=True)
class DavisCoefficients:
    """Running resistance R(v) = A + B*v + C*v^2, with v in m/s and R in newtons."""

    A: float
    B: float
    C: float
    provenance: str = "modelled"

    def __call__(self, v_ms: float) -> float:
        """Resistance in newtons at speed `v_ms`. Always non-negative."""
        v = abs(v_ms)
        return self.A + self.B * v + self.C * v * v

    def specific_n_per_tonne(self, v_ms: float, mass_kg: float) -> float:
        """Resistance per tonne, the form railway sources usually quote."""
        return self(v_ms) / (mass_kg / 1000.0)

    @classmethod
    def estimate_for_emu(
        cls,
        mass_kg: float,
        n_cars: int,
        frontal_area_m2: float,
        *,
        journal_n_per_tonne: float = 10.0,
        flange_n_per_tonne_per_kmh: float = 0.06,
        cd_ends: float = 0.80,
        cd_per_car: float = 0.06,
    ) -> DavisCoefficients:
        """Estimate Davis coefficients for a multiple unit."""
        mass_t = mass_kg / 1000.0

        A = journal_n_per_tonne * mass_t

        # Convert the per-km/h figure into per-(m/s).
        B = flange_n_per_tonne_per_kmh * mass_t * 3.6

        cd_effective = cd_ends + cd_per_car * n_cars
        C = 0.5 * RHO_AIR * cd_effective * frontal_area_m2

        return cls(
            A=A,
            B=B,
            C=C,
            provenance=(
                f"modelled: journal={journal_n_per_tonne} N/t, "
                f"flange={flange_n_per_tonne_per_kmh} N/t/(km/h), "
                f"Cd_eff={cd_effective:.2f} over {frontal_area_m2:.2f} m^2 "
                f"({n_cars} cars). No published E235 coefficient set available."
            ),
        )


def grade_resistance(mass_kg: float, grade_permille: float) -> float:
    """Gravitational resistance on a gradient, in newtons."""
    return mass_kg * G * (grade_permille / 1000.0)


def curve_resistance(mass_kg: float, radius_m: float | None,
                     coefficient: float = 600.0) -> float:
    """Extra resistance on a curve, in newtons."""
    if radius_m is None or radius_m <= 0.0:
        return 0.0
    return coefficient * (mass_kg / 1000.0) / radius_m
