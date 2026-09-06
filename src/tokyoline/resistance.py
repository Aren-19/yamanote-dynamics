"""Running, grade and curve resistance.

The Davis coefficients are modelled, not measured. Per-class sets exist inside
Japanese operators but are not released, and none for the E235 could be found.
These are estimated from mass, car count and frontal area with each
contribution named separately, so the assumption can be audited rather than
taken as a magic number.

This is the largest single uncertainty in the physics. Any agreement with the
timetable is conditional on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .units import RHO_AIR, G


@dataclass(frozen=True)
class DavisCoefficients:
    """Running resistance R(v) = A + B*v + C*v^2, with v in m/s and R in newtons.

    Railway literature usually writes Davis in km/h and kgf/t. Everything here is
    SI so it drops straight into F = ma.

    A is journal, bearing and rolling resistance: roughly proportional to mass,
    dominant at low speed, and the only term left at rest. B is flange contact and
    track interaction. C is aerodynamic drag, proportional to frontal area and
    dominant above about 80 km/h.
    """

    A: float
    B: float
    C: float
    provenance: str = "modelled"

    def __call__(self, v_ms: float) -> float:
        """Resistance in newtons at speed `v_ms`. Always non-negative."""
        v = abs(v_ms)
        return self.A + self.B * v + self.C * v * v

    def specific_n_per_tonne(self, v_ms: float, mass_kg: float) -> float:
        """Resistance per tonne, the form railway sources usually quote.

        Useful for sanity-checking the estimate: a commuter EMU on level track
        should land around 10-15 N/t at rest and 25-35 N/t at 90 km/h.
        """
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
        """Estimate Davis coefficients for a multiple unit.

        Each keyword is a separate physical claim, so each can be challenged on its
        own.

        journal_n_per_tonne
            Rolling and bearing resistance at rest. Modern roller-bearing stock sits
            around 8-15 N/t; 10 is central.
        flange_n_per_tonne_per_kmh
            Flange and track interaction, per km/h because that is how it is
            tabulated, converted here.
        cd_ends
            Combined nose and tail pressure drag. A commuter EMU is far blunter than a
            Shinkansen, hence 0.8 rather than 0.2.
        cd_per_car
            Skin friction per car. Over 11 cars this roughly doubles the total drag
            coefficient, which is why a long train is not one bluff body.

        The aerodynamic term is much higher in tunnels: irrelevant for the Yamanote,
        which is overwhelmingly surface running, and load-bearing for anything
        Shinkansen.
        """
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
    """Gravitational resistance on a gradient, in newtons.

    Japanese practice gives gradient in per-mille, and at these angles sin(theta)
    is grade/1000 to well inside any other error here. Positive grade is uphill,
    so a positive retarding force.

    Negligible on the Yamanote, which is close to flat and has no public
    per-segment gradient data at usable precision. Implemented for correctness, and
    so the model carries over to lines where it matters.
    """
    return mass_kg * G * (grade_permille / 1000.0)


def curve_resistance(mass_kg: float, radius_m: float | None,
                     coefficient: float = 600.0) -> float:
    """Extra resistance on a curve, in newtons.

    Empirical, of the form k / R. Several formulations are in use; `coefficient` is
    in N*m per tonne and 600 is common for standard gauge.

    Returns zero when `radius_m` is None, which is the honest default: per-segment
    curve radii for the Yamanote are not available, so the term is carried in the
    code and evaluated at zero. The effect is small.
    """
    if radius_m is None or radius_m <= 0.0:
        return 0.0
    return coefficient * (mass_kg / 1000.0) / radius_m
