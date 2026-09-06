"""Tractive effort against speed."""

from __future__ import annotations

from dataclasses import dataclass

from .resistance import DavisCoefficients
from .stock import TrainSpec


@dataclass(frozen=True)
class TractionModel:
    """Tractive effort curve for one train at one load condition."""

    spec: TrainSpec
    davis: DavisCoefficients
    load_factor: float = 1.0
    power_factor: float = 1.0
    """Multiplier on the published one-hour power rating."""

    @property
    def resistance_at_rest_n(self) -> float:
        return self.davis(0.0)

    @property
    def max_effort_n(self) -> float:
        """Flat effort in the constant-effort region, in newtons."""
        return self.spec.tractive_effort_limit(self.resistance_at_rest_n, self.load_factor)

    @property
    def power_w(self) -> float:
        return self.spec.total_power_w * self.power_factor

    @property
    def base_speed_ms(self) -> float:
        """Corner between the constant-effort and constant-power regions."""
        return self.power_w / self.max_effort_n

    def capability_n(self, v_ms: float) -> float:
        """Greatest tractive effort the machine can produce at this speed."""
        v = max(v_ms, 0.0)
        current_limited = self.max_effort_n
        if v <= self.base_speed_ms:
            power_limited = current_limited
        else:
            power_limited = self.power_w / max(v, 1e-9)
        adhesion_limited = self.spec.adhesion_limit_n(v, self.load_factor)
        return min(current_limited, power_limited, adhesion_limited)

    def effort_n(self, v_ms: float) -> float:
        """Tractive effort applied when accelerating at full power, in newtons."""
        if max(v_ms, 0.0) >= self.spec.max_speed_ms:
            return 0.0
        return self.capability_n(v_ms)

    def adhesion_binds(self, v_ms: float) -> bool:
        """Whether the adhesion ceiling is the active constraint at this speed."""
        v = max(v_ms, 0.0)
        if v <= self.base_speed_ms:
            unconstrained = self.max_effort_n
        else:
            unconstrained = self.power_w / max(v, 1e-9)
        return self.spec.adhesion_limit_n(v, self.load_factor) < unconstrained

    def holding_force_n(self, v_ms: float, *, grade_permille: float = 0.0,
                        curve_radius_m: float | None = None) -> float:
        """Tractive effort needed to hold a constant speed, in newtons."""
        from .resistance import curve_resistance, grade_resistance

        mass = self.spec.mass_kg(self.load_factor)
        required = (
            self.davis(v_ms)
            + grade_resistance(mass, grade_permille)
            + curve_resistance(mass, curve_radius_m)
        )
        return min(max(required, 0.0), self.capability_n(v_ms))

    def net_force_n(self, v_ms: float, *, grade_permille: float = 0.0,
                    curve_radius_m: float | None = None,
                    powering: bool = True) -> float:
        """Net accelerating force, in newtons. Positive accelerates the train."""
        from .resistance import curve_resistance, grade_resistance

        mass = self.spec.mass_kg(self.load_factor)
        traction = self.effort_n(v_ms) if powering else 0.0
        return (
            traction
            - self.davis(v_ms)
            - grade_resistance(mass, grade_permille)
            - curve_resistance(mass, curve_radius_m)
        )

    def acceleration_ms2(self, v_ms: float, **kwargs) -> float:
        """Instantaneous acceleration, in m/s^2, using effective mass."""
        return self.net_force_n(v_ms, **kwargs) / self.spec.effective_mass_kg(self.load_factor)
