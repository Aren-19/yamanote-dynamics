"""Rail surface condition, and what it does to braking."""

from __future__ import annotations

from dataclasses import dataclass

from .stock import adhesion_coefficient
from .units import G


@dataclass(frozen=True)
class RailCondition:
    """A rail surface state and its adhesion penalty."""

    name: str
    adhesion_multiplier: float
    description: str

    def coefficient(self, v_ms: float) -> float:
        """Available adhesion coefficient at this speed and condition."""
        return adhesion_coefficient(v_ms) * self.adhesion_multiplier

    def max_braking_ms2(self, v_ms: float) -> float:
        """Deceleration ceiling set by adhesion, in m/s^2."""
        return self.coefficient(v_ms) * G

    def effective_brake_rate_ms2(self, service_rate_ms2: float,
                                 v_ms: float = 0.0) -> float:
        """Deceleration actually achievable: the lesser of service rate and grip."""
        return min(service_rate_ms2, self.max_braking_ms2(v_ms))

    def is_adhesion_limited(self, service_rate_ms2: float, v_ms: float = 0.0) -> bool:
        return self.max_braking_ms2(v_ms) < service_rate_ms2


#: Rail conditions, ordered from best to worst grip.
CONDITIONS: dict[str, RailCondition] = {
    "dry": RailCondition(
        "Dry", 1.00,
        "Clean dry rail. Service brake rate governs; adhesion is not a factor.",
    ),
    "damp": RailCondition(
        "Damp", 0.80,
        "Light moisture or early drizzle. Slight loss of grip, still well clear "
        "of the service brake rate.",
    ),
    "wet": RailCondition(
        "Wet", 0.60,
        "Steady rain. Grip noticeably reduced but the service rate still governs "
        "for this class.",
    ),
    "leaf_fall": RailCondition(
        "Leaf fall / frost", 0.35,
        "Crushed leaf film or frost. The classic low-adhesion condition. Grip "
        "may fall below the service brake rate, making braking adhesion-limited.",
    ),
}

DEFAULT_CONDITION = "dry"
