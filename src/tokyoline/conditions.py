"""Rail surface condition, and what it does to braking.

Modelled as rail condition rather than weather because only the wheel-rail
friction coefficient enters the equations. Temperature and wind would decorate
the interface without changing a number.

The asymmetry is the point. For traction the adhesion ceiling never binds on
this class: only 6 of 11 cars are powered, but demanded effort is about half
what even wet rail supplies. For braking every axle contributes, so the ceiling
is mu*g on the full mass, around 3.2 m/s^2 dry against a 1.17 m/s^2 service
rate. On badly contaminated rail it drops below the service rate and becomes
the binding constraint.

So rail condition barely touches acceleration and decides stopping distance.
Every multiplier here is modelled; there are no published figures for this line.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stock import adhesion_coefficient
from .units import G


@dataclass(frozen=True)
class RailCondition:
    """A rail surface state and its adhesion penalty.

    Attributes
    ----------
    name : display name.
    adhesion_multiplier : factor applied to the dry-rail Curtius-Kniffler
        coefficient. 1.0 is dry.
    description : what the condition means operationally.
    """

    name: str
    adhesion_multiplier: float
    description: str

    def coefficient(self, v_ms: float) -> float:
        """Available adhesion coefficient at this speed and condition."""
        return adhesion_coefficient(v_ms) * self.adhesion_multiplier

    def max_braking_ms2(self, v_ms: float) -> float:
        """Deceleration ceiling set by adhesion, in m/s^2.

        Braking uses every axle, so the full train mass is available and the
        ceiling is simply mu * g -- no adhesive-mass fraction, unlike traction.
        """
        return self.coefficient(v_ms) * G

    def effective_brake_rate_ms2(self, service_rate_ms2: float,
                                 v_ms: float = 0.0) -> float:
        """Deceleration actually achievable: the lesser of service rate and grip.

        Evaluated at low speed by default because adhesion rises as speed falls,
        so the ceiling is least restrictive at the end of the stop. Evaluating
        at the start of braking would be the conservative choice; evaluating at
        rest matches how the published service rate is quoted.
        """
        return min(service_rate_ms2, self.max_braking_ms2(v_ms))

    def is_adhesion_limited(self, service_rate_ms2: float, v_ms: float = 0.0) -> bool:
        return self.max_braking_ms2(v_ms) < service_rate_ms2


#: Rail conditions, ordered from best to worst grip.
#:
#: Multipliers are MODELLED. The literature gives wet rail adhesion at roughly
#: 50-70 percent of dry, and contaminated rail (crushed leaves, frost, light
#: drizzle on a dry-weather film) substantially lower still -- leaf contamination
#: is the classic autumn adhesion problem on railways worldwide. Values here are
#: representative of those ranges, not measurements from this line.
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
