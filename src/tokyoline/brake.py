"""Jerk-limited braking.

A real brake application cannot step instantly to full deceleration -- that
would be violently uncomfortable for standing passengers, and on a line where
most riders stand, comfort is a hard operational constraint rather than a nicety.
So a stop is three phases:

    ramp in at the jerk limit  ->  hold at maximum deceleration  ->  ramp out

The ramp-out matters as much as the ramp-in: deceleration must reach zero at the
same moment speed does, otherwise the train stops with a lurch.

Consequences that make this worth modelling from the start rather than bolting
on later:

* Stopping takes longer than the idealised instant application by exactly
  ``a_max / j`` seconds in the trapezoidal case -- independent of initial speed.
  For the E235 at 4.2 km/h/s with a 0.75 m/s^3 jerk limit that is about 1.6 s
  per stop, so roughly 47 s over a 30-station circuit against a scheduled
  3948 s. About 1.2 percent: real, measurable, and smaller than the recovery
  margin it must not be confused with.
* It changes the structure of the solver. The brake application point is no
  longer a simple v^2/2a lookup, so it has to be found by intersecting a
  backward-integrated brake curve with the forward run curve.

Sign convention throughout: `a_max` and `jerk` are positive magnitudes.
Deceleration is applied as a negative acceleration internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Comfort jerk limit for standing passengers, m/s^3.
#: Literature puts the acceptable range at roughly 0.5-1.0; 0.75 is the midpoint.
#: MODELLED, not published for this line or class.
DEFAULT_JERK_LIMIT = 0.75


@dataclass(frozen=True)
class BrakeProfile:
    """A jerk-limited deceleration profile to a stand.

    Parameters
    ----------
    a_max : maximum deceleration magnitude, m/s^2.
    jerk : maximum rate of change of deceleration, m/s^3.

    The published deceleration figure for Japanese stock (4.2 km/h/s for the
    E235) is a NET rate -- what the train actually achieves. Running resistance
    is therefore not added on top of it during braking, unlike during coasting
    where resistance is the only retarding force. Treating the published figure
    as net is the standard reading and is the conservative one here: it does not
    credit the model with free deceleration.
    """

    a_max: float
    jerk: float = DEFAULT_JERK_LIMIT

    def __post_init__(self) -> None:
        if self.a_max <= 0.0:
            raise ValueError("a_max must be a positive magnitude")
        if self.jerk <= 0.0:
            raise ValueError("jerk must be a positive magnitude")

    @property
    def speed_threshold_for_full_braking(self) -> float:
        """Initial speed below which the profile is triangular, not trapezoidal.

        Below ``a_max^2 / j`` there is not enough speed to reach full
        deceleration before the ramp-out must begin, so the hold phase vanishes.
        """
        return self.a_max ** 2 / self.jerk

    def is_trapezoidal(self, v0: float) -> bool:
        return v0 >= self.speed_threshold_for_full_braking

    def peak_deceleration(self, v0: float) -> float:
        """Actual peak deceleration reached when stopping from `v0`.

        Equals `a_max` for a trapezoidal stop; less for a triangular one.
        """
        if self.is_trapezoidal(v0):
            return self.a_max
        return math.sqrt(max(v0, 0.0) * self.jerk)

    # ------------------------------------------------------- closed solution

    def phase_durations(self, v0: float) -> tuple[float, float, float]:
        """Durations of (ramp-in, hold, ramp-out) when stopping from `v0`."""
        v0 = max(v0, 0.0)
        if v0 == 0.0:
            return (0.0, 0.0, 0.0)

        a_pk = self.peak_deceleration(v0)
        t_ramp = a_pk / self.jerk

        if self.is_trapezoidal(v0):
            # Speed shed by the two ramps together is a_max^2 / j.
            t_hold = (v0 - self.a_max ** 2 / self.jerk) / self.a_max
        else:
            t_hold = 0.0

        return (t_ramp, t_hold, t_ramp)

    def time_to_stop(self, v0: float) -> float:
        """Total time to come to a stand from `v0`, in seconds.

        For a trapezoidal stop this reduces to ``v0 / a_max + a_max / j`` --
        the idealised time plus a fixed jerk penalty.
        """
        t1, t2, t3 = self.phase_durations(v0)
        return t1 + t2 + t3

    def distance_to_stop(self, v0: float) -> float:
        """Distance required to come to a stand from `v0`, in metres.

        Closed form, integrated phase by phase. This is the function the segment
        solver inverts to find the brake application point, and computing it
        analytically rather than by forward trial-and-error is what stops the
        solver overshooting the platform.
        """
        v0 = max(v0, 0.0)
        if v0 == 0.0:
            return 0.0

        j = self.jerk
        t1, t2, t3 = self.phase_durations(v0)
        a_pk = self.peak_deceleration(v0)

        # Phase 1: a ramps 0 -> -a_pk.  v = v0 - 0.5*j*t^2
        d1 = v0 * t1 - j * t1 ** 3 / 6.0
        v1 = v0 - 0.5 * j * t1 ** 2

        # Phase 2: constant -a_pk.
        d2 = v1 * t2 - 0.5 * a_pk * t2 ** 2
        v2 = v1 - a_pk * t2

        # Phase 3: a ramps -a_pk -> 0, ending exactly at v = 0.
        d3 = v2 * t3 - 0.5 * a_pk * t3 ** 2 + j * t3 ** 3 / 6.0

        return d1 + d2 + d3

    def idealised_distance_to_stop(self, v0: float) -> float:
        """Distance under an instant full-brake application: v0^2 / (2*a_max).

        Only useful for quantifying what the jerk limit costs. Never used in the
        simulation itself.
        """
        return max(v0, 0.0) ** 2 / (2.0 * self.a_max)

    def jerk_penalty_distance(self, v0: float) -> float:
        """Extra stopping distance attributable to the jerk limit, in metres."""
        return self.distance_to_stop(v0) - self.idealised_distance_to_stop(v0)

    # ------------------------------------------------------------ trajectory

    def trajectory(self, v0: float, dt: float = 0.01) -> dict[str, np.ndarray]:
        """Sampled deceleration trajectory from `v0` to rest.

        Integrated backwards in spirit but evaluated forwards from the closed
        form, so it carries no integration error of its own. Returned distances
        are measured from the brake application point.

        Returns arrays keyed 't', 'v', 'x', 'a'.
        """
        t_total = self.time_to_stop(v0)
        n = max(math.ceil(t_total / dt) + 1, 2)
        t = np.linspace(0.0, t_total, n)

        j = self.jerk
        t1, t2, _t3 = self.phase_durations(v0)
        a_pk = self.peak_deceleration(v0)

        a = np.empty_like(t)
        v = np.empty_like(t)
        x = np.empty_like(t)

        v1 = v0 - 0.5 * j * t1 ** 2
        d1 = v0 * t1 - j * t1 ** 3 / 6.0
        v2 = v1 - a_pk * t2
        d2 = v1 * t2 - 0.5 * a_pk * t2 ** 2

        in1 = t <= t1
        in2 = (t > t1) & (t <= t1 + t2)
        in3 = t > t1 + t2

        # Ramp in.
        tau = t[in1]
        a[in1] = -j * tau
        v[in1] = v0 - 0.5 * j * tau ** 2
        x[in1] = v0 * tau - j * tau ** 3 / 6.0

        # Hold.
        tau = t[in2] - t1
        a[in2] = -a_pk
        v[in2] = v1 - a_pk * tau
        x[in2] = d1 + v1 * tau - 0.5 * a_pk * tau ** 2

        # Ramp out.
        tau = t[in3] - t1 - t2
        a[in3] = -a_pk + j * tau
        v[in3] = v2 - a_pk * tau + 0.5 * j * tau ** 2
        x[in3] = d1 + d2 + v2 * tau - 0.5 * a_pk * tau ** 2 + j * tau ** 3 / 6.0

        # Clamp the final sample: floating point can leave v a hair below zero.
        v[-1] = 0.0
        a[-1] = 0.0

        return {"t": t, "v": np.maximum(v, 0.0), "x": x, "a": a}


class SpeedDependentBrakeProfile:
    """A jerk-limited stop whose deceleration limit varies with speed.

    High-speed stock does not brake at a constant rate. The E7 publishes
    2.69 km/h/s below 70 km/h falling to 1.44 at 275 km/h, because regenerative
    brake effort is power-limited in the same way tractive effort is, and
    adhesion falls with speed. Treating that as constant understates the
    stopping distance from line speed by about 30 percent.

    For a 90 km/h commuter EMU the constant assumption is harmless, since the
    whole speed range sits in the flat part of the curve. That is why the
    Yamanote model uses BrakeProfile. This class exists so high-speed stock can
    be simulated on the same solver.

    The interface deliberately matches BrakeProfile, so simulate_segment
    accepts either without modification.

    There is no closed form here, so stops are integrated numerically once at
    construction and cached as a lookup table. distance_to_stop is called at
    every solver step; integrating on each call would dominate the run time.

    ACCURACY. Validated against the closed-form BrakeProfile by giving this
    class a flat curve: the two then describe identical physics, and agree to
    under a microsecond and a few tens of micrometres on the raw integration.
    Table lookup adds a little, leaving under a millimetre of platform error
    through the segment solver.

    Two details carry most of that accuracy. The ramp-out is closed-form rather
    than stepped, because deceleration tends to zero there and the phase is
    correspondingly sensitive to where it is entered -- stepping into it cost
    0.1 s on every stop, independent of speed. And the lookup table is
    interpolated against v-squared rather than v, since stopping distance is
    dominated by the v^2/2a term.

    Prefer :class:`BrakeProfile` anyway whenever the rate is constant: it is
    exact by construction rather than by validation.
    """

    #: Speeds sampled when building the lookup table. Stopping distance varies
    #: smoothly with initial speed, so interpolating between these is accurate
    #: to well under a metre.
    TABLE_POINTS = 80

    def __init__(self, points, jerk=DEFAULT_JERK_LIMIT, *, dt=0.005,
                 v_max_ms=None):
        """
        Parameters
        ----------
        points : (speed km/h, deceleration km/h/s) pairs.
        jerk : comfort jerk limit, m/s^3.
        dt : integration step used to build the internal table, seconds.
        v_max_ms : highest speed the table covers. Defaults to the fastest
            point supplied.
        """
        if not points:
            raise ValueError("points must not be empty")
        if jerk <= 0.0:
            raise ValueError("jerk must be a positive magnitude")

        ordered = sorted(points)
        self._v_kmh = np.array([p[0] for p in ordered], dtype=float)
        self._a_kmh_s = np.array([p[1] for p in ordered], dtype=float)
        if np.any(self._a_kmh_s <= 0.0):
            raise ValueError("deceleration values must be positive")

        self.points = tuple(ordered)
        self.jerk = float(jerk)
        self.dt = float(dt)
        self.v_max_ms = float(
            v_max_ms if v_max_ms is not None else self._v_kmh[-1] / 3.6)

        self._build_table()

    # ---------------------------------------------------------------- rate

    def rate_ms2(self, v_ms):
        """Deceleration limit at this speed, m/s^2.

        Held flat outside the published range rather than extrapolated. The
        curve is empirical, and continuing its slope past the last point would
        invent behaviour that eventually goes negative.
        """
        v_kmh = max(float(v_ms), 0.0) * 3.6
        return float(np.interp(v_kmh, self._v_kmh, self._a_kmh_s)) / 3.6

    @property
    def a_max(self):
        """Greatest deceleration anywhere on the curve, m/s^2.

        Present so code written against BrakeProfile keeps working.
        """
        return float(self._a_kmh_s.max()) / 3.6

    @property
    def speed_threshold_for_full_braking(self):
        return self.a_max ** 2 / self.jerk

    # --------------------------------------------------------- integration

    def _integrate(self, v0):
        """Integrate one stop from v0, returning sampled t, v, x, a."""
        v0 = max(float(v0), 0.0)
        if v0 <= 0.0:
            z = np.zeros(1)
            return {"t": z, "v": z, "x": z, "a": z}

        dt = self.dt
        j = self.jerk
        v = v0
        x = 0.0
        a = 0.0
        t = 0.0
        ts = [t]
        vs = [v]
        xs = [x]
        acc = [0.0]

        # Cap the run so a pathological curve cannot loop forever.
        max_steps = int(600.0 / dt)
        reached_rest = False
        for _ in range(max_steps):
            target = self.rate_ms2(v)
            a_prev = a

            # The ramp-out threshold: the speed at which easing deceleration
            # off at the jerk limit brings the train exactly to rest.
            v_threshold = a * a / (2.0 * j)

            if v <= v_threshold:
                # Close the stop out analytically. Once easing off, the rate
                # curve no longer governs -- deceleration simply ramps down at
                # the jerk limit -- so this phase is exact:
                #     duration = a / j,  distance = a^3 / (6 j^2)
                # Doing it in closed form matters because deceleration tends to
                # zero here, which makes the phase extremely sensitive to where
                # it is entered. Stepping into it on a step boundary instead
                # cost roughly 0.1 s on every stop.
                t += a / j
                x += a ** 3 / (6.0 * j * j)
                ts.append(t)
                vs.append(0.0)
                xs.append(x)
                acc.append(0.0)
                reached_rest = True
                break

            if a < target:
                a = min(a + j * dt, target)
            else:
                a = max(a - j * dt, target)

            # Trapezoidal in the deceleration: the average of the values at
            # each end of the step rather than the end value alone, which
            # would over-estimate the mean rate while jerk is ramping in.
            a_mean = 0.5 * (a_prev + a)
            dv = a_mean * dt

            # Step exactly onto the threshold rather than past it, so the
            # closed-form phase above starts where it should.
            next_threshold = a * a / (2.0 * j)
            if v - dv <= next_threshold and a_mean > 0.0:
                step = (v - next_threshold) / a_mean
                if step > 0.0:
                    x += v * step - 0.5 * a_mean * step * step
                    t += step
                v = next_threshold
                ts.append(t)
                vs.append(v)
                xs.append(x)
                acc.append(-a)
                continue

            if dv >= v:
                if a_mean > 0.0:
                    x += v * v / (2.0 * a_mean)
                    t += v / a_mean
                ts.append(t)
                vs.append(0.0)
                xs.append(x)
                acc.append(-a)
                reached_rest = True
                break

            x += v * dt - 0.5 * a_mean * dt * dt
            v -= dv
            t += dt
            ts.append(t)
            vs.append(v)
            xs.append(x)
            acc.append(-a)

        if not reached_rest:
            raise RuntimeError("braking integration did not reach rest")

        return {"t": np.array(ts), "v": np.array(vs),
                "x": np.array(xs), "a": np.array(acc)}

    def _build_table(self):
        speeds = np.linspace(0.0, self.v_max_ms, self.TABLE_POINTS)
        dist = []
        time = []
        for v0 in speeds:
            r = self._integrate(v0)
            dist.append(float(r["x"][-1]))
            time.append(float(r["t"][-1]))
        self._table_v = speeds
        self._table_v2 = speeds * speeds
        self._table_d = np.array(dist)
        self._table_t = np.array(time)

    # ----------------------------------------------------------- interface

    def distance_to_stop(self, v0):
        # Interpolated against v-squared, not v. Stopping distance is dominated
        # by the v^2/2a term, so linear interpolation in v under-resolves the
        # curvature and left about 2 cm of error -- enough to show up as
        # platform overshoot in the segment solver. In v^2 the relationship is
        # very nearly straight, and the error drops by two orders of magnitude.
        v = min(max(float(v0), 0.0), self.v_max_ms)
        return float(np.interp(v * v, self._table_v2, self._table_d))

    def time_to_stop(self, v0):
        # Time is close to linear in v (v/a plus a fixed ramp penalty), so
        # interpolating against v is already appropriate here.
        v = min(max(float(v0), 0.0), self.v_max_ms)
        return float(np.interp(v, self._table_v, self._table_t))

    def idealised_distance_to_stop(self, v0):
        """Distance under an instant application of the low-speed rate.

        Useful only for quantifying what the speed dependence costs.
        """
        return max(float(v0), 0.0) ** 2 / (2.0 * self.rate_ms2(0.0))

    def jerk_penalty_distance(self, v0):
        return self.distance_to_stop(v0) - self.idealised_distance_to_stop(v0)

    def is_trapezoidal(self, v0):
        return float(v0) >= self.speed_threshold_for_full_braking

    def peak_deceleration(self, v0):
        return float(np.abs(self._integrate(v0)["a"]).max())

    def trajectory(self, v0, dt=0.01):
        """Sampled stop from v0, resampled onto a uniform grid of step dt.

        Resampling matters because the caller splices this onto the end of a
        fixed-step run, and a variable grid would leave a discontinuity in
        sample spacing at the join.
        """
        raw = self._integrate(v0)
        total = float(raw["t"][-1])
        if total <= 0.0:
            z = np.zeros(2)
            return {"t": z, "v": z, "x": z, "a": z}

        n = max(math.ceil(total / dt) + 1, 2)
        grid = np.linspace(0.0, total, n)
        out = {
            "t": grid,
            "v": np.maximum(np.interp(grid, raw["t"], raw["v"]), 0.0),
            "x": np.interp(grid, raw["t"], raw["x"]),
            "a": np.interp(grid, raw["t"], raw["a"]),
        }
        out["v"][-1] = 0.0
        out["a"][-1] = 0.0
        return out

    def __repr__(self):
        return ("SpeedDependentBrakeProfile("
                f"{self._a_kmh_s.max():.2f}-{self._a_kmh_s.min():.2f} km/h/s, "
                f"jerk={self.jerk})")
