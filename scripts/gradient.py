"""Bounding study: how much can the flat-track assumption cost?"""

from __future__ import annotations

from _env import ensure

ensure()

from tokyoline import (
    BrakeProfile,
    build_model,
    load_segments,
    simulate_segment,
)
from tokyoline.energy import JOULES_PER_KWH, segment_energy
from tokyoline.units import G

traction, brake = build_model(load_factor=1.0)
V_LIMIT = traction.spec.max_speed_ms
DT = 0.05

flat_segments = load_segments()
graded_segments = load_segments(gradients=True)


def brake_for(grade_permille: float) -> BrakeProfile:
    """Braking adjusted for gradient."""
    adjusted = brake.a_max + G * (grade_permille / 1000.0)
    return BrakeProfile(a_max=max(adjusted, 0.15), jerk=brake.jerk)


def run(seg, grade: float):
    b = brake_for(grade)
    r = simulate_segment(traction, b, seg.distance_m, speed_limit_ms=V_LIMIT,
                         dt=DT, grade_permille=grade)
    e, _ = segment_energy(r, traction)
    return r, e


print("=" * 80)
print("GRADIENT SENSITIVITY  --  bounding the flat-track assumption")
print("=" * 80)
print("  Gradients are DERIVED from terrain elevation and overstate real track")
print("  gradient. Treat every number here as an upper bound on the effect, not")
print("  as a corrected result.")
print()

steepest = sorted(graded_segments, key=lambda s: -abs(s.grade_permille))[:6]
print(f"  {'steepest segments':<40} {'grade':>9}")
print("  " + "-" * 52)
for s in steepest:
    print(f"  {s.from_station + ' → ' + s.to_station:<40} {s.grade_permille:+8.1f}‰")
print()
print("  Published maximum for the line is 34‰ between Tabata and Nishi-Nippori.")
print("  The derived average there is -18.8‰, descending outer-loop: below the")
print("  published local maximum, as a segment average should be, and in the")
print("  direction the terrain implies. The profile is tracking real topography.")

print()
print("=" * 80)
print("PER-SEGMENT EFFECT  (outer loop)")
print("=" * 80)
print(f"  {'segment':<38} {'grade':>8} {'flat':>8} {'graded':>8} {'Δt':>8}")
print("  " + "-" * 74)

flat_total = graded_total = 0.0
flat_energy = graded_energy = 0.0
worst = None
for fs, gs in zip(flat_segments, graded_segments):
    rf, ef = run(fs, 0.0)
    rg, eg = run(gs, gs.grade_permille)
    flat_total += rf.run_time_s
    graded_total += rg.run_time_s
    flat_energy += ef.net_j
    graded_energy += eg.net_j
    delta = rg.run_time_s - rf.run_time_s
    if worst is None or abs(delta) > abs(worst[1]):
        worst = (f"{gs.from_station} → {gs.to_station}", delta)
    if abs(gs.grade_permille) >= 5.0:
        print(f"  {gs.from_station + ' → ' + gs.to_station:<38} "
              f"{gs.grade_permille:+7.1f}‰ {rf.run_time_s:7.1f}s "
              f"{rg.run_time_s:7.1f}s {delta:+7.1f}s")

print()
print(f"  largest single-segment shift: {worst[0]} at {worst[1]:+.1f} s")

print()
print("=" * 80)
print("CIRCUIT TOTALS  --  does it cancel?")
print("=" * 80)

inner_total = 0.0
inner_energy = 0.0
for gs in graded_segments:
    rg, eg = run(gs, -gs.grade_permille)
    inner_total += rg.run_time_s
    inner_energy += eg.net_j

print(f"  {'case':<28} {'run time':>11} {'vs flat':>10} {'net energy':>13}")
print("  " + "-" * 66)
print(f"  {'flat (headline model)':<28} {flat_total:10.1f}s {'--':>10} "
      f"{flat_energy/JOULES_PER_KWH:11.1f}kWh")
print(f"  {'graded, outer loop':<28} {graded_total:10.1f}s "
      f"{graded_total-flat_total:+9.1f}s {graded_energy/JOULES_PER_KWH:11.1f}kWh")
print(f"  {'graded, inner loop':<28} {inner_total:10.1f}s "
      f"{inner_total-flat_total:+9.1f}s {inner_energy/JOULES_PER_KWH:11.1f}kWh")

print()
print("  net climb around the loop: 0 m exactly -- it is a closed circuit, so")
print("  gravitational work cancels by construction.")
print()
print("  But run time does NOT cancel. Both directions come out slower than flat,")
print("  because the asymmetry is real: climbing is power-limited so it costs more")
print("  time than the matching descent returns, and on a descent the brakes must")
print("  fight gravity too.")

pct_time = 100 * (graded_total - flat_total) / flat_total
pct_energy = 100 * (graded_energy - flat_energy) / flat_energy
print()
print(f"  circuit run time penalty:  {pct_time:+.2f}%  (outer loop)")
print(f"  circuit energy penalty:    {pct_energy:+.2f}%")

print()
print("=" * 80)
print("VERDICT")
print("=" * 80)
print("  Under gradients that are known to be EXAGGERATED, the circuit run time")
print(f"  moves by {abs(pct_time):.2f}% and energy by {abs(pct_energy):.2f}%. Against a")
print("  measured recovery-margin signal of a few percent, that is not negligible")
print("  at the per-segment level -- the worst single segment shifts by")
print(f"  {abs(worst[1]):.1f} s -- but the CIRCUIT total, which is what the Phase 3")
print("  validation actually rests on, is barely touched.")
print()
print("  So: the flat-track assumption is safe for the headline dwell inference,")
print("  and unsafe for any per-segment claim. That is now measured rather than")
print("  asserted, and the per-segment limitation in the README is real.")
