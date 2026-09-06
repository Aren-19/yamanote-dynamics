"""Bounding study: how much can the no-restrictions assumption cost?"""

from __future__ import annotations

from _env import ensure

ensure()

from tokyoline import build_model, load_segments, simulate_segment

traction, brake = build_model(load_factor=1.0)
LINE_SPEED_KMH = traction.spec.max_speed_ms * 3.6
DT = 0.1

segments = load_segments()
RESIDUAL_MARGIN_S = 598.0  # from the decomposition ladder, central dwell


def circuit(cap_kmh: float) -> float:
    return sum(
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=cap_kmh / 3.6, dt=DT).run_time_s
        for s in segments
    )


baseline = circuit(LINE_SPEED_KMH)

print()
print("Where a restriction could bind at all")
print("-" * 37)
print("  A speed limit costs nothing where the train would not have reached it.")
print("  Short segments never do: the train is still accelerating when it has to")
print("  start braking, so its peak is set by the distance, not by the limit.")
print()
print(f"  {'length':>8}  {'count':>5}  {'natural peak':>13}")
print("  " + "-" * 34)

lengths: dict[float, int] = {}
for s in segments:
    lengths[s.distance_m] = lengths.get(s.distance_m, 0) + 1

never_reach = 0
for length in sorted(lengths):
    peak = simulate_segment(traction, brake, length,
                            speed_limit_ms=LINE_SPEED_KMH / 3.6,
                            dt=0.05).max_speed_kmh
    if peak < LINE_SPEED_KMH - 0.1:
        never_reach += lengths[length]
    print(f"  {length:6.0f} m  {lengths[length]:5d}  {peak:10.1f} km/h")

print()
print(f"  {never_reach} of {len(segments)} segments never reach line speed unrestricted.")
print("  On those, any restriction at or above the natural peak is free.")

print()
print()
print("Cost of a uniform cap")
print("-" * 21)
print("  An upper bound on any set of per-segment restrictions no lower than the")
print("  cap, since run time falls strictly with the speed limit.")
print()
print(f"  {'cap':>10}  {'circuit run':>12}  {'vs 90 km/h':>11}  "
      f"{'share of the 598 s residual':>28}")
print("  " + "-" * 70)
for cap in (85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0):
    total = circuit(cap)
    delta = total - baseline
    print(f"  {cap:7.0f} km/h  {total:10.1f} s  {delta:+10.1f} s  "
          f"{delta / RESIDUAL_MARGIN_S:27.0%}")

print()
print()
print("Inverted: what would restrictions have to be to explain the residual?")
print("-" * 69)

lo, hi = 20.0, LINE_SPEED_KMH
for _ in range(24):
    mid = 0.5 * (lo + hi)
    if circuit(mid) - baseline > RESIDUAL_MARGIN_S:
        lo = mid
    else:
        hi = mid
implied_cap = 0.5 * (lo + hi)

loop_km = sum(s.distance_m for s in segments) / 1000.0
print(f"  Published line speed                  {LINE_SPEED_KMH:.0f} km/h")
print(f"  Cap needed to absorb the 598 s        {implied_cap:.0f} km/h")
print(f"  Shortfall against the published limit "
      f"{100 * (1 - implied_cap / LINE_SPEED_KMH):.0f}%, line-wide")
print()
print("  It has to be said plainly: at the circuit level a uniform restriction")
print("  is OBSERVATIONALLY EQUIVALENT to recovery margin. Both add time to the")
print("  same total, and the loop time cannot tell them apart. The moving")
print("  averages make that concrete:")
print()
for label, seconds in (
    (f"model at the {LINE_SPEED_KMH:.0f} km/h line speed", baseline),
    (f"model under the {implied_cap:.0f} km/h cap", circuit(implied_cap)),
    ("timetable, less 32 s dwell per stop", 3948.0 - 32.0 * len(segments)),
):
    print(f"    {label:<38}{loop_km / (seconds / 3600):5.1f} km/h")
print()
print("  So this does not refute the restriction explanation on its own. What")
print(f"  does is the published line speed: {LINE_SPEED_KMH:.0f} km/h is itself a "
      f"published figure, and an")
print(f"  effective line-wide limit "
      f"{100 * (1 - implied_cap / LINE_SPEED_KMH):.0f} percent below it would "
      f"contradict it. Curve")
print("  and turnout restrictions are also local by")
print("  nature -- they apply at the curve, not for 34.5 km -- so a line-wide")
print("  derate is not the shape real restrictions take.")
print()
print(f"  A line-wide 75 km/h cap, already severe, accounts for "
      f"{(circuit(75.0) - baseline) / RESIDUAL_MARGIN_S:.0%} of the residual.")
print("  That is the realistic order of the effect.")

print()
print()
print("Verdict")
print("-" * 7)
print("  Restrictions belong in the residual, not outside it. The residual was")
print("  already defined as everything left over -- added margin, driver")
print("  conservatism, signal restrictions -- so this does not move the")
print("  decomposition. What it adds is a size: on any plausible reading,")
print("  restrictions are a minority share of the 598 s, and it would take a")
print("  line-wide derate contradicting the published line speed for them to be")
print("  the whole of it.")
print()
print("  The dwell inference is unaffected either way. It fixes margin at")
print("  documented levels and solves for dwell, and restrictions sit inside")
print("  that margin term rather than beside it.")
print()
print("  What this does NOT do is measure real restrictions. None are published.")
print("  It bounds how much they could matter, which is a different and weaker")
print("  claim, and the only one the public record supports.")
