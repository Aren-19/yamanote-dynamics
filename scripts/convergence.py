"""How precise is the simulated run time, and what limits it?"""

from __future__ import annotations

from _env import ensure

ensure()

import numpy as np

from tokyoline import build_model, load_segments, simulate_segment

traction, brake = build_model(load_factor=1.0)
segments = load_segments()
v_limit = traction.spec.max_speed_ms

STEPS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

print("Loop total run time vs integration step")
print()
print("    dt (s)     loop run time (s)      change vs finer      worst platform err")
print("  " + "-" * 76)

results = []
for dt in STEPS:
    total = 0.0
    worst = 0.0
    for seg in segments:
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=v_limit, dt=dt)
        total += r.run_time_s
        worst = max(worst, abs(r.position_error_m))
    results.append((dt, total, worst))

reference = results[-1][1]
for i, (dt, total, worst) in enumerate(results):
    delta = total - results[i + 1][1] if i + 1 < len(results) else 0.0
    print(f"  {dt:8.3f}   {total:14.4f}       {delta:+12.5f} s      {worst:.2e} m")

print()
print(f"  reference (dt = {STEPS[-1]} s): {reference:.4f} s")
for dt, total, _ in results:
    err = total - reference
    print(f"  dt = {dt:6.3f} s  ->  error {err:+9.5f} s  "
          f"({100 * abs(err) / reference:.5f}% of loop run time)")

print()
print("  RK4 is 4th order, so halving dt should cut the error by roughly 16x")
print("  in the smooth regions. Phase-switch handling limits the observed order.")


print()
print()
print("Distance quantization: what the integrator cannot fix")
print("-" * 53)
print("  Published kilometrage is given to 0.1 km, so each station position is")
print("  known to within 50 m and the segment distances inherit that. Sampling")
print("  each position uniformly across its rounding interval gives the spread")
print("  the published data alone permits.")
print()

distances = np.array([seg.distance_m for seg in segments])
loop_m = distances.sum()
cumulative = np.concatenate([[0.0], np.cumsum(distances)])[:-1]

print("  cost of 50 m, by segment length:")
for d in (500.0, 1200.0, 2000.0):
    t_short = simulate_segment(traction, brake, d, speed_limit_ms=v_limit,
                               dt=0.05).run_time_s
    t_long = simulate_segment(traction, brake, d + 50.0, speed_limit_ms=v_limit,
                              dt=0.05).run_time_s
    print(f"    {d:6.0f} m segment  ->  {t_long - t_short:5.2f} s")
print()


# dt = 0.2 s: the step error is far below the spread being measured.
def loop_time(seg_distances: np.ndarray, dt: float = 0.2) -> float:
    return sum(
        simulate_segment(traction, brake, float(d), speed_limit_ms=v_limit,
                         dt=dt).run_time_s
        for d in seg_distances
    )


# The seed is fixed so the figure quoted in the README is reproducible.
rng = np.random.default_rng(20260905)
SAMPLES = 300
totals = np.empty(SAMPLES)
for i in range(SAMPLES):
    positions = cumulative + rng.uniform(-50.0, 50.0, size=cumulative.size)
    # Around a closed loop the errors largely cancel.
    perturbed = np.diff(np.concatenate([positions, [positions[0] + loop_m]]))
    totals[i] = loop_time(perturbed)

published = loop_time(distances)
print(f"  {SAMPLES} samples, published distances give {published:.1f} s")
print(f"    mean            {totals.mean():8.1f} s   ({totals.mean() - published:+.1f})")
print(f"    std             {totals.std():8.2f} s")
print(f"    5th to 95th     {np.percentile(totals, 5):8.1f} to "
      f"{np.percentile(totals, 95):.1f} s")
print(f"    full range      {totals.min():8.1f} to {totals.max():.1f} s")
print()
print("  The loop total is insensitive because it is closed: the rounding")
print("  errors are differences of positions that each appear twice, with")
print("  opposite sign. A single segment is not protected that way.")
print()

step_error = abs(results[2][1] - reference)  # dt = 0.1 s, the working step
print(f"  step size (dt = 0.1 s) contributes   {step_error:.5f} s")
print(f"  distance rounding contributes        {totals.std():.5f} s  (1 sigma)")
ratio = totals.std() / step_error if step_error else float("inf")
print(f"  input rounding is {ratio:,.0f}x the numerical error")
print()
print("  So the integrator is not what limits the result, and refining it")
print("  further would buy nothing. At the loop level both are negligible")
print("  against the 598 s residual margin. At the per-segment level 50 m is")
print("  worth 2 to 3 s, which is a second independent reason -- alongside the")
print("  minute-rounded timetable -- that no per-segment claim is supported.")
