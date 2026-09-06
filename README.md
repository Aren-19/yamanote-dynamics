# Yamanote Line Train Dynamics Model

[![tests](https://github.com/VolitantTrack70/yamanote-dynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/VolitantTrack70/yamanote-dynamics/actions/workflows/tests.yml)

A first-principles physics simulation of the JR East Yamanote Line, compared
against the published timetable.

The simulation on its own is not the point. The point is the gap between what
the physics permits and what the railway actually schedules, and what that gap
can be made to measure.

## Result

Simulating all 30 interstation segments from published rolling stock
specifications gives a **minimum circuit time of 2,309 s**. The published
circuit time is **3,948 s**, a gap of 42%.

That gap is not a recovery-margin measurement. It is dominated by station
dwell, which JR East does not publish. Circuit time = run time + margin +
dwell is one equation in two unknowns, so margin and dwell are **not
separately identifiable** from public data.

Margin practice *is* documented (MLIT surveys every major Japanese operator);
dwell is not. Constraining margin to documented levels and solving for dwell
instead gives:

| Assumed margin | Implied mean dwell |
|---|---|
| 3% | 45-50 s |
| 5% | 43-48 s |
| 10% | 39-44 s |

**Implied mean station dwell is 39-50 s.** The estimate is well-conditioned:
physics error enters multiplied by the run time (2,390 s) while dwell is
multiplied by 30 stations, so a 5% physics error moves it by about 4 s.

A mean dwell of 51.9 s or more would imply no recovery margin at all, which
documented JR East practice excludes. That is an upper bound requiring no
modelling assumption.

## Running it

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install --upgrade pip
```

```bash
.venv\Scripts\python.exe -m pip install -e ".[app]"
```

The pip upgrade is not optional on Python 3.10, which ships pip 21. Editable
installs of a project with no `setup.py` need PEP 660 support, added in pip 21.3;
older pip fails with *"editable mode currently requires a setuptools-based
build"*. Verified from a clean checkout: after upgrading, the install, the lint
and the full suite pass.

```bash
.venv\Scripts\python.exe launcher.py
```

On Windows, `run.bat` does the same by double-click. The launcher picks a free
port, waits until the server is actually serving, and opens a native desktop
window. `--browser` opens a browser tab instead.

Analysis scripts must use the virtual environment's interpreter:

```bash
.venv\Scripts\python.exe scripts\validate.py
```

| Script | Output |
|---|---|
| `smoke_check.py` | Physics sanity checks against published figures |
| `convergence.py` | Integrator step-size study |
| `validate.py` | Circuit decomposition and dwell inference |
| `fleet.py` | Cross-class comparison |
| `energy.py` | Energy accounting and balance check |
| `coasting.py` | Four-phase solver and Pareto front |
| `gradient.py` | Bounding study for the flat-track assumption |
| `restrictions.py` | Bounding study for the no-speed-restrictions assumption |
| `measure_dwell.py` | Live dwell measurement (needs an ODPT key) |

### Tests

```bash
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

```bash
.venv\Scripts\python.exe -m pytest
```

91% coverage. Most tests assert identities and bounds rather than pinning
numbers: energy conservation, exact platform arrival, monotonicity of
resistance and run time, the jerk penalty being independent of initial speed,
the adhesion ceiling never binding for this class. A separate group pins the
figures this README states, so they cannot drift out of date unnoticed, and
another enforces the data layer's provenance tags.

CI runs lint, the suite on Python 3.10 and 3.12 with an 80% coverage floor,
then the analysis scripts, which exercise more of the code than the unit tests
do.

## Method

| Quantity | Expression |
|---|---|
| Equation of motion | `m_eff · dv/dt = F(v) − R_run(v) − R_grade − R_curve` |
| Effective mass | `m_eff = m(1 + λ)`, λ ≈ 0.08 |
| Running resistance | `R(v) = A + Bv + Cv²` |
| Tractive effort | `F(v) = min(F_max, P/v, μ(v)·m_adh·g)` |
| Adhesion | `μ(v) = 0.161 + 7.5/(v_kmh + 44)` (Curtius-Kniffler) |
| Jerk-limited stop | `T = v₀/a_max + a_max/j` |
| Ramp-out (closed form) | `t = a/j`, `d = a³/6j²` |
| Energy balance | `E_traction = E_resistance + E_brake + E_grade` |

Fixed-step RK4 with explicit control-phase switching; bisection for root
finding; closed-form braking. scipy is not a dependency.

The brake application point is found by intersecting the forward run curve
with a brake curve solved backwards from rest at the platform: a forward-only
solver overshoots.

### Numerical validation

| Check | Result |
|---|---|
| Final position error | 0.0 m on all 30 segments |
| Circuit time convergence | 8 ms across dt = 0.5 → 0.005 s |
| Energy balance residual | 9.8 × 10⁻⁴ worst case, first order in dt |
| Distance rounding | ±0.8 s (1σ) on the circuit, ±2-3 s per segment |

The energy balance is a real test. Both sides are computed independently, so
it can fail, and did, at 7.3%, on a genuine bug. The gravitational term was
added after the identity turned out to fail by up to 104% on a graded segment
while holding exactly on level track, the only regime it had been tested in.

Precision is limited by the input, not the integrator. Published kilometrage
(営業キロ) is stated to 0.1 km, so each station position carries up to 50 m of
rounding. Sampling across that interval moves the circuit minimum by 0.83 s
(1σ), about 17,000× the step-size error at dt = 0.1 s. The closed loop absorbs
it, because each position appears in two segments with opposite sign. A single
segment has no cancelling partner, and 50 m is worth 2-3 s there: a second
reason, alongside the minute-rounded timetable, that no per-segment claim is
supported.

## Other results

**Energy.** 92% of traction work is dissipated in braking, which is what 30
stops in 34.5 km costs. Net consumption is 1.82 kWh per car-km, below the
published 2-4 heavy-metro band. That exposes a conflict: matching JR East's
回生率 of 59.0% needs a regeneration receptivity around 0.76, while the energy
band needs 0.3-0.5. Two published anchors disagree about the same unpublished
parameter. Recorded, not resolved.

**Coasting.** The energy-minimal profile is accelerate-cruise-coast-brake, so
with the shape fixed the strategy is one parameter. At equal run time coasting
beats simply cruising slower by 3-4%. Allowing 10% more run time cuts energy by
about a third: the shape of the Pareto front matters more than where you sit on
it.

**Gradient.** Under a terrain-derived profile known to overstate real track
gradient, circuit run time moves 0.46% while the worst single segment shifts
7.7 s. The flat-track assumption is safe for the circuit-level result and
unsafe for any per-segment claim.

**Speed restrictions.** The model runs the whole line at 90 km/h; real curve,
turnout and signal limits are not published. Run time falls strictly with the
speed limit, so a uniform cap bounds any set of per-segment restrictions above
it. A severe line-wide 75 km/h cap costs 85 s, 14% of the residual, and
absorbing the whole 598 s would take 51 km/h, 43% below the published line
speed. At circuit level a uniform restriction is indistinguishable from
recovery margin; the published line speed is what separates them, not the loop
time. Eleven of 30 segments never reach 90 km/h anyway, the shortest peaking at
67, so a restriction there is free.

**Passenger load.** Every headline figure assumes 100% of published capacity,
while the loop time it is compared against averages all services, most of them
far lighter. Empty to the MLIT-measured 125% peak moves the standard run time
by 105 s and implied dwell by 3.7 s, from 48.0 s to 50.2 s. So the bias runs
against the result: a realistic daily mean load pushes implied dwell up, toward
the 51.9 s zero-margin bound, not down.

## Data sources

| Quantity | Source |
|---|---|
| Station sequence, operating distance | [Yamanote Line, Wikipedia](https://en.wikipedia.org/wiki/Yamanote_Line) |
| Rolling stock specifications | Japanese Wikipedia, per class |
| Margin and dwell practice | [MLIT 運行計画 survey (PDF)](https://www.mlit.go.jp/kisha/kisha05/08/080722_3/01.pdf) |
| Station boardings FY2024 | [JR East 各駅の乗車人員](https://www.jreast.co.jp/company/data/passenger/) |
| Peak congestion | MLIT 混雑率 survey FY2023 |
| Station coordinates | [OpenStreetMap via Overpass](https://overpass-api.de/api/interpreter) |
| Station elevations | [Open-Elevation](https://api.open-elevation.com/api/v1/lookup) |
| Real-time positions | [ODPT](https://api.odpt.org/api/v4/odpt:Train) |

Every parameter in `data/` is tagged `published`, `derived`, `modelled`,
`bounded` or `unknown`, and the tag is carried through the code. The
application's Sources tab lists every equation with the module implementing it.

## Assumptions and limitations

- Level track. Known false on one segment; bounded above.
- No signal or ATC restrictions modelled: not public, and the largest
  unmodelled contributor to run time.
- Dry rail, no wind. Published deceleration treated as a net rate.
- One-hour power rating rather than short-term overload, which makes the model
  slower than the real machine in the mid-speed region.
- **Dwell is inferred, not measured.** JR East real-time data requires an ODPT
  consumer key; the unauthenticated mirror carries other operators only.
- **Davis coefficients are modelled.** No published set exists for this class;
  they are estimated from mass, car count and frontal area, and are the largest
  single uncertainty in the physics core.
- Per-segment scheduled times are published only to the minute, so ±30 s
  exceeds the effect being measured. No per-segment timetable is committed.

## Licence

MIT. See `LICENSE`.
