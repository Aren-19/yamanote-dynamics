# Yamanote Line Train Dynamics Model

A first-principles physics simulation of the JR East Yamanote Line, compared
against the published timetable.

The simulation on its own is not the point. The point is the gap between what
the physics permits and what the railway actually schedules, and what that gap
can be made to measure.

## Result

Simulating all 30 interstation segments from published rolling stock
specifications gives a **minimum circuit time of 2,309 s**. The published
circuit time is **3,948 s** — a gap of 42%.

That gap is not a recovery-margin measurement. It is dominated by station
dwell, which JR East does not publish. Circuit time = run time + margin +
dwell is one equation in two unknowns, so margin and dwell are **not
separately identifiable** from public data.

Margin practice *is* documented (MLIT surveys every major Japanese operator);
dwell is not. Constraining margin to documented levels and solving for dwell
instead gives:

| Assumed margin | Implied mean dwell |
|---|---|
| 3% | 45–50 s |
| 5% | 43–48 s |
| 10% | 39–44 s |

**Implied mean station dwell is 39–50 s.** The estimate is well-conditioned:
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
.venv\Scripts\python.exe -m pip install -e ".[app]"
```

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
| `measure_dwell.py` | Live dwell measurement (needs an ODPT key) |

## Method

| Quantity | Expression |
|---|---|
| Equation of motion | `m_eff · dv/dt = F(v) − R_run(v) − R_grade − R_curve` |
| Effective mass | `m_eff = m(1 + λ)`, λ ≈ 0.08 |
| Running resistance | `R(v) = A + Bv + Cv²` |
| Tractive effort | `F(v) = min(F_max, P/v, μ(v)·m_adh·g)` |
| Adhesion | `μ(v) = 0.161 + 7.5/(v_kmh + 44)` (Curtius–Kniffler) |
| Jerk-limited stop | `T = v₀/a_max + a_max/j` |
| Energy balance | `E_traction = E_resistance + E_brake` |

Fixed-step RK4 with explicit control-phase switching; bisection for root
finding; closed-form braking. scipy is not a dependency.

The brake application point is found by intersecting the forward run curve
with a brake curve solved backwards from rest at the platform — a forward-only
solver overshoots.

### Numerical validation

| Check | Result |
|---|---|
| Final position error | 0.0 m on all 30 segments |
| Circuit time convergence | 8 ms across dt = 0.5 → 0.005 s |
| Energy balance residual | 2.5 × 10⁻⁴, first order in dt |

The energy balance is a real test, not a restatement: a segment starts and
ends at rest, so `E_traction = E_resistance + E_brake` must hold, and both
sides are computed independently. It initially failed at 7.3% and exposed a
genuine bug.

## Other results

**Energy.** 92% of traction work is dissipated in braking — a consequence of 30
stops in 34.5 km. Net consumption is 1.82 kWh per car-km, which sits *below*
the published 2–4 heavy-metro band. That exposes a conflict: matching JR East's
published 回生率 of 59.0% needs a higher regeneration receptivity (~0.76), while
the energy band needs a lower one (0.3–0.5). The two published anchors disagree
about the same unpublished parameter. The conflict is recorded, not resolved.

**Coasting.** The energy-minimal profile is accelerate–cruise–coast–brake, and
with the shape fixed the strategy is one parameter. Coasting beats simply
cruising slower by 3–4% at equal run time. Allowing 10% more run time cuts
energy by about a third — the shape of the Pareto front matters more than the
strategy along it.

**Gradient.** Under a terrain-derived profile known to overstate real track
gradient, circuit run time moves 0.46% while the worst single segment shifts
7.7 s. The flat-track assumption is safe for the circuit-level result and
unsafe for any per-segment claim.

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
- No signal or ATC restrictions modelled — not public, and the largest
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
